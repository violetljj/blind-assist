"""Verify and seal the three completed N3 native-dynamics source traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from dtr_carla_c2_rich_scene import canonical_json_bytes, sha256_bytes, write_json_atomic
from dtr_carla_n3_multitown_native_dynamics import SCENE_ORDER, SUITE_SCHEMA, sha256_file


SOURCE_FILES = (
    "behavior_trace.jsonl",
    "actor_manifest.json",
    "frozen_plan.json",
    "event_receipts.json",
    "result.json",
)
RESULT_STATUS = "DTR_CARLA_N3_MULTITOWN_NATIVE_DYNAMICS_MATERIALIZED"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(bool(line.strip()), f"blank trace row at {path}:{line_number}")
            count += 1
    return count


def seal_suite(suite_manifest_path: Path, traces_root: Path, output_root: Path) -> dict[str, Any]:
    suite_manifest_path = suite_manifest_path.resolve(strict=True)
    traces_root = traces_root.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    suite = read_json(suite_manifest_path)
    _require(suite.get("schema_version") == SUITE_SCHEMA, "suite schema differs")
    _require(tuple(suite.get("scene_order", [])) == SCENE_ORDER, "scene order differs")
    scene_receipts: list[dict[str, Any]] = []
    total_frames = 0
    total_actors = 0
    for expected_ordinal, scene in enumerate(suite["scenes"]):
        scene_id = str(scene["scene_id"])
        _require(int(scene["ordinal"]) == expected_ordinal, "scene ordinal differs")
        source_root = (traces_root / scene_id).resolve(strict=True)
        try:
            source_root.relative_to(traces_root)
        except ValueError as exc:
            raise ValueError(f"source root escapes trace root: {source_root}") from exc
        files: dict[str, dict[str, Any]] = {}
        for name in SOURCE_FILES:
            path = source_root / name
            _require(path.is_file(), f"missing N3 source file: {path}")
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        result = read_json(source_root / "result.json")
        plan = read_json(source_root / "frozen_plan.json")
        events = read_json(source_root / "event_receipts.json")
        _require(
            result.get("status") == "DTR_CARLA_N1_NATURAL_DYNAMICS_MATERIALIZED",
            f"source status differs for {scene_id}",
        )
        failed_checks = [
            name for name, value in result.get("checks", {}).items() if value is not True
        ]
        _require(not failed_checks, f"source checks failed for {scene_id}: {failed_checks}")
        _require(result.get("map") == scene.get("map"), f"map differs for {scene_id}")
        _require(plan.get("plan_id") == scene.get("plan_id"), f"plan id differs for {scene_id}")
        _require(
            plan.get("plan_fingerprint_sha256") == scene.get("plan_fingerprint_sha256"),
            f"plan fingerprint differs for {scene_id}",
        )
        _require(
            files["frozen_plan.json"]["sha256"] == scene.get("plan_file_sha256"),
            f"plan file hash differs for {scene_id}",
        )
        expected_frames = int(scene["expected_trace_frames"])
        trace_frames = count_jsonl(source_root / "behavior_trace.jsonl")
        _require(trace_frames == expected_frames, f"trace denominator differs for {scene_id}")
        _require(
            int(result.get("vehicle_count", -1)) + int(result.get("walker_count", -1))
            == int(scene["expected_actor_count"]),
            f"actor denominator differs for {scene_id}",
        )
        _require(
            set(result.get("required_native_vehicle_classes", []))
            == {"heavy_vehicle", "two_wheeler"}
            and set(result.get("moving_native_vehicle_classes", []))
            >= {"heavy_vehicle", "two_wheeler"},
            f"native class motion differs for {scene_id}",
        )
        _require(
            len(events.get("tail_events", [])) == 4
            and all(bool(value.get("observed_effect")) for value in events["tail_events"]),
            f"tail-event effects differ for {scene_id}",
        )
        total_frames += trace_frames
        total_actors += int(scene["expected_actor_count"])
        scene_receipts.append(
            {
                "ordinal": expected_ordinal,
                "scene_id": scene_id,
                "map": scene["map"],
                "scenario_class": scene["scenario_class"],
                "source_root": f"source-traces/{scene_id}",
                "plan_id": scene["plan_id"],
                "plan_fingerprint_sha256": scene["plan_fingerprint_sha256"],
                "expected_trace_frames": expected_frames,
                "expected_actor_count": int(scene["expected_actor_count"]),
                "route_view_distance_m": float(scene["route_view_distance_m"]),
                "maximum_event_view_range_m": float(
                    scene["maximum_event_view_range_m"]
                ),
                "maximum_wearer_speed_mps": float(scene["maximum_wearer_speed_mps"]),
                "event_actor_bindings": scene["event_actor_bindings"],
                "files": files,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "dtr-carla-n3-multitown-native-source-receipt-v1",
        "authority": "FROZEN_N3_MULTITOWN_NATIVE_SOURCE_SUITE_VERIFIED",
        "suite_manifest_sha256": sha256_file(suite_manifest_path),
        "scene_count": len(scene_receipts),
        "scene_order": list(SCENE_ORDER),
        "scenes": scene_receipts,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    receipt_path = output_root / "source_suite_receipt.json"
    write_json_atomic(receipt_path, receipt)
    checks = {
        "exact_three_scene_roster_verified": len(scene_receipts) == 3,
        "town01_town04_town05_maps_verified": tuple(
            value["map"] for value in scene_receipts
        )
        == ("Carla/Maps/Town01", "Carla/Maps/Town04", "Carla/Maps/Town05"),
        "all_native_trace_denominators_exact": total_frames
        == sum(int(value["expected_trace_frames"]) for value in scene_receipts),
        "heavy_and_two_wheeler_native_motion_in_each_scene": True,
        "all_twelve_tail_event_effects_observed": True,
        "frozen_source_receipt_materialized": receipt_path.is_file(),
    }
    result = {
        "schema_version": "dtr-carla-n3-multitown-native-dynamics-result-v1",
        "status": RESULT_STATUS if all(checks.values()) else "DTR_CARLA_N3_GATE_NOT_MET",
        "checks": checks,
        "scene_count": len(scene_receipts),
        "trace_frame_count": total_frames,
        "actor_count": total_actors,
        "tail_event_count": 4 * len(scene_receipts),
        "suite_manifest_sha256": sha256_file(suite_manifest_path),
        "source_suite_receipt_sha256": sha256_file(receipt_path),
        "source_suite_receipt_authority_sha256": receipt["receipt_sha256"],
        "claim_boundary": [
            "This is a three-map synthetic Development source materialization.",
            "Native CARLA controllers and authored interventions do not establish a natural traffic distribution, source-disjoint confirmation, real-world benefit, or safety.",
        ],
    }
    write_json_atomic(output_root / "result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = seal_suite(args.suite_manifest, args.traces_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == RESULT_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())

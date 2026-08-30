"""Join the three N4 map shards into one frozen four-modal replay result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from dtr_carla_c2_rich_scene import write_json_atomic
from dtr_carla_n3_multitown_native_dynamics import SCENE_ORDER, sha256_file
from build_dtr_carla_n4_multitown_frozen_replay import BUNDLE_SCHEMA


RESULT_STATUS = "DTR_CARLA_N4_MULTITOWN_FROZEN_FOUR_MODAL_REPLAY_COMPLETE"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def join_replay(bundle_path: Path, child_root: Path, output_root: Path) -> dict[str, Any]:
    bundle_path = bundle_path.resolve(strict=True)
    child_root = child_root.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    bundle = read_json(bundle_path)
    _require(bundle.get("schema_version") == BUNDLE_SCHEMA, "N4 bundle schema differs")
    _require(tuple(bundle.get("scene_order", [])) == SCENE_ORDER, "N4 scene order differs")
    _require(int(bundle.get("replay_invocation_budget", -1)) == 1, "replay budget differs")
    child_summaries: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    total_frames = 0
    total_payloads = 0
    total_events = 0
    for expected_ordinal, child in enumerate(bundle["children"]):
        scene_id = str(child["scene_id"])
        _require(int(child["ordinal"]) == expected_ordinal, "child ordinal differs")
        protocol_path = bundle_path.parent / str(child["protocol_path"])
        route_audit_path = bundle_path.parent / str(child["route_audit_path"])
        _require(
            sha256_file(protocol_path) == str(child["protocol_sha256"]),
            f"protocol hash differs for {scene_id}",
        )
        _require(
            sha256_file(route_audit_path) == str(child["route_audit_sha256"]),
            f"route audit hash differs for {scene_id}",
        )
        route_audit = read_json(route_audit_path)
        _require(
            all(value is True for value in route_audit.get("checks", {}).values()),
            f"route audit failed for {scene_id}",
        )
        evidence_root = child_root / scene_id
        result_path = evidence_root / "result.json"
        frozen_protocol_path = evidence_root / "frozen_protocol.json"
        sealed_model_path = evidence_root / "sealed_model_manifest.json"
        sealed_evidence_path = evidence_root / "sealed_evidence_manifest.json"
        for path in (
            result_path,
            frozen_protocol_path,
            sealed_model_path,
            sealed_evidence_path,
        ):
            _require(path.is_file(), f"missing child evidence file: {path}")
        result = read_json(result_path)
        failed_checks = [
            name for name, value in result.get("checks", {}).items() if value is not True
        ]
        _require(
            result.get("status") == "DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_COMPLETE",
            f"child replay status differs for {scene_id}",
        )
        _require(not failed_checks, f"child checks failed for {scene_id}: {failed_checks}")
        _require(result.get("map") == child.get("map"), f"child map differs for {scene_id}")
        _require(
            result.get("protocol_sha256") == child.get("protocol_sha256")
            and sha256_file(frozen_protocol_path) == child.get("protocol_sha256"),
            f"frozen protocol differs for {scene_id}",
        )
        visibility = list(result.get("tail_event_visibility", []))
        _require(len(visibility) == 4, f"tail-event visibility roster differs for {scene_id}")
        _require(
            all(
                int(value.get("visible_frames", 0)) >= 1
                and int(value.get("maximum_instance_pixels", 0)) >= 16
                for value in visibility
            ),
            f"tail event did not enter wearable view for {scene_id}",
        )
        frames = int(result["frames"])
        payloads = int(result["total_sensor_payloads"])
        _require(frames == int(child["expected_trace_frames"]), "frame count differs")
        _require(payloads == frames * 4, "four-modal payload count differs")
        total_frames += frames
        total_payloads += payloads
        total_events += len(visibility)
        child_summaries.append(
            {
                "ordinal": expected_ordinal,
                "scene_id": scene_id,
                "map": child["map"],
                "scenario_class": child["scenario_class"],
                "frames": frames,
                "sensor_payloads": payloads,
                "tail_event_visibility": visibility,
                "alignment_receipt_sha256": result["alignment_receipt_sha256"],
                "sealed_model_manifest_sha256": result[
                    "sealed_model_manifest_sha256"
                ],
                "sealed_evidence_manifest_sha256": result[
                    "sealed_evidence_manifest_sha256"
                ],
                "contact_sheet": result["contact_sheet"],
            }
        )
        for label, path in (
            ("result", result_path),
            ("frozen_protocol", frozen_protocol_path),
            ("sealed_model_manifest", sealed_model_path),
            ("sealed_evidence_manifest", sealed_evidence_path),
            ("route_audit", route_audit_path),
        ):
            manifest_entries.append(
                {
                    "scene_id": scene_id,
                    "role": label,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest_entries.sort(key=lambda value: (value["scene_id"], value["role"]))
    sealed_replay_manifest_path = output_root / "sealed_replay_manifest.json"
    write_json_atomic(sealed_replay_manifest_path, manifest_entries)
    checks = {
        "single_outer_replay_invocation_consumed": True,
        "exact_three_map_shards_complete": len(child_summaries) == 3,
        "town01_town04_town05_order_verified": tuple(
            value["map"] for value in child_summaries
        )
        == ("Carla/Maps/Town01", "Carla/Maps/Town04", "Carla/Maps/Town05"),
        "all_four_modal_payload_denominators_exact": total_payloads == total_frames * 4,
        "all_twelve_long_tail_events_enter_wearable_view": total_events == 12,
        "all_child_same_world_frame_alignment_gates_passed": all(
            bool(value["alignment_receipt_sha256"]) for value in child_summaries
        ),
        "sealed_replay_manifest_materialized": sealed_replay_manifest_path.is_file(),
    }
    result = {
        "schema_version": "dtr-carla-n4-multitown-frozen-four-modal-replay-result-v1",
        "status": RESULT_STATUS if all(checks.values()) else "DTR_CARLA_N4_REPLAY_NOT_EVALUABLE",
        "checks": checks,
        "replay_invocation_count": 1,
        "map_shard_count": len(child_summaries),
        "scene_order": list(SCENE_ORDER),
        "frames": total_frames,
        "sensor_modalities": ["instance", "wearable", "depth", "witness"],
        "sensor_payloads": total_payloads,
        "tail_event_count": total_events,
        "children": child_summaries,
        "bundle_sha256": sha256_file(bundle_path),
        "sealed_replay_manifest_sha256": sha256_file(sealed_replay_manifest_path),
        "claim_boundary": [
            "This is one frozen three-map synthetic Development source replay, implemented as three map shards because one CARLA world loads one map.",
            "Instance visibility establishes only that each authored event occupied the wearable camera field during replay.",
            "No obstacle-algorithm, source-disjoint, natural-distribution, real-world, product-benefit, or safety claim follows.",
        ],
    }
    write_json_atomic(output_root / "result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--child-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = join_replay(args.bundle, args.child_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == RESULT_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Pure validation and evidence helpers for N1 frozen-trace C2 replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from dtr_carla_c2_rich_scene import (
    camera_intrinsics,
    canonical_json_bytes,
    forbidden_model_paths,
    sha256_bytes,
    sha256_file,
    validate_model_record,
    write_json_atomic,
)


EXPERIMENT_ID = "DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_V1"
TRACE_SCHEMA = "dtr-carla-n1-behavior-trace-frame-v1"
SENSOR_ORDER = ("instance", "wearable", "depth", "witness")
SOURCE_FILES = (
    "behavior_trace.jsonl",
    "actor_manifest.json",
    "frozen_plan.json",
    "event_receipts.json",
    "result.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            values.append(value)
    return values


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_protocol(protocol: dict[str, Any]) -> None:
    _require(protocol.get("experiment_id") == EXPERIMENT_ID, "unexpected experiment")
    _require(
        protocol.get("schema_version")
        == "dtr-carla-n2-frozen-trace-replay-protocol-v1",
        "unexpected protocol schema",
    )
    source = protocol["source"]
    _require(tuple(source["files"]) == SOURCE_FILES, "source file roster/order differs")
    for name, digest in source["files"].items():
        _require(name in SOURCE_FILES, f"unknown source file: {name}")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and digest == digest.upper(),
            f"invalid frozen SHA-256 for {name}",
        )
    _require(int(source["expected_trace_frames"]) > 0, "trace frame count must be positive")
    _require(int(source["expected_actor_count"]) > 0, "actor count must be positive")

    environment = protocol["environment"]
    _require(environment["carla_version"] == "0.9.16", "CARLA version differs")
    _require(
        environment["map"] == "Carla/Maps/Town10HD_Opt", "CARLA map differs"
    )
    fixed_delta = _finite_number(
        environment["fixed_delta_seconds"], "environment.fixed_delta_seconds"
    )
    _require(abs(fixed_delta - 0.05) <= 1e-12, "fixed delta must remain 0.05 s")

    capture = protocol["capture"]
    _require(tuple(capture["sensor_order"]) == SENSOR_ORDER, "sensor order differs")
    _require(capture["same_world_frame"] is True, "sensors must share a world frame")
    _require(capture["resolution"] == [1280, 720], "capture must remain 1280x720")
    _require(
        capture["witness_transform_authority"] == "source_actor_manifest_camera",
        "witness must bind the source camera transform",
    )
    expected_k = camera_intrinsics(1280, 720, float(capture["fov_degrees"]))
    calibration = capture["camera_calibration"]
    _require(
        [float(value) for value in calibration["principal_point"]]
        == [expected_k[0][2], expected_k[1][2]],
        "principal point differs",
    )
    _require(
        all(
            abs(float(actual) - float(expected)) <= 1e-9
            for actual, expected in zip(
                calibration["focal_length_pixels"],
                [expected_k[0][0], expected_k[1][1]],
                strict=True,
            )
        ),
        "focal length differs",
    )
    _require(
        protocol["model_contract"]["include_current_actors"] is False,
        "model contract must exclude actors",
    )
    _require(
        protocol["episode"]["issued_plan_authority"] == "NO_PLAN",
        "N2 must not invent a navigation plan",
    )


def actor_roster(actor_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = actor_manifest.get("actors")
    if not isinstance(values, list):
        raise ValueError("actor manifest actors must be a list")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("actor manifest entry must be an object")
        actor_id = str(value.get("actor_id", ""))
        if not actor_id or actor_id in result:
            raise ValueError(f"invalid or duplicate actor id: {actor_id}")
        result[actor_id] = value
    return result


def validate_trace_rows(
    rows: list[dict[str, Any]],
    actor_manifest: dict[str, Any],
    *,
    expected_frames: int,
    expected_actors: int,
    fixed_delta_seconds: float,
) -> dict[str, Any]:
    _require(len(rows) == expected_frames, "trace frame denominator differs")
    roster = actor_roster(actor_manifest)
    _require(len(roster) == expected_actors, "actor manifest denominator differs")
    roster_ids = set(roster)
    source_frames: list[int] = []
    actor_type_ids: dict[str, str] = {}
    actor_kinds: dict[str, str] = {}
    for sample_index, row in enumerate(rows):
        _require(row.get("schema_version") == TRACE_SCHEMA, "trace schema differs")
        _require(int(row.get("sample_index", -1)) == sample_index, "sample index gap")
        expected_time = sample_index * fixed_delta_seconds
        _require(
            abs(_finite_number(row.get("time_s"), "trace.time_s") - expected_time)
            <= 1e-8,
            f"logical time differs at sample {sample_index}",
        )
        source_frames.append(int(row["world_frame"]))
        actors = row.get("actors")
        _require(isinstance(actors, dict), "trace actors must be an object")
        _require(set(actors) == roster_ids, f"actor roster differs at {sample_index}")
        for actor_id in sorted(roster_ids):
            state = actors[actor_id]
            _require(state.get("actor_id") == actor_id, "state actor id differs")
            type_id = str(state.get("type_id", ""))
            kind = str(state.get("kind", ""))
            _require(
                type_id == str(roster[actor_id]["blueprint_id"]),
                f"blueprint differs for {actor_id}",
            )
            _require(kind == str(roster[actor_id]["kind"]), f"kind differs for {actor_id}")
            actor_type_ids.setdefault(actor_id, type_id)
            actor_kinds.setdefault(actor_id, kind)
            _require(actor_type_ids[actor_id] == type_id, "blueprint changes within trace")
            _require(actor_kinds[actor_id] == kind, "actor kind changes within trace")
            transform = state.get("transform")
            _require(isinstance(transform, dict), f"missing transform for {actor_id}")
            for key in ("x", "y", "z", "pitch", "yaw", "roll"):
                _finite_number(transform.get(key), f"{actor_id}.transform.{key}")
    _require(
        all(right == left + 1 for left, right in zip(source_frames, source_frames[1:])),
        "source CARLA frames are not contiguous",
    )
    return {
        "frames": len(rows),
        "actors": len(roster),
        "source_world_frame_first": source_frames[0],
        "source_world_frame_last": source_frames[-1],
        "actor_ids": sorted(roster),
    }


def verify_source_bundle(protocol: dict[str, Any], source_root: Path) -> dict[str, Any]:
    validate_protocol(protocol)
    root = source_root.resolve(strict=True)
    verified_files: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source file: {path}")
        digest = sha256_file(path)
        expected = str(protocol["source"]["files"][name])
        if digest != expected:
            raise ValueError(f"frozen source SHA-256 differs for {name}")
        verified_files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }

    result = read_json(root / "result.json")
    manifest = read_json(root / "actor_manifest.json")
    plan = read_json(root / "frozen_plan.json")
    events = read_json(root / "event_receipts.json")
    rows = read_jsonl(root / "behavior_trace.jsonl")
    source = protocol["source"]
    environment = protocol["environment"]
    _require(result["status"] == source["required_result_status"], "source status differs")
    _require(result["plan_id"] == source["plan_id"], "result plan id differs")
    _require(
        result["plan_fingerprint_sha256"] == source["plan_fingerprint_sha256"],
        "result plan fingerprint differs",
    )
    _require(plan["plan_id"] == source["plan_id"], "frozen plan id differs")
    _require(
        plan["plan_fingerprint_sha256"] == source["plan_fingerprint_sha256"],
        "frozen plan fingerprint differs",
    )
    _require(manifest["map"] == environment["map"], "manifest map differs")
    _require(result["map"] == environment["map"], "result map differs")
    _require(
        abs(float(manifest["fixed_delta_seconds"]) - float(environment["fixed_delta_seconds"]))
        <= 1e-12,
        "manifest fixed delta differs",
    )
    _require(events["plan_id"] == source["plan_id"], "event plan id differs")
    _require(len(events["tail_events"]) == 4, "tail-event denominator differs")
    _require(
        {str(value["type"]) for value in events["tail_events"]}
        == {"occluded_jaywalk", "sudden_brake", "reverse_pullout", "door_open"},
        "tail-event roster differs",
    )
    trace_summary = validate_trace_rows(
        rows,
        manifest,
        expected_frames=int(source["expected_trace_frames"]),
        expected_actors=int(source["expected_actor_count"]),
        fixed_delta_seconds=float(environment["fixed_delta_seconds"]),
    )
    receipt: dict[str, Any] = {
        "schema_version": "dtr-carla-n2-source-bundle-receipt-v1",
        "experiment_id": EXPERIMENT_ID,
        "authority": "FROZEN_N1_SOURCE_BUNDLE_VERIFIED",
        "plan_id": source["plan_id"],
        "plan_fingerprint_sha256": source["plan_fingerprint_sha256"],
        "files": verified_files,
        "trace": trace_summary,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return {
        "root": root,
        "result": result,
        "actor_manifest": manifest,
        "plan": plan,
        "events": events,
        "rows": rows,
        "receipt": receipt,
    }


def active_tail_event_ids(events: dict[str, Any], time_s: float) -> list[str]:
    return sorted(
        str(value["event_id"])
        for value in events["tail_events"]
        if float(value["applied_time_s"]) - 1e-9
        <= time_s
        < float(value["ended_time_s"]) - 1e-9
    )


def door_event(events: dict[str, Any]) -> dict[str, Any]:
    matches = [value for value in events["tail_events"] if value["type"] == "door_open"]
    if len(matches) != 1:
        raise ValueError("expected exactly one frozen door-open event")
    return matches[0]


def build_alignment_receipt(
    source_receipt_sha256: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    projection: list[dict[str, Any]] = []
    replay_frames: list[int] = []
    for expected_index, row in enumerate(rows):
        if int(row["sample_index"]) != expected_index:
            raise ValueError("alignment sample indices differ")
        sensor_frames = {str(key): int(value) for key, value in row["sensor_world_frames"].items()}
        if tuple(sensor_frames) != SENSOR_ORDER:
            raise ValueError("alignment sensor roster/order differs")
        if len(set(sensor_frames.values())) != 1:
            raise ValueError(f"sensor world frames differ at {expected_index}")
        replay_frame = next(iter(sensor_frames.values()))
        if int(row["replay_world_frame"]) != replay_frame:
            raise ValueError("replay world frame differs from sensors")
        replay_frames.append(replay_frame)
        projection.append(
            {
                "sample_index": expected_index,
                "time_s": round(float(row["time_s"]), 8),
                "source_world_frame": int(row["source_world_frame"]),
                "replay_world_frame": replay_frame,
            }
        )
    if any(right != left + 1 for left, right in zip(replay_frames, replay_frames[1:])):
        raise ValueError("replay world frames are not contiguous")
    receipt: dict[str, Any] = {
        "schema_version": "dtr-carla-n2-four-modal-alignment-receipt-v1",
        "experiment_id": EXPERIMENT_ID,
        "authority": "SAME_WORLD_FRAME_FOUR_MODAL_REPLAY_VERIFIED",
        "source_bundle_receipt_sha256": source_receipt_sha256,
        "sensor_order": list(SENSOR_ORDER),
        "frames": len(projection),
        "replay_world_frame_first": replay_frames[0],
        "replay_world_frame_last": replay_frames[-1],
        "alignment_projection_sha256": sha256_bytes(canonical_json_bytes(projection)),
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def safe_relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def seal_tree(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": safe_relative(path, root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    ]


def scan_model_tree(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("*.json*")):
        values: Iterable[Any]
        if path.suffix == ".jsonl":
            values = read_jsonl(path)
        else:
            values = [read_json(path)]
        for index, value in enumerate(values):
            for forbidden in forbidden_model_paths(value):
                failures.append(f"{safe_relative(path, root)}[{index}]{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = read_json(args.protocol.resolve(strict=True))
    bundle = verify_source_bundle(protocol, args.source_root)
    if args.receipt is not None:
        write_json_atomic(args.receipt.resolve(), bundle["receipt"])
    print(
        json.dumps(
            {
                "status": bundle["receipt"]["authority"],
                "receipt_sha256": bundle["receipt"]["receipt_sha256"],
                "frames": len(bundle["rows"]),
                "actors": len(actor_roster(bundle["actor_manifest"])),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

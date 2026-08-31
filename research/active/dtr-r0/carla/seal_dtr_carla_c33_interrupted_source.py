"""Seal the interrupted C33 source as terminal in-doubt evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL_SHA256 = (
    "421CA0C5B5518B87E2AA55679561E99F03BCA45CE5CE6AD7255DCB1B94446EF5"
)
TERMINAL_STATUS = "DTR_CARLA_C33_SOURCE_NOT_EVALUABLE_IN_DOUBT_PARTIAL_DEPTH"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    protocol = args.protocol.resolve(strict=True)
    if sha256_file(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("C33 protocol hash drift")
    frozen_protocol = root / "frozen_protocol.json"
    if not frozen_protocol.is_file() or sha256_file(frozen_protocol) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("C33 captured protocol identity drift")

    completed: dict[str, Any] = {}
    for sensor in ("instance", "wearable"):
        shard_root = root / "shards" / sensor
        shard_result_path = shard_root / "result.json"
        shard_result = read_json(shard_result_path)
        if (
            shard_result.get("status")
            != "DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE"
            or int(shard_result.get("payload_count", 0)) != 728
        ):
            raise RuntimeError(f"C33 completed shard drift: {sensor}")
        completed[sensor] = {
            "status": shard_result["status"],
            "payload_count": 728,
            "result_sha256": sha256_file(shard_result_path),
            "payload_inventory_sha256": sha256_file(
                shard_root / "payload_inventory.json"
            ),
        }

    depth_root = root / "shards" / "depth"
    depth_pngs = sorted(depth_root.rglob("*.png"))
    if not depth_pngs:
        raise RuntimeError("C33 interruption was not a nonzero partial depth shard")
    if (depth_root / "result.json").exists():
        raise RuntimeError("C33 depth shard unexpectedly completed")
    if (root / "shards" / "witness").exists():
        raise RuntimeError("C33 witness shard unexpectedly started")

    partial_inventory = {
        "schema": "dtr-carla-c33-partial-depth-inventory-v1",
        "status": "SEALED_IN_DOUBT_PARTIAL_SHARD",
        "sensor": "depth",
        "payload_count": len(depth_pngs),
        "files": [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in depth_pngs
        ],
    }
    partial_inventory_path = root / "partial_depth_inventory.json"
    write_json_exclusive(partial_inventory_path, partial_inventory)

    evidence_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "terminal_partial_manifest.json"
    )
    manifest = {
        "schema": "dtr-carla-c33-terminal-partial-manifest-v1",
        "status": TERMINAL_STATUS,
        "files": [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in evidence_files
        ],
    }
    manifest_path = root / "terminal_partial_manifest.json"
    write_json_exclusive(manifest_path, manifest)

    result = {
        "schema": "dtr-carla-c33-interrupted-source-result-v1",
        "status": TERMINAL_STATUS,
        "cohort_id": "DTR_CARLA_C33_X65_RENDER_TRANSFER_CONFIRMATION_V1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "completed_sensors": completed,
        "partial_sensor": {
            "sensor": "depth",
            "durable_payload_count": len(depth_pngs),
            "result_exists": False,
            "partial_inventory_sha256": sha256_file(partial_inventory_path),
        },
        "unstarted_sensors": ["witness"],
        "terminal_reason": (
            "THE_CAPTURE_SUPERVISION_SESSION_WAS_INTERRUPTED_AFTER_NONZERO_"
            "DEPTH_PAYLOADS_BECAME_DURABLE"
        ),
        "capture_retry_allowed": False,
        "capture_retry_rule": (
            "ANY_NONZERO_PARTIAL_SHARD_IS_SOURCE_NOT_EVALUABLE_AND_MUST_NOT_RETRY"
        ),
        "model_predictions_created": False,
        "evaluator_opened": False,
        "carla_processes_remaining": 0,
        "carla_ports_remaining": 0,
        "storage_lease_released": True,
        "terminal_partial_manifest_sha256": sha256_file(manifest_path),
        "claim_boundary": [
            "C33 is a terminal source interruption, not an X65 algorithm result.",
            "The 21 durable depth payloads make the interrupted shard in_doubt and forbid retry under the frozen protocol.",
            "No C33 model prediction, evaluator score, metric, or promotion claim exists.",
            "Completed and partial payloads are retained only as audit evidence.",
        ],
    }
    result_path = root / "result.json"
    write_json_exclusive(result_path, result)
    print(
        json.dumps(
            {
                "status": TERMINAL_STATUS,
                "root": str(root),
                "depth_payload_count": len(depth_pngs),
                "result_sha256": sha256_file(result_path),
                "terminal_partial_manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

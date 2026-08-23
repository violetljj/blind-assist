"""Validate copied ABotN official-render pixels against the local freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any


SCHEMA = "blindassist_abotn_official_render_canary_local_audit_v0"
HTTP_RENDER_SUCCESS = re.compile(r'"POST /render_gs HTTP/1\.1" 200')


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit(
    *,
    freeze_path: Path,
    remote_output: Path,
    server_log: Path,
    finalizer_path: Path,
    frozen_render_helper: Path,
    frozen_download_helper: Path,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    receipt_path = remote_output / "terminal-receipt.json"
    journal_path = remote_output / "render-journal.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "blindassist_abotn_official_render_canary_freeze_v0":
        raise ValueError("freeze schema mismatch")
    if receipt.get("schema_version") != "blindassist_abotn_official_render_canary_v0":
        raise ValueError("remote receipt schema mismatch")
    if journal.get("schema_version") != "blindassist_abotn_official_render_journal_v0":
        raise ValueError("render journal schema mismatch")

    expected_poses = freeze["execution"]["pose_indices"]
    if receipt.get("pose_indices") != expected_poses:
        raise ValueError("receipt pose roster drift")
    if [row.get("pose_index") for row in journal.get("calls", [])] != expected_poses:
        raise ValueError("journal pose roster drift")
    expected_calls = freeze["execution"]["render_calls_authorized"]
    if any(
        value != expected_calls
        for value in (
            receipt.get("render_calls_dispatched"),
            receipt.get("render_calls_completed"),
            journal.get("render_calls_dispatched"),
            journal.get("render_calls_completed"),
        )
    ) or receipt.get("render_calls_in_doubt") != 0:
        raise ValueError("render call accounting mismatch")
    if any(row.get("status") != "COMPLETED" for row in journal["calls"]):
        raise ValueError("journal contains a non-completed call")
    if any(
        receipt.get(field) != 0
        for field in ("teacher_calls", "provider_calls", "baseline_calls", "sealed_episode_reruns")
    ):
        raise ValueError("zero-model or no-rerun firewall violated")

    if receipt.get("official_repository_commit") != freeze["official_renderer"]["commit"]:
        raise ValueError("official repository revision drift")
    if receipt.get("annotation_sha256") != freeze["inputs"]["annotation"]["sha256"]:
        raise ValueError("annotation identity drift")
    if receipt.get("scene_id") != freeze["inputs"]["scene_id"]:
        raise ValueError("scene identity drift")
    expected_camera = freeze["official_renderer"]["camera"] | {
        "server_render_scale": freeze["official_renderer"]["render_scale"]
    }
    if receipt.get("camera") != expected_camera:
        raise ValueError("official camera configuration drift")
    runtime = receipt["runtime"]
    for field in ("diff_plane_rasterization_sha256", "simple_knn_sha256"):
        if runtime.get(field) != freeze["worker"][field]:
            raise ValueError(f"runtime extension drift: {field}")
    if _sha256(frozen_render_helper) != freeze["helpers"]["remote_official_render_canary_sha256"]:
        raise ValueError("frozen remote render helper hash mismatch")
    if _sha256(frozen_download_helper) != freeze["helpers"]["remote_pinned_range_download_sha256"]:
        raise ValueError("frozen remote download helper hash mismatch")

    frame_receipts = []
    by_pose = {row["pose_index"]: row for row in receipt["frames"]}
    if set(by_pose) != set(expected_poses):
        raise ValueError("receipt frame roster drift")
    for call in journal["calls"]:
        frame = remote_output / str(call["output"])
        actual_sha = _sha256(frame)
        width, height = _png_size(frame)
        remote_frame = by_pose[call["pose_index"]]
        if actual_sha != call["output_sha256"] or actual_sha != remote_frame["sha256"]:
            raise ValueError(f"frame hash mismatch: {frame.name}")
        if (width, height) != (
            remote_frame["pixel_stats"]["width"],
            remote_frame["pixel_stats"]["height"],
        ):
            raise ValueError(f"frame dimensions mismatch: {frame.name}")
        frame_receipts.append({
            "pose_index": call["pose_index"],
            "path": str(frame.resolve()),
            "bytes": frame.stat().st_size,
            "sha256": actual_sha,
            "width": width,
            "height": height,
            "pixel_stats": remote_frame["pixel_stats"],
        })

    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    http_success_count = len(HTTP_RENDER_SUCCESS.findall(log_text))
    if http_success_count != expected_calls:
        raise ValueError(f"server HTTP render count mismatch: {http_success_count}")
    result = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "terminal": "ABOTN_OFFICIAL_RENDER_CANARY_LOCAL_AUDIT_PASS",
        "freeze_manifest": {
            "path": str(freeze_path.resolve()),
            "sha256": _sha256(freeze_path),
        },
        "remote_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": _sha256(receipt_path),
            "finalized_from_existing_outputs": receipt["receipt_finalized_from_existing_outputs"],
            "finalizer_sha256": _sha256(finalizer_path),
        },
        "helpers": {
            "frozen_render_helper": str(frozen_render_helper.resolve()),
            "frozen_render_helper_sha256": _sha256(frozen_render_helper),
            "frozen_download_helper": str(frozen_download_helper.resolve()),
            "frozen_download_helper_sha256": _sha256(frozen_download_helper),
        },
        "render_journal": {
            "path": str(journal_path.resolve()),
            "sha256": _sha256(journal_path),
            "calls_dispatched": journal["render_calls_dispatched"],
            "calls_completed": journal["render_calls_completed"],
            "calls_in_doubt": 0,
        },
        "server_log": {
            "path": str(server_log.resolve()),
            "sha256": _sha256(server_log),
            "http_render_200_count": http_success_count,
        },
        "frames": frame_receipts,
        "teacher_calls": 0,
        "provider_calls": 0,
        "baseline_calls": 0,
        "sealed_episode_reruns": 0,
        "claim_ceiling": "PINNED_OFFICIAL_RENDERER_PIXEL_TRANSPORT_ONLY",
        "next_action": "ADJUDICATE_RENDERER_FIDELITY_CONFOUND_WITHOUT_MODEL_REPLAY",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--remote-output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--frozen-render-helper", type=Path, required=True)
    parser.add_argument("--frozen-download-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        freeze_path=args.freeze.resolve(),
        remote_output=args.remote_output.resolve(),
        server_log=args.server_log.resolve(),
        finalizer_path=args.finalizer.resolve(),
        frozen_render_helper=args.frozen_render_helper.resolve(),
        frozen_download_helper=args.frozen_download_helper.resolve(),
    )
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "terminal": result["terminal"],
        "frames": len(result["frames"]),
        "render_calls": result["render_journal"]["calls_completed"],
        "provider_calls": result["provider_calls"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

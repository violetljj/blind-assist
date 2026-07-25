#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_rows(data: bytes, expected_columns: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in data.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != expected_columns:
            raise ValueError(f"unexpected index row: {line}")
        rows.append(values)
    timestamps = [float(row[0]) for row in rows]
    if not rows or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("index timestamps must be non-empty and strictly increasing")
    return rows


def audit_archive(path: Path, sequence_id: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failure: {bad_member}")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        root = f"{sequence_id}/"
        if not names or any(not name.startswith(root) for name in names):
            raise ValueError("archive root identity mismatch")
        required = {
            "rgb.txt": f"{root}rgb.txt",
            "depth.txt": f"{root}depth.txt",
            "groundtruth.txt": f"{root}groundtruth.txt",
        }
        for member in required.values():
            if names.count(member) != 1:
                raise ValueError(f"required member count mismatch: {member}")
        text = {key: archive.read(member) for key, member in required.items()}
        rgb_rows = index_rows(text["rgb.txt"], 2)
        depth_rows = index_rows(text["depth.txt"], 2)
        pose_rows = index_rows(text["groundtruth.txt"], 8)
        rgb_png_members = {
            name
            for name in names
            if name.startswith(f"{root}rgb/") and name.endswith(".png")
        }
        depth_png_members = {
            name
            for name in names
            if name.startswith(f"{root}depth/") and name.endswith(".png")
        }
        rgb_png_count = len(rgb_png_members)
        depth_png_count = len(depth_png_members)
        rgb_references = {f"{root}{row[1]}" for row in rgb_rows}
        depth_references = {f"{root}{row[1]}" for row in depth_rows}
        name_set = set(names)
        missing_rgb_references = sorted(rgb_references - name_set)
        missing_depth_references = sorted(depth_references - name_set)
        manifest_lines = [
            "\t".join(
                [
                    info.filename,
                    str(info.file_size),
                    str(info.compress_size),
                    f"{info.CRC:08x}",
                ]
            )
            for info in infos
        ]
        member_manifest_sha256 = sha256_bytes(
            ("\n".join(manifest_lines) + "\n").encode("utf-8")
        )
        return {
            "sequence_id": sequence_id,
            "archive_filename": path.name,
            "archive_bytes": path.stat().st_size,
            "archive_sha256": sha256_path(path),
            "zip_member_count": len(infos),
            "zip_crc_all_members_valid": True,
            "member_manifest_sha256": member_manifest_sha256,
            "rgb_frame_count": len(rgb_rows),
            "depth_frame_count": len(depth_rows),
            "rgb_png_member_count": rgb_png_count,
            "depth_png_member_count": depth_png_count,
            "unindexed_rgb_png_count": len(rgb_png_members - rgb_references),
            "unindexed_depth_png_count": len(depth_png_members - depth_references),
            "missing_indexed_rgb_png_count": len(missing_rgb_references),
            "missing_indexed_depth_png_count": len(missing_depth_references),
            "missing_indexed_rgb_members": missing_rgb_references,
            "missing_indexed_depth_members": missing_depth_references,
            "indexed_rgb_png_available_count": len(rgb_references)
            - len(missing_rgb_references),
            "indexed_depth_png_available_count": len(depth_references)
            - len(missing_depth_references),
            "pose_sample_count": len(pose_rows),
            "rgb_duration_seconds": float(rgb_rows[-1][0])
            - float(rgb_rows[0][0]),
            "text_member_sha256": {
                key: sha256_bytes(value) for key, value in sorted(text.items())
            },
            "image_member_decode_count": 0,
        }


def build_receipt(freeze: dict[str, Any], archive_dir: Path) -> dict[str, Any]:
    selected = freeze["selection_contract"]["selected"]
    discovery = [item for item in selected if item["role"] == "discovery"]
    if len(discovery) != 2:
        raise ValueError("exactly two discovery sequences required")
    forbidden = [
        item["sequence_id"]
        for item in selected
        if item["role"] in {"validation", "sealed_holdout"}
        and (archive_dir / f"{item['sequence_id']}.zip").exists()
    ]
    if forbidden:
        raise ValueError(f"sealed archive present: {forbidden}")
    archives = [
        audit_archive(archive_dir / f"{item['sequence_id']}.zip", item["sequence_id"])
        for item in discovery
    ]
    return {
        "schema_version": "bonn_claim_scoped_discovery_archive_audit_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "cohort_identity_sha256": freeze["selection_contract"][
            "cohort_identity_sha256"
        ],
        "archives": archives,
        "totals": {
            "archive_count": len(archives),
            "archive_bytes": sum(item["archive_bytes"] for item in archives),
            "rgb_frame_count": sum(item["rgb_frame_count"] for item in archives),
            "depth_frame_count": sum(item["depth_frame_count"] for item in archives),
            "pose_sample_count": sum(item["pose_sample_count"] for item in archives),
        },
        "sealed_roles": {
            "validation_archive_read_count": 0,
            "holdout_archive_read_count": 0,
        },
        "read_firewall": {
            "old_15_pair_window_selection_tuning_acceptance_reads": 0,
            "prior_bonn_outcome_reads": 0,
            "image_member_decode_count": 0,
            "candidate_signal_computed": False,
        },
        "terminal": "BONN_DISCOVERY_ARCHIVES_ACQUIRED_METADATA_VALID_EXTRACTION_NOT_RUN",
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    receipt = build_receipt(freeze, args.archive_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                "archive_bytes": receipt["totals"]["archive_bytes"],
                "rgb_frames": receipt["totals"]["rgb_frame_count"],
                "image_decode_count": receipt["read_firewall"][
                    "image_member_decode_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

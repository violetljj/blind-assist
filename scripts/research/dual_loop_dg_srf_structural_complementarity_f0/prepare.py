"""Materialize the truth-minimized 520-frame DG-SRF F0 inference manifest."""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    PROTOCOL_ID,
    ensure_artifact_output,
    read_json,
    read_jsonl,
    resolve_repo_path,
    sha256_file,
    validate_config,
    verify_file,
    write_json,
    write_jsonl,
)


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_prepare(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    output_root = ensure_artifact_output(repo_root, output_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")

    config = read_json(config_path)
    validate_config(config)
    contract = config["input_contract"]
    all_frames: list[dict[str, Any]] = []
    for source in contract["frame_sources"]:
        path = resolve_repo_path(repo_root, source["path"])
        verify_file(path, source["sha256"])
        rows = read_jsonl(path)
        if len(rows) != int(source["row_count"]):
            raise ValueError(f"row count mismatch: {path}")
        for row_index, row in enumerate(rows):
            all_frames.append(
                {
                    "frame_source_path": source["path"],
                    "frame_source_row_index": row_index,
                    "source_role": source["role"],
                    "view_row_id": row["view_row_id"],
                    "session_id": row["session_id"],
                    "frame_id": int(row["frame_id"]),
                    "image_sha256": row["image_sha256"],
                }
            )

    if len(all_frames) != int(contract["expected_frame_count"]):
        raise ValueError("unexpected selected frame count")
    view_ids = [row["view_row_id"] for row in all_frames]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("duplicate view_row_id")

    manifest_spec = contract["canonical_manifest"]
    manifest_path = resolve_repo_path(repo_root, manifest_spec["path"])
    verify_file(manifest_path, manifest_spec["sha256"])
    manifest_rows = read_jsonl(manifest_path)
    if len(manifest_rows) != int(manifest_spec["row_count"]):
        raise ValueError("canonical manifest row count mismatch")
    manifest_index: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in manifest_rows:
        key = (
            row["session_id"],
            int(row["frame_id"]),
            row["image_sha256"],
        )
        manifest_index.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for index, selected in enumerate(all_frames):
        key = (
            selected["session_id"],
            selected["frame_id"],
            selected["image_sha256"],
        )
        matches = manifest_index.get(key, [])
        if len(matches) != 1:
            raise ValueError(f"canonical mapping count {len(matches)} for {key}")
        match = matches[0]
        image_path = resolve_repo_path(repo_root, match["image_repo_relative_path"])
        verify_file(image_path, match["image_sha256"])
        output_rows.append(
            {
                "schema_version": (
                    "blindassist.dg_srf_image_space_structural_"
                    "complementarity_f0.inference_input.v1"
                ),
                "protocol_id": PROTOCOL_ID,
                "index": index,
                **selected,
                "image_repo_relative_path": match["image_repo_relative_path"],
                "formal_authority": False,
            }
        )

    session_counts = Counter(row["session_id"] for row in output_rows)
    if len(session_counts) != int(contract["expected_source_session_count"]):
        raise ValueError("unexpected source-session count")

    output_root.mkdir(parents=True)
    inference_manifest_path = output_root / "inference_manifest.jsonl"
    write_jsonl(inference_manifest_path, output_rows)
    receipt = {
        "schema_version": (
            "blindassist.dg_srf_image_space_structural_"
            "complementarity_f0.prepare_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "git_head": _git_head(repo_root),
        "config_path": str(config_path.relative_to(repo_root)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "frame_count": len(output_rows),
        "source_session_count": len(session_counts),
        "source_session_frame_counts": dict(sorted(session_counts.items())),
        "inference_manifest": {
            "path": str(inference_manifest_path.relative_to(repo_root)).replace(
                "\\", "/"
            ),
            "sha256": sha256_file(inference_manifest_path),
            "row_count": len(output_rows),
        },
        "truth_minimized_inference_fields": [
            "identity",
            "source_role",
            "image_path_and_hash",
        ],
        "preparation_source_rows_are_consumed_and_contain_unused_outcome_fields": True,
        "scientific_truth_or_packed_masks_emitted_to_inference_manifest": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    write_json(output_root / "prepare_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_prepare(
        repo_root=args.repo_root,
        config_path=args.config,
        output_root=args.output_root,
    )
    print(
        f"{receipt['status']} frames={receipt['frame_count']} "
        f"sessions={receipt['source_session_count']}"
    )


if __name__ == "__main__":
    main()

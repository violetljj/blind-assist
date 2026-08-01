"""Materialize a compact, hash-closed device view of the frozen event cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from scripts.research.dual_loop_segmentation_r2_p0.canonicalizer import (
    canonicalize_array,
    load_contract,
)


PROTOCOL_ID = "RISKSEG_R0_EVENT_EVAL_V1"
EXPECTED_LEDGER_SHA256 = (
    "d9c6a1881096ddcb1a312b7a3f04497350c5f1034e04c2344d2211686e935012"
)
EXPECTED_COHORT_RECEIPT_SHA256 = (
    "2da321dfc56498baac2989c9ee751a612615376984d55743d83d530c53643345"
)
EXPECTED_CANDIDATE_INDEX_SHA256 = (
    "99157f256d4c6555dc1593700823cc5211acf15b190a2a1e4329ec95853190fb"
)
CANONICALIZATION_CONTRACT = Path(
    "configs/dual_loop_segmentation_r2_p0/canonicalization_contract.json"
)
EXPECTED_CANONICALIZATION_SHA256 = (
    "a1818b8b52ece55cc046c424defc4451a800790e15905c9c8f4b18a630fe67e1"
)
LEGACY_TO_RISKSEG = np.asarray([0, 2, 1, 3], dtype=np.uint8)
CLASS_ORDER = [
    "walkable",
    "blocking_obstacle",
    "boundary_level_change",
    "unknown_nonwalkable",
]
POSITIVE_BUCKETS = {
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _manifest_path(raw: str) -> Path:
    path = Path(raw).resolve()
    return path / "manifest.draft.jsonl" if path.is_dir() else path


def _mask_path(dataset_root: Path, row: dict[str, Any]) -> Path:
    image_relative = Path(str(row["image_path"]))
    if not image_relative.parts or image_relative.parts[0] != "images":
        raise ValueError(f"unexpected image_path layout: {image_relative}")
    return dataset_root / "source_masks" / Path(*image_relative.parts[1:])


def _save_rgb(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB").resize(
            (512, 288),
            resample=Image.Resampling.BILINEAR,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(target, format="PNG", optimize=False, compress_level=6)


def _save_oracle_mask(
    source: Path,
    target: Path,
    canonicalization_contract: dict[str, Any],
) -> Counter[int]:
    legacy = canonicalize_array(
        source,
        "source_native",
        canonicalization_contract,
    )
    riskseg = LEGACY_TO_RISKSEG[legacy]
    unique = set(int(value) for value in np.unique(riskseg))
    if not unique.issubset({0, 1, 2, 3}):
        raise ValueError(f"oracle mask contains invalid IDs: {sorted(unique)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(riskseg, mode="L").save(
        target,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    counts = np.bincount(riskseg.reshape(-1), minlength=4)
    return Counter({index: int(counts[index]) for index in range(4)})


def materialize(
    *,
    repo_root: Path,
    truth_ledger_path: Path,
    cohort_receipt_path: Path,
    candidate_index_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    artifacts_root = (repo_root / "artifacts.local").resolve()
    if output_root == artifacts_root or artifacts_root not in output_root.parents:
        raise ValueError("output root must be a child of artifacts.local")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_root}")

    truth_ledger_path = truth_ledger_path.resolve()
    cohort_receipt_path = cohort_receipt_path.resolve()
    candidate_index_path = candidate_index_path.resolve()
    if sha256_file(truth_ledger_path) != EXPECTED_LEDGER_SHA256:
        raise ValueError("frozen truth ledger SHA-256 mismatch")
    if sha256_file(cohort_receipt_path) != EXPECTED_COHORT_RECEIPT_SHA256:
        raise ValueError("cohort receipt SHA-256 mismatch")
    if sha256_file(candidate_index_path) != EXPECTED_CANDIDATE_INDEX_SHA256:
        raise ValueError("candidate index SHA-256 mismatch")

    cohort_receipt = _read_object(cohort_receipt_path)
    if (
        cohort_receipt.get("status") != "EVENT_EVAL_FROZEN_ADEQUATE"
        or cohort_receipt.get("truth_ledger_sha256") != EXPECTED_LEDGER_SHA256
        or cohort_receipt.get("candidate_output_visible_before_freeze") is not False
    ):
        raise ValueError("cohort receipt does not authorize frozen evaluation")
    candidate_index = _read_object(candidate_index_path)
    if (
        candidate_index.get("schema_version")
        != "blindassist.riskseg_r0.event_candidate_index.v1"
        or candidate_index.get("candidate_output_visible") is not False
    ):
        raise ValueError("candidate index output firewall mismatch")
    items = candidate_index.get("items")
    if not isinstance(items, list):
        raise ValueError("candidate index items missing")
    item_by_id = {str(item["event_candidate_id"]): item for item in items}
    if len(item_by_id) != len(items):
        raise ValueError("duplicate candidate IDs")

    contract_path = (repo_root / CANONICALIZATION_CONTRACT).resolve()
    if sha256_file(contract_path) != EXPECTED_CANONICALIZATION_SHA256:
        raise ValueError("canonicalization contract SHA-256 mismatch")
    canonicalization_contract = load_contract(contract_path)

    ledger = _read_jsonl(truth_ledger_path)
    if len(ledger) != 30:
        raise ValueError(f"expected 30 frozen parent events, found {len(ledger)}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    ).resolve()
    published = False
    try:
        events: list[dict[str, Any]] = []
        bucket_counts: Counter[str] = Counter()
        session_ids: set[str] = set()
        oracle_pixels: Counter[int] = Counter()
        frame_total = 0
        for event_index, truth in enumerate(ledger):
            if (
                truth.get("schema_version")
                != "blindassist.riskseg_r0.event_truth_item.v1"
                or truth.get("outcome_access_state")
                != "PIDNET_YOLO_AND_ORACLE_UNOPENED"
            ):
                raise ValueError("truth ledger output firewall/schema mismatch")
            candidate_id = str(truth["event_candidate_id"])
            candidate = item_by_id.get(candidate_id)
            if candidate is None:
                raise ValueError(f"{candidate_id}: absent from candidate index")
            for field in (
                "sequence_id",
                "source_session_id",
                "source_frame_start",
                "source_frame_end",
            ):
                if candidate.get(field) != truth.get(field):
                    raise ValueError(f"{candidate_id}: {field} mismatch")
            for field in ("rgb_sha256s", "source_mask_sha256s"):
                if candidate.get(field) != truth.get(field):
                    raise ValueError(f"{candidate_id}: {field} mismatch")

            manifest_path = _manifest_path(str(candidate["draft_manifest_path"]))
            if sha256_file(manifest_path) != candidate.get("draft_manifest_sha256"):
                raise ValueError(f"{candidate_id}: draft manifest SHA-256 mismatch")
            source_rows = [
                row
                for row in _read_jsonl(manifest_path)
                if row.get("sequence_id") == truth["sequence_id"]
            ]
            source_rows.sort(key=lambda row: int(row["frame_index"]))
            if len(source_rows) != int(candidate["frame_count"]):
                raise ValueError(f"{candidate_id}: exact frame count mismatch")
            if [int(row["frame_index"]) for row in source_rows] != list(
                range(len(source_rows))
            ):
                raise ValueError(f"{candidate_id}: non-contiguous frame indices")
            dataset_root = manifest_path.parent
            scenarios = {str(row.get("assist_scenario")) for row in source_rows}
            if len(scenarios) != 1 or next(iter(scenarios)) not in {
                "GENERAL",
                "INDOOR",
                "CORRIDOR",
                "CROWDED",
                "OUTDOOR_SLOW",
            }:
                raise ValueError(f"{candidate_id}: invalid/inconsistent assist scenario")
            scenario = next(iter(scenarios))
            frames: list[dict[str, Any]] = []
            for frame_index, (row, rgb_sha, mask_sha) in enumerate(
                zip(
                    source_rows,
                    truth["rgb_sha256s"],
                    truth["source_mask_sha256s"],
                    strict=True,
                )
            ):
                source_info = row.get("source")
                if not isinstance(source_info, dict):
                    raise ValueError(f"{candidate_id}/{frame_index}: source missing")
                if source_info.get("sha256") != rgb_sha:
                    raise ValueError(f"{candidate_id}/{frame_index}: RGB ledger mismatch")
                if source_info.get("mask_sha256") != mask_sha:
                    raise ValueError(f"{candidate_id}/{frame_index}: mask ledger mismatch")
                source_image = (dataset_root / str(row["image_path"])).resolve()
                source_mask = _mask_path(dataset_root, row).resolve()
                if sha256_file(source_image) != rgb_sha:
                    raise ValueError(f"{candidate_id}/{frame_index}: RGB file mismatch")
                if sha256_file(source_mask) != mask_sha:
                    raise ValueError(f"{candidate_id}/{frame_index}: mask file mismatch")

                stem = f"{event_index:02d}_{frame_index:03d}"
                image_relative = Path("images") / f"{stem}.png"
                mask_relative = Path("oracle_masks") / f"{stem}.png"
                _save_rgb(source_image, staging / image_relative)
                counts = _save_oracle_mask(
                    source_mask,
                    staging / mask_relative,
                    canonicalization_contract,
                )
                oracle_pixels.update(counts)
                frames.append(
                    {
                        "frame_index": frame_index,
                        "source_frame_index": int(row["source_frame_index"]),
                        "timestamp_ms": int(row["frame_timestamp_ms"]),
                        "image_path": image_relative.as_posix(),
                        "image_sha256": sha256_file(staging / image_relative),
                        "oracle_mask_path": mask_relative.as_posix(),
                        "oracle_mask_sha256": sha256_file(staging / mask_relative),
                        "source_rgb_sha256": rgb_sha,
                        "source_mask_sha256": mask_sha,
                    }
                )
            bucket = str(truth["bucket"])
            positive = bucket in POSITIVE_BUCKETS
            if positive:
                alertable = [int(value) for value in truth["alertable_interval_frames"]]
                passed = [int(value) for value in truth["passed_interval_frames"]]
                for name, interval in (("alertable", alertable), ("passed", passed)):
                    if len(interval) != 2 or not (
                        0 <= interval[0] <= interval[1] < len(frames)
                    ):
                        raise ValueError(f"{candidate_id}: invalid {name} interval")
            else:
                alertable = None
                passed = None
            events.append(
                {
                    "parent_event_id": truth["parent_event_id"],
                    "event_candidate_id": candidate_id,
                    "source_session_id": truth["source_session_id"],
                    "sequence_id": truth["sequence_id"],
                    "bucket": bucket,
                    "positive": positive,
                    "assist_scenario": scenario,
                    "alertable_interval_frames": alertable,
                    "passed_interval_frames": passed,
                    "frames": frames,
                }
            )
            bucket_counts[bucket] += 1
            session_ids.add(str(truth["source_session_id"]))
            frame_total += len(frames)

        expected_buckets = {
            "blocking_obstacle_positive": 8,
            "boundary_level_change_positive": 8,
            "parallel_curb_negative": 7,
            "normal_walkable_negative": 7,
        }
        if dict(bucket_counts) != expected_buckets or len(session_ids) != 30:
            raise ValueError("frozen cohort bucket/session count drift")
        manifest = {
            "schema_version": "blindassist.riskseg_r0.device_event_view.v1",
            "protocol_id": PROTOCOL_ID,
            "class_order": CLASS_ORDER,
            "image_transform": {
                "size_wh": [512, 288],
                "mode": "RGB",
                "resize": "PIL_BILINEAR",
                "format": "PNG",
                "compress_level": 6,
            },
            "oracle_transform": {
                "size_wh": [256, 256],
                "source_native_to_legacy_contract_sha256": (
                    EXPECTED_CANONICALIZATION_SHA256
                ),
                "legacy_to_riskseg_r0": {"0": 0, "1": 2, "2": 1, "3": 3},
                "format": "PNG_L",
            },
            "event_count": len(events),
            "source_session_count": len(session_ids),
            "frame_count": frame_total,
            "bucket_counts": expected_buckets,
            "events": events,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt = {
            "schema_version": "blindassist.riskseg_r0.device_event_view_receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "RISKSEG_R0_DEVICE_EVENT_VIEW_MATERIALIZED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "truth_ledger": {
                "path": str(truth_ledger_path),
                "sha256": sha256_file(truth_ledger_path),
            },
            "cohort_receipt": {
                "path": str(cohort_receipt_path),
                "sha256": sha256_file(cohort_receipt_path),
            },
            "candidate_index": {
                "path": str(candidate_index_path),
                "sha256": sha256_file(candidate_index_path),
            },
            "canonicalization_contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "manifest": "manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
            "event_count": len(events),
            "source_session_count": len(session_ids),
            "frame_count": frame_total,
            "bucket_counts": expected_buckets,
            "oracle_class_pixel_counts": {
                str(class_id): oracle_pixels[class_id] for class_id in range(4)
            },
            "candidate_outputs_read": False,
            "yolo_outputs_read": False,
            "oracle_event_outcomes_read": False,
            "atomic_publish": True,
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
        published = True
        return receipt
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--truth-ledger", type=Path, required=True)
    parser.add_argument("--cohort-receipt", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    receipt = materialize(
        repo_root=repo_root,
        truth_ledger_path=resolve(args.truth_ledger),
        cohort_receipt_path=resolve(args.cohort_receipt),
        candidate_index_path=resolve(args.candidate_index),
        output_root=resolve(args.output_root),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

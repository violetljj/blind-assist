#!/usr/bin/env python3
"""Materialize a public-RGB/private-door-depth NYU Depth V2 cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


EXPECTED_SIZE = 2_972_037_809
EXPECTED_MD5 = "520609c519fba3ba5ac58c8fefcc3530"
EXPECTED_SHA256 = "2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa"
SOURCE_RECEIPT_SCHEMA = "blindassist_nyuv2_source_receipt_v1"
ROSTER_SCHEMA = "blindassist_nyuv2_door_depth_roster_v1"
PUBLIC_SCHEMA = "blindassist_completion_nearness_public_input_v1"
PRIVATE_SCHEMA = "blindassist_completion_nearness_private_eval_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _decode_cell(handle: Any, reference: Any) -> str:
    return "".join(chr(int(value)) for value in handle[reference][()].ravel())


def verify_source(source_lock_path: Path, dataset_path: Path, output_path: Path) -> dict[str, Any]:
    import h5py

    _require(not output_path.exists(), "source receipt is immutable and already exists")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    _require(source_lock.get("created_before_dataset_payload_access") is True, "source lock payload precedence missing")
    _require(dataset_path.is_file() and dataset_path.stat().st_size == EXPECTED_SIZE, "NYUv2 dataset size mismatch")
    _require(file_hash(dataset_path, "md5") == EXPECTED_MD5, "NYUv2 MD5 mismatch")
    _require(file_hash(dataset_path, "sha256") == EXPECTED_SHA256, "NYUv2 SHA-256 mismatch")
    with h5py.File(dataset_path, "r") as handle:
        required_shapes = {
            "images": (1449, 3, 640, 480),
            "labels": (1449, 640, 480),
            "instances": (1449, 640, 480),
            "rawDepths": (1449, 640, 480),
            "depths": (1449, 640, 480),
            "rawRgbFilenames": (1, 1449),
            "names": (1, 894),
        }
        for key, shape in required_shapes.items():
            _require(key in handle and tuple(handle[key].shape) == shape, f"NYUv2 {key} shape drift")
        names = [_decode_cell(handle, reference) for reference in handle["names"][0]]
        exact_door_ids = [index + 1 for index, name in enumerate(names) if name == "door"]
        _require(exact_door_ids == [28], "NYUv2 exact door taxonomy drift")
    receipt = {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "source_lock_sha256": file_hash(source_lock_path, "sha256"),
        "dataset_size_bytes": dataset_path.stat().st_size,
        "dataset_md5": EXPECTED_MD5,
        "dataset_sha256": EXPECTED_SHA256,
        "sample_count": 1449,
        "global_class_count": len(names),
        "global_names_sha256": _json_hash(names),
        "exact_door_class_name": "door",
        "exact_door_class_id": 28,
        "per_sample_rgb_access": False,
        "per_sample_label_instance_depth_access": False,
    }
    _atomic_json(output_path, receipt)
    return receipt


def _target_for_sample(labels: np.ndarray, instances: np.ndarray, raw_depth: np.ndarray, rule: Mapping[str, Any]) -> dict[str, Any] | None:
    door_mask = labels == 28
    if not np.any(door_mask):
        return None
    minimum_pixels = int(rule["minimum_instance_pixels"])
    candidates = []
    for instance_id in np.unique(instances[door_mask]):
        if int(instance_id) <= 0:
            continue
        mask = door_mask & (instances == instance_id)
        ys, xs = np.nonzero(mask)
        if len(xs) < minimum_pixels:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        candidates.append((float((bbox[0] + bbox[2]) / 2.0), int(instance_id), mask, bbox))
    if not candidates:
        return None
    _, instance_id, mask, bbox = min(candidates, key=lambda item: (item[0], item[1]))
    minimum_depth, maximum_depth = [float(value) for value in rule["valid_raw_depth_range_m"]]
    valid = mask & np.isfinite(raw_depth) & (raw_depth >= minimum_depth) & (raw_depth <= maximum_depth)
    valid_fraction = float(valid.sum() / mask.sum())
    if valid_fraction < float(rule["minimum_raw_depth_valid_fraction_in_target"]):
        return None
    depth_values = raw_depth[valid]
    return {
        "instance_id": instance_id,
        "mask": mask,
        "bbox_xyxy": bbox,
        "target_pixel_count": int(mask.sum()),
        "raw_depth_valid_pixel_count": int(valid.sum()),
        "raw_depth_valid_fraction": valid_fraction,
        "target_raw_depth_median_m": float(np.median(depth_values)),
        "target_raw_depth_p10_m": float(np.quantile(depth_values, 0.10)),
        "target_raw_depth_p90_m": float(np.quantile(depth_values, 0.90)),
    }


def freeze_roster(source_lock_path: Path, source_receipt_path: Path, c0_path: Path, dataset_path: Path, output_path: Path) -> dict[str, Any]:
    import h5py

    _require(not output_path.exists(), "roster is immutable and already exists")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    _require(source_receipt.get("schema_version") == SOURCE_RECEIPT_SCHEMA, "source receipt schema mismatch")
    _require(source_receipt.get("source_lock_sha256") == file_hash(source_lock_path, "sha256"), "source receipt lock binding mismatch")
    _require(source_receipt.get("dataset_sha256") == EXPECTED_SHA256 and dataset_path.stat().st_size == EXPECTED_SIZE, "dataset identity drift")
    _require(source_lock.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "source lock/C0 binding mismatch")
    rule = source_lock["eligibility_rule"]
    near_threshold = float(rule["near_threshold_m"])
    eligible = []
    with h5py.File(dataset_path, "r") as handle:
        for index in range(1449):
            labels = np.asarray(handle["labels"][index]).T
            if not np.any(labels == 28):
                continue
            instances = np.asarray(handle["instances"][index]).T
            raw_depth = np.asarray(handle["rawDepths"][index]).T
            target = _target_for_sample(labels, instances, raw_depth, rule)
            if target is None:
                continue
            filename = _decode_cell(handle, handle["rawRgbFilenames"][0, index])
            eligible.append({
                "dataset_index": index,
                "raw_rgb_filename": filename,
                "raw_rgb_filename_sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
                "stratum": "NEAR" if target["target_raw_depth_median_m"] <= near_threshold else "FAR",
                "target": {key: value for key, value in target.items() if key != "mask"},
            })
    near = sorted((row for row in eligible if row["stratum"] == "NEAR"), key=lambda row: row["raw_rgb_filename_sha256"])
    far = sorted((row for row in eligible if row["stratum"] == "FAR"), key=lambda row: row["raw_rgb_filename_sha256"])
    near_take, far_take = int(rule["near_take"]), int(rule["far_take"])
    _require(len(near) >= near_take and len(far) >= far_take, f"NYUv2 stratum denominator insufficient: near={len(near)} far={len(far)}")
    selected_source = near[:near_take] + far[:far_take]
    _require(len(selected_source) == len(c0.get("episodes", [])), "selected roster/C0 size mismatch")
    cases = []
    for ordinal, (row, episode) in enumerate(zip(selected_source, c0["episodes"], strict=True), start=1):
        cases.append({
            "case_id": f"nyuv2-door-depth-case-{ordinal:03d}",
            "episode_id": episode["episode_id"],
            **row,
        })
    roster = {
        "schema_version": ROSTER_SCHEMA,
        "source_lock_sha256": file_hash(source_lock_path, "sha256"),
        "source_receipt_sha256": file_hash(source_receipt_path, "sha256"),
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "eligible_case_count": len(eligible),
        "eligible_near_count": len(near),
        "eligible_far_count": len(far),
        "selection_rule": rule,
        "private_truth_access_for_frozen_selection": True,
        "provider_truth_access": False,
        "cases": cases,
    }
    roster["roster_body_sha256"] = _json_hash(roster)
    _atomic_json(output_path, roster)
    return roster


def materialize_inputs(roster_path: Path, c0_path: Path, dataset_path: Path, public_root: Path, private_root: Path, public_output: Path, private_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import h5py

    _require(not public_output.exists() and not private_output.exists(), "NYUv2 inputs are immutable and already exist")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    declared = roster.pop("roster_body_sha256")
    _require(declared == _json_hash(roster), "roster body hash mismatch")
    roster["roster_body_sha256"] = declared
    c0_by_episode = {case["episode_id"]: case for case in c0["episodes"]}
    public_cases = []
    private_cases = []
    with h5py.File(dataset_path, "r") as handle:
        for row in roster["cases"]:
            index = int(row["dataset_index"])
            rgb = np.asarray(handle["images"][index]).transpose(2, 1, 0)
            rgb_path = (public_root / f"{row['case_id']}.jpg").resolve()
            _require(not rgb_path.exists(), "refusing to overwrite public RGB")
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(rgb_path, format="JPEG", quality=95, subsampling=0)
            labels = np.asarray(handle["labels"][index]).T
            instances = np.asarray(handle["instances"][index]).T
            target = _target_for_sample(labels, instances, np.asarray(handle["rawDepths"][index]).T, roster["selection_rule"])
            _require(target is not None and target["instance_id"] == row["target"]["instance_id"], "private target drift")
            mask_path = (private_root / f"{row['case_id']}-target-mask.png").resolve()
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((target["mask"] * 255).astype(np.uint8)).save(mask_path)
            goal = c0_by_episode[row["episode_id"]]
            public_cases.append({
                "case_id": row["case_id"],
                "episode_id": row["episode_id"],
                "goal_contract": goal["goal_contract"] | {"canonical_prompt": goal["canonical_prompt"]},
                "query": {"image_path": str(rgb_path), "image_sha256": file_hash(rgb_path, "sha256")},
            })
            private_cases.append({
                "case_id": row["case_id"],
                "target_bbox_xyxy": target["bbox_xyxy"],
                "target_mask_path": str(mask_path),
                "target_mask_sha256": file_hash(mask_path, "sha256"),
                "target_pixel_count": target["target_pixel_count"],
                "raw_depth_valid_pixel_count": target["raw_depth_valid_pixel_count"],
                "raw_depth_valid_fraction": target["raw_depth_valid_fraction"],
                "target_raw_depth_median_m": target["target_raw_depth_median_m"],
                "target_raw_depth_p10_m": target["target_raw_depth_p10_m"],
                "target_raw_depth_p90_m": target["target_raw_depth_p90_m"],
                "true_interaction_range": target["target_raw_depth_median_m"] <= float(roster["selection_rule"]["near_threshold_m"]),
                "stratum": row["stratum"],
            })
    public = {
        "schema_version": PUBLIC_SCHEMA,
        "protocol_id": "BLINDASSIST_COMPLETION_NEARNESS_NYUV2_DOOR_V1",
        "roster_body_sha256": roster["roster_body_sha256"],
        "private_truth_access": False,
        "cases": public_cases,
    }
    _atomic_json(public_output, public)
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "protocol_id": "BLINDASSIST_COMPLETION_NEARNESS_NYUV2_DOOR_V1",
        "public_input_sha256": file_hash(public_output, "sha256"),
        "interaction_range_m": float(roster["selection_rule"]["near_threshold_m"]),
        "cases": private_cases,
    }
    _atomic_json(private_output, private)
    return public, private


def download(source_lock_path: Path, output_path: Path, workers: int) -> Path:
    import requests

    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    _require(source_lock.get("created_before_dataset_payload_access") is True, "source lock payload precedence missing")
    source = source_lock.get("source", {})
    _require(source.get("size_bytes") == EXPECTED_SIZE, "NYUv2 size drift")
    _require(source.get("md5") == EXPECTED_MD5 and source.get("sha256") == EXPECTED_SHA256, "NYUv2 hash contract drift")
    url = str(source.get("download_url", ""))
    _require(url == "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat", "NYUv2 URL drift")
    _require(1 <= workers <= 16, "workers must be in [1, 16]")
    if output_path.is_file() and output_path.stat().st_size == EXPECTED_SIZE and file_hash(output_path, "sha256") == EXPECTED_SHA256:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts_root = output_path.with_suffix(output_path.suffix + ".parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    segment_size = (EXPECTED_SIZE + workers - 1) // workers

    def fetch(index: int) -> Path:
        start = index * segment_size
        end = min(EXPECTED_SIZE - 1, start + segment_size - 1)
        expected = end - start + 1
        part = parts_root / f"part-{index:02d}.bin"
        if part.is_file() and part.stat().st_size == expected:
            return part
        temporary = part.with_suffix(".tmp")
        if temporary.is_file() and temporary.stat().st_size == expected:
            os.replace(temporary, part)
            return part
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=180)
                _require(response.status_code == 206, f"range {index} HTTP {response.status_code}")
                _require(response.headers.get("content-range", "").startswith(f"bytes {start}-{end}/"), f"range {index} content-range mismatch")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                _require(temporary.stat().st_size == expected, f"range {index} size mismatch")
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt < 3:
                    time.sleep(float(attempt))
        if last_error is not None:
            raise last_error
        os.replace(temporary, part)
        return part

    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(fetch, range(workers)))
    temporary_output = output_path.with_suffix(output_path.suffix + ".complete.tmp")
    with temporary_output.open("wb") as target:
        for part in parts:
            with part.open("rb") as source_handle:
                for chunk in iter(lambda: source_handle.read(4 * 1024 * 1024), b""):
                    target.write(chunk)
    _require(temporary_output.stat().st_size == EXPECTED_SIZE, "assembled NYUv2 size mismatch")
    _require(file_hash(temporary_output, "md5") == EXPECTED_MD5, "assembled NYUv2 MD5 mismatch")
    _require(file_hash(temporary_output, "sha256") == EXPECTED_SHA256, "assembled NYUv2 SHA-256 mismatch")
    os.replace(temporary_output, output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("--source-lock", required=True, type=Path)
    download_parser.add_argument("--output", required=True, type=Path)
    download_parser.add_argument("--workers", type=int, default=8)
    verify = sub.add_parser("verify-source")
    verify.add_argument("--source-lock", required=True, type=Path)
    verify.add_argument("--dataset", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    roster = sub.add_parser("freeze-roster")
    roster.add_argument("--source-lock", required=True, type=Path)
    roster.add_argument("--source-receipt", required=True, type=Path)
    roster.add_argument("--c0", required=True, type=Path)
    roster.add_argument("--dataset", required=True, type=Path)
    roster.add_argument("--output", required=True, type=Path)
    inputs = sub.add_parser("materialize-inputs")
    inputs.add_argument("--roster", required=True, type=Path)
    inputs.add_argument("--c0", required=True, type=Path)
    inputs.add_argument("--dataset", required=True, type=Path)
    inputs.add_argument("--public-root", required=True, type=Path)
    inputs.add_argument("--private-root", required=True, type=Path)
    inputs.add_argument("--public-output", required=True, type=Path)
    inputs.add_argument("--private-output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.source_lock, args.output, args.workers)
    elif args.command == "verify-source":
        verify_source(args.source_lock, args.dataset, args.output)
    elif args.command == "freeze-roster":
        freeze_roster(args.source_lock, args.source_receipt, args.c0, args.dataset, args.output)
    else:
        materialize_inputs(args.roster, args.c0, args.dataset, args.public_root, args.private_root, args.public_output, args.private_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

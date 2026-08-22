#!/usr/bin/env python3
"""Materialize an ancestry-disjoint SUN RGB-D door/depth completion cohort."""

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

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_nyuv2_door_depth import PRIVATE_SCHEMA, PUBLIC_SCHEMA


SOURCE_RECEIPT_SCHEMA = "blindassist_sunrgbd_source_receipt_v1"
ROSTER_SCHEMA = "blindassist_sunrgbd_door_depth_roster_v1"
PROTOCOL_ID = "BLINDASSIST_COMPLETION_NEARNESS_SUNRGBD_DOOR_V1"
ALLOWED_PATH_MARKERS = ("/kv1/b3dodata/", "/xtion/sun3ddata/", "/kv2/kinect2data/")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def _download_ranges(url: str, expected_size: int, expected_etag: str, output: Path, workers: int) -> Path:
    import requests

    _require(1 <= workers <= 16, "workers must be in [1,16]")
    if output.is_file() and output.stat().st_size == expected_size:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_root = output.with_suffix(output.suffix + ".parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    segment_size = (expected_size + workers - 1) // workers

    def fetch(index: int) -> Path:
        start = index * segment_size
        end = min(expected_size - 1, start + segment_size - 1)
        expected = end - start + 1
        part = parts_root / f"part-{index:02d}.bin"
        temporary = part.with_suffix(".tmp")
        if part.is_file() and part.stat().st_size == expected:
            return part
        if temporary.is_file() and temporary.stat().st_size == expected:
            os.replace(temporary, part)
            return part
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=240)
                _require(response.status_code == 206, f"range {index} HTTP {response.status_code}")
                _require(response.headers.get("content-range", "").startswith(f"bytes {start}-{end}/"), f"range {index} content-range mismatch")
                response_etag = response.headers.get("etag", "").strip('"')
                _require(not response_etag or response_etag == expected_etag, f"range {index} ETag mismatch")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
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
    temporary_output = output.with_suffix(output.suffix + ".complete.tmp")
    with temporary_output.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(chunk)
    _require(temporary_output.stat().st_size == expected_size, "assembled SUN RGB-D archive size mismatch")
    os.replace(temporary_output, output)
    return output


def download(source_lock_path: Path, output_root: Path, workers: int) -> None:
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    _require(lock.get("created_before_dataset_payload_access") is True, "source lock payload precedence missing")
    source = lock.get("source", {})
    _download_ranges(str(source["data_url"]), int(source["data_size_bytes"]), str(source["data_etag"]), output_root / "SUNRGBD.zip", workers)
    _download_ranges(str(source["toolbox_url"]), int(source["toolbox_size_bytes"]), str(source["toolbox_etag"]), output_root / "SUNRGBDtoolbox.zip", workers)


def verify(source_lock_path: Path, data_archive: Path, toolbox_archive: Path, output: Path) -> dict[str, Any]:
    import zipfile

    _require(not output.exists(), "source receipt is immutable and already exists")
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source = lock["source"]
    _require(data_archive.stat().st_size == int(source["data_size_bytes"]), "SUNRGBD.zip size mismatch")
    _require(toolbox_archive.stat().st_size == int(source["toolbox_size_bytes"]), "SUNRGBDtoolbox.zip size mismatch")
    with zipfile.ZipFile(data_archive) as archive:
        bad = archive.testzip()
        _require(bad is None, f"SUNRGBD.zip CRC failure: {bad}")
        data_members = len(archive.infolist())
    with zipfile.ZipFile(toolbox_archive) as archive:
        bad = archive.testzip()
        _require(bad is None, f"SUNRGBDtoolbox.zip CRC failure: {bad}")
        toolbox_members = len(archive.infolist())
    receipt = {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "source_lock_sha256": file_hash(source_lock_path),
        "data_archive_size_bytes": data_archive.stat().st_size,
        "data_archive_sha256": file_hash(data_archive),
        "data_archive_member_count": data_members,
        "toolbox_archive_size_bytes": toolbox_archive.stat().st_size,
        "toolbox_archive_sha256": file_hash(toolbox_archive),
        "toolbox_archive_member_count": toolbox_members,
        "selected_rgb_access": False,
        "private_segmentation_depth_access": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return receipt


def _members_for_root(names: set[str], root: str) -> tuple[str, str]:
    rgb = sorted(name for name in names if name.startswith(root + "/image/") and name.lower().endswith((".jpg", ".jpeg")))
    depth = sorted(name for name in names if name.startswith(root + "/depth/") and name.lower().endswith(".png"))
    _require(len(rgb) == 1 and len(depth) == 1, f"SUNRGBD sample member drift: {root}")
    return rgb[0], depth[0]


def _decode_depth_png(raw: bytes) -> np.ndarray:
    import io

    encoded = np.asarray(Image.open(io.BytesIO(raw)), dtype=np.uint16)
    # Official SUN RGB-D toolbox convention: rotate the 16-bit value right by 3,
    # then convert millimetres to metres.
    decoded = ((encoded >> 3) | ((encoded << 13) & np.uint16(0xFFFF))).astype(np.float32) / 1000.0
    return decoded


def _door_targets(seglabel: np.ndarray, names: np.ndarray, depth: np.ndarray, rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(seglabel.shape == depth.shape, "SUNRGBD segmentation/depth shape mismatch")
    minimum_pixels = int(rule["minimum_connected_region_pixels"])
    minimum_depth, maximum_depth = map(float, rule["valid_depth_range_m"])
    targets: list[dict[str, Any]] = []
    for label_id, value in enumerate(np.atleast_1d(names), start=1):
        if str(value).strip().lower() != "door":
            continue
        mask = seglabel == label_id
        ys, xs = np.nonzero(mask)
        if xs.size < minimum_pixels:
            continue
        valid = mask & np.isfinite(depth) & (depth >= minimum_depth) & (depth <= maximum_depth)
        valid_fraction = float(valid.sum() / mask.sum())
        if valid_fraction < float(rule["minimum_valid_depth_fraction_in_region"]):
            continue
        values = depth[valid]
        targets.append({
            "label_id": label_id,
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "target_pixel_count": int(mask.sum()),
            "depth_valid_pixel_count": int(valid.sum()),
            "depth_valid_fraction": valid_fraction,
            "target_depth_median_m": float(np.median(values)),
            "target_depth_p10_m": float(np.quantile(values, 0.10)),
            "target_depth_p90_m": float(np.quantile(values, 0.90)),
            "mask": mask,
        })
    return targets


def freeze_roster(source_lock_path: Path, source_receipt_path: Path, c0_path: Path, data_archive: Path, output: Path) -> dict[str, Any]:
    import io
    import scipy.io as sio
    import zipfile

    _require(not output.exists(), "roster is immutable and already exists")
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    _require(receipt.get("schema_version") == SOURCE_RECEIPT_SCHEMA, "source receipt schema mismatch")
    _require(receipt.get("source_lock_sha256") == file_hash(source_lock_path), "source receipt lock binding mismatch")
    _require(receipt.get("data_archive_sha256") == file_hash(data_archive), "SUNRGBD archive identity drift")
    _require(lock.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "source lock/C0 binding mismatch")
    rule = lock["eligibility_rule"]
    eligible: list[dict[str, Any]] = []
    with zipfile.ZipFile(data_archive) as archive:
        member_names = set(archive.namelist())
        seg_members = sorted(name for name in member_names if name.endswith("/seg.mat") and any(marker in name for marker in ALLOWED_PATH_MARKERS))
        for seg_member in seg_members:
            root = seg_member.rsplit("/", 1)[0]
            rgb_member, depth_member = _members_for_root(member_names, root)
            mat = sio.loadmat(io.BytesIO(archive.read(seg_member)), squeeze_me=True, struct_as_record=False)
            names = np.atleast_1d(mat["names"])
            if not any(str(value).strip().lower() == "door" for value in names):
                continue
            depth = _decode_depth_png(archive.read(depth_member))
            targets = _door_targets(np.asarray(mat["seglabel"]), names, depth, rule)
            if not targets:
                continue
            near = any(target["target_depth_median_m"] <= float(rule["near_threshold_m"]) for target in targets)
            eligible.append({
                "sample_root": root,
                "sample_root_sha256": hashlib.sha256(root.encode("utf-8")).hexdigest(),
                "rgb_member": rgb_member,
                "depth_member": depth_member,
                "seg_member": seg_member,
                "stratum": "NEAR" if near else "FAR",
                "targets": [{key: value for key, value in target.items() if key != "mask"} for target in targets],
            })
    near = sorted((row for row in eligible if row["stratum"] == "NEAR"), key=lambda row: row["sample_root_sha256"])
    far = sorted((row for row in eligible if row["stratum"] == "FAR"), key=lambda row: row["sample_root_sha256"])
    near_take, far_take = int(rule["near_take"]), int(rule["far_take"])
    _require(len(near) >= near_take and len(far) >= far_take, f"SUNRGBD denominator insufficient: near={len(near)} far={len(far)}")
    selected = near[int(rule["near_skip"]):int(rule["near_skip"]) + near_take] + far[int(rule["far_skip"]):int(rule["far_skip"]) + far_take]
    _require(len(selected) == len(c0.get("episodes", [])), "selected roster/C0 size mismatch")
    cases = [{"case_id": f"sunrgbd-door-depth-case-{index:03d}", "episode_id": episode["episode_id"], **row} for index, (row, episode) in enumerate(zip(selected, c0["episodes"], strict=True), start=1)]
    roster = {
        "schema_version": ROSTER_SCHEMA,
        "source_lock_sha256": file_hash(source_lock_path),
        "source_receipt_sha256": file_hash(source_receipt_path),
        "goal_receipt_body_sha256": c0["receipt_body_sha256"],
        "eligible_case_count": len(eligible),
        "eligible_near_count": len(near),
        "eligible_far_count": len(far),
        "selection_rule": rule,
        "ancestry_exclusion_applied": "NYUdata and all non-predeclared source families excluded",
        "provider_truth_access": False,
        "cases": cases,
    }
    roster["roster_body_sha256"] = _json_hash(roster)
    _atomic_json(output, roster)
    return roster


def materialize_inputs(roster_path: Path, c0_path: Path, data_archive: Path, public_root: Path, private_root: Path, public_output: Path, private_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import io
    import scipy.io as sio
    import zipfile

    _require(not public_output.exists() and not private_output.exists(), "SUNRGBD inputs are immutable and already exist")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    declared = roster.pop("roster_body_sha256")
    _require(declared == _json_hash(roster), "roster body hash mismatch")
    roster["roster_body_sha256"] = declared
    goals = {episode["episode_id"]: episode for episode in c0["episodes"]}
    public_cases, private_cases = [], []
    with zipfile.ZipFile(data_archive) as archive:
        for row in roster["cases"]:
            rgb_path = (public_root / f"{row['case_id']}.jpg").resolve()
            _require(not rgb_path.exists(), "refusing to overwrite public RGB")
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_path.write_bytes(archive.read(row["rgb_member"]))
            mat = sio.loadmat(io.BytesIO(archive.read(row["seg_member"])), squeeze_me=True, struct_as_record=False)
            depth = _decode_depth_png(archive.read(row["depth_member"]))
            targets = _door_targets(np.asarray(mat["seglabel"]), np.atleast_1d(mat["names"]), depth, roster["selection_rule"])
            _require([target["label_id"] for target in targets] == [target["label_id"] for target in row["targets"]], "private target drift")
            legal_targets = []
            for target_index, target in enumerate(targets, start=1):
                mask_path = (private_root / f"{row['case_id']}-target-{target_index:02d}-mask.png").resolve()
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((target["mask"] * 255).astype(np.uint8)).save(mask_path)
                legal_targets.append({
                    "target_bbox_xyxy": target["bbox_xyxy"],
                    "target_mask_path": str(mask_path),
                    "target_mask_sha256": file_hash(mask_path),
                    "target_pixel_count": target["target_pixel_count"],
                    "depth_valid_pixel_count": target["depth_valid_pixel_count"],
                    "depth_valid_fraction": target["depth_valid_fraction"],
                    "target_depth_median_m": target["target_depth_median_m"],
                    "target_depth_p10_m": target["target_depth_p10_m"],
                    "target_depth_p90_m": target["target_depth_p90_m"],
                })
            goal = goals[row["episode_id"]]
            public_cases.append({"case_id": row["case_id"], "episode_id": row["episode_id"], "goal_contract": goal["goal_contract"] | {"canonical_prompt": goal["canonical_prompt"]}, "query": {"image_path": str(rgb_path), "image_sha256": file_hash(rgb_path)}})
            private_cases.append({"case_id": row["case_id"], "legal_targets": legal_targets, "true_interaction_range": any(target["target_depth_median_m"] <= float(roster["selection_rule"]["near_threshold_m"]) for target in legal_targets), "stratum": row["stratum"]})
    public = {"schema_version": PUBLIC_SCHEMA, "protocol_id": PROTOCOL_ID, "roster_body_sha256": roster["roster_body_sha256"], "private_truth_access": False, "cases": public_cases}
    _atomic_json(public_output, public)
    private = {"schema_version": PRIVATE_SCHEMA, "protocol_id": PROTOCOL_ID, "public_input_sha256": file_hash(public_output), "interaction_range_m": float(roster["selection_rule"]["near_threshold_m"]), "cases": private_cases}
    _atomic_json(private_output, private)
    return public, private


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("--source-lock", required=True, type=Path)
    download_parser.add_argument("--output-root", required=True, type=Path)
    download_parser.add_argument("--workers", type=int, default=12)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--source-lock", required=True, type=Path)
    verify_parser.add_argument("--data-archive", required=True, type=Path)
    verify_parser.add_argument("--toolbox-archive", required=True, type=Path)
    verify_parser.add_argument("--output", required=True, type=Path)
    roster_parser = sub.add_parser("freeze-roster")
    for name in ("source-lock", "source-receipt", "c0", "data-archive", "output"):
        roster_parser.add_argument(f"--{name}", required=True, type=Path)
    inputs_parser = sub.add_parser("materialize-inputs")
    for name in ("roster", "c0", "data-archive", "public-root", "private-root", "public-output", "private-output"):
        inputs_parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.source_lock, args.output_root, args.workers)
    elif args.command == "verify":
        verify(args.source_lock, args.data_archive, args.toolbox_archive, args.output)
    elif args.command == "freeze-roster":
        freeze_roster(args.source_lock, args.source_receipt, args.c0, args.data_archive, args.output)
    else:
        materialize_inputs(args.roster, args.c0, args.data_archive, args.public_root, args.private_root, args.public_output, args.private_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

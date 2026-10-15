"""Align and rescore the official StreamingRVOS-1B MeViS_U cached masks.

This is prediction replay, never local inference or measured runtime. Archive
entry offsets are pinned to HF revision in constants; only ~65 MB are needed.
Native empty/nonempty masks have no confidence; score=1 is an output indicator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

from acquire_pilot_data import rle_counts, rle_bbox
from analyze_pilot import analyze, load_manifest, load_predictions, write_csv, write_results

REVISION = "3a86dc3b7eb362202d3a4d2feb36961537cca0c9"
URL = f"https://huggingface.co/winking636/StreamingRVOS/resolve/{REVISION}/eval_results/1b.zip"
ENTRY_OFFSET = 64189375
COMPRESSED_SIZE = 64671596
UNCOMPRESSED_SIZE = 124097530
ARCHIVE_SIZE = 746224776
COMPRESSED_SHA256 = "14b1dfa11be0e00fa31a810b42646daef648ad10be296ea70c57712de3633e9d"
JSON_SHA256 = "e8ece89977c184dfe073da8a8cdce95356e81470e7871ae25d52434c5f69b50f"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(out: Path) -> None:
    if (out / "results.json").exists():
        if sha((out / "results.json").read_bytes()) != JSON_SHA256:
            raise ValueError("existing official JSON checksum mismatch")
        return
    out.mkdir(parents=True, exist_ok=True)

    def byte_range(start: int, end: int):
        response = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 90))
        response.raise_for_status()
        expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("official archive did not honor exact range")
        if len(response.content) != end - start + 1:
            raise ValueError("range response length mismatch")
        return response

    started = time.monotonic()
    header = byte_range(ENTRY_OFFSET, ENTRY_OFFSET + 29).content
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04" or fields[3] != 8:
        raise ValueError("unexpected ZIP local header or compression method")
    name_length, extra_length = fields[-2:]
    body_start = ENTRY_OFFSET + 30 + name_length + extra_length
    names = byte_range(ENTRY_OFFSET + 30, body_start - 1).content
    if names[:name_length].decode() != "1b/MEVIS_U/results.json":
        raise ValueError("unexpected archive entry identity")
    response = byte_range(body_start, body_start + COMPRESSED_SIZE - 1)
    compressed = response.content
    if sha(compressed) != COMPRESSED_SHA256:
        raise ValueError("compressed official entry checksum mismatch")
    plain = zlib.decompress(compressed, -15)
    if len(plain) != UNCOMPRESSED_SIZE or sha(plain) != JSON_SHA256:
        raise ValueError("official JSON checksum/length mismatch")
    (out / "mevis_u_results.deflate").write_bytes(compressed)
    (out / "results.json").write_bytes(plain)
    receipt = {
        "model": "StreamingRVOS-1B", "evidence_kind": "official precomputed prediction replay, not local model inference",
        "url": URL, "revision": REVISION, "archive_entry": "1b/MEVIS_U/results.json",
        "entry_local_header_offset": ENTRY_OFFSET, "entry_compressed_size": COMPRESSED_SIZE,
        "body_start": body_start, "range": response.headers["Content-Range"],
        "compressed_sha256": sha(compressed), "json_sha256": sha(plain),
        "payload_download_bytes": 30 + name_length + extra_length + len(compressed),
        "uncompressed_size": len(plain), "download_wall_s": time.monotonic() - started,
        "score_semantics": "Official masks have no confidence; presence is nonempty mask; score=1 is only an output indicator.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def decode_rle(rle: dict | None, shape: tuple[int, int]) -> np.ndarray:
    if rle is None:
        return np.zeros(shape, dtype=bool)
    if tuple(rle["size"]) != shape:
        raise ValueError("RLE/image dimensions differ")
    counts = rle_counts(rle)
    if min(counts) < 0 or sum(counts) != math.prod(shape):
        raise ValueError("invalid COCO RLE counts")
    flat = np.zeros(math.prod(shape), dtype=bool)
    offset = 0
    for index, length in enumerate(counts):
        if index % 2:
            flat[offset:offset + length] = True
        offset += length
    return flat.reshape(shape, order="F")


def boundary(mask: np.ndarray) -> np.ndarray:
    # Same one-pixel boundary convention as official MeViS tools/metrics.py.
    east = np.zeros_like(mask)
    south = np.zeros_like(mask)
    southeast = np.zeros_like(mask)
    east[:, :-1] = mask[:, 1:]
    south[:-1, :] = mask[1:, :]
    southeast[:-1, :-1] = mask[1:, 1:]
    result = (mask ^ east) | (mask ^ south) | (mask ^ southeast)
    result[-1, :] = mask[-1, :] ^ east[-1, :]
    result[:, -1] = mask[:, -1] ^ south[:, -1]
    result[-1, -1] = False
    return result


def mask_jf(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    union = np.count_nonzero(gt | pred)
    j = np.count_nonzero(gt & pred) / union if union else 1.0
    gb, pb = boundary(gt), boundary(pred)
    ng, npred = np.count_nonzero(gb), np.count_nonzero(pb)
    if not ng or not npred:
        return j, 1.0 if ng == npred else 0.0
    radius = int(np.ceil(0.008 * np.linalg.norm(gt.shape)))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx * xx + yy * yy <= radius * radius).astype(np.uint8)
    gd = cv2.dilate(gb.astype(np.uint8), disk).astype(bool)
    pd = cv2.dilate(pb.astype(np.uint8), disk).astype(bool)
    precision = np.count_nonzero(pb & gd) / npred
    recall = np.count_nonzero(gb & pd) / ng
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return j, f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    out = args.root / "official_streamingrvos"
    if args.fetch:
        fetch(out)
    source_bytes = (out / "results.json").read_bytes()
    if sha(source_bytes) != JSON_SHA256:
        raise ValueError("official source checksum mismatch")
    data = json.loads(source_bytes)
    rows = load_manifest(args.root / "data/samples.jsonl")
    meta = json.loads((args.root / "data/mevis_meta_expressions_v2.json").read_text(encoding="utf-8"))["videos"]
    masks = json.loads((args.root / "data/mevis_mask_dict.json").read_text(encoding="utf-8"))
    matched, missing, predictions, native = [], [], [], []
    for row in rows:
        vid, query_id = str(row["source_id"]), str(row["query_id"])
        item = data.get(vid, {}).get(query_id)
        if item is None:
            missing.append({"sequence_id": row["sequence_id"], "frame_idx": row["frame_idx"],
                            "reason": "official prediction expression not provided; NOT_EVALUABLE"})
            continue
        if item["video_id"] != vid or str(item["exp_id"]) != query_id:
            raise ValueError("official video/expression ID mismatch")
        wrappers = {f"<image>\nPlease segment {row['query']}.", f"<image>\\nPlease segment {row['query']}."}
        if item["exp"] not in wrappers:
            raise ValueError(f"expression text mismatch at {vid}/{query_id}")
        if len(set(item["frames"])) != len(item["frames"]) or len(item["frames"]) != len(item["prediction_masks"]):
            raise ValueError("official frame identity/mask length mismatch")
        frame = Path(row["image_path"]).stem
        frame_map = dict(zip(item["frames"], item["prediction_masks"]))
        if frame not in frame_map:
            missing.append({"sequence_id": row["sequence_id"], "frame_idx": row["frame_idx"],
                            "reason": "official prediction frame not provided; NOT_EVALUABLE"})
            continue
        if int(frame) != row["frame_idx"]:
            raise ValueError("manifest frame index differs from source basename")
        gt_index = meta[vid]["frames"].index(frame)
        aid = row.get("annotation_id")
        gt_rle = masks[str(aid)][gt_index] if aid is not None else None
        pred_rle = frame_map[frame]
        box = rle_bbox(pred_rle)
        if rle_bbox(gt_rle) != row["gt_bbox_xyxy"]:
            raise ValueError(f"GT frame/box mismatch at {vid}/{frame}")
        shape = int(row["height"]), int(row["width"])
        gt_mask, pred_mask = decode_rle(gt_rle, shape), decode_rle(pred_rle, shape)
        j, f = mask_jf(gt_mask, pred_mask)
        mask_area = int(np.count_nonzero(pred_mask))
        predictions.append({
            "sequence_id": row["sequence_id"], "frame_idx": row["frame_idx"], "query": row["query"],
            "candidates": [{"bbox_xyxy": box, "score": 1.0}] if box is not None else [],
            "presence_score": 1.0 if box is not None else 0.0,
            "score_semantics": "binary_nonempty_mask_indicator_not_confidence",
            "native_mask_present": box is not None, "elapsed_s": None,
            "selection_method": "official_StreamingRVOS_1B_mask_bbox_replay",
            "source_video_id": vid, "source_expression_id": query_id, "source_frame": frame,
        })
        matched.append(row)
        native.append({"sequence_id": row["sequence_id"], "source_id": vid,
                       "frame_idx": row["frame_idx"], "gt_present": row["gt_present"],
                       "pred_mask_nonempty": box is not None,
                       "pred_mask_area_pixels": mask_area,
                       "pred_mask_area_image_fraction": mask_area / math.prod(shape),
                       "J": j, "F": f, "JF": (j + f) / 2})
        if len(matched) % 36 == 0:
            print(json.dumps({"matched_scored": len(matched), "total_rows": len(rows)}), flush=True)
    for name, records in (("matched_manifest.jsonl", matched), ("official_predictions.jsonl", predictions),
                          ("not_evaluable_rows.jsonl", missing)):
        (out / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    observations, provenance = load_predictions([out / "official_predictions.jsonl"])
    result = analyze(matched, observations)
    result["inputs"] = provenance
    result["official_cache_identity_alignment"] = {
        "original_manifest_frames": len(rows), "matched_frames": len(matched), "not_evaluable_frames": len(missing),
        "join_keys": ["source_id/video_id", "query_id/exp_id", "image basename/official frames"],
        "expression_text_exact_match_with_official_prompt_wrapper": True,
        "gt_bbox_verified_against_native_mask": True,
        "scores": "not provided; binary nonempty-mask indicators, no confidence curve",
        "runtime": "not measured; official precomputed mask replay only",
        "missing_predictions": "excluded as NOT_EVALUABLE; never interpreted as absent",
        "source_sha256": JSON_SHA256, "source_revision": REVISION,
    }
    by_seq = defaultdict(list)
    for row in native:
        by_seq[row["sequence_id"]].append(row)
    seq_metrics = [{"sequence_id": seq, "frames": len(sub),
                    **{metric: sum(row[metric] for row in sub) / len(sub) for metric in ("J", "F", "JF")}}
                   for seq, sub in by_seq.items()]
    result["native_mask_metrics"] = {
        "metric_source": "https://github.com/henghuiding/MeViS/blob/main/tools/metrics.py",
        "boundary_tolerance_fraction_image_diagonal": 0.008,
        "empty_empty_J_and_F": 1.0,
        "frame_weighted": {metric: sum(row[metric] for row in native) / len(native) for metric in ("J", "F", "JF")},
        "query_macro": {metric: sum(row[metric] for row in seq_metrics) / len(seq_metrics) for metric in ("J", "F", "JF")},
        "visible_frame_weighted": {metric: sum(row[metric] for row in native if row["gt_present"]) /
                                    sum(row["gt_present"] for row in native) for metric in ("J", "F", "JF")},
        "absent_mask_area_descriptive_not_threshold_selection": {
            "absent_denominator": sum(not row["gt_present"] for row in native),
            "nonempty": sum(not row["gt_present"] and row["pred_mask_nonempty"] for row in native),
            "ge_10_pixels": sum(not row["gt_present"] and row["pred_mask_area_pixels"] >= 10 for row in native),
            "ge_100_pixels": sum(not row["gt_present"] and row["pred_mask_area_pixels"] >= 100 for row in native),
            "ge_0_1pct_image": sum(not row["gt_present"] and row["pred_mask_area_image_fraction"] >= 0.001 for row in native),
        },
        "limits": "Pilot windows selected from annotations; not full official benchmark or fresh held-out confirmation.",
    }
    write_results(result, out / "analysis")
    write_csv(out / "analysis/native_mask_per_frame.csv", native)
    write_csv(out / "analysis/native_mask_per_sequence.csv", seq_metrics)
    print(json.dumps({"alignment": result["official_cache_identity_alignment"],
                      "bbox_metrics": result["baseline_thresholded"], "native": result["native_mask_metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

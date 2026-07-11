#!/usr/bin/env python3
"""Discover public SANPO-Real continuous-sequence candidates without downloading RGB video."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sanpo_sequence_evalset import (  # noqa: E402
    GCS_PREFIX,
    DEFAULT_CAMERA,
    DEFAULT_LENS,
    GCS_API,
    fetch_json,
    fetch_text,
    frame_number,
    get_gcs_object,
    list_gcs_objects,
    media_url,
)


LABELS = {
    "curb": 2,
    "stairs": 15,
    "inaccessible_surface": 18,
    "generic_obstacle": 20,
    "pole": 24,
}


def mask_class_counts(url: str, wanted: set[int]) -> Counter[int]:
    import io
    from urllib.request import urlopen

    with urlopen(url, timeout=60) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")
        ids = np.asarray(image, dtype=np.uint8)[:, :, 0]
    values, counts = np.unique(ids, return_counts=True)
    return Counter({int(value): int(count) for value, count in zip(values, counts, strict=True) if int(value) in wanted})


def session_ids(split: str) -> list[str]:
    name = f"{GCS_PREFIX}/sanpo-real/splits/{split}_session_ids.txt"
    item = get_gcs_object(name)
    return [line.strip() for line in fetch_text(media_url(name, item.get("generation"))).splitlines() if line.strip()]


def first_mask_page(prefix: str) -> list[dict]:
    """One GCS page is sufficient for discovery; full inventory is deferred to download."""
    payload = fetch_json(f"{GCS_API}?{urlencode({'prefix': prefix, 'maxResults': 1000})}")
    return [item for item in payload.get("items", []) if item["name"].endswith(".png")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--max-sessions", type=int, default=0, help="0 scans every session in the selected split")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--minimum-hits", type=int, default=3)
    parser.add_argument("--labels", nargs="+", choices=sorted(LABELS), default=sorted(LABELS))
    args = parser.parse_args()
    if args.sample_count <= 0 or args.minimum_hits <= 0:
        raise SystemExit("--sample-count and --minimum-hits must be positive")

    splits = ("train", "test") if args.split == "all" else (args.split,)
    wanted_by_name = {name: LABELS[name] for name in args.labels}
    wanted = set(wanted_by_name.values())
    records: list[dict] = []
    failures: list[dict] = []
    for split in splits:
        ids = session_ids(split)
        if args.max_sessions:
            ids = ids[: args.max_sessions]
        for index, session_id in enumerate(ids, start=1):
            try:
                prefix = f"{GCS_PREFIX}/sanpo-real/{session_id}/{DEFAULT_CAMERA}/{DEFAULT_LENS}/segmentation_masks/"
                items = first_mask_page(prefix)
                numbers = sorted(frame_number(item["name"]) for item in items)
                if len(numbers) < args.sample_count:
                    continue
                sampled = [numbers[round(i * (len(numbers) - 1) / (args.sample_count - 1))] for i in range(args.sample_count)]
                by_number = {frame_number(item["name"]): item for item in items}
                class_frames: dict[str, list[int]] = {name: [] for name in wanted_by_name}
                for frame in sampled:
                    item = by_number[frame]
                    counts = mask_class_counts(media_url(item["name"], item.get("generation")), wanted)
                    for name, class_id in wanted_by_name.items():
                        if counts[class_id] > 0:
                            class_frames[name].append(frame)
                for name, frames in class_frames.items():
                    if len(frames) >= args.minimum_hits:
                        records.append({
                            "session_id": session_id,
                            "official_split": split,
                            "camera": DEFAULT_CAMERA,
                            "lens": DEFAULT_LENS,
                            "target_label": name,
                            "sampled_source_frames": sampled,
                            "matching_source_frames": frames,
                            "recommended_start_frame": max(0, frames[len(frames) // 2] - 15),
                            "license": "Creative Commons Attribution 4.0 International",
                            "dataset_page": "https://google-research-datasets.github.io/sanpo_dataset/",
                        })
                print(f"scanned={split}:{index}/{len(ids)} records={len(records)}", flush=True)
            except Exception as error:  # network/data errors are reported, not hidden
                failures.append({"session_id": session_id, "official_split": split, "error": f"{type(error).__name__}: {error}"})
    payload = {
        "source": "SANPO-Real v0 public GCS metadata and sparse segmentation-mask scan",
        "license": "Creative Commons Attribution 4.0 International",
        "camera": DEFAULT_CAMERA,
        "lens": DEFAULT_LENS,
        "sample_count": args.sample_count,
        "minimum_hits": args.minimum_hits,
        "labels": wanted_by_name,
        "candidates": records,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"candidates={len(records)} failures={len(failures)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

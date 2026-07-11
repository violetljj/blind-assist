#!/usr/bin/env python3
"""Discover public SANPO-Real candidates with auditable mask geometry.

This sparse pass never downloads RGB.  It is intentionally a shortlist only:
every shortlisted session still needs a downloaded, exact 50-frame geometry
gate and model review before it can enter the dense-annotation queue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
from PIL import Image

from select_sanpo_sequence_by_geometry import (
    CENTER_HAZARD_IDS,
    PROFILE_TARGETS,
    components_for_mask,
)

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
    "pedestrian": 12,
    "rider": 13,
    "stairs": 15,
    "inaccessible_surface": 18,
    "generic_obstacle": 20,
    "vehicle": 21,
    "pole": 24,
}


def mask_geometry(url: str) -> tuple[dict[int, list[dict]], dict[str, float]]:
    import io
    from urllib.request import urlopen

    with urlopen(url, timeout=60) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")
        return components_for_mask(np.asarray(image, dtype=np.uint8))


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def sparse_profile_evidence(components: dict[int, list[dict]], path: dict[str, float]) -> dict[str, Any]:
    """Classify one sparse mask without treating mere label presence as evidence."""
    central_hazards = [
        component for class_id in CENTER_HAZARD_IDS for component in components.get(class_id, [])
        if component["corridor_target_ratio"] >= 0.12 and component["bottom_ratio"] >= 0.45
    ]
    center_targets = [
        component for class_id in PROFILE_TARGETS["center_obstacle"] for component in components.get(class_id, [])
        if component["corridor_target_ratio"] >= 0.12 and component["bottom_ratio"] >= 0.45
    ]
    lateral_targets = [
        component for class_id in PROFILE_TARGETS["lateral_pedestrian_or_ebike"] for component in components.get(class_id, [])
        if component["corridor_target_ratio"] <= 0.01
        and (component["center_x_ratio"] <= 0.35 or component["center_x_ratio"] >= 0.65)
        and component["bottom_ratio"] >= 0.35
    ]
    center_lateral_targets = [
        component for class_id in PROFILE_TARGETS["lateral_pedestrian_or_ebike"] for component in components.get(class_id, [])
        if component["corridor_target_ratio"] >= 0.12 and component["bottom_ratio"] >= 0.45
    ]
    path_ok = path["walkable_corridor_ratio"] >= 0.18
    return {
        "path_geometry_usable": path_ok,
        "walkable_corridor_ratio": path["walkable_corridor_ratio"],
        "center_obstacle": bool(center_targets) and path_ok,
        "lateral_pedestrian_or_ebike": bool(lateral_targets) and not center_lateral_targets and not central_hazards and path_ok,
        "has_center_hazard": bool(central_hazards),
        "best_center_target": max(center_targets, key=lambda item: item["corridor_blocking_ratio"], default=None),
        "best_lateral_target": max(lateral_targets, key=lambda item: item["bottom_ratio"], default=None),
    }


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
                frame_evidence: list[dict] = []
                for frame in sampled:
                    item = by_number[frame]
                    components, path = mask_geometry(media_url(item["name"], item.get("generation")))
                    profile = sparse_profile_evidence(components, path)
                    frame_evidence.append({"source_frame": frame, "profiles": profile})
                for profile_name in ("center_obstacle", "lateral_pedestrian_or_ebike"):
                    matches = [item for item in frame_evidence if item["profiles"][profile_name]]
                    match_bools = [item["profiles"][profile_name] for item in frame_evidence]
                    if len(matches) >= args.minimum_hits and longest_run(match_bools) >= 2:
                        frames = [item["source_frame"] for item in matches]
                        records.append({
                            "session_id": session_id,
                            "official_split": split,
                            "camera": DEFAULT_CAMERA,
                            "lens": DEFAULT_LENS,
                            "selection_profile": profile_name,
                            "sampled_source_frames": sampled,
                            "geometry_matching_source_frames": frames,
                            "sparse_longest_consecutive_sample_run": longest_run(match_bools),
                            "sparse_frame_evidence": frame_evidence,
                            "recommended_start_frame": max(0, frames[len(frames) // 2] - 15),
                            "next_gate": "download an exact 50-frame draft, run select_sanpo_sequence_by_geometry.py, then model review",
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
        "selection_method": {
            "version": "corridor-path-persistence-v1",
            "center_obstacle": "sparse source-mask target intrudes into the near-field center corridor; path has walkable support; repeated in >=minimum-hits sampled frames",
            "lateral_pedestrian_or_ebike": "pedestrian/rider stays outside the corridor, path has walkable support, and no other center hazard occurs in sampled frames",
            "important_limit": "sparse results are not a 50-frame acceptance. The exact draft selector is mandatory before model review.",
        },
        "candidates": records,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"candidates={len(records)} failures={len(failures)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

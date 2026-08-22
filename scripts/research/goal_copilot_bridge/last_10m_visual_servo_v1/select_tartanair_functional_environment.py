#!/usr/bin/env python3
"""Select the smallest untouched exact-door TartanAir environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require


REQUIRED_SUFFIXES = ("Data_easy/seg_lcam_front.zip", "Data_easy/depth_lcam_front.zip")


def exact_door_environments(label_root: Path) -> set[str]:
    result = set()
    for path in label_root.glob("*/seg_label_map.json"):
        labels = _read(path).get("name_map", {})
        names = labels.keys() if isinstance(labels, dict) else ()
        if "door" in names:
            result.add(path.parent.name)
    return result


def rank_environments(files, allowed: set[str], consumed: set[str]) -> list[dict]:
    sizes: dict[str, dict[str, int]] = {}
    for file in files:
        path = getattr(file, "path", "")
        environment = path.split("/", 1)[0]
        suffix = next((item for item in REQUIRED_SUFFIXES if path.endswith(item)), None)
        if suffix is None or environment not in allowed or environment in consumed:
            continue
        sizes.setdefault(environment, {})[suffix] = int(getattr(file, "size", 0) or 0)
    rows = [
        {"environment": environment, "required_archive_bytes": sum(modalities.values()), "modalities": modalities}
        for environment, modalities in sizes.items()
        if set(modalities) == set(REQUIRED_SUFFIXES)
    ]
    return sorted(rows, key=lambda row: (row["required_archive_bytes"], row["environment"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--consumed", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "environment selection receipt already exists")
    api = HfApi()
    print("listing official TartanAir repository tree", flush=True)
    info = api.dataset_info("theairlabcmu/tartanair2")
    files = list(api.list_repo_tree("theairlabcmu/tartanair2", repo_type="dataset", recursive=True, expand=True))
    ranking = rank_environments(files, exact_door_environments(args.label_root), set(args.consumed))
    _require(bool(ranking), "no untouched exact-door environment available")
    payload = {
        "schema_version": "blindassist_tartanair_functional_environment_selection_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "theairlabcmu/tartanair2",
        "repository_revision": info.sha,
        "consumed_environments": sorted(set(args.consumed)),
        "selection_rule": "minimum public seg+depth required archive bytes among untouched exact-door environments",
        "selected": ranking[0],
        "ranking": ranking,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"repository_revision": info.sha, "selected": ranking[0], "candidate_count": len(ranking)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

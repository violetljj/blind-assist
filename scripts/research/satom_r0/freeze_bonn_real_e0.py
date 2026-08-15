#!/usr/bin/env python3
"""Print the metadata-only Bonn roster section for the SATOM-R0 pre-outcome lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bonn import frozen_frame_rows, rank_hash, sha256_file


def build(metadata_root: Path, contract: dict) -> dict:
    excluded = set(contract["excluded_sequence_ids"])
    candidates = sorted(
        (path for path in metadata_root.iterdir() if path.is_dir() and path.name not in excluded),
        key=lambda path: (rank_hash(path.name, str(contract["parent_rank_seed"])), path.name),
    )
    selected = candidates[: int(contract["parent_count"])]
    if len(selected) != int(contract["parent_count"]):
        raise ValueError("insufficient Bonn parent candidates")
    parents = []
    for rank, root in enumerate(selected):
        frames = frozen_frame_rows(root, contract)
        parents.append(
            {
                "rank": rank,
                "sequence_id": root.name,
                "rank_sha256": rank_hash(root.name, str(contract["parent_rank_seed"])),
                "official_url": str(contract["official_url_template"]).format(sequence_id=root.name),
                "metadata": {
                    name: {
                        "bytes": (root / name).stat().st_size,
                        "sha256": sha256_file(root / name),
                    }
                    for name in ("rgb.txt", "depth.txt", "groundtruth.txt")
                },
                "selected_frame_count": len(frames),
                "first_rgb_timestamp_s": frames[0]["rgb_timestamp_s"],
                "last_rgb_timestamp_s": frames[-1]["rgb_timestamp_s"],
                "selected_frame_identity_sha256": rank_hash(
                    "|".join(
                        f'{row["rgb_timestamp_s"]:.6f}:{row["rgb_relative_path"]}:{row["depth_timestamp_s"]:.6f}:{row["depth_relative_path"]}'
                        for row in frames
                    ),
                    "SATOM_R0_BONN_FRAME_IDENTITY_V1",
                ),
            }
        )
    return {"parents": parents, "candidate_parent_count": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))["source_contract"]
    print(json.dumps(build(args.metadata_root, contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

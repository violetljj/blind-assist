#!/usr/bin/env python3
"""Freeze a physical-target-disjoint 3RScan dual-surface confirmation cohort."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_center_target_door_retrieval as base  # noqa: E402


class PhysicalTargetExclusions(set[tuple[str, str, int]]):
    def __init__(
        self,
        triples: set[tuple[str, str, int]],
        physical_targets: set[tuple[str, int]],
    ) -> None:
        super().__init__(triples)
        self.physical_targets = physical_targets

    def __contains__(self, value: object) -> bool:
        if super().__contains__(value):
            return True
        if not isinstance(value, tuple) or len(value) != 3:
            return False
        return (str(value[0]), int(value[2])) in self.physical_targets


def consumed_physical_targets(
    protocol: dict[str, Any],
) -> tuple[PhysicalTargetExclusions, list[dict[str, Any]]]:
    triples: set[tuple[str, str, int]] = set()
    physical_targets: set[tuple[str, int]] = set()
    receipts: list[dict[str, Any]] = []
    for item in protocol["source"]["consumed_target_cohorts"]:
        path = HERE / item["path"]
        base.verify_path(path, item["sha256"], "CONSUMED_COHORT")
        cohort = base.load_json(path)
        found = 0
        for episode in cohort.get("episodes", []):
            if not all(
                key in episode
                for key in ("reference_scan_id", "rescan_id", "target_instance_id")
            ):
                continue
            triple = (
                str(episode["reference_scan_id"]),
                str(episode["rescan_id"]),
                int(episode["target_instance_id"]),
            )
            triples.add(triple)
            physical_targets.add((triple[0], triple[2]))
            found += 1
        receipts.append(
            {
                "path": path.name,
                "sha256": item["sha256"],
                "target_triples": found,
                "physical_targets": len(
                    {
                        (str(row["reference_scan_id"]), int(row["target_instance_id"]))
                        for row in cohort.get("episodes", [])
                        if all(
                            key in row
                            for key in ("reference_scan_id", "rescan_id", "target_instance_id")
                        )
                    }
                ),
            }
        )
    return PhysicalTargetExclusions(triples, physical_targets), receipts


@contextmanager
def confirmation_surface():
    saved_file = base.__file__
    saved_consumed = base.consumed_triples
    base.__file__ = str(Path(__file__).resolve())
    base.consumed_triples = consumed_physical_targets
    try:
        yield
    finally:
        base.__file__ = saved_file
        base.consumed_triples = saved_consumed


def freeze(protocol: Path, artifact_root: Path, output: Path) -> None:
    with confirmation_surface():
        base.freeze(protocol, artifact_root, output)
    cohort = base.load_json(output)
    cohort["authority"] = "FROZEN_PRE_RGB_PHYSICAL_TARGET_DISJOINT_3RSCAN_DUAL_SURFACE_CONFIRMATION_COHORT"
    cohort["selection"]["physical_target_exclusion"] = True
    cohort["selection"]["model_calls"] = 0
    base.write_json(output, cohort)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.artifact_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

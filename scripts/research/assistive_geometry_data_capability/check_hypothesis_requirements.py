#!/usr/bin/env python3
"""Evaluate a versioned hypothesis requirement contract against an existing AG-DCA atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.assistive_geometry_data_capability.build_capability_atlas import (
    REPO_ROOT,
    REQUIREMENTS_RELATIVE,
    evaluate_requirements,
    load_json,
    require,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        type=Path,
        default=REPO_ROOT / "artifacts.local/evidence/assistive-geometry-data-capability/r0/atlas.json",
    )
    parser.add_argument("--requirements", type=Path, default=REPO_ROOT / REQUIREMENTS_RELATIVE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    atlas = load_json(args.atlas.resolve())
    requirements = load_json(args.requirements.resolve())
    require(atlas.get("schema") == "blindassist.assistive_geometry_dca.r0_atlas.v1", "atlas schema drift")
    decisions = evaluate_requirements(atlas, requirements, atlas["authority_facts"])
    result = {
        "schema": "blindassist.assistive_geometry_dca.requirement_check.v1",
        "atlas_sha256": sha256_file(args.atlas.resolve()),
        "requirements_sha256": sha256_file(args.requirements.resolve()),
        "decisions": decisions,
        "execution_authorized": False,
        "claim_ceiling": "Requirement admission only. A PASS permits a separate protocol lock, never direct execution.",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        require(not output.exists(), f"output collision: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

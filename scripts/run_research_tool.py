"""Stable adapter for research tools whose implementations live below scripts/research/."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DOMAINS = {
    "egomotion-compensated-looming": SCRIPTS_DIR / "research" / "egomotion_compensated_looming",
    "public-video": SCRIPTS_DIR / "research" / "public_video",
    "ustrf-crosscam-codex": SCRIPTS_DIR / "research" / "ustrf_crosscam_codex",
    "ustrf-route-target-evidence-closure": SCRIPTS_DIR / "research" / "ustrf_route_target_evidence_closure",
    "ustrf-sensor-replay": SCRIPTS_DIR / "research" / "ustrf_sensor_replay",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a research CLI through a stable repository entry point."
    )
    parser.add_argument("domain", choices=sorted(DOMAINS))
    parser.add_argument("tool", help="Python filename inside the selected domain")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    tool_name = Path(args.tool)
    if tool_name.name != args.tool or tool_name.suffix != ".py":
        parser.error("tool must be one Python filename without directory components")

    domain_dir = DOMAINS[args.domain].resolve()
    target = (domain_dir / tool_name.name).resolve()
    if target.parent != domain_dir or not target.is_file():
        parser.error(f"unknown {args.domain} tool: {args.tool}")

    # Keep historical sibling imports and stable root helpers behind this Adapter.
    sys.path[:0] = [str(domain_dir), str(SCRIPTS_DIR), str(REPO_ROOT)]
    sys.argv = [str(target), *args.arguments]
    os.chdir(REPO_ROOT)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Private evaluator entry point for PA3 semantic candidate availability."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("PA3 evaluation already exists; refusing replay")
    payload = evaluate(args.public, args.private, args.prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

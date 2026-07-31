"""Validate the current B Development implementation lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .create_implementation_lock import build_lock


def validate(lock_path: Path, repo_root: Path) -> dict[str, object]:
    actual = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = build_lock(repo_root)
    if actual != expected:
        raise ValueError("implementation lock drifted from current module or contract")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    result = validate(args.lock, args.repo_root or Path(__file__).resolve().parents[3])
    print(json.dumps({"status": result["status"], "implementation_id": result["implementation_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

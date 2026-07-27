from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .validator import validate


def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(
        args.repo_root.resolve(),
        args.contract.resolve(),
        args.cache_dir.resolve(),
        args.output_dir.resolve(),
    )
    write_exclusive(
        args.output_dir.resolve() / "validation.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())

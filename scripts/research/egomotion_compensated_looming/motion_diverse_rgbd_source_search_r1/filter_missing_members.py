"""Create an exact include file containing only locally missing members."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    members = [
        line.strip()
        for line in args.members.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    root = args.root.resolve()
    missing = [
        member
        for member in members
        if not root.joinpath(*Path(member).parts).is_file()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(json.dumps({"member_count": len(members), "missing_member_count": len(missing)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

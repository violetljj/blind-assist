#!/usr/bin/env python3
"""Collect the fresh R1C-G0 cohort through the frozen R1C-L renderer mechanics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_r1cl_collector(path: Path) -> Any:
    import sys

    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("grail_r1cl_collector_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen collector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-r1cl-collector", type=Path, required=True)
    parser.add_argument("--house-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blindassist_grail_r1c_g0_manifest_v1":
        raise ValueError("R1C-G0 manifest schema mismatch")
    collector = _load_r1cl_collector(args.frozen_r1cl_collector.resolve())
    result = collector.collect(
        args.dataset,
        args.manifest,
        "validation",
        args.output,
        args.house_limit,
        True,
        args.shard_index,
        args.shard_count,
    )
    result["schema"] = "blindassist_grail_r1c_g0_collection_v1"
    result["inherited_renderer"] = {
        "route": "GRAIL-R1C-L",
        "file": args.frozen_r1cl_collector.name,
        "semantic_change": "none; only the fresh roster and downstream evaluator differ",
    }
    collection_path = args.output / "validation" / "collection.json"
    collection_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

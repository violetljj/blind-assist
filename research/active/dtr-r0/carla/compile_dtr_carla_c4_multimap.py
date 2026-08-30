"""Compile the C4 multimap registries into one C2-compatible protocol per map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dtr_carla_c2_rich_scene import write_json_atomic
from dtr_carla_c4_scene import compile_multimap, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-c2-protocol", type=Path, required=True)
    parser.add_argument("--source-c3-asset-registry", type=Path, required=True)
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing C4 compile overwrite: {output_dir}")
    base_c2_protocol = load_json(args.base_c2_protocol.resolve(strict=True))
    source_c3_registry = load_json(args.source_c3_asset_registry.resolve(strict=True))
    asset_registry = load_json(args.asset_registry.resolve(strict=True))
    scene_registry = load_json(args.scene_registry.resolve(strict=True))
    protocols, index, receipt = compile_multimap(
        base_c2_protocol,
        source_c3_registry,
        asset_registry,
        scene_registry,
    )
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "dtr_carla_c4_asset_registry.json", asset_registry)
    write_json_atomic(output_dir / "dtr_carla_c4_scene_registry.json", scene_registry)
    for protocol_id, protocol in sorted(protocols.items()):
        write_json_atomic(output_dir / f"{protocol_id}.c2-protocol.json", protocol)
    write_json_atomic(output_dir / "compiled-index.json", index)
    write_json_atomic(output_dir / "compiler-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

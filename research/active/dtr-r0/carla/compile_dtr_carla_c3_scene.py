"""Compile one admitted C3 scene into an immutable C2 capture protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dtr_carla_c2_rich_scene import write_json_atomic
from dtr_carla_c3_scene import compile_scene, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-c2-protocol", type=Path, required=True)
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-protocol", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = [args.output_protocol.resolve(), args.output_receipt.resolve()]
    if outputs[0] == outputs[1]:
        raise ValueError("protocol and receipt outputs must differ")
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing output overwrite: {outputs}")
    protocol, receipt = compile_scene(
        load_json(args.base_c2_protocol.resolve(strict=True)),
        load_json(args.asset_registry.resolve(strict=True)),
        load_json(args.scene_registry.resolve(strict=True)),
        str(args.scene_id),
    )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(outputs[0], protocol)
    write_json_atomic(outputs[1], receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

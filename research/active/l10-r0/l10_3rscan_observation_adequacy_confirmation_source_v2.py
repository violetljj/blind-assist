#!/usr/bin/env python3
"""Run the second frozen 3RScan observation-adequacy source gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import l10_3rscan_observation_adequacy_confirmation_source as base


base.PROTOCOL_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-confirmation-source-protocol-v2"
base.RESULT_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-confirmation-source-result-v2"
base.__file__ = __file__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = base.pixel.load_json(protocol_path)
    dependency = protocol["base_implementation"]
    base.pixel.require(
        base.pixel.sha256(base.HERE / dependency["path"]) == dependency["sha256"],
        "BASE_IMPLEMENTATION_HASH",
    )
    base.run(protocol_path, args.output.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen vacancy scorer and give its one-family result exact Development authority."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_axis_completion_vacancy_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-axis-completion-vacancy-posthoc-protocol-v1"


def run(protocol_path: Path, raw_path: Path, output_path: Path) -> None:
    protocol = base.consensus.base.pixel.load_json(protocol_path)
    base.consensus.base.pixel.require(
        protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA"
    )
    adapter = protocol["development_adapter"]
    base.consensus.base.pixel.require(
        base.consensus.base.pixel.sha256(Path(__file__)) == adapter["sha256"],
        "ADAPTER_HASH",
    )
    base.run(protocol_path, raw_path)
    raw = base.consensus.base.pixel.load_json(raw_path)
    expected = int(adapter["expected_unique_query_count"])
    base.consensus.base.pixel.require(
        raw["metrics"]["family_count"] == 1, "FAMILY_COUNT"
    )
    base.consensus.base.pixel.require(
        raw["metrics"]["query_count"] == expected, "UNIQUE_QUERY_COUNT"
    )
    result = deepcopy(raw)
    result["schema"] = (
        "blindassist-l10-3rscan-asymmetric-reference-vacancy-development-result-v1"
    )
    result["authority"] = (
        "CONSUMED_NINTH_FAMILY_ASYMMETRIC_REFERENCE_VACANCY_DEVELOPMENT"
    )
    result["conclusion"] = (
        "L10_3RSCAN_ASYMMETRIC_REFERENCE_VACANCY_DEVELOPMENT_GATE_MET"
        if raw["gate_met"]
        else "L10_3RSCAN_ASYMMETRIC_REFERENCE_VACANCY_DEVELOPMENT_GATE_NOT_MET"
    )
    result["implementation"] = {
        "path": Path(__file__).name,
        "sha256": base.consensus.base.pixel.sha256(Path(__file__)),
        "base_path": Path(base.__file__).name,
        "base_sha256": base.consensus.base.pixel.sha256(Path(base.__file__)),
    }
    result["base_result"] = {
        "path": raw_path.name,
        "sha256": base.consensus.base.pixel.sha256(raw_path),
    }
    result["claim_boundary"] = protocol["claim_boundary"]
    base.consensus.base.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.raw_output.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

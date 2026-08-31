#!/usr/bin/env python3
"""V2 fresh-panel selector with prior-evidence item exclusions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import l10_panolab_track_lexical_fresh_panel as base


SCHEMA = "blindassist-l10-panolab-track-lexical-fresh-source-protocol-v2"


def main() -> None:
    base.require("--protocol" in sys.argv, "PROTOCOL_ARGUMENT_REQUIRED")
    protocol_path = Path(sys.argv[sys.argv.index("--protocol") + 1]).resolve()
    protocol = base.load(protocol_path)
    base.require(protocol.get("schema") == SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    base_path = base.verify(protocol["inputs"]["base_v1_evaluator"])
    base.require(base_path.resolve() == Path(base.__file__).resolve(), "LOADED_BASE_EVALUATOR_MISMATCH")
    base.verify(protocol["inputs"]["rejected_v1_selection"])
    base.verify(protocol["inputs"]["v1_rejection_audit"])

    original_schema = base.SCHEMA
    original_file = base.__file__
    original_candidate_pairs = base.candidate_pairs

    def candidate_pairs_v2(
        candidates: dict[str, Any],
        orientation_protocol: dict[str, Any],
        prior_source: dict[str, Any],
        prior_local_ids: set[str],
        contract: dict[str, Any],
    ) -> list[dict[str, Any]]:
        excluded = set(contract["additional_prior_evidence_item_ids"])
        return original_candidate_pairs(
            candidates,
            orientation_protocol,
            prior_source,
            prior_local_ids | excluded,
            contract,
        )

    base.SCHEMA = SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    base.candidate_pairs = candidate_pairs_v2
    try:
        base.main()
    finally:
        base.SCHEMA = original_schema
        base.__file__ = original_file
        base.candidate_pairs = original_candidate_pairs


if __name__ == "__main__":
    main()

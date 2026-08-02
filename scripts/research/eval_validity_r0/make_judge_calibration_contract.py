"""Derive an explicit burned-calibration contract from the formal contract.

The derived file is an execution receipt for the 8-12 event pilot only.  It
does not relax the formal contract: category counts are zero only because the
pilot intentionally keeps scenario coverage unclassified and outside the
formal denominator.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common import PROTOCOL_ID, read_json, sha256_file
from .judge_audit import JUDGE_CONTRACT_SCHEMA, _contract


def derive(formal: dict, *, formal_path: Path) -> dict:
    _contract(formal)
    result = copy.deepcopy(formal)
    result["mode"] = "CALIBRATION_BURNED"
    result["cohort_role"] = "CALIBRATION_BURNED"
    result["minimum_events"] = 8
    result["maximum_events"] = 12
    result["minimum_counterfactual_pairs"] = 3
    result["maximum_counterfactual_pairs"] = 4
    result["required_coverage_min_counts"] = {category: 0 for category in result["required_coverage_min_counts"]}
    result["required_coverage_min_source_sessions"] = {category: 0 for category in result["required_coverage_min_source_sessions"]}
    result["calibration_only"] = True
    result["formal_denominator_inclusion"] = False
    result["formal_category_coverage_required"] = False
    result["derived_from_formal_contract_sha256"] = sha256_file(formal_path)
    result["next_gate"] = "After review seal, run only the burned pilot; do not use this derived contract for formal model comparison."
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite calibration contract: {args.output}")
    result = derive(read_json(args.formal_contract), formal_path=args.formal_contract)
    _contract(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status=CALIBRATION_CONTRACT_DERIVED protocol={PROTOCOL_ID} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_LOCK = Path("docs/research/taro/TARO_O1R_R7_FRESH_PARENT_DISJOINT_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json")
EXPECTED_BYTES = 4583
EXPECTED_SHA256 = "1419070D09951AE7251C9832EF006C329F82D1DA1C46DB8F759ABBF6ECA11A01"


def validate(path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    errors: list[str] = []
    if len(raw) != EXPECTED_BYTES or hashlib.sha256(raw).hexdigest().upper() != EXPECTED_SHA256:
        errors.append("fresh confirmation protocol identity mismatch")
    lock = json.loads(raw.decode("utf-8"))
    if lock.get("status") != "FROZEN" or lock.get("lock_id") != "TARO_O1R_R7_FRESH_PARENT_DISJOINT_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK":
        errors.append("fresh confirmation protocol status/id drift")
    candidate = lock.get("frozen_candidate", {})
    if (candidate.get("minimum_connected_confidence2_pixels"), candidate.get("minimum_component_max_height_m"), candidate.get("maximum_component_min_forward_m")) != (2, 0.08, 2.0):
        errors.append("fresh positive-occupancy candidate drift")
    if candidate.get("positive_evidence_output") != "OCCUPIED_OBSERVED" or candidate.get("absence_of_positive_evidence_output") != "UNKNOWN" or candidate.get("clear_output_allowed") is not False or candidate.get("threshold_search_or_selector_fit") is not False:
        errors.append("fresh reducer authority drift")
    cohort = lock.get("fresh_cohort", {})
    if cohort.get("parent_and_visit_disjoint_from_all_prior_taro_roles") is not True or cohort.get("no_replacement_after_outcome") is not True or cohort.get("authorization_not_implied_by_this_protocol") is not True:
        errors.append("fresh cohort firewall drift")
    evaluability = lock.get("dual_class_evaluability_gates", {})
    if evaluability.get("minimum_parents_with_definite_clear_label") != 4 or evaluability.get("minimum_definite_clear_query_count") != 50:
        errors.append("dual-class negative-control gate drift")
    gates = lock.get("confirmation_gates", {})
    if gates.get("maximum_clear_outputs") != 0 or gates.get("unknown_is_negative") is not False or gates.get("all_gates_required") is not True:
        errors.append("fresh confirmation gate/UNKNOWN drift")
    if lock.get("unique_successor") != "TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK":
        errors.append("fresh confirmation successor drift")
    return {
        "schema": "blindassist.taro.o1r.r7_fresh_confirmation_protocol_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O1R_R7_FRESH_CONFIRMATION_PROTOCOL_VALID" if not errors else "TARO_O1R_R7_FRESH_CONFIRMATION_PROTOCOL_INVALID",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    result = validate(args.lock)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

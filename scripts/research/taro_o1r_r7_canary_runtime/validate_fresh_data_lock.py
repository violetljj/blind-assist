from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as cohort


DEFAULT_LOCK = Path("docs/research/taro/TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK_2026-08-12.json")
EXPECTED_BYTES = 7157
EXPECTED_SHA256 = "4CC1C00ACB049C0622CCB834EF4709DCEE88343E2702EB02187DD2311B2957AB"


def validate(path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    lock_path = path.resolve()
    root = Path(__file__).resolve().parents[3]
    errors: list[str] = []
    raw = lock_path.read_bytes()
    if len(raw) != EXPECTED_BYTES or hashlib.sha256(raw).hexdigest().upper() != EXPECTED_SHA256:
        errors.append("fresh data lock identity mismatch")
    lock = json.loads(raw.decode("utf-8"))
    if lock.get("lock_id") != "TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK" or lock.get("status") != "COHORT_AND_DATA_USE_FROZEN_HEAD_EXECUTION_FALSE":
        errors.append("fresh data lock status/id drift")
    for binding in lock.get("predecessor_bindings", []):
        bound = root / binding["path"]
        if not bound.is_file():
            errors.append(f"missing predecessor: {binding['role']}")
            continue
        payload = bound.read_bytes()
        if len(payload) != binding["bytes"] or hashlib.sha256(payload).hexdigest().upper() != binding["sha256"]:
            errors.append(f"predecessor binding drift: {binding['role']}")
    plan = cohort.build_plan(root)
    selection = lock.get("selection", {})
    expected_roster = plan["selection"]["roster"]
    if selection.get("roster") != expected_roster or selection.get("planner_output_sha256") != adapter.canonical_sha256(plan):
        errors.append("fresh roster/planner output drift")
    for field in ("exclusion_snapshot_commit", "matched_official_identity_count", "matched_official_identities_sha256", "selection_salt", "eligible_row_count"):
        if selection.get(field) != plan["selection"].get(field):
            errors.append(f"fresh selection field drift: {field}")
    asset = lock.get("asset_plan", {})
    if asset.get("parent_count") != 8 or asset.get("request_count") != 24 or asset.get("request_method") != "HEAD" or asset.get("response_body_bytes_allowed") != 0 or asset.get("expanded_requests_sha256") != plan["request_plan"]["expanded_requests_sha256"]:
        errors.append("fresh asset request plan drift")
    authorization = lock.get("user_authorization", {})
    if authorization.get("confirmation_verbatim") != "授权" or authorization.get("authorization_does_not_itself_activate_execution") is not True:
        errors.append("fresh user authorization receipt drift")
    firewalls = lock.get("scientific_firewalls", {})
    if firewalls.get("parent_and_visit_disjoint_from_all_prior_taro_roles") is not True or firewalls.get("selection_reads_model_output_or_truth") is not False or firewalls.get("replacement_after_head_or_body_access") is not False or firewalls.get("unknown_is_negative") is not False or firewalls.get("clear_branch_enabled") is not False:
        errors.append("fresh scientific firewall drift")
    authority = lock.get("authority", {})
    if authority.get("exact_cohort_frozen") is not True or authority.get("data_use_authorized") is not True:
        errors.append("fresh data-use authority missing")
    if any(authority.get(field) is not False for field in ("head_execution_lock", "head_requests", "source_download", "source_decode", "model_execution", "truth_scoring", "training", "network", "device", "product", "safety")):
        errors.append("fresh data lock improperly activates execution")
    if lock.get("unique_successor") != "TARO_O1R_R7_FRESH_CONFIRMATION_CONTENT_LENGTH_HEAD_ONE_SHOT_EXECUTION_LOCK":
        errors.append("fresh data lock successor drift")
    return {
        "schema": "blindassist.taro.o1r.r7_fresh_confirmation_data_lock_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O1R_R7_FRESH_CONFIRMATION_DATA_LOCK_VALID" if not errors else "TARO_O1R_R7_FRESH_CONFIRMATION_DATA_LOCK_INVALID",
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

#!/usr/bin/env python3
"""Validate the TARO R6 exact untouched cohort and data-use lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_untouched_cohort as cohort
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o0r.r6_untouched_cohort_data_use_lock.v1"
LOCK_ID = "TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK"
SUCCESSOR = "TARO_O0R_R6_UNTOUCHED_CONTENT_LENGTH_HEAD_ONE_SHOT_EXECUTION_LOCK"
EXPECTED_BINDINGS = {
    "R6_PROTOCOL": ("docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_PARENT_CONFIRMATION_PROTOCOL_LOCK_2026-08-11.json", 4570, "5F2802F2585861F4D2D1EB002D1AFA7050278CBD33732F665DB6AF9CA32A101C"),
    "R6_IMPLEMENTATION": ("docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK_2026-08-11.json", 5026, "34D1C30193183F8406D5A4CA5EF7598E7EE933B4D62008500A70523A5EE3C90B"),
    "R6_COHORT_PLANNER": ("scripts/research/taro_o0r_candidate_scale_runtime/r6_untouched_cohort.py", 11960, "04528590F193CEE87E07F8F1B72C34AFA8AF0F088CAFA378A7643D84D16DFF14"),
    "R6_COHORT_PLANNER_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_r6_untouched_cohort.py", 2448, "E79C19B1961A48A8A1E8648E2D830409FD918EB0575AF361C9E854853AB204FD"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def validate_payload(payload: Mapping[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    expected_keys = {"schema", "lock_id", "date", "research_mode", "status", "predecessor_bindings", "user_authorization", "source", "selection", "asset_plan", "scientific_firewalls", "authority", "unique_successor", "claim_ceiling"}
    _require(errors, set(payload) == expected_keys, "R6_DATA_LOCK_KEY_SET_DRIFT")
    _require(errors, payload.get("schema") == SCHEMA and payload.get("lock_id") == LOCK_ID, "R6_DATA_LOCK_IDENTITY_DRIFT")
    _require(errors, payload.get("status") == "COHORT_AND_DATA_USE_FROZEN_HEAD_EXECUTION_FALSE", "R6_DATA_LOCK_STATUS_DRIFT")
    _require(errors, payload.get("unique_successor") == SUCCESSOR, "R6_DATA_LOCK_SUCCESSOR_DRIFT")

    observed_bindings = {}
    for row in payload.get("predecessor_bindings", []):
        if isinstance(row, Mapping) and set(row) == {"role", "path", "bytes", "sha256"}:
            observed_bindings[row["role"]] = (row["path"], row["bytes"], row["sha256"])
    _require(errors, observed_bindings == EXPECTED_BINDINGS, "R6_DATA_LOCK_BINDING_SET_DRIFT")

    authorization = payload.get("user_authorization", {})
    _require(errors, authorization.get("confirmed_by") == "user" and authorization.get("confirmation_verbatim") == "授权", "R6_DATA_LOCK_USER_AUTHORITY_DRIFT")
    _require(errors, authorization.get("authorization_does_not_itself_activate_execution") is True, "R6_DATA_LOCK_EXECUTION_AUTOACTIVATED")

    plan = cohort.build_plan(REPO_ROOT) if verify_files else None
    selection = payload.get("selection", {})
    observed_roster = [(row.get("visit_id"), row.get("video_id"), row.get("selection_rank_sha256")) for row in selection.get("roster", [])]
    _require(errors, observed_roster == cohort.EXPECTED_ROSTER, "R6_DATA_LOCK_ROSTER_DRIFT")
    _require(errors, selection.get("planner_output_sha256") == "E6737BBEB7B1E6289531E486A8865C523A15776D3A1EC8622FECC2323B5A9387", "R6_DATA_LOCK_PLAN_HASH_DRIFT")
    _require(errors, selection.get("exclusion_snapshot_commit") == cohort.EXCLUSION_COMMIT and selection.get("matched_official_identity_count") == 186 and selection.get("matched_official_identities_sha256") == "D451B3AB0493FE7878BD627A9E71D618AD1D93996B2BD225D4F18627769AD493", "R6_DATA_LOCK_EXCLUSION_DRIFT")

    assets = payload.get("asset_plan", {})
    _require(errors, assets.get("parent_count") == 8 and assets.get("asset_count_per_parent") == 3 and assets.get("request_count") == 24, "R6_DATA_LOCK_REQUEST_COUNT_DRIFT")
    _require(errors, assets.get("request_method") == "HEAD" and assets.get("response_body_bytes_allowed") == 0 and assets.get("off_host_redirect_allowed") is False, "R6_DATA_LOCK_HEAD_SCOPE_DRIFT")
    _require(errors, assets.get("expanded_requests_sha256") == "25B710DD39823754C08FB147FBA74E577FFCB7137CE55A7F82312BC896F2B2B4", "R6_DATA_LOCK_REQUEST_HASH_DRIFT")

    authority = payload.get("authority", {})
    _require(errors, authority.get("exact_cohort_frozen") is True and authority.get("data_use_authorized") is True, "R6_DATA_LOCK_AUTHORITY_MISSING")
    for field in ("head_execution_lock", "head_requests", "source_download", "source_decode", "model_execution", "truth_scoring", "training", "network", "device", "product", "safety"):
        _require(errors, authority.get(field) is False, f"R6_DATA_LOCK_AUTHORITY_DRIFT:{field}")

    if verify_files:
        _require(errors, plan is not None and adapter.canonical_sha256(plan) == selection.get("planner_output_sha256"), "R6_DATA_LOCK_PLAN_REPLAY_DRIFT")
        for relative, size, digest in EXPECTED_BINDINGS.values():
            path = REPO_ROOT / relative
            _require(errors, path.is_file() and path.stat().st_size == size and _sha256(path) == digest, f"R6_DATA_LOCK_BOUND_FILE_DRIFT:{relative}")
        license_path = REPO_ROOT / str(payload.get("source", {}).get("license_path", ""))
        _require(errors, license_path.is_file() and _sha256(license_path) == payload.get("source", {}).get("license_sha256"), "R6_DATA_LOCK_LICENSE_DRIFT")
    return errors


def validate_file(path: Path = DEFAULT_LOCK_PATH, *, verify_files: bool = True) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"R6_DATA_LOCK_READ_FAILED:{error}"]
    return validate_payload(payload, verify_files=verify_files) if isinstance(payload, Mapping) else ["R6_DATA_LOCK_NOT_OBJECT"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_file(args.lock, verify_files=not args.skip_file_verification)
    print(json.dumps({"passed": not errors, "error_count": len(errors), "errors": errors, "terminal": "TARO_O0R_R6_UNTOUCHED_DATA_LOCK_VALID" if not errors else "TARO_O0R_R6_UNTOUCHED_DATA_LOCK_INVALID"}, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independently validate a controlled synthetic stress package."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = Path(__file__).with_name("protocol_r0.json")
SCHEMA = "blindassist.controlled_synthetic_stress.r0.result"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL row is not an object")
            rows.append(value)
    return rows


def validate(output: Path) -> dict[str, Any]:
    required = [
        "protocol_copy.json",
        "case_manifest.json",
        "boundary_results.json",
        "summary.json",
        "run_manifest.json",
        "case_results.jsonl",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        return {
            "schema": "blindassist.controlled_synthetic_stress.r0.validation",
            "status": "INVALID",
            "output": str(output.resolve()),
            "case_count": 0,
            "errors": [f"MISSING_REQUIRED_FILE:{name}" for name in missing],
            "synthetic_only": True,
        }
    protocol = load_json(PROTOCOL_PATH)
    copied_protocol = load_json(output / "protocol_copy.json")
    case_manifest = load_json(output / "case_manifest.json")
    boundary = load_json(output / "boundary_results.json")
    summary = load_json(output / "summary.json")
    manifest = load_json(output / "run_manifest.json")
    rows = load_jsonl(output / "case_results.jsonl")
    errors: list[str] = []

    if copied_protocol != protocol:
        errors.append("PROTOCOL_COPY_MISMATCH")
    protocol_sha = sha256_file(PROTOCOL_PATH)
    if manifest.get("protocol_sha256") != protocol_sha or summary.get("protocol_sha256") != protocol_sha:
        errors.append("PROTOCOL_SHA256_MISMATCH")
    expected_count = int(case_manifest.get("case_count", -1))
    if expected_count != len(rows) or expected_count != int(manifest.get("case_count", -2)):
        errors.append("CASE_COUNT_MISMATCH")
    case_ids = [str(case.get("case_id")) for case in case_manifest.get("cases", [])]
    result_ids = [str(row.get("case_id")) for row in rows]
    if len(case_ids) != expected_count or len(set(case_ids)) != expected_count:
        errors.append("CASE_MANIFEST_ID_UNIQUENESS")
    if set(case_ids) != set(result_ids) or len(set(result_ids)) != len(result_ids):
        errors.append("RESULT_CASE_ID_SET_MISMATCH")
    if [int(row.get("ordinal", -1)) for row in rows] != list(range(1, len(rows) + 1)):
        errors.append("ORDINAL_SEQUENCE_MISMATCH")
    if boundary.get("status") != "PASS" or int(boundary.get("passed", -1)) != int(boundary.get("total", -2)):
        errors.append("BOUNDARY_NOT_PASS")
    if summary.get("schema") != SCHEMA:
        errors.append("SUMMARY_SCHEMA_MISMATCH")
    if summary.get("authority", {}).get("synthetic_only") is not True:
        errors.append("SYNTHETIC_AUTHORITY_MISSING")

    expected_hashes = {
        "case_manifest_sha256": sha256_file(output / "case_manifest.json"),
        "case_results_sha256": sha256_file(output / "case_results.jsonl"),
        "boundary_results_sha256": sha256_file(output / "boundary_results.json"),
        "summary_sha256": sha256_file(output / "summary.json"),
    }
    for key, expected in expected_hashes.items():
        if manifest.get(key) != expected:
            errors.append(f"{key.upper()}_MISMATCH")

    rcle_status = Counter()
    field_status = Counter()
    d2_status = Counter()
    unknown_violations = 0
    for row in rows:
        rcle = row.get("rcle", {})
        field = row.get("field_transport", {})
        rcle_status[str(rcle.get("status"))] += 1
        field_status[str(field.get("status"))] += 1
        d2_status[str(row.get("d2_time_status"))] += 1
        unknown_violations += int(field.get("unknown_to_safe_violations", 0))
        if field.get("status") != "EVALUABLE":
            errors.append(f"FIELD_CASE_NOT_EVALUABLE:{row.get('case_id')}")
        if row.get("ttc_proxy", {}).get("proxy_is_physical_ttc") is True:
            errors.append(f"TTC_PROXY_OVERCLAIM:{row.get('case_id')}")
        if row.get("motion_family") not in {"scale", "frontal_approach"} and "expansion_sign" in rcle:
            errors.append(f"UNDECLARED_SIGN_TRUTH:{row.get('case_id')}")
        for horizon, stratum in field.get("horizons", {}).items():
            if stratum.get("status") == "EVALUABLE":
                if int(stratum.get("common_known_cells", 0)) <= 0:
                    errors.append(f"COMMON_KNOWN_ZERO:{row.get('case_id')}:{horizon}")
                for key in ("persistence_mae_m", "advected_mae_m"):
                    if not isinstance(stratum.get(key), (int, float)):
                        errors.append(f"NONNUMERIC_FIELD_METRIC:{row.get('case_id')}:{horizon}:{key}")
    if unknown_violations != 0:
        errors.append("UNKNOWN_TO_SAFE_NONZERO")
    if dict(rcle_status) != summary.get("rcle_status"):
        errors.append("RCLE_STATUS_SUMMARY_MISMATCH")
    if dict(field_status) != summary.get("field_status"):
        errors.append("FIELD_STATUS_SUMMARY_MISMATCH")
    if dict(d2_status) != summary.get("d2_time_status"):
        errors.append("D2_STATUS_SUMMARY_MISMATCH")

    return {
        "schema": "blindassist.controlled_synthetic_stress.r0.validation",
        "status": "VALID" if not errors else "INVALID",
        "output": str(output.resolve()),
        "case_count": len(rows),
        "errors": errors,
        "rcle_status": dict(rcle_status),
        "field_status": dict(field_status),
        "d2_time_status": dict(d2_status),
        "unknown_to_safe_violations": unknown_violations,
        "synthetic_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.output.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "INVALID", "errors": [f"{type(error).__name__}:{error}"]}, ensure_ascii=False))
        return 2
    if args.receipt is not None:
        if args.receipt.exists():
            print(json.dumps({"status": "INVALID", "errors": ["RECEIPT_EXISTS"]}, ensure_ascii=False))
            return 2
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(canonical_bytes(result))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())

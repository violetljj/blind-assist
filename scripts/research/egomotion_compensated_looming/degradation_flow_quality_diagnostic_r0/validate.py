from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .analyze import (
    PAIR_COUNT,
    PROTOCOL_ID,
    SESSIONS,
    read_jsonl,
    sha256_file,
    summarize_session,
)


FORBIDDEN_KEYS = {
    "auroc",
    "f1",
    "false_positive_rate",
    "risk_label",
    "obstacle_label",
    "pooled_pair_count",
}


def walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key).lower())
            keys.extend(walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(walk_keys(item))
    return keys


def validate(
    proxy_root: Path,
    r3_runs_root: Path,
    contract_path: Path,
    analysis_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if analysis.get("protocol_id") != PROTOCOL_ID:
        failures.append("PROTOCOL_ID_MISMATCH")
    if analysis.get("contract_sha256") != sha256_file(contract_path):
        failures.append("CONTRACT_HASH_MISMATCH")
    if analysis.get("risk_label_accessed") is not False:
        failures.append("RISK_LABEL_FIREWALL_FAILED")
    if analysis.get("sealed_session_accessed") is not False:
        failures.append("SEALED_SESSION_FIREWALL_FAILED")
    if analysis.get("threshold_per_s") != 0.01:
        failures.append("THRESHOLD_CHANGED")
    if analysis.get("required_consecutive_pairs") != 3:
        failures.append("THREE_PAIR_RULE_CHANGED")
    if (
        analysis.get("pair_records_are_longitudinal_not_independent_samples")
        is not True
    ):
        failures.append("PAIR_GRAIN_MISDECLARED")
    forbidden_present = sorted(FORBIDDEN_KEYS.intersection(walk_keys(analysis)))
    if forbidden_present:
        failures.append(f"FORBIDDEN_KEYS:{forbidden_present}")

    recomputed: list[dict[str, Any]] = []
    proxy_hashes: dict[str, str] = {}
    for session in SESSIONS:
        proxy_path = proxy_root / f"advio-{session:02d}" / "proxy_ledger.jsonl"
        summary_path = proxy_root / f"advio-{session:02d}" / "proxy_summary.json"
        r3_path = (
            r3_runs_root
            / f"advio-{session:02d}_r3_fixed_601"
            / "pair_ledger.jsonl"
        )
        for path in (proxy_path, summary_path, r3_path):
            if not path.is_file():
                failures.append(f"MISSING_INPUT:{path.as_posix()}")
                continue
        if failures:
            continue
        proxy_rows = read_jsonl(proxy_path)
        if len(proxy_rows) != PAIR_COUNT:
            failures.append(f"PROXY_PAIR_COUNT:{session}:{len(proxy_rows)}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        proxy_hash = sha256_file(proxy_path)
        proxy_hashes[str(session)] = proxy_hash
        if summary.get("proxy_ledger_sha256") != proxy_hash:
            failures.append(f"PROXY_LEDGER_HASH:{session}")
        if (
            summary.get("risk_label_accessed") is not False
            or summary.get("response_accessed_during_extraction") is not False
            or summary.get("sealed_session_accessed") is not False
        ):
            failures.append(f"PROXY_FIREWALL:{session}")
        recomputed.append(
            summarize_session(session, proxy_rows, read_jsonl(r3_path))
        )
    if not failures and recomputed != analysis.get("session_results"):
        failures.append("SESSION_RESULTS_RECOMPUTE_MISMATCH")

    status = "VALID" if not failures else "INVALID"
    receipt = {
        "schema": "rcle.degradation_flow_quality.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "failures": failures,
        "contract_sha256": sha256_file(contract_path),
        "analysis_sha256": sha256_file(analysis_path),
        "proxy_ledger_sha256_by_session": proxy_hashes,
        "risk_label_accessed": False,
        "sealed_session_accessed": False,
        "independent_recomputation": not failures,
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--r3-runs-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.proxy_root.resolve(),
        args.r3_runs_root.resolve(),
        args.contract.resolve(),
        args.analysis.resolve(),
        args.receipt.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())

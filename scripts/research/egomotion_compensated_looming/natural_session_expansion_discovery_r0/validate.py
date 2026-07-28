from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.egomotion_compensated_looming.natural_session_expansion_discovery_r0 import (
    analyze,
)


FORBIDDEN_KEYS = {"auroc", "f1", "roc_auc", "pooled_pair_count"}


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                errors.append(f"FORBIDDEN_KEY:{path}.{key}")
            errors.extend(find_forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return errors


def validate(
    evidence_root: Path, result_path: Path, contract_path: Path
) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    errors: list[str] = []
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_sessions = {
        int(item["session_id"].split("SEQUENCE", 1)[1].split("_", 1)[0]):
        item
        for item in contract.get("sessions", [])
    }
    if (
        set(contract_sessions) != {13, 14, 15, 16, 17}
        or contract_sessions[16].get("pre_run_access_state")
        != "SEALED_UNSEEN"
        or contract_sessions[16].get("algorithm_execution_authorized")
        is not False
    ):
        errors.append("CONTRACT_SESSION_SET_OR_SEALED_ROLE_MISMATCH")
    forbidden_sealed_paths = [
        path.as_posix()
        for path in evidence_root.rglob("*")
        if "advio-16" in path.name.lower()
        or "advio_16" in path.name.lower()
    ]
    if forbidden_sealed_paths:
        errors.append(
            "SEALED_SESSION_ARTIFACT_PRESENT:"
            + ",".join(sorted(forbidden_sealed_paths))
        )
    session_dirs = {
        session: evidence_root
        / "runs"
        / f"advio-{session:02d}_r3_fixed_601"
        for session in analyze.DISCOVERY_SESSIONS
    }
    source_receipt_hashes: dict[str, str] = {}
    for session in analyze.DISCOVERY_SESSIONS:
        receipt_path = (
            evidence_root / f"source_receipt_advio_{session:02d}.json"
        )
        if not receipt_path.is_file():
            errors.append(f"MISSING_SOURCE_RECEIPT:{session}")
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        frozen = contract_sessions[session]
        if (
            receipt.get("session_number") != session
            or receipt.get("archive_bytes") != frozen.get("archive_bytes")
            or receipt.get("archive_md5") != frozen.get("archive_md5")
            or receipt.get("sealed_session_touched") is not False
        ):
            errors.append(f"SOURCE_RECEIPT_CONTRACT_MISMATCH:{session}")
        source_root = Path(receipt["source_root"])
        members = receipt.get("members", {})
        for relative, identity in members.items():
            path = source_root / relative
            if (
                not path.is_file()
                or digest_file(path) != identity.get("sha256")
            ):
                errors.append(
                    f"SOURCE_MEMBER_RECOMPUTATION_MISMATCH:{session}:{relative}"
                )
        source_receipt_hashes[f"session_{session}"] = digest_file(
            receipt_path
        )
    try:
        recomputed = analyze.analyze(session_dirs)
    except Exception as exc:
        errors.append(f"RECOMPUTATION_FAILED:{type(exc).__name__}:{exc}")
        recomputed = None
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    if recomputed is not None and stored != recomputed:
        errors.append("STORED_RESULT_DIFFERS_FROM_LEDGER_RECOMPUTATION")
    errors.extend(find_forbidden_keys(stored))
    if (
        stored.get("analysis_unit") != "CAPTURE_SESSION"
        or stored.get(
            "pair_records_are_longitudinal_not_independent_samples"
        )
        is not True
        or stored.get("session_count") != 4
        or stored.get("sealed_session", {}).get("accessed") is not False
    ):
        errors.append("ANALYSIS_UNIT_OR_SEALED_STATE_MISMATCH")
    artifact_hashes: dict[str, str] = {}
    for session, directory in session_dirs.items():
        for name in ("pair_ledger.jsonl", "summary.json"):
            path = directory / name
            if not path.is_file():
                errors.append(f"MISSING_ARTIFACT:{session}:{name}")
            else:
                artifact_hashes[f"session_{session}_{name}"] = digest_file(
                    path
                )
    artifact_hashes["session_analysis_r0.json"] = digest_file(result_path)
    artifact_hashes["contract"] = digest_file(contract_path)
    return {
        "schema": "rcle.natural_session_expansion.validation.v1",
        "protocol_id": analyze.PROTOCOL_ID,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "session_set": list(analyze.DISCOVERY_SESSIONS),
        "sealed_session": analyze.SEALED_SESSION,
        "sealed_artifact_paths": forbidden_sealed_paths,
        "recomputed_exact_match": recomputed == stored,
        "source_receipt_sha256": source_receipt_hashes,
        "artifact_sha256": artifact_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.evidence_root,
        args.result.resolve(),
        args.contract.resolve(),
    )
    args.output.resolve().write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

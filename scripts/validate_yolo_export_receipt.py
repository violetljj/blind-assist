"""Fail-closed validator for blindassist_yolo_export_receipt_v1."""

import json
import math
import sys
from pathlib import Path

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
SCHEMA_VERSION = "blindassist_yolo_export_receipt_v1"


class ValidationError(ValueError):
    pass


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _verdict(decision, reason):
    return {"decision": decision, "reason_codes": [reason]}


def derive(receipt):
    source = receipt["source"]
    exporter = receipt["exporter"]
    artifacts = receipt["artifacts"]
    inspection = receipt["inspection"]
    equivalence = receipt["equivalence"]

    byte_missing = not all([
        _sha(source.get("checkpoint_sha256")),
        source.get("immutable_revision"),
        source.get("license_expression"),
        source.get("license_evidence_ref"),
        _sha(exporter.get("repository_commit")),
        _sha(exporter.get("script_sha256")),
        exporter.get("python_version"),
        exporter.get("platform"),
        _sha(exporter.get("lock_sha256")),
        _sha(artifacts["tflite"].get("sha256")),
        _sha(artifacts["pinned_tflite_sha256"]),
    ])
    if byte_missing:
        byte = _verdict(UNKNOWN, "BYTE_IDENTITY_NOT_ESTABLISHED")
    elif artifacts["tflite"]["sha256"] != artifacts["pinned_tflite_sha256"]:
        byte = _verdict(FAIL, "TFLITE_SHA256_MISMATCH")
    else:
        byte = _verdict(PASS, "BYTE_IDENTITY_ESTABLISHED")

    if not inspection.get("expected") or not inspection.get("observed"):
        structural = _verdict(UNKNOWN, "TENSOR_CONTRACT_NOT_ESTABLISHED")
    elif inspection["expected"] != inspection["observed"]:
        structural = _verdict(FAIL, "TENSOR_CONTRACT_MISMATCH")
    else:
        structural = _verdict(PASS, "TENSOR_CONTRACT_MATCH")

    outputs = equivalence.get("outputs")
    numerical_missing = not all([
        _sha(equivalence.get("fixture_set_sha256")),
        isinstance(equivalence.get("denominator"), int) and equivalence["denominator"] > 0,
        _sha(equivalence.get("reference_runtime_sha256")),
        _finite(equivalence.get("max_abs_error_bound")),
        isinstance(outputs, list) and outputs,
    ])
    if numerical_missing:
        numerical = _verdict(UNKNOWN, "NUMERICAL_EVIDENCE_NOT_ESTABLISHED")
    elif any(not _finite(item.get("max_abs_error")) for item in outputs):
        numerical = _verdict(FAIL, "NON_FINITE_OUTPUT")
    elif any(item["max_abs_error"] > equivalence["max_abs_error_bound"] for item in outputs):
        numerical = _verdict(FAIL, "NUMERICAL_BOUND_EXCEEDED")
    else:
        numerical = _verdict(PASS, "NUMERICAL_EQUIVALENCE_ESTABLISHED")
    return {
        "byte_reproducibility": byte,
        "structural_equivalence": structural,
        "numerical_equivalence": numerical,
    }


def validate(receipt):
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("schema_version mismatch")
    for key in ("source", "exporter", "parameters", "artifacts", "inspection", "equivalence", "verdicts", "authority"):
        if not isinstance(receipt.get(key), dict):
            raise ValidationError(f"{key} missing")
    if receipt["authority"] != {
        "application_behavior": False,
        "accuracy": False,
        "accessibility_effectiveness": False,
        "product": False,
        "safety": False,
    }:
        raise ValidationError("receipt must grant no downstream authority")
    expected = derive(receipt)
    if receipt["verdicts"] != expected:
        raise ValidationError(f"misleading verdicts: expected {expected!r}")


def main(argv):
    if len(argv) != 2:
        return 2
    try:
        validate(json.loads(Path(argv[1]).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


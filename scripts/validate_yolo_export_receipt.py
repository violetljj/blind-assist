"""Fail-closed validator for blindassist_yolo_export_receipt_v1."""

import json
import math
import re
import sys
from pathlib import Path

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
SCHEMA_VERSION = "blindassist_yolo_export_receipt_v1"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "blindassist_yolo_export_receipt_v1.schema.json"


class ValidationError(ValueError):
    pass


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _git_object_id(value):
    return isinstance(value, str) and len(value) in (40, 64) and all(c in "0123456789abcdef" for c in value)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _verdict(decision, reason):
    return {"decision": decision, "reason_codes": [reason]}


def _resolve_ref(root, ref):
    if not ref.startswith("#/"):
        raise ValidationError(f"unsupported schema reference: {ref}")
    value = root
    for token in ref[2:].split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def _is_type(value, expected):
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _finite(value)
    raise ValidationError(f"unsupported schema type: {expected}")


def _validate_schema(value, schema, root, path="$"):
    if "$ref" in schema:
        _validate_schema(value, _resolve_ref(root, schema["$ref"]), root, path)
        return
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _validate_schema(value, option, root, path)
                return
            except ValidationError:
                pass
        raise ValidationError(f"{path}: no anyOf branch matched")
    if "const" in schema and value != schema["const"]:
        raise ValidationError(f"{path}: const mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: value is not in enum")
    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_is_type(value, expected) for expected in expected_types):
            raise ValidationError(f"{path}: type mismatch")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValidationError(f"{path}: missing required fields {missing}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValidationError(f"{path}: unexpected fields {extra}")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], root, f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValidationError(f"{path}: too few items")
        if schema.get("uniqueItems"):
            normalized = [json.dumps(item, sort_keys=True, allow_nan=False) for item in value]
            if len(normalized) != len(set(normalized)):
                raise ValidationError(f"{path}: duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], root, f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValidationError(f"{path}: string is too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValidationError(f"{path}: pattern mismatch")

    if _finite(value):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(f"{path}: value is above maximum")


def _validate_schema_surface(receipt):
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load receipt schema: {exc}") from exc
    _validate_schema(receipt, schema, schema)


def derive(receipt):
    source = receipt["source"]
    exporter = receipt["exporter"]
    artifacts = receipt["artifacts"]
    inspection = receipt["inspection"]
    equivalence = receipt["equivalence"]

    byte_missing = not all([
        _sha(source.get("checkpoint_sha256")),
        _nonempty(source.get("immutable_revision")),
        _nonempty(source.get("license_expression")),
        _nonempty(source.get("license_evidence_ref")),
        _nonempty(exporter.get("repository_url")),
        _git_object_id(exporter.get("repository_commit")),
        _sha(exporter.get("script_sha256")),
        _nonempty(exporter.get("python_version")),
        _nonempty(exporter.get("platform")),
        _sha(exporter.get("lock_sha256")),
        _sha(artifacts["checkpoint"].get("sha256")),
        _sha(artifacts["labels"].get("sha256")),
        _sha(artifacts["tflite"].get("sha256")),
        _sha(artifacts["pinned_tflite_sha256"]),
    ])
    if byte_missing:
        byte = _verdict(UNKNOWN, "BYTE_IDENTITY_NOT_ESTABLISHED")
    elif source["checkpoint_sha256"] != artifacts["checkpoint"]["sha256"]:
        byte = _verdict(FAIL, "CHECKPOINT_SHA256_MISMATCH")
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
    expected_output_names = [item["name"] for item in inspection.get("expected", []) if item.get("role") == "output"]
    numerical_missing = not all([
        _sha(equivalence.get("fixture_set_sha256")),
        isinstance(equivalence.get("denominator"), int) and not isinstance(equivalence.get("denominator"), bool) and equivalence["denominator"] > 0,
        _sha(equivalence.get("reference_runtime_sha256")),
        _finite(equivalence.get("max_abs_error_bound")),
        equivalence.get("max_abs_error_bound") is not None and equivalence["max_abs_error_bound"] >= 0,
        isinstance(outputs, list) and outputs,
    ])
    if numerical_missing:
        numerical = _verdict(UNKNOWN, "NUMERICAL_EVIDENCE_NOT_ESTABLISHED")
    else:
        output_names = [item.get("name") for item in outputs]
        if len(output_names) != len(set(output_names)) or len(output_names) != len(expected_output_names) or set(output_names) != set(expected_output_names):
            numerical = _verdict(FAIL, "OUTPUT_IDENTITY_MISMATCH")
        elif any(not _finite(item.get("max_abs_error")) for item in outputs):
            numerical = _verdict(FAIL, "NON_FINITE_OUTPUT")
        elif any(item["max_abs_error"] < 0 for item in outputs):
            numerical = _verdict(FAIL, "INVALID_NUMERICAL_EVIDENCE")
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
    _validate_schema_surface(receipt)
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

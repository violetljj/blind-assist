"""Fail-closed validator for blindassist_yolo_export_receipt_v1."""

import json
import math
import re
import sys
from pathlib import Path

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
SCHEMA_VERSION = "blindassist_yolo_export_receipt_v1"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "blindassist_yolo_export_receipt_v1.schema.json"

PUBLIC_ASSETS_PATH = Path(__file__).resolve().parents[2] / "config" / "public_release_assets.json"


class ValidationError(ValueError):
    pass


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _git_object_id(value):
    return isinstance(value, str) and len(value) in (40, 64) and all(c in "0123456789abcdef" for c in value)


def _finite(value):
    return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and math.isfinite(value))


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


def _pinned_asset_sha256():
    path = PUBLIC_ASSETS_PATH
    try:
        assets = json.loads(path.read_text(encoding="utf-8"))["assets"]
        matches = [item for item in assets if item["path"] == "app/src/main/assets/yolo11n_fp16_320.tflite"]
        if len(matches) != 1 or not _sha(matches[0]["sha256"].lower()):
            raise ValueError("expected exactly one valid YOLO asset identity")
        return matches[0]["sha256"].lower()
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValidationError(f"cannot load pinned asset identity: {exc}") from exc


def _artifact_complete(artifact):
    size = artifact.get("size_bytes")
    return (_nonempty(artifact.get("path")) and isinstance(size, int)
            and not isinstance(size, bool) and size > 0 and _sha(artifact.get("sha256")))


def _tensor_inventory_valid(tensors):
    identities = [(item["role"], item["name"]) for item in tensors]
    return (len(identities) == len(set(identities))
            and sum(item["role"] == "input" for item in tensors) == 1
            and sum(item["role"] == "output" for item in tensors) == 1)


def _parameters_match(parameters, tensors):
    inputs = [item for item in tensors if item["role"] == "input"]
    outputs = [item for item in tensors if item["role"] == "output"]
    return (len(inputs) == len(outputs) == 1
            and inputs[0]["shape"] == [parameters["batch"], parameters["imgsz"], parameters["imgsz"], 3]
            and inputs[0]["dtype"] == parameters["input_dtype"]
            and len(outputs[0]["shape"]) == 3
            and outputs[0]["shape"][0] == parameters["batch"]
            and outputs[0]["dtype"] == "float32")


def derive(receipt):
    source = receipt["source"]
    exporter = receipt["exporter"]
    artifacts = receipt["artifacts"]
    inspection = receipt["inspection"]
    equivalence = receipt["equivalence"]

    pinned_sha256 = _pinned_asset_sha256()
    byte_missing = not all([
        _nonempty(source.get("checkpoint_url")),
        _nonempty(source.get("upstream_project")),
        _nonempty(exporter.get("lock_path")),
        bool(exporter.get("packages")),
        all(_artifact_complete(artifacts[name]) for name in ("checkpoint", "labels", "tflite")),
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
    if _sha(artifacts["pinned_tflite_sha256"]) and artifacts["pinned_tflite_sha256"] != pinned_sha256:
        byte = _verdict(FAIL, "PINNED_ASSET_IDENTITY_MISMATCH")
    elif (_sha(source["checkpoint_sha256"]) and _sha(artifacts["checkpoint"]["sha256"])
          and source["checkpoint_sha256"] != artifacts["checkpoint"]["sha256"]):
        byte = _verdict(FAIL, "CHECKPOINT_SHA256_MISMATCH")
    elif _sha(artifacts["tflite"]["sha256"]) and artifacts["tflite"]["sha256"] != pinned_sha256:
        byte = _verdict(FAIL, "TFLITE_SHA256_MISMATCH")
    elif byte_missing:
        byte = _verdict(UNKNOWN, "BYTE_IDENTITY_NOT_ESTABLISHED")
    else:
        byte = _verdict(PASS, "BYTE_IDENTITY_ESTABLISHED")

    parameters = receipt["parameters"]
    if not inspection["expected"] or not inspection["observed"]:
        structural = _verdict(UNKNOWN, "TENSOR_CONTRACT_NOT_ESTABLISHED")
    elif not all(_tensor_inventory_valid(inspection[key]) for key in ("expected", "observed")):
        structural = _verdict(FAIL, "INVALID_TENSOR_INVENTORY")
    elif inspection["expected"] != inspection["observed"]:
        structural = _verdict(FAIL, "TENSOR_CONTRACT_MISMATCH")
    elif not _parameters_match(parameters, inspection["expected"]):
        structural = _verdict(FAIL, "PARAMETER_TENSOR_MISMATCH")
    elif not all(_sha(parameters[key]) for key in ("preprocess_contract", "postprocess_contract", "metadata_sha256")):
        structural = _verdict(UNKNOWN, "EXPORT_CONTRACT_NOT_ESTABLISHED")
    else:
        structural = _verdict(PASS, "TENSOR_CONTRACT_MATCH")

    outputs = equivalence.get("outputs")
    expected_output_names = [item["name"] for item in inspection.get("expected", []) if item.get("role") == "output"]
    numerical_missing = not all([
        bool(inspection["expected"]) and bool(inspection["observed"]),
        _sha(equivalence.get("fixture_set_sha256")),
        isinstance(equivalence.get("denominator"), int) and not isinstance(equivalence.get("denominator"), bool) and equivalence["denominator"] > 0,
        _sha(equivalence.get("reference_runtime_sha256")),
        _finite(equivalence.get("max_abs_error_bound")),
        equivalence.get("max_abs_error_bound") is not None and equivalence["max_abs_error_bound"] >= 0,
        isinstance(outputs, list) and outputs,
    ])
    observed_output_names = [item["name"] for item in inspection["observed"] if item["role"] == "output"]
    if numerical_missing:
        numerical = _verdict(UNKNOWN, "NUMERICAL_EVIDENCE_NOT_ESTABLISHED")
    else:
        output_names = [item.get("name") for item in outputs]
        if output_names != observed_output_names or len(output_names) != len(set(output_names)) or len(output_names) != len(expected_output_names) or set(output_names) != set(expected_output_names):
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
    packages = receipt["exporter"]["packages"]
    package_names = [re.sub(r"[-_.]+", "-", item.split("==", 1)[0]).lower() for item in packages]
    if len(package_names) != len(set(package_names)):
        raise ValidationError("package inventory contains duplicate project names")
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

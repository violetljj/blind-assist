from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.validate_repair_implementation_lock import (
    LOCK_ID,
    ImplementationLockError,
    validate_lock_document,
    validate_lock_file,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
LOCK_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_CONTROL_FORMAT_AND_RUNTIME_BINDING_REPAIR_IMPLEMENTATION_LOCK_2026-08-12.json"
)


def _document() -> dict[str, object]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_frozen_repair_implementation_lock_passes() -> None:
    result = validate_lock_file(LOCK_PATH, REPO_ROOT)
    assert result["lock_id"] == LOCK_ID
    assert result["implementation_binding_count"] == 18
    assert result["depthart_source_file_count"] == 29


def test_any_execution_authority_mutation_fails_closed() -> None:
    document = _document()
    document["execution_authority"]["model_inference"] = True  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="execution authority"):
        validate_lock_document(document, REPO_ROOT)


def test_access_receipt_mutation_fails_closed() -> None:
    document = _document()
    document["access_receipt"]["archive_member_reads"] = 1  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="access receipt"):
        validate_lock_document(document, REPO_ROOT)


def test_successor_cannot_be_self_authorizing() -> None:
    document = _document()
    document["unique_successor"]["execution_authority"] = True  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="unique successor"):
        validate_lock_document(document, REPO_ROOT)


def test_status_cannot_claim_scientific_confirmation() -> None:
    document = _document()
    document["status"] = "CONFIRM_PASS"
    with pytest.raises(ImplementationLockError, match="status"):
        validate_lock_document(document, REPO_ROOT)


def test_implementation_binding_hash_mutation_fails_closed() -> None:
    document = copy.deepcopy(_document())
    document["implementation_bindings"][0]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="SHA"):
        validate_lock_document(document, REPO_ROOT)

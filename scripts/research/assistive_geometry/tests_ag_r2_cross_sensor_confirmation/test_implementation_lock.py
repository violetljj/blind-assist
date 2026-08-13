from __future__ import annotations

import copy
import hashlib
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _currentized_document() -> dict[str, object]:
    document = copy.deepcopy(_document())
    for group in ("predecessor_bindings", "implementation_bindings"):
        for row in document[group]:  # type: ignore[index]
            path = REPO_ROOT / row["path"]
            row["bytes"] = path.stat().st_size
            row["sha256"] = _sha(path)
    return document


def test_frozen_repair_implementation_lock_is_preserved_and_superseded() -> None:
    document = _document()
    assert document["lock_id"] == LOCK_ID
    frozen_contract = next(
        row for row in document["implementation_bindings"] if row["role"] == "EXECUTION_CONTRACT_V2"  # type: ignore[index]
    )
    current_contract = REPO_ROOT / frozen_contract["path"]
    assert (frozen_contract["bytes"], frozen_contract["sha256"]) == (
        21051,
        "4FB63A7A2EFEACCBC7A0F8957F81C82EB9FA07A791188B54A5B5D5645C53B5D1",
    )
    assert (current_contract.stat().st_size, _sha(current_contract)) != (
        frozen_contract["bytes"],
        frozen_contract["sha256"],
    )
    with pytest.raises(ImplementationLockError, match="EXECUTION_CONTRACT_V2"):
        validate_lock_file(LOCK_PATH, REPO_ROOT)


def test_historical_validator_accepts_a_currentized_synthetic_document() -> None:
    result = validate_lock_document(_currentized_document(), REPO_ROOT)
    assert result["lock_id"] == LOCK_ID
    assert result["implementation_binding_count"] == 18
    assert result["depthart_source_file_count"] == 29


def test_any_execution_authority_mutation_fails_closed() -> None:
    document = _currentized_document()
    document["execution_authority"]["model_inference"] = True  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="execution authority"):
        validate_lock_document(document, REPO_ROOT)


def test_access_receipt_mutation_fails_closed() -> None:
    document = _currentized_document()
    document["access_receipt"]["archive_member_reads"] = 1  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="access receipt"):
        validate_lock_document(document, REPO_ROOT)


def test_successor_cannot_be_self_authorizing() -> None:
    document = _currentized_document()
    document["unique_successor"]["execution_authority"] = True  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="unique successor"):
        validate_lock_document(document, REPO_ROOT)


def test_status_cannot_claim_scientific_confirmation() -> None:
    document = _currentized_document()
    document["status"] = "CONFIRM_PASS"
    with pytest.raises(ImplementationLockError, match="status"):
        validate_lock_document(document, REPO_ROOT)


def test_implementation_binding_hash_mutation_fails_closed() -> None:
    document = _currentized_document()
    document["implementation_bindings"][0]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ImplementationLockError, match="SHA"):
        validate_lock_document(document, REPO_ROOT)

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.validate_depthart_source_manifest import (
    EXPECTED_PATHS,
    ValidationError,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_DEPTHART_SOURCE_MANIFEST_2026-08-12.json"
)


def test_real_runtime_source_manifest_is_exact_and_hash_bound() -> None:
    result = validate(MANIFEST)
    assert result["valid"] is True
    assert result["file_count"] == 29 == len(EXPECTED_PATHS)
    assert result["model_or_checkpoint_loaded"] is False


def test_manifest_mutations_fail_closed(tmp_path: Path) -> None:
    original = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, mutate, code in (
        ("path", lambda value: value["files"][0].__setitem__("path", "metric/missing.py"), "F2_DEPTHART_MANIFEST_MEMBER_MISSING"),
        ("bytes", lambda value: value["files"][0].__setitem__("bytes", 1), "F2_DEPTHART_MANIFEST_BYTES"),
        ("sha", lambda value: value["files"][0].__setitem__("sha256", "0" * 64), "F2_DEPTHART_MANIFEST_SHA"),
    ):
        changed = deepcopy(original)
        mutate(changed)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValidationError, match=code):
            validate(path)

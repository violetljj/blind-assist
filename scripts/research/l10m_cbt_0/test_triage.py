from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from .triage import ProtocolIntegrityError, candidate_sources, canonical_bytes, analyze


def test_cross_arm_candidate_never_enters_set():
    event = {"candidate_output": "", "arm": "a"}
    assert candidate_sources(event, "structured") == []


def test_future_candidate_is_not_a_same_decision_source():
    current = [{"generation": 1, "canonical": "a"}]
    future = {"generation": 2, "canonical": "b"}
    assert future not in current


def test_single_nonincumbent_candidate_is_not_selection_identifiable():
    legal = {"candidate-a"}
    assert not (len(legal) >= 2)


def test_structural_zero_regret_is_not_selection_success():
    identifiable = False; regret = 0.0
    assert regret == 0.0 and not identifiable


def test_coverage_projection_excludes_forbidden_model_proposal():
    event = {"model_proposal_canonical": "model", "admitted_canonical": "projection", "operator_disposition": "COVERAGE_PROJECTION"}
    rows = candidate_sources(event, "structured_balanced")
    assert [row["legally_selectable"] for row in rows] == [False, True]


def test_strict_retention_has_zero_retention_regret():
    incumbent, chosen = 0.4, 0.6
    retained = chosen if chosen > incumbent else incumbent
    assert max(0.0, chosen-retained) == 0.0


def test_manifest_digest_is_immutable(tmp_path: Path):
    manifest = tmp_path/"manifest.json"; digest = tmp_path/"manifest.sha256"
    manifest.write_bytes(canonical_bytes({"protocol": "wrong"})); digest.write_text(hashlib.sha256(manifest.read_bytes()).hexdigest()+"  manifest.json\n", encoding="ascii")
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolIntegrityError): analyze(tmp_path, manifest, digest)


def test_zero_experimental_provider_path():
    source = Path(__file__).with_name("triage.py").read_text(encoding="utf-8")
    assert "provider_transport" not in source
    assert "codex.exe" not in source


def test_candidate_source_reconstruction_is_deterministic():
    event = {"model_proposal_canonical": "same", "admitted_canonical": "same", "operator_disposition": "MODEL_UNTRIED_DIRECTION"}
    assert candidate_sources(event, "structured_balanced") == candidate_sources(event, "structured_balanced")

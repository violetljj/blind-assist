from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, all_specs
from .triage import ProtocolIntegrityError, analyze, canonical_bytes, changed_fields, classify


def test_representation_cardinality_is_frozen():
    assert len(all_specs()) == 162


def test_changed_field_taxonomy_is_exact():
    other = next(state for state in all_specs() if state.action_selection_turn_threshold != INITIAL_SPEC.action_selection_turn_threshold and state.fallback_action == INITIAL_SPEC.fallback_action)
    assert "action_selection_turn_threshold" in changed_fields(INITIAL_SPEC, other)


def test_representation_ceiling_has_first_priority():
    assert classify(representation_supported=False, operator_supported=False, target_trajectories=36, target_completion_candidates=0) == "REPRESENTATION_EXPRESSIBILITY_CEILING"


def test_operator_ceiling_has_second_priority():
    assert classify(representation_supported=True, operator_supported=False, target_trajectories=36, target_completion_candidates=0) == "OPERATOR_SUPPORT_CEILING"


def test_generation_ceiling_requires_supported_paths_and_zero_completion():
    assert classify(representation_supported=True, operator_supported=True, target_trajectories=36, target_completion_candidates=0) == "GENERATION_COVERAGE_CEILING"


def test_completion_observation_prevents_generation_ceiling():
    assert classify(representation_supported=True, operator_supported=True, target_trajectories=36, target_completion_candidates=1) == "CEILING_SOURCE_NOT_IDENTIFIABLE"


def test_manifest_digest_is_immutable(tmp_path: Path):
    manifest=tmp_path/"manifest.json"; digest=tmp_path/"manifest.sha256"
    manifest.write_bytes(canonical_bytes({"protocol":"wrong"})); digest.write_text(hashlib.sha256(manifest.read_bytes()).hexdigest()+"  oracle_manifest.json\n",encoding="ascii")
    manifest.write_text("{}\n",encoding="utf-8")
    with pytest.raises(ProtocolIntegrityError): analyze(tmp_path,manifest,digest)


def test_zero_experimental_provider_path():
    source=Path(__file__).with_name("triage.py").read_text(encoding="utf-8")
    assert "provider_transport" not in source and "codex.exe" not in source


def test_classification_is_deterministic():
    args=dict(representation_supported=True,operator_supported=True,target_trajectories=36,target_completion_candidates=0)
    assert classify(**args)==classify(**args)

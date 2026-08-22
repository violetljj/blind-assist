"""Mechanical truth-authority contract for the public-real pilot."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class TruthAuthorityTier(str, Enum):
    NATIVE_GT = "NATIVE_GT"
    MAP_TRAJECTORY_DERIVED = "MAP_TRAJECTORY_DERIVED"
    TEACHER_SUPPORTED = "TEACHER_SUPPORTED"
    TEACHER_ONLY_WEAK = "TEACHER_ONLY_WEAK"
    UNKNOWN = "UNKNOWN"


class TeacherAgreement(str, Enum):
    AGREE = "AGREE"
    PARTIAL = "PARTIAL"
    DISAGREE = "DISAGREE"


class FunctionalAuthority(str, Enum):
    ESTABLISHED = "ESTABLISHED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


TEACHER_KEYS = ("teacher_A", "teacher_B", "teacher_C")
STRONG_FUNCTIONAL_SOURCES = {
    TruthAuthorityTier.NATIVE_GT.value,
    TruthAuthorityTier.MAP_TRAJECTORY_DERIVED.value,
}


def empty_teacher_outputs() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "teacher_id": key,
            "implementation_id": None,
            "status": "NOT_RUN",
            "raw_output": None,
            "independent_of_evaluated_provider": None,
            "provider_family_overlap": None,
        }
        for key in TEACHER_KEYS
    }


def validate_observation_truth(row: Mapping[str, Any], *, finalized: bool) -> None:
    tier = TruthAuthorityTier(row.get("truth_authority_tier"))
    authority = FunctionalAuthority(row.get("functional_authority"))
    outputs = row.get("teacher_outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(TEACHER_KEYS):
        raise ValueError("teacher_outputs must preserve exactly teacher_A/B/C")

    successes = 0
    teacher_ids = set()
    for key in TEACHER_KEYS:
        output = outputs[key]
        if not isinstance(output, Mapping):
            raise ValueError(f"{key} output must be an object")
        teacher_id = str(output.get("teacher_id") or "")
        if not teacher_id or teacher_id in teacher_ids:
            raise ValueError("teacher identities must be present and distinct")
        teacher_ids.add(teacher_id)
        status = output.get("status")
        if status not in {"NOT_RUN", "RUN_SUCCESS", "RUN_FAILED"}:
            raise ValueError(f"invalid teacher status: {status}")
        if status == "RUN_SUCCESS":
            successes += 1
            if output.get("implementation_id") in {None, ""}:
                raise ValueError("successful teacher output requires implementation_id")
            if output.get("raw_output") is None:
                raise ValueError("successful teacher output requires preserved raw_output")
            if output.get("independent_of_evaluated_provider") is not True:
                raise ValueError("successful teacher must be independent of the evaluated provider")
            if output.get("provider_family_overlap") is not False:
                raise ValueError("teacher/provider family overlap must be absent")

    agreement_raw = row.get("teacher_agreement")
    agreement = TeacherAgreement(agreement_raw) if agreement_raw is not None else None
    if finalized and successes > 0 and agreement is None:
        raise ValueError("finalized teacher evidence requires teacher_agreement")
    if successes == 0 and agreement is not None:
        raise ValueError("teacher_agreement must be null when no teacher succeeded")
    sources = row.get("functional_authority_sources")
    if not isinstance(sources, list):
        raise ValueError("functional_authority_sources must be a list")
    source_set = {str(value) for value in sources}

    if authority is FunctionalAuthority.ESTABLISHED and not source_set.intersection(STRONG_FUNCTIONAL_SOURCES):
        raise ValueError("functional authority requires native or map/trajectory support")
    if tier is TruthAuthorityTier.NATIVE_GT and TruthAuthorityTier.NATIVE_GT.value not in source_set:
        raise ValueError("NATIVE_GT tier requires native authority source")
    if tier is TruthAuthorityTier.MAP_TRAJECTORY_DERIVED and TruthAuthorityTier.MAP_TRAJECTORY_DERIVED.value not in source_set:
        raise ValueError("MAP_TRAJECTORY_DERIVED tier requires map/trajectory source")
    if tier is TruthAuthorityTier.TEACHER_SUPPORTED:
        if authority is not FunctionalAuthority.ESTABLISHED:
            raise ValueError("TEACHER_SUPPORTED requires independently established functional authority")
        if successes < 2 or agreement not in {TeacherAgreement.AGREE, TeacherAgreement.PARTIAL}:
            raise ValueError("TEACHER_SUPPORTED requires at least two usable agreeing teachers")
    if tier is TruthAuthorityTier.TEACHER_ONLY_WEAK:
        if authority is not FunctionalAuthority.NOT_ESTABLISHED:
            raise ValueError("teacher-only consensus cannot establish functional truth")
        if successes < 2:
            raise ValueError("TEACHER_ONLY_WEAK requires at least two usable teacher outputs")
    if tier is TruthAuthorityTier.UNKNOWN and authority is not FunctionalAuthority.NOT_ESTABLISHED:
        raise ValueError("UNKNOWN cannot carry established functional authority")


def validate_annotation(annotation: Mapping[str, Any]) -> None:
    if annotation.get("schema_version") != "blindassist_real_episode_annotation_v1":
        raise ValueError("annotation schema mismatch")
    if annotation.get("truth_frozen") is not True:
        raise ValueError("evaluator accepts only frozen truth")
    for episode in annotation.get("episodes", []):
        for row in episode.get("observations", []):
            validate_observation_truth(row, finalized=True)

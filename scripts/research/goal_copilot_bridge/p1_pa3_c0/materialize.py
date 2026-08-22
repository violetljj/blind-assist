#!/usr/bin/env python3
"""Materialize provider-public PA3 Goal Contracts without target truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


INTAKE_SCHEMA = "blindassist_p1_pa3_c0_goal_intake_v1"
PROMPT_MAP_SCHEMA = "blindassist_p1_pa3_c0_prompt_map_v1"
OUTPUT_SCHEMA = "blindassist_p1_pa3_c0_public_goal_cohort_v1"
REFERENCE_MODES = {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}
SOURCE_AUTHORITIES = {"USER_TASK_INPUT", "PRODUCT_TASK_INPUT"}
FORBIDDEN_PROVIDER_KEYS = {
    "bbox",
    "bbox_xyxy",
    "mask",
    "target_bbox_xyxy",
    "target_mask",
    "target_category",
    "category_label",
    "instance_name",
    "object_uid",
    "referent_id",
    "truth",
    "evaluator",
    "canonical_prompt",
}


class MaterializationError(ValueError):
    """Raised when an intake cannot establish a public, pre-truth Goal Contract."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    return value.strip()


def _reject_forbidden_keys(value: Any, path: str = "episode") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(str(key) not in FORBIDDEN_PROVIDER_KEYS, f"{path}.{key} is evaluator/target-derived or provider-forbidden")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _prompt_lookup(prompt_map: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    _require(prompt_map.get("schema_version") == PROMPT_MAP_SCHEMA, "prompt-map schema mismatch")
    _require(prompt_map.get("mapping_rule") == "EXACT_GLOBAL_GOAL_TYPE_LOOKUP_NO_EPISODE_OVERRIDE", "prompt-map rule mismatch")
    mapping_id = _text(prompt_map.get("mapping_id"), "mapping_id")
    entries = prompt_map.get("entries")
    _require(isinstance(entries, list) and bool(entries), "prompt map must contain entries")
    lookup: dict[str, str] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "prompt-map entry must be an object")
        goal_type = _text(entry.get("goal_type"), "goal_type")
        prompt = _text(entry.get("canonical_prompt"), "canonical_prompt")
        _require(goal_type not in lookup, f"duplicate prompt mapping for {goal_type}")
        lookup[goal_type] = prompt
    return lookup, mapping_id


def materialize(intake: Mapping[str, Any], prompt_map: Mapping[str, Any]) -> dict[str, Any]:
    _require(intake.get("schema_version") == INTAKE_SCHEMA, "intake schema mismatch")
    intake_id = _text(intake.get("intake_id"), "intake_id")
    provenance = intake.get("provenance_contract")
    _require(isinstance(provenance, Mapping), "provenance_contract is required")
    _require(provenance.get("truth_state_at_goal_recording") == "NOT_CREATED", "goal must be recorded before truth exists")
    _require(provenance.get("capture_state_at_goal_recording") == "NOT_STARTED", "goal must be recorded before episode capture")
    declared_authorities = provenance.get("allowed_source_authorities")
    _require(isinstance(declared_authorities, list) and set(declared_authorities) == SOURCE_AUTHORITIES, "source-authority contract drift")
    prompt_lookup, mapping_id = _prompt_lookup(prompt_map)

    raw_episodes = intake.get("episodes")
    _require(isinstance(raw_episodes, list) and bool(raw_episodes), "prospective intake has no Goal Contract episodes")
    episodes: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    for index, raw in enumerate(raw_episodes):
        _require(isinstance(raw, Mapping), f"episode {index} must be an object")
        _reject_forbidden_keys(raw, f"episodes[{index}]")
        episode_id = _text(raw.get("episode_id"), "episode_id")
        _require(episode_id not in episode_ids, f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)
        goal_text = _text(raw.get("goal_text_original"), "goal_text_original")
        goal_recorded_at = _text(raw.get("goal_recorded_at_utc"), "goal_recorded_at_utc")
        source = raw.get("goal_source")
        _require(isinstance(source, Mapping), "goal_source is required")
        authority = _text(source.get("authority"), "goal_source.authority")
        _require(authority in SOURCE_AUTHORITIES, "goal source is not user/product task input")
        source_record_sha256 = _text(source.get("source_record_sha256"), "goal_source.source_record_sha256")
        _require(len(source_record_sha256) == 64 and all(character in "0123456789abcdef" for character in source_record_sha256), "source_record_sha256 must be lowercase SHA-256")
        contract = raw.get("goal_contract")
        _require(isinstance(contract, Mapping), "goal_contract is required")
        goal_type = _text(contract.get("goal_type"), "goal_contract.goal_type")
        reference_mode = _text(contract.get("reference_mode"), "goal_contract.reference_mode")
        _require(reference_mode in REFERENCE_MODES, "invalid reference_mode")
        _require(goal_type in prompt_lookup, f"no frozen canonical prompt for {goal_type}")
        public_contract = {
            "episode_id": episode_id,
            "goal_text_original": goal_text,
            "goal_contract": {
                "goal_type": goal_type,
                "reference_mode": reference_mode,
                "task_semantics": _text(contract.get("task_semantics"), "goal_contract.task_semantics"),
            },
            "canonical_prompt": prompt_lookup[goal_type],
            "goal_provenance": {
                "goal_recorded_at_utc": goal_recorded_at,
                "source_authority": authority,
                "source_record_sha256": source_record_sha256,
                "capture_state_at_goal_recording": "NOT_STARTED",
                "truth_state_at_goal_recording": "NOT_CREATED",
            },
        }
        episodes.append(public_contract)

    prompt_map_sha256 = content_sha256(prompt_map)
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol_id": "P1-PA3-C0-PUBLIC-GOAL-CONTRACT-COHORT-MATERIALIZATION-V1",
        "intake_id": intake_id,
        "intake_sha256": content_sha256(intake),
        "prompt_mapping_id": mapping_id,
        "prompt_map_sha256": prompt_map_sha256,
        "provider_public_fields": [
            "episode_id",
            "goal_text_original",
            "goal_contract",
            "canonical_prompt",
            "goal_provenance",
        ],
        "private_truth_access": False,
        "created_before_truth": "PENDING_FUTURE_TRUTH_BINDING_TO_THIS_RECEIPT",
        "future_truth_must_bind_receipt_body_sha256": True,
        "episodes": episodes,
        "episode_count": len(episodes),
        "pa3_inference_authorized": False,
        "claim_ceiling": "PUBLIC_GOAL_CONTRACT_PROVENANCE_MECHANICS_ONLY_NO_PROPOSAL_IDENTITY_MODEL_PRODUCT_OR_SAFETY_CLAIM",
        "terminal": "P1_PA3_C0_PUBLIC_GOAL_CONTRACT_COHORT_MATERIALIZED_PA3_NOT_AUTHORIZED",
    }
    payload["receipt_body_sha256"] = content_sha256(payload)
    return payload


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intake", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise MaterializationError("output already exists; prospective goal receipts are immutable")
    intake = json.loads(args.intake.read_text(encoding="utf-8"))
    prompt_map = json.loads(args.prompt_map.read_text(encoding="utf-8"))
    atomic_json(args.output, materialize(intake, prompt_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

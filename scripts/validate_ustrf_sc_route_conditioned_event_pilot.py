#!/usr/bin/env python3
"""Audit the 10-episode USTRF route-conditioned collection pilot.

The pilot exists only to prove that the collection pipeline can produce a
hash-bound, frame-aligned package. It can never authorize full-matrix GPT/Codex consensus truth,
U0 evaluation, training, Android changes, or production replacement.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_ustrf_sc_route_conditioned_event_pilot_audit_v1"
PILOT_MANIFEST_SCHEMA = "blindassist_ustrf_sc_route_conditioned_event_pilot_manifest_v1"
PILOT_SCOPE = "pipeline_audit_pilot"
PILOT_AUTHORITY = "collection-pipeline-audit-only"


class ContractError(ValueError):
    """The pilot package is not valid under the frozen collection contract."""


_BASE_PATH = Path(__file__).with_name("validate_sanpo_counterfactual_episodes.py")
_BASE_SPEC = importlib.util.spec_from_file_location("ustrf_pilot_base_validator", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load base validator: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)

_BINDING_PATH = Path(__file__).with_name("validate_ustrf_sc_capture_frame_ledger.py")
_BINDING_SPEC = importlib.util.spec_from_file_location("ustrf_pilot_binding_validator", _BINDING_PATH)
if _BINDING_SPEC is None or _BINDING_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load frame binding validator: {_BINDING_PATH}")
_BINDING = importlib.util.module_from_spec(_BINDING_SPEC)
_BINDING_SPEC.loader.exec_module(_BINDING)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _validate_frame_ledger(
    row: dict[str, Any], *, root: Path, policy: dict[str, Any], endpoint_tolerance_ms: int, where: str,
) -> dict[str, Any]:
    try:
        return _BINDING.validate_episode_binding(
            row, root=root, policy=policy, endpoint_tolerance_ms=endpoint_tolerance_ms, where=where,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ContractError(str(error)) from error


def validate_pilot(config: dict[str, Any], manifest: dict[str, Any], *, root: Path) -> dict[str, Any]:
    if config.get("schema") != "blindassist_sanpo_counterfactual_episode_collection_v1":
        raise ContractError("unexpected collection config schema")
    if config.get("collection_scope") != "official_full_matrix":
        raise ContractError("pilot must derive from the official_full_matrix collection config")
    design = config.get("design")
    pilot = design.get("pilot_before_full_matrix") if isinstance(design, dict) else None
    if not isinstance(pilot, dict) or pilot.get("authority") != PILOT_AUTHORITY:
        raise ContractError("config lacks the collection-pipeline-audit-only pilot policy")
    pilot_contract_id = _text(pilot.get("contract_id"), where="config.design.pilot_before_full_matrix.contract_id")
    if manifest.get("schema") != PILOT_MANIFEST_SCHEMA:
        raise ContractError("unexpected pilot manifest schema")
    if manifest.get("contract_id") != pilot_contract_id:
        raise ContractError("pilot manifest contract_id mismatch")
    if manifest.get("source_truth_contract_id") != config.get("contract_id"):
        raise ContractError("pilot manifest source_truth_contract_id mismatch")
    if manifest.get("collection_scope") != PILOT_SCOPE or manifest.get("collection_status") != "pilot_complete":
        raise ContractError("pilot manifest must declare pipeline_audit_pilot and pilot_complete")
    if manifest.get("pilot_authority") != PILOT_AUTHORITY:
        raise ContractError("pilot manifest authority mismatch")
    for key in (
        "route_conditioned_truth_eligible", "u0_evaluation_eligible", "s0_probe_eligible",
        "training_eligible", "android_runtime_change_authorized", "production_model_replacement_authorized",
    ):
        if manifest.get(key) is not False:
            raise ContractError(f"pilot manifest must declare {key}=false")

    base_config = copy.deepcopy(config)
    base_config["contract_id"] = pilot_contract_id
    # The base validator is reused only for evidence semantics.  Removing the
    # official scope prevents a pilot from masquerading as full-matrix truth.
    base_config.pop("collection_scope", None)
    base_config["source_receipt_schema"]["required_origin_scope"] = pilot.get("origin_scope")
    base_manifest = copy.deepcopy(manifest)
    base_manifest["schema"] = "blindassist_sanpo_counterfactual_episode_manifest_v1"
    try:
        base_report = _BASE.validate(base_config, base_manifest, root=root, require_complete=False)
    except (ValueError, KeyError, TypeError) as error:
        raise ContractError(f"pilot evidence validation failed: {error}") from error
    if base_report.get("route_conditioned_truth_eligible") is not False:
        raise ContractError("pilot must never pass the route-conditioned truth gate")

    sessions = config.get("sessions")
    scenes = config.get("scenes")
    if not isinstance(sessions, list) or not isinstance(scenes, list):
        raise ContractError("config sessions and scenes must be lists")
    session_count = pilot.get("session_count")
    pair_count = pilot.get("matched_pairs_per_scene")
    expected_episode_count = pilot.get("episode_count")
    if not isinstance(session_count, int) or session_count <= 0 or not isinstance(pair_count, int) or pair_count <= 0:
        raise ContractError("pilot matrix counts are invalid")
    pilot_sessions = [_text(row.get("session_id") if isinstance(row, dict) else None, where="config.sessions.session_id") for row in sessions[:session_count]]
    scene_ids = [_text(row.get("scene_id") if isinstance(row, dict) else None, where="config.scenes.scene_id") for row in scenes]
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episode_count:
        raise ContractError("pilot episode count does not match the frozen matrix")
    expected_matrix = Counter((session_id, scene_id, role) for session_id in pilot_sessions for scene_id in scene_ids for role in ("positive", "matched_negative") for _ in range(pair_count))
    actual_matrix = Counter((row.get("session_id"), row.get("scene_id"), row.get("pair_role")) for row in episodes if isinstance(row, dict))
    if actual_matrix != expected_matrix:
        raise ContractError("pilot episodes do not exactly cover the frozen session/scene/role matrix")
    episode_ids = [row.get("episode_id") for row in episodes]
    risk_event_ids = [row.get("risk_event_id") for row in episodes]
    if len(set(episode_ids)) != len(episode_ids) or len(set(risk_event_ids)) != len(risk_event_ids):
        raise ContractError("pilot episode_id and risk_event_id values must be unique")

    endpoint_tolerance = config.get("route_conditioning_policy", {}).get("endpoint_coverage_tolerance_ms")
    if not isinstance(endpoint_tolerance, int) or endpoint_tolerance < 0:
        raise ContractError("route endpoint tolerance is invalid")
    binding_by_episode: dict[str, dict[str, Any]] = {}
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(episodes):
        if not isinstance(row, dict):
            raise ContractError(f"episodes[{index}] must be an object")
        binding_by_episode[row["episode_id"]] = _validate_frame_ledger(
            row,
            root=root,
            policy=config.get("capture_frame_ledger_policy", {}),
            endpoint_tolerance_ms=endpoint_tolerance,
            where=f"episodes[{index}]",
        )
        pairs[_text(row.get("matched_pair_id"), where=f"episodes[{index}].matched_pair_id")].append(row)
    expected_pair_count = session_count * len(scene_ids) * pair_count
    if len(pairs) != expected_pair_count:
        raise ContractError("pilot matched-pair count does not match the frozen matrix")
    for pair_id, members in pairs.items():
        bindings = [binding_by_episode[member["episode_id"]] for member in members]
        pair_semantics = [
            (binding["route_plan_id"], binding["provider_policy"], binding["route_choice"])
            for binding in bindings
        ]
        if len(pair_semantics) != 2 or pair_semantics[0] != pair_semantics[1]:
            raise ContractError(f"matched_pair {pair_id} does not share route_plan_id, provider policy, and route choice")

    return {
        "schema": SCHEMA,
        "contract_id": pilot_contract_id,
        "source_truth_contract_id": config.get("contract_id"),
        "collection_scope": PILOT_SCOPE,
        "collection_status": "pilot_complete",
        "pilot_collection_pipeline_audit_passed": True,
        "episode_count": len(episodes),
        "matched_pair_count": len(pairs),
        "route_bound_episode_count": base_report.get("route_bound_episode_count"),
        "capture_frame_count": sum(binding["frame_count"] for binding in binding_by_episode.values()),
        "route_conditioned_truth_eligible": False,
        "u0_evaluation_eligible": False,
        "s0_probe_eligible": False,
        "training_eligible": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_pilot(_load(args.config), _load(args.manifest), root=args.manifest.resolve().parent)
    except (ContractError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "pilot_collection_pipeline_audit_passed": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

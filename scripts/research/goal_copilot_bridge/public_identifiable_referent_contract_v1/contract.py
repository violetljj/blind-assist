"""Freeze a public referent contract separately from evaluator-private identity truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "PUBLIC_IDENTIFIABLE_REFERENT_CONTRACT_V1"
FREEZE_SCHEMA = "blindassist_public_identifiable_referent_freeze_v1"
PUBLIC_SCHEMA = "blindassist_public_identifiable_referent_contract_v1"
PRIVATE_SCHEMA = "blindassist_private_referent_identity_lock_v1"
TRUTH_SCHEMA = "blindassist_private_referent_observation_truth_v1"
REPORT_SCHEMA = "blindassist_public_identifiable_referent_truth_audit_v1"
REFERENCE_MODES = {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}
MODALITIES = {"REFERENCE_IMAGE_INSTANCE", "LANGUAGE_REFERRING_EXPRESSION"}
BINDING_AUTHORITIES = {
    "SOURCE_NATIVE_REFERENCE_LINK",
    "NATIVE_INSTANCE_ID",
    "INDEPENDENT_PREOBSERVATION_REVIEW",
}
FORBIDDEN_PUBLIC_KEYS = {
    "physical_instance_id",
    "physical_instance_ids",
    "legal_physical_instance_ids",
    "world_anchor",
    "world_anchors",
    "target_region",
    "target_regions",
    "visibility_truth",
    "private_truth",
    "evaluator",
    "object_uid",
}
CLAIM_CEILING = (
    "PUBLIC_REFERENT_CONTRACT_AND_FIREWALL_MECHANICS_ONLY_NO_COHORT_BASELINE_"
    "IDENTITY_ALGORITHM_ACTIVE_SEARCH_CONTROL_SAFETY_OR_PRODUCT_CLAIM"
)


class ContractError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    _require(len(text) == 64 and all(character in "0123456789abcdef" for character in text), f"{label} must be lowercase SHA-256")
    return text


def _utc_timestamp(value: Any, label: str) -> str:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{label} must be an ISO-8601 timestamp") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed), f"{label} must be UTC")
    return text


def _normalized_bbox(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 4, f"{label} must contain four coordinates")
    _require(all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value), f"{label} must be numeric")
    x0, y0, x1, y1 = [float(item) for item in value]
    _require(0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0, f"{label} must be positive normalized XYXY")
    return [x0, y0, x1, y1]


def _reject_private_public_keys(value: Any, path: str = "public_contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(str(key) not in FORBIDDEN_PUBLIC_KEYS, f"{path}.{key} is evaluator-private")
            _reject_private_public_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_public_keys(child, f"{path}[{index}]")


def _verify_body_hash(value: Mapping[str, Any], label: str) -> None:
    claimed = value.get("body_sha256")
    payload = dict(value)
    payload.pop("body_sha256", None)
    _require(isinstance(claimed, str) and content_sha256(payload) == claimed, f"{label} body hash mismatch")


def _validate_reference_image(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "reference_image is required for reference-image goals")
    selector = _text(value.get("target_selector"), "reference_image.target_selector")
    _require(selector in {"FULL_FRAME_SINGLE_INSTANCE", "PUBLIC_TARGET_REGION"}, "unsupported reference target selector")
    region = value.get("public_target_region_xyxy")
    if selector == "FULL_FRAME_SINGLE_INSTANCE":
        _require(region is None, "full-frame selector cannot carry a target region")
    else:
        region = _normalized_bbox(region, "reference_image.public_target_region_xyxy")
    width = value.get("width")
    height = value.get("height")
    _require(isinstance(width, int) and not isinstance(width, bool) and width > 0, "reference image width must be positive")
    _require(isinstance(height, int) and not isinstance(height, bool) and height > 0, "reference image height must be positive")
    return {
        "reference_image_id": _text(value.get("reference_image_id"), "reference_image.reference_image_id"),
        "image_sha256": _sha256(value.get("image_sha256"), "reference_image.image_sha256"),
        "width": width,
        "height": height,
        "target_selector": selector,
        "public_target_region_xyxy": region,
        "role": "TARGET_INSTANCE_REFERENCE",
    }


def _validate_world_anchors(value: Any, legal_ids: list[str], mode: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "private_binding.world_anchors must be an array")
    if mode == "AMBIGUOUS":
        _require(not value, "AMBIGUOUS cannot carry world anchors")
        return []
    _require(len(value) == len(legal_ids), "every legal physical instance requires one world anchor")
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        _require(isinstance(raw, Mapping), "world anchor must be an object")
        instance_id = _text(raw.get("physical_instance_id"), "world_anchor.physical_instance_id")
        _require(instance_id in legal_ids and instance_id not in seen, "world anchor instance mismatch or duplicate")
        position = raw.get("position_xyz_m")
        _require(isinstance(position, list) and len(position) == 3, "world anchor position must contain XYZ")
        _require(all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in position), "world anchor position must be numeric")
        seen.add(instance_id)
        anchors.append(
            {
                "physical_instance_id": instance_id,
                "coordinate_frame_id": _text(raw.get("coordinate_frame_id"), "world_anchor.coordinate_frame_id"),
                "position_xyz_m": [float(item) for item in position],
                "authority": _text(raw.get("authority"), "world_anchor.authority"),
            }
        )
    return anchors


def freeze_contract(bundle: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an outcome-blind freeze bundle and split public/private receipts."""

    _require(bundle.get("schema_version") == FREEZE_SCHEMA, "freeze schema mismatch")
    _require(bundle.get("protocol_id") == PROTOCOL_ID, "protocol mismatch")
    contract_id = _text(bundle.get("contract_id"), "contract_id")
    frozen_at = _utc_timestamp(bundle.get("frozen_at_utc"), "frozen_at_utc")
    states = bundle.get("freeze_states")
    _require(isinstance(states, Mapping), "freeze_states is required")
    _require(states.get("episode_observation_pixels") == "NOT_CAPTURED", "contract must precede episode pixels")
    _require(states.get("provider_output") == "NOT_CREATED", "contract must precede provider output")
    _require(states.get("candidate_output") == "NOT_CREATED", "contract must precede candidates")
    _require(states.get("outcome_access") == "NONE", "contract freeze cannot access outcomes")

    goal = bundle.get("public_goal")
    _require(isinstance(goal, Mapping), "public_goal is required")
    modality = _text(goal.get("modality"), "public_goal.modality")
    mode = _text(goal.get("reference_mode"), "public_goal.reference_mode")
    _require(modality in MODALITIES, "unsupported public goal modality")
    _require(mode in REFERENCE_MODES, "invalid reference mode")
    language_description = goal.get("language_description")
    if language_description is not None:
        language_description = _text(language_description, "public_goal.language_description")
    reference_image = goal.get("reference_image")
    if modality == "REFERENCE_IMAGE_INSTANCE":
        _require(mode == "UNIQUE", "reference-image instance goals must be UNIQUE")
        reference_image = _validate_reference_image(reference_image)
    else:
        _require(reference_image is None, "language-only goals cannot carry a reference image")
        _require(language_description is not None, "language referring expression is required")

    binding = bundle.get("private_binding")
    _require(isinstance(binding, Mapping), "private_binding is required")
    authority = _text(binding.get("binding_authority"), "private_binding.binding_authority")
    _require(authority in BINDING_AUTHORITIES, "binding authority is not admissible")
    _require(binding.get("binding_created_before_episode_observations") is True, "identity binding must precede episode observations")
    _require(binding.get("binding_created_before_provider_output") is True, "identity binding must precede provider output")
    _require(binding.get("model_or_teacher_used_for_binding") is False, "model or teacher cannot be identity authority")
    binding_created_at = _utc_timestamp(binding.get("binding_created_at_utc"), "private_binding.binding_created_at_utc")
    _require(
        datetime.fromisoformat(binding_created_at.replace("Z", "+00:00"))
        <= datetime.fromisoformat(frozen_at.replace("Z", "+00:00")),
        "identity binding cannot postdate contract freeze",
    )
    bound_reference_sha = binding.get("bound_reference_image_sha256")
    if modality == "REFERENCE_IMAGE_INSTANCE":
        bound_reference_sha = _sha256(bound_reference_sha, "private_binding.bound_reference_image_sha256")
        _require(bound_reference_sha == reference_image["image_sha256"], "private identity lock is not bound to the public reference image")
    else:
        _require(bound_reference_sha is None, "language-only binding cannot claim a reference image")
    legal_raw = binding.get("legal_physical_instance_ids")
    _require(isinstance(legal_raw, list), "legal physical instance ids must be an array")
    legal_ids = [_text(item, "legal physical instance id") for item in legal_raw]
    _require(len(legal_ids) == len(set(legal_ids)), "legal physical instance ids must be unique")
    if mode == "UNIQUE":
        _require(len(legal_ids) == 1, "UNIQUE requires exactly one physical instance")
    elif mode == "SET_VALUED":
        _require(len(legal_ids) >= 2, "SET_VALUED requires at least two legal physical instances")
    else:
        _require(not legal_ids, "AMBIGUOUS cannot carry legal physical instances")
    world_anchors = _validate_world_anchors(binding.get("world_anchors"), legal_ids, mode)
    binding_body = {
        "binding_authority": authority,
        "binding_created_at_utc": binding_created_at,
        "binding_created_before_episode_observations": True,
        "binding_created_before_provider_output": True,
        "model_or_teacher_used_for_binding": False,
        "source_record_sha256": _sha256(binding.get("source_record_sha256"), "private_binding.source_record_sha256"),
        "bound_reference_image_sha256": bound_reference_sha,
        "legal_physical_instance_ids": legal_ids,
        "world_anchors": world_anchors,
    }
    private_commitment = content_sha256(binding_body)

    public_receipt = {
        "schema_version": PUBLIC_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "contract_id": contract_id,
        "frozen_at_utc": frozen_at,
        "goal_text": _text(goal.get("goal_text"), "public_goal.goal_text"),
        "modality": modality,
        "reference_mode": mode,
        "reference_anchor_id": _text(goal.get("reference_anchor_id"), "public_goal.reference_anchor_id"),
        "reference_image": reference_image,
        "language_description": language_description,
        "language_role": "SUPPLEMENTARY_RECOGNITION_EVIDENCE_NOT_IDENTITY_AUTHORITY" if language_description else "NONE",
        "private_identity_commitment_sha256": private_commitment,
        "provider_may_access_private_identity": False,
        "created_before_episode_observations": True,
        "created_before_provider_output": True,
        "cohort_freeze_authorized": False,
        "passive_baseline_authorized": False,
        "algorithm_authorized": False,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "PUBLIC_IDENTIFIABLE_REFERENT_CONTRACT_V1_FROZEN_EXECUTION_NOT_AUTHORIZED",
    }
    _reject_private_public_keys(public_receipt)
    public_bytes = canonical_bytes(public_receipt)
    _require(all(instance_id.encode("utf-8") not in public_bytes for instance_id in legal_ids), "physical instance id leaked into public contract")
    public_receipt["body_sha256"] = content_sha256(public_receipt)

    private_receipt = {
        "schema_version": PRIVATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "contract_id": contract_id,
        "reference_anchor_id": public_receipt["reference_anchor_id"],
        "reference_mode": mode,
        "public_contract_body_sha256": public_receipt["body_sha256"],
        "private_identity_commitment_sha256": private_commitment,
        "private_binding": binding_body,
        "provider_access": "FORBIDDEN",
        "observation_truth_state": "NOT_CREATED",
    }
    private_receipt["body_sha256"] = content_sha256(private_receipt)
    return public_receipt, private_receipt


def validate_observation_truth(
    public_receipt: Mapping[str, Any], private_receipt: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate later evaluator truth against the pre-observation identity lock."""

    _verify_body_hash(public_receipt, "public contract")
    _verify_body_hash(private_receipt, "private identity lock")
    _require(public_receipt.get("schema_version") == PUBLIC_SCHEMA, "public schema mismatch")
    _require(private_receipt.get("schema_version") == PRIVATE_SCHEMA, "private schema mismatch")
    _require(truth.get("schema_version") == TRUTH_SCHEMA, "observation truth schema mismatch")
    contract_id = public_receipt["contract_id"]
    _require(private_receipt.get("contract_id") == contract_id == truth.get("contract_id"), "contract id mismatch")
    _require(private_receipt.get("public_contract_body_sha256") == public_receipt["body_sha256"], "private lock is not bound to public contract")
    _require(truth.get("public_contract_body_sha256") == public_receipt["body_sha256"], "truth is not bound to public contract")
    _require(truth.get("private_identity_lock_body_sha256") == private_receipt["body_sha256"], "truth is not bound to private identity lock")
    _require(truth.get("provider_access_to_truth") is False, "provider cannot access observation truth")
    _require(truth.get("truth_created_after_contract_freeze") is True, "truth timing must follow contract freeze")

    mode = public_receipt["reference_mode"]
    legal_ids = private_receipt["private_binding"]["legal_physical_instance_ids"]
    observations = truth.get("observations")
    _require(isinstance(observations, list) and bool(observations), "observation truth must be non-empty")
    rows = []
    seen: set[str] = set()
    for raw in observations:
        _require(isinstance(raw, Mapping), "observation truth row must be an object")
        observation_id = _text(raw.get("observation_id"), "observation_id")
        _require(observation_id not in seen, "duplicate observation id")
        seen.add(observation_id)
        visibility = _text(raw.get("visibility"), "visibility")
        _require(visibility in {"VISIBLE", "NOT_VISIBLE", "UNKNOWN"}, "invalid visibility")
        regions_raw = raw.get("target_regions")
        _require(isinstance(regions_raw, list), "target_regions must be an array")
        regions = []
        region_ids: set[str] = set()
        for region in regions_raw:
            _require(isinstance(region, Mapping), "target region must be an object")
            instance_id = _text(region.get("physical_instance_id"), "target_region.physical_instance_id")
            _require(instance_id in legal_ids and instance_id not in region_ids, "target region instance is illegal or duplicated")
            region_ids.add(instance_id)
            regions.append({"physical_instance_id": instance_id, "bbox_xyxy_normalized": _normalized_bbox(region.get("bbox_xyxy_normalized"), "target region bbox")})

        if mode == "AMBIGUOUS":
            _require(visibility == "UNKNOWN" and not regions, "AMBIGUOUS truth cannot create a scored target")
            evaluable = False
        elif visibility == "VISIBLE":
            _require(bool(regions), "VISIBLE requires at least one legal target region")
            if mode == "UNIQUE":
                _require(len(regions) == 1 and regions[0]["physical_instance_id"] == legal_ids[0], "UNIQUE visible truth must bind the locked instance")
            evaluable = True
        elif visibility == "NOT_VISIBLE":
            _require(not regions, "NOT_VISIBLE cannot carry target regions")
            evaluable = True
        else:
            _require(not regions, "UNKNOWN cannot carry target regions")
            evaluable = False
        rows.append(
            {
                "observation_id": observation_id,
                "frame_sha256": _sha256(raw.get("frame_sha256"), "frame_sha256"),
                "visibility": visibility,
                "target_regions": regions,
                "primary_evaluable": evaluable,
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "contract_id": contract_id,
        "public_contract_body_sha256": public_receipt["body_sha256"],
        "private_identity_lock_body_sha256": private_receipt["body_sha256"],
        "observation_truth_body_sha256": content_sha256(truth),
        "reference_mode": mode,
        "observation_count": len(rows),
        "primary_evaluable_count": sum(item["primary_evaluable"] for item in rows),
        "rows": rows,
        "cohort_freeze_authorized": False,
        "passive_baseline_authorized": False,
        "algorithm_authorized": False,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "PUBLIC_IDENTIFIABLE_REFERENT_TRUTH_BINDING_MECHANICS_PASS_EXECUTION_NOT_AUTHORIZED",
    }
    report["body_sha256"] = content_sha256(report)
    return report


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-bundle", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args(argv)
    _require(not args.public_output.exists() and not args.private_output.exists(), "freeze outputs must be new")
    bundle = json.loads(args.freeze_bundle.read_text(encoding="utf-8"))
    public_receipt, private_receipt = freeze_contract(bundle)
    atomic_json(args.public_output, public_receipt)
    atomic_json(args.private_output, private_receipt)
    print(json.dumps({"public": str(args.public_output.resolve()), "private": str(args.private_output.resolve()), "terminal": public_receipt["terminal"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

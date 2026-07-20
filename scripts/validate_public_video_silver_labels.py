#!/usr/bin/env python3
"""Validate hash-bound GPT/VLM supervision for public navigation videos.

Legacy v1 manifests remain comparison-only.  The explicit v2 path permits
provisional model training from licensed, hash-bound, machine-labelled public
RGB, while preserving the distinction from human event truth and forbidding
calibration, blind evaluation, and production replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class SilverLabelError(ValueError):
    """A public-video silver-label manifest breaks its quarantine contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SilverLabelError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SilverLabelError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SilverLabelError(f"{where} must be a non-empty string")
    return value


def boolean_false(value: Any, *, where: str) -> None:
    if value is not False:
        raise SilverLabelError(f"{where} must be false")


def number_01(value: Any, *, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1:
        raise SilverLabelError(f"{where} must be a number in [0, 1]")
    return float(value)


def known_hashes(source_manifest: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    rows = source_manifest.get("frames")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("sha256"), str):
                hashes.add(row["sha256"])
    if not hashes:
        raise SilverLabelError("source manifest contains no frame SHA256 values")
    return hashes


def validate_v2_source_license(source_manifest: dict[str, Any]) -> None:
    source_info = source_manifest.get("source")
    if not isinstance(source_info, dict):
        raise SilverLabelError("v2 provisional training requires source license metadata")
    license_name = source_info.get("license")
    if license_name in {"CC-BY-4.0", "CC0-1.0"}:
        return
    if license_name != "CC-BY-3.0":
        raise SilverLabelError(
            "v2 provisional training requires CC-BY-4.0, reviewed CC-BY-3.0, or CC0-1.0"
        )
    review = source_manifest.get("license_review")
    if not isinstance(review, dict):
        raise SilverLabelError(
            "CC-BY-3.0 provisional training requires a bound Wikimedia license review"
        )
    required = {
        "status": "license_confirmed_by_youtube_review_bot",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
    }
    for field, expected in required.items():
        if review.get(field) != expected:
            raise SilverLabelError(
                f"CC-BY-3.0 license_review.{field} must be {expected!r}"
            )
    for field in ("reviewed_at", "file_page_url", "original_source_url", "author"):
        text(review.get(field), where=f"license_review.{field}")
    if not str(review["file_page_url"]).startswith("https://commons.wikimedia.org/"):
        raise SilverLabelError(
            "CC-BY-3.0 license_review.file_page_url must bind a Wikimedia Commons page"
        )
    if source_info.get("author") != review.get("author"):
        raise SilverLabelError(
            "CC-BY-3.0 source author must match the bound license review attribution"
        )


def validate_negative_decision_quality(episode: dict[str, Any], *, where: str) -> None:
    """Fail closed when a VLM claims a no-alert corridor despite ambiguity.

    A model-only no-alert claim is especially vulnerable to a camera pan hiding
    a near seated person or obstacle. Such clips remain useful as abstentions,
    but cannot be included in the candidate-agreement denominator.
    """
    quality = episode.get("negative_decision_quality")
    if not isinstance(quality, dict):
        raise SilverLabelError(f"{where}.negative_decision_quality is required for candidate_no_alert")
    heading = number_01(quality.get("corridor_heading_stability"), where=f"{where}.negative_decision_quality.corridor_heading_stability")
    visibility = number_01(quality.get("near_field_visibility"), where=f"{where}.negative_decision_quality.near_field_visibility")
    clearance = number_01(quality.get("corridor_clearance"), where=f"{where}.negative_decision_quality.corridor_clearance")
    lateral = number_01(quality.get("near_field_lateral_intrusion_absent"), where=f"{where}.negative_decision_quality.near_field_lateral_intrusion_absent")
    if min(heading, visibility, clearance, lateral) < 0.7:
        raise SilverLabelError(f"{where} candidate_no_alert requires stable heading, visible near field, clear corridor, and no near lateral intrusion >= 0.7; otherwise abstain")


def validate_causal_actionability(episode: dict[str, Any], *, where: str) -> None:
    actionability = episode.get("silver_actionability")
    allowed = {
        "no_attention",
        "context_only",
        "intervention_then_route_clear",
        "persistent_intervention",
        "uncertain",
    }
    if actionability not in allowed:
        raise SilverLabelError(f"{where}.silver_actionability is invalid")
    if episode.get("causal_evidence_basis") != "past_or_current_only":
        raise SilverLabelError(f"{where}.causal_evidence_basis must be past_or_current_only")
    verdict = episode.get("silver_should_alert")
    expected_verdicts = {
        "no_attention": {"candidate_no_alert"},
        "context_only": {"candidate_alert"},
        "intervention_then_route_clear": {"candidate_alert"},
        "persistent_intervention": {"candidate_alert"},
        "uncertain": {"abstain"},
    }
    if verdict not in expected_verdicts[actionability]:
        raise SilverLabelError(
            f"{where} eventual outcome cannot redefine causal actionability; "
            f"{actionability} requires {sorted(expected_verdicts[actionability])}"
        )
    eventual = episode.get("eventual_outcome")
    if eventual not in {"unknown", "safe_pass", "route_changed", "contact_or_blocked", "not_applicable"}:
        raise SilverLabelError(f"{where}.eventual_outcome is invalid")


def validate(manifest: dict[str, Any], *, source_manifest_path: Path) -> dict[str, Any]:
    schema = manifest.get("schema")
    if schema not in {"blindassist_public_video_silver_labels_v1", "blindassist_public_video_silver_labels_v2", "blindassist_public_video_silver_labels_v3"}:
        raise SilverLabelError("unexpected silver-label schema")
    source_manifest = load_json(source_manifest_path)
    source = manifest.get("source")
    labeler = manifest.get("labeler")
    labels = manifest.get("episodes")
    if not isinstance(source, dict) or not isinstance(labeler, dict) or not isinstance(labels, list) or not labels:
        raise SilverLabelError("manifest requires source, labeler, and non-empty episodes")
    text(source.get("source_id"), where="source.source_id")
    if source.get("source_manifest_sha256") != sha256_file(source_manifest_path):
        raise SilverLabelError("source manifest SHA256 does not match")
    if source.get("human_event_truth_present") is not False:
        raise SilverLabelError("source must declare human_event_truth_present=false")
    if source.get("privacy_audit_required") is not True:
        raise SilverLabelError("source must keep privacy_audit_required=true")
    if schema in {"blindassist_public_video_silver_labels_v2", "blindassist_public_video_silver_labels_v3"}:
        validate_v2_source_license(source_manifest)
        if source_manifest.get("provisional_training_authorized") is not True:
            raise SilverLabelError("v2 provisional training requires an authorized source manifest")
    for field in ("provider", "model", "prompt_id", "prompt_sha256", "review_mode"):
        text(labeler.get(field), where=f"labeler.{field}")
    if labeler.get("review_mode") != "multiframe_temporal":
        raise SilverLabelError("labeler.review_mode must be multiframe_temporal")
    hashes = known_hashes(source_manifest)
    identifiers: set[str] = set()
    for index, episode in enumerate(labels):
        where = f"episodes[{index}]"
        if not isinstance(episode, dict):
            raise SilverLabelError(f"{where} must be an object")
        identifier = text(episode.get("episode_id"), where=f"{where}.episode_id")
        if identifier in identifiers:
            raise SilverLabelError(f"duplicate episode_id: {identifier}")
        identifiers.add(identifier)
        verdict = episode.get("silver_should_alert")
        if verdict not in {"candidate_alert", "candidate_no_alert", "abstain"}:
            raise SilverLabelError(f"{where}.silver_should_alert is invalid")
        confidence = number_01(episode.get("confidence"), where=f"{where}.confidence")
        evidence = episode.get("evidence_frame_sha256")
        if not isinstance(evidence, list) or len(evidence) < 2 or not all(isinstance(item, str) and item in hashes for item in evidence):
            raise SilverLabelError(f"{where}.evidence_frame_sha256 needs at least two known frame hashes")
        reasons = episode.get("uncertainty_reasons")
        if not isinstance(reasons, list) or not all(isinstance(item, str) and item for item in reasons):
            raise SilverLabelError(f"{where}.uncertainty_reasons must be a string list")
        if verdict == "abstain" and confidence > 0.5:
            raise SilverLabelError(f"{where} abstentions must have confidence <= 0.5")
        if verdict != "abstain" and not isinstance(episode.get("risk_profile"), dict):
            raise SilverLabelError(f"{where}.risk_profile is required for a non-abstaining verdict")
        if verdict == "candidate_no_alert":
            validate_negative_decision_quality(episode, where=where)
        if schema == "blindassist_public_video_silver_labels_v3":
            validate_causal_actionability(episode, where=where)
    for field in ("calibration_authorized", "blind_evaluation_authorized", "production_model_replacement_authorized"):
        boolean_false(manifest.get(field), where=field)
    if schema == "blindassist_public_video_silver_labels_v1":
        boolean_false(manifest.get("training_execution_authorized"), where="training_execution_authorized")
        training_authorized = False
    else:
        if manifest.get("training_execution_authorized") is not True:
            raise SilverLabelError("v2 provisional training requires training_execution_authorized=true")
        if manifest.get("training_mode") != "provisional_model_supervision":
            raise SilverLabelError("v2 provisional training requires training_mode=provisional_model_supervision")
        training_authorized = True
    return {
        "ok": True,
        "episode_count": len(labels),
        "candidate_alert_count": sum(item["silver_should_alert"] == "candidate_alert" for item in labels),
        "candidate_no_alert_count": sum(item["silver_should_alert"] == "candidate_no_alert" for item in labels),
        "abstain_count": sum(item["silver_should_alert"] == "abstain" for item in labels),
        "model_assisted_only": True,
        "provisional_model_supervision": schema in {"blindassist_public_video_silver_labels_v2", "blindassist_public_video_silver_labels_v3"},
        "causal_actionability_supervision": schema == "blindassist_public_video_silver_labels_v3",
        "training_execution_authorized": training_authorized,
        "production_model_replacement_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(load_json(args.manifest), source_manifest_path=args.source_manifest), ensure_ascii=False))
    except SilverLabelError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

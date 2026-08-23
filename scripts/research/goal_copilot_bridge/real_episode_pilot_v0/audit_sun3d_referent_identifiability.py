"""Read-only identifiability autopsy for the consumed SUN3D door episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0 import run_sun3d_door_approach_v0 as sun3d


SCHEMA_VERSION = "sun3d_referent_identifiability_audit_v0"
TERMINAL = "PUBLIC_GOAL_TO_PRIVATE_REFERENT_AMBIGUOUS_SELECTION_NOT_EVALUABLE"


class AuditError(RuntimeError):
    pass


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_content_hash(value: Mapping[str, Any], label: str) -> None:
    claimed = value.get("content_sha256")
    payload = dict(value)
    payload.pop("content_sha256", None)
    if not claimed or materializer.content_sha256(payload) != claimed:
        raise AuditError(f"{label} content hash mismatch")


def _door_family_ids(annotation: Mapping[str, Any]) -> list[int]:
    return [
        index
        for index, item in enumerate(annotation["objects"])
        if item is not None and item["name"].strip().lower().split(":", 1)[0] == "door"
    ]


def _proposal_rank(candidate_id: str) -> int:
    try:
        return int(candidate_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise AuditError(f"malformed candidate id: {candidate_id}") from error


def build_audit(
    roster: Mapping[str, Any],
    pixels_manifest: Mapping[str, Any],
    proposal_output: Mapping[str, Any],
    final_report: Mapping[str, Any],
    annotation: Mapping[str, Any],
    annotation_sha256: str,
    review: Mapping[str, Any],
) -> dict[str, Any]:
    _verify_content_hash(roster, "roster")
    _verify_content_hash(final_report, "final report")
    if annotation_sha256 != roster["source"]["annotation_sha256"]:
        raise AuditError("annotation SHA-256 differs from the frozen source binding")
    if review["source_roster_content_sha256"] != roster["content_sha256"]:
        raise AuditError("visual review is not bound to this roster")
    if roster["goal_text"] != "the door" or review["goal_text"] != roster["goal_text"]:
        raise AuditError("audit is defined only for the frozen generic goal")

    truth_by_id = {item["observation_id"]: item for item in roster["observations"]}
    pixel_by_id = {item["observation_id"]: item for item in pixels_manifest["observations"]}
    proposal_by_id = {item["image_id"]: item for item in proposal_output["outputs"]}
    result_by_id = {item["observation_id"]: item for item in final_report["evaluation"]["observations"]}
    review_by_id = {item["observation_id"]: item for item in review["records"]}
    visible_ids = {key for key, value in truth_by_id.items() if value["visibility"] == "VISIBLE"}
    wrong_absent_ids = {
        key
        for key, value in result_by_id.items()
        if value["outcome"] == "WRONG_CONFIDENT_GUIDANCE_ON_NOT_VISIBLE"
    }
    required_review_ids = visible_ids | wrong_absent_ids
    if set(review_by_id) != required_review_ids:
        raise AuditError("review must cover exactly target-visible frames and target-absent confident commits")

    door_ids = _door_family_ids(annotation)
    if roster["target_object_id"] not in door_ids or len(door_ids) < 2:
        raise AuditError("expected the frozen target plus at least one other native door-family object")

    rows = []
    for observation_id in sorted(required_review_ids):
        truth = truth_by_id[observation_id]
        pixel = pixel_by_id[observation_id]
        result = result_by_id[observation_id]
        reviewed = review_by_id[observation_id]
        proposal = proposal_by_id[observation_id]
        if reviewed["image_sha256"] != pixel["sha256"] or proposal["image_sha256"] != pixel["sha256"]:
            raise AuditError(f"image binding mismatch for {observation_id}")
        if reviewed["private_target_visibility"] != truth["visibility"]:
            raise AuditError(f"review visibility mismatch for {observation_id}")
        if reviewed["current_frame_resolution_for_private_target"] not in {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}:
            raise AuditError(f"invalid resolution for {observation_id}")
        if int(reviewed["plausible_door_instance_lower_bound"]) < 1:
            raise AuditError(f"door lower bound must be positive for {observation_id}")

        frame = annotation["frames"][truth["frame_id"]]
        native_door_ids = sorted(
            {int(polygon["object"]) for polygon in frame.get("polygon", []) if int(polygon["object"]) in door_ids}
        )
        correct_ranks = sorted(_proposal_rank(item) for item in result["correct_candidate_ids"])
        rows.append(
            {
                "observation_id": observation_id,
                "frame_id": truth["frame_id"],
                "private_target_visibility": truth["visibility"],
                "native_door_family_object_ids_in_annotation": native_door_ids,
                "plausible_door_instance_lower_bound": reviewed["plausible_door_instance_lower_bound"],
                "current_frame_resolution_for_private_target": reviewed[
                    "current_frame_resolution_for_private_target"
                ],
                "private_target_correct_proposal_ranks": correct_ranks,
                "brain_action": result["brain_action"],
                "sealed_private_target_outcome": result["outcome"],
                "public_goal_scoring_disposition": (
                    "NOT_EVALUABLE_GENERIC_GOAL_DOES_NOT_BIND_PRIVATE_TARGET"
                    if reviewed["current_frame_resolution_for_private_target"] == "AMBIGUOUS"
                    else "FRAME_LOCAL_UNIQUE_ONLY"
                ),
                "evidence_note": reviewed["evidence_note"],
            }
        )

    visible_rows = [item for item in rows if item["private_target_visibility"] == "VISIBLE"]
    usable_visible = [item for item in visible_rows if item["private_target_correct_proposal_ranks"]]
    ambiguous_usable = [
        item for item in usable_visible if item["current_frame_resolution_for_private_target"] == "AMBIGUOUS"
    ]
    wrong_absent_rows = [item for item in rows if item["observation_id"] in wrong_absent_ids]
    if len(visible_rows) != 4 or len(usable_visible) != 3 or len(ambiguous_usable) != 3:
        raise AuditError("consumed 4-visible/3-usable identifiability invariant changed")
    if len(wrong_absent_rows) != 3 or any(item["plausible_door_instance_lower_bound"] < 1 for item in wrong_absent_rows):
        raise AuditError("target-absent confident-commit audit invariant changed")

    target_id = int(roster["target_object_id"])
    other_doors = [
        {"object_id": index, "name": annotation["objects"][index]["name"]}
        for index in door_ids
        if index != target_id
    ]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "terminal": TERMINAL,
        "source_roster_content_sha256": roster["content_sha256"],
        "source_final_report_content_sha256": final_report["content_sha256"],
        "annotation_url": roster["source"]["annotation_url"],
        "annotation_sha256": annotation_sha256,
        "goal_text_public": roster["goal_text"],
        "private_target": {"object_id": target_id, "name": roster["target_object_name"]},
        "other_native_door_family_objects": other_doors,
        "review_authority": "CODEX_VISUAL_REVIEWER_DERIVED_READ_ONLY_NOT_NATIVE_GT",
        "episode_referent_resolution": "AMBIGUOUS",
        "set_valued_alternative": (
            "If every visually valid door is legal under the literal generic goal, the legal target is a set; "
            "the sealed evaluator nevertheless accepts only object 45."
        ),
        "critical_counts": {
            "target_visible_frames": len(visible_rows),
            "target_visible_with_usable_proposal": len(usable_visible),
            "usable_proposal_frames_ambiguous_for_private_target": len(ambiguous_usable),
            "private_target_absent_confident_commits": len(wrong_absent_rows),
            "such_commits_with_another_plausible_door_visible": sum(
                item["plausible_door_instance_lower_bound"] >= 1 for item in wrong_absent_rows
            ),
        },
        "private_target_correct_proposal_ranks_on_visible_frames": {
            item["observation_id"]: item["private_target_correct_proposal_ranks"] for item in visible_rows
        },
        "claim_disposition": {
            "object_45_visibility_4_of_15": "DESCRIPTIVE_PRIVATE_TARGET_TRUTH_ONLY",
            "object_45_not_visible_11_of_15": "NOT_PUBLIC_GOAL_ABSENCE",
            "selection_given_usable_proposal_0_of_3": "NOT_EVALUABLE_PUBLIC_REFERENT_NOT_IDENTIFIABLE",
            "wrong_confident_guidance_4_of_15": "NOT_EVALUABLE_AS_PUBLIC_GOAL_WRONGNESS",
            "active_referent_search_successor": "NOT_AUTHORIZED_BY_THIS_EPISODE",
        },
        "required_successor_frontdoor": (
            "Freeze an independently public-identifiable referent contract before pixels or provider output; "
            "preserve UNIQUE, SET_VALUED, and AMBIGUOUS and score only contract-legal targets."
        ),
        "calls": {"new_benchmark_model": 0, "provider": 0, "teacher": 0, "fresh_episode": 0},
        "rows": rows,
    }
    audit["content_sha256"] = materializer.content_sha256(audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--pixels-manifest", type=Path, required=True)
    parser.add_argument("--proposal-output", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    roster = _load(args.roster)
    response = requests.get(roster["source"]["annotation_url"], timeout=90)
    response.raise_for_status()
    annotation_bytes = response.content
    audit = build_audit(
        roster,
        _load(args.pixels_manifest),
        _load(args.proposal_output),
        _load(args.final_report),
        json.loads(annotation_bytes),
        sun3d._sha256_bytes(annotation_bytes),
        _load(args.review),
    )
    sun3d._atomic_json(args.output, audit)
    print(json.dumps({"output": str(args.output.resolve()), "terminal": audit["terminal"], "sha256": sun3d._sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_TRUE = (
    "sign_owned_by_facade",
    "entrance_owned_by_target",
    "same_facade_as_frozen_peer",
    "terminal_endpoint_visible",
    "terminal_orientation_valid",
    "terminal_standoff_valid",
    "visual_handoff_ready",
)
REQUIRED_IDENTIFIERS = (
    "facade_id",
    "entrance_instance_id",
    "sign_instance_id",
    "terminal_frame_id",
    "frame_sha256",
    "adjudication_source",
)


def sha256_document(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def template_for(cohort: dict[str, Any], cohort_sha256: str) -> dict[str, Any]:
    scenes = []
    for frozen_scene in cohort["cohort"]:
        targets = []
        for frozen_target in frozen_scene["frozen_pair"]:
            target = {"annotation_path": frozen_target["annotation_path"]}
            target.update({field: None for field in REQUIRED_IDENTIFIERS})
            target.update({field: "UNKNOWN" for field in REQUIRED_TRUE})
            targets.append(target)
        scenes.append(
            {
                "scene_id": frozen_scene["scene_id"],
                "frozen_targets": targets,
                "target_absent_control": {
                    "goal_label": frozen_scene["target_absent_control"]["goal_label"],
                    "target_present_truth": "UNKNOWN",
                    "adjudication_source": None,
                },
                "scene_disposition": "NOT_EVALUABLE",
            }
        )
    return {
        "schema": "blindassist_l10_abotn_authority_addendum_v1",
        "cohort_canonical_json_sha256": cohort_sha256,
        "scenes": scenes,
        "claim_boundary": (
            "UNKNOWN remains NOT_EVALUABLE. VISUAL_HANDOFF_READY does not establish user completion, "
            "safe access, navigation success, deployment performance, or safety."
        ),
    }


def evaluate(cohort: dict[str, Any], addendum: dict[str, Any], cohort_sha256: str) -> dict[str, Any]:
    global_issues: list[str] = []
    if addendum.get("schema") != "blindassist_l10_abotn_authority_addendum_v1":
        global_issues.append("schema_mismatch")
    if addendum.get("cohort_canonical_json_sha256") != cohort_sha256:
        global_issues.append("cohort_canonical_json_sha256_mismatch")

    supplied = {scene.get("scene_id"): scene for scene in addendum.get("scenes", [])}
    expected_ids = [scene["scene_id"] for scene in cohort["cohort"]]
    if len(supplied) != len(addendum.get("scenes", [])):
        global_issues.append("duplicate_scene_id")
    if set(supplied) != set(expected_ids):
        global_issues.append("scene_roster_mismatch")

    scene_results = []
    for frozen_scene in cohort["cohort"]:
        scene_id = frozen_scene["scene_id"]
        scene = supplied.get(scene_id, {})
        issues: list[str] = []
        expected_paths = [row["annotation_path"] for row in frozen_scene["frozen_pair"]]
        targets = scene.get("frozen_targets", [])
        if [row.get("annotation_path") for row in targets] != expected_paths:
            issues.append("frozen_target_roster_or_order_mismatch")
        else:
            for index, target in enumerate(targets):
                for field in REQUIRED_IDENTIFIERS:
                    if not target.get(field):
                        issues.append(f"target_{index}:{field}_missing")
                for field in REQUIRED_TRUE:
                    if target.get(field) != "TRUE":
                        issues.append(f"target_{index}:{field}_not_true")

        control = scene.get("target_absent_control", {})
        expected_control = frozen_scene["target_absent_control"]["goal_label"]
        if control.get("goal_label") != expected_control:
            issues.append("target_absent_control_mismatch")
        if control.get("target_present_truth") != "FALSE":
            issues.append("target_absent_truth_not_false")
        if not control.get("adjudication_source"):
            issues.append("target_absent_adjudication_source_missing")

        disposition = "ADMITTED" if not issues else "NOT_EVALUABLE"
        if scene.get("scene_disposition") != disposition:
            issues.append("declared_scene_disposition_mismatch")
            disposition = "NOT_EVALUABLE"
        scene_results.append({"scene_id": scene_id, "disposition": disposition, "issues": issues})

    admitted = not global_issues and all(row["disposition"] == "ADMITTED" for row in scene_results)
    return {
        "schema": "blindassist_l10_abotn_authority_admission_result_v1",
        "admitted": admitted,
        "verdict": "AUTHORITY_COHORT_ADMITTED" if admitted else "AUTHORITY_COHORT_NOT_ADMITTED",
        "global_issues": global_issues,
        "scenes": scene_results,
        "claim_boundary": (
            "Admission establishes annotation-contract completeness only; it is not an algorithm, "
            "action-utility, recovery, handoff-usefulness, deployment, or safety result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--addendum", type=Path)
    parser.add_argument("--write-template", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cohort_sha256 = sha256_document(args.cohort)
    if args.write_template:
        payload = template_for(cohort, cohort_sha256)
    elif args.addendum:
        addendum = json.loads(args.addendum.read_text(encoding="utf-8"))
        payload = evaluate(cohort, addendum, cohort_sha256)
    else:
        parser.error("provide --write-template or --addendum")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    destination = args.write_template or args.output
    if destination:
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

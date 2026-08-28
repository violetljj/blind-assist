from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from functional_part_binding import FunctionalPartCandidate, TaskRelationalFunctionalSelector
from scenefun3d_functional_handoff_ceiling import _load_json
from scenefun3d_functional_set_integrity import (
    PARENT_CONTAINMENT_INPUT_LABEL,
    _build_proposals,
    _score,
    _sha256,
    _source_paths,
    _verify_source,
    apply_integrity,
)


def infer_task_action_witness(
    task_description: str, semantic_mode: str = "EXACT_ONLY"
) -> tuple[tuple[str, ...], str]:
    text = " ".join(task_description.casefold().replace("'", " ").split())
    if "unplug" in text:
        return ("unplug",), "EXACT"
    if "plug" in text or "connect" in text:
        return ("plug_in",), "EXACT"
    if "remote" in text:
        return ("key_press",), "EXACT"
    if any(
        cue in text
        for cue in (
            "intensity",
            "dial",
            "temperature",
            " heat",
            "lock ",
            " tap",
            "program",
            "timer",
            "setting",
        )
    ):
        return ("key_press", "rotate"), "EXACT"
    if (
        ("turn on" in text and ("lamp" in text or "light" in text))
        or "switch" in text
    ):
        return ("tip_push",), "EXACT"
    if "fan" in text and ("start" in text or "oscillation" in text):
        return ("key_press", "tip_push"), "EXACT"
    if semantic_mode in {
        "EXACT_THEN_ACTION_FAMILY",
        "EXACT_THEN_REDUNDANCY_GATED_ACTION_FAMILY",
    }:
        if "open" in text or "close" in text:
            return (
                ("foot_push", "hook_pull", "hook_turn", "pinch_pull"),
                "FAMILY",
            )
        if any(cue in text for cue in ("turn on", "start", "activate", "control")):
            return ("key_press", "rotate", "tip_push"), "FAMILY"
    return (), "NONE"


def infer_task_action_labels(
    task_description: str, semantic_mode: str = "EXACT_ONLY"
) -> tuple[str, ...]:
    return infer_task_action_witness(task_description, semantic_mode)[0]


def _annotation_labels(path: Path) -> dict[str, str]:
    return {
        row["annot_id"]: row["label"]
        for row in _load_json(path)["annotations"]
        if row["label"] != "exclude"
    }


def _select_with_integrity(
    selector: TaskRelationalFunctionalSelector,
    task_description: str,
    parent: Any,
    candidates: dict[str, Any],
    *,
    link_radius: float,
    minimum_component_size: int,
) -> tuple[Any, Any]:
    relational = selector.select(
        task_description,
        parent.binding_id,
        [
            FunctionalPartCandidate(
                proposal.candidate_id,
                parent.binding_id,
                tuple(float(value) for value in proposal.center),
            )
            for proposal in candidates.values()
        ],
    )
    integrity = apply_integrity(
        relational,
        parent,
        candidates,
        link_radius=link_radius,
        minimum_component_size=minimum_component_size,
    )
    return relational, integrity


def build_provider(
    data_root: Path, protocol: dict[str, Any], source_protocol: dict[str, Any]
) -> dict[str, Any]:
    selector = TaskRelationalFunctionalSelector()
    algorithm = protocol["frozen_algorithm"]
    semantic_mode = str(algorithm.get("semantic_mode", "EXACT_ONLY"))
    link_radius = float(algorithm["normalized_component_link_radius"])
    minimum_component_size = int(algorithm["minimum_dominant_component_size"])
    scenes: list[dict[str, Any]] = []
    for cohort in source_protocol["cohort"]:
        visit_id = cohort["visit_id"]
        video_id = cohort["video_id"]
        paths = _source_paths(data_root, visit_id, video_id)
        _verify_source(paths, cohort["sha256"])
        proposals, unmatched = _build_proposals(paths)
        labels = _annotation_labels(paths["annotations"])
        tasks: list[dict[str, Any]] = []
        not_evaluable: list[dict[str, Any]] = []
        for description in _load_json(paths["descriptions"])["descriptions"]:
            target_proposals = [
                proposals[target_id]
                for target_id in description["annot_id"]
                if target_id in proposals
            ]
            if len(target_proposals) != len(description["annot_id"]):
                not_evaluable.append(
                    {"desc_id": description["desc_id"], "reason": "NOT_EVALUABLE_PARENT_BINDING"}
                )
                continue
            parent_ids = {proposal.parent.binding_id for proposal in target_proposals}
            if len(parent_ids) != 1:
                not_evaluable.append(
                    {"desc_id": description["desc_id"], "reason": "NOT_EVALUABLE_MULTI_PARENT_TASK"}
                )
                continue
            parent = target_proposals[0].parent
            parent_candidates = {
                candidate_id: proposal
                for candidate_id, proposal in proposals.items()
                if proposal.parent.binding_id == parent.binding_id
            }
            baseline_relational, baseline = _select_with_integrity(
                selector,
                description["description"],
                parent,
                parent_candidates,
                link_radius=link_radius,
                minimum_component_size=minimum_component_size,
            )

            requested_labels, semantic_witness = infer_task_action_witness(
                description["description"], semantic_mode
            )
            compatible_candidates = {
                candidate_id: proposal
                for candidate_id, proposal in parent_candidates.items()
                if labels[candidate_id] in requested_labels
            }
            redundancy_gate = (
                semantic_mode != "EXACT_THEN_REDUNDANCY_GATED_ACTION_FAMILY"
                or semantic_witness != "FAMILY"
                or len(compatible_candidates) >= 2
            )
            semantic_admitted = bool(
                requested_labels and compatible_candidates and redundancy_gate
            )
            semantic_candidates = (
                compatible_candidates if semantic_admitted else parent_candidates
            )
            successor_relational, successor = _select_with_integrity(
                selector,
                description["description"],
                parent,
                semantic_candidates,
                link_radius=link_radius,
                minimum_component_size=minimum_component_size,
            )
            tasks.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "parent_binding_id": parent.binding_id,
                    "parent_binding_authority": PARENT_CONTAINMENT_INPUT_LABEL,
                    "candidate_labels": {
                        candidate_id: labels[candidate_id]
                        for candidate_id in sorted(parent_candidates)
                    },
                    "requested_action_labels": list(requested_labels),
                    "semantic_witness": semantic_witness,
                    "compatible_candidate_count": len(compatible_candidates),
                    "redundancy_gate_passed": redundancy_gate,
                    "semantic_admitted": semantic_admitted,
                    "semantic_no_match_policy_used": bool(
                        requested_labels and not semantic_admitted
                    ),
                    "baseline": {
                        "relational_state": baseline_relational.state.value,
                        "selected_candidate_ids": list(baseline.selected_candidate_ids),
                        "quarantined_candidate_ids": list(baseline.quarantined_candidate_ids),
                        "component_sizes": list(baseline.component_sizes),
                    },
                    "successor": {
                        "relational_state": successor_relational.state.value,
                        "selected_candidate_ids": list(successor.selected_candidate_ids),
                        "quarantined_candidate_ids": list(successor.quarantined_candidate_ids),
                        "component_sizes": list(successor.component_sizes),
                    },
                }
            )
        scenes.append(
            {
                "visit_id": visit_id,
                "video_id": video_id,
                "source_sha256": cohort["sha256"],
                "functional_annotations_parent_bound": len(proposals),
                "functional_annotations_parent_unmatched": unmatched,
                "tasks": tasks,
                "not_evaluable": not_evaluable,
            }
        )
    return {
        "schema_version": 1,
        "provider": "L10-SC24-TASK-SEMANTIC-TOPOLOGY-INTEGRITY-PROVIDER",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_protocol_sha256": source_protocol["protocol_sha256"],
        "truth_isolation": (
            "Target IDs are used only to derive the authorized exact parent binding. "
            "The successor receives task text, opaque parent identity, proposal geometry, "
            "and privileged functional action labels, but never evaluator target membership."
        ),
        "frozen_algorithm": protocol["frozen_algorithm"],
        "scenes": scenes,
    }


def evaluate_provider(
    data_root: Path,
    protocol: dict[str, Any],
    provider: dict[str, Any],
    provider_sha256: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cross_parent_violations = 0
    for scene in provider["scenes"]:
        paths = _source_paths(data_root, scene["visit_id"], scene["video_id"])
        descriptions = {
            row["desc_id"]: row
            for row in _load_json(paths["descriptions"])["descriptions"]
        }
        for task in scene["tasks"]:
            target_set = set(descriptions[task["desc_id"]]["annot_id"])
            candidate_set = set(task["candidate_labels"])
            baseline_selected = task["baseline"]["selected_candidate_ids"]
            successor_selected = task["successor"]["selected_candidate_ids"]
            if not set(successor_selected).issubset(candidate_set):
                cross_parent_violations += 1
            rows.append(
                {
                    "visit_id": scene["visit_id"],
                    "desc_id": task["desc_id"],
                    "description": task["description"],
                    "requested_action_labels": task["requested_action_labels"],
                    "semantic_admitted": task["semantic_admitted"],
                    "semantic_no_match_policy_used": task["semantic_no_match_policy_used"],
                    "baseline": _score(baseline_selected, target_set),
                    "successor": _score(successor_selected, target_set),
                }
            )

    def aggregate(arm: str) -> dict[str, Any]:
        legal = sum(row[arm]["legal_commit"] for row in rows)
        wrong = sum(row[arm]["wrong_part_count"] for row in rows)
        recall = float(np.mean([row[arm]["target_set_recall"] for row in rows])) if rows else 0.0
        return {
            "legal_commit_count": legal,
            "legal_commit_rate": legal / len(rows) if rows else 0.0,
            "mean_target_set_recall": recall,
            "wrong_part_count": wrong,
        }

    baseline = aggregate("baseline")
    successor = aggregate("successor")
    semantic_admissions = sum(row["semantic_admitted"] for row in rows)
    changed_tasks = sum(row["baseline"] != row["successor"] for row in rows)
    gate = protocol["frozen_gate"]
    decision_labels = protocol.get(
        "decision_labels",
        {
            "insufficient_tasks": "SC24_NOT_EVALUABLE_INSUFFICIENT_PARENT_BOUND_TASKS",
            "insufficient_semantic_admissions": "SC24_NOT_EVALUABLE_INSUFFICIENT_SEMANTIC_ADMISSIONS",
            "pass": "SC24_SEMANTIC_TOPOLOGY_INTEGRITY_CONSUMED_MECHANICS_SIGNAL",
            "fail": "SC24_SEMANTIC_TOPOLOGY_INTEGRITY_GATE_NOT_MET",
        },
    )
    wrong_reduction = baseline["wrong_part_count"] - successor["wrong_part_count"]
    if len(rows) < int(gate["minimum_evaluable_tasks"]):
        decision = decision_labels["insufficient_tasks"]
    elif semantic_admissions < int(gate["minimum_semantic_admissions"]):
        decision = decision_labels["insufficient_semantic_admissions"]
    elif (
        wrong_reduction >= int(gate["minimum_wrong_part_reduction"])
        and successor["legal_commit_count"] >= baseline["legal_commit_count"]
        and successor["mean_target_set_recall"] >= baseline["mean_target_set_recall"]
        and cross_parent_violations <= int(gate["maximum_cross_parent_violations"])
    ):
        decision = decision_labels["pass"]
    else:
        decision = decision_labels["fail"]
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "claim_layer": protocol.get(
            "claim_layer", "CONSUMED_PRIVILEGED_SEMANTIC_PROPOSAL_MECHANICS"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "provider_sha256": provider_sha256,
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "scenes": len(provider["scenes"]),
            "tasks_evaluable": len(rows),
            "tasks_not_evaluable": sum(
                len(scene["not_evaluable"]) for scene in provider["scenes"]
            ),
            "semantic_admissions": semantic_admissions,
            "changed_tasks": changed_tasks,
        },
        "baseline": baseline,
        "successor": successor,
        "wrong_part_reduction": wrong_reduction,
        "cross_parent_violations": cross_parent_violations,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = _load_json(args.protocol)
    protocol["protocol_sha256"] = _sha256(args.protocol)
    source_protocol = _load_json(args.source_protocol)
    source_protocol["protocol_sha256"] = _sha256(args.source_protocol)
    expected_source_hash = protocol["source"]["cohort_protocol_sha256"]
    if source_protocol["protocol_sha256"] != expected_source_hash:
        raise ValueError("Source cohort protocol hash mismatch")

    provider = build_provider(
        args.data_root.resolve(), protocol, source_protocol
    )
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(
        json.dumps(provider, indent=2) + "\n", encoding="utf-8"
    )
    provider_sha256 = _sha256(args.provider_output)
    result = evaluate_provider(
        args.data_root.resolve(), protocol, provider, provider_sha256
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "baseline": result["baseline"],
                "successor": result["successor"],
                "wrong_part_reduction": result["wrong_part_reduction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

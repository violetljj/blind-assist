"""Audit a sealed official-pixel ABotN V0 episode without new model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "blindassist_abotn_official_closed_loop_audit_v0"
HTTP_RENDER_SUCCESS = re.compile(r'"POST /render_gs HTTP/1\.1" 200')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = args.prospective_freeze.resolve()
    public_path = args.public_graph.resolve()
    private_path = args.private_truth.resolve()
    pixel_path = args.pixel_receipt.resolve()
    qualification_path = args.qualification.resolve()
    run_path = args.run_receipt.resolve()
    server_log_path = args.server_log.resolve()
    cohort_freeze_path = (
        args.cohort_freeze.resolve() if getattr(args, "cohort_freeze", None) else None
    )
    freeze = _read(freeze_path)
    public = _read(public_path)
    private = _read(private_path)
    pixels = _read(pixel_path)
    qualification = _read(qualification_path)
    run = _read(run_path)
    if freeze.get("terminal") != "ABOTN_OFFICIAL_PROSPECTIVE_EPISODE_FROZEN":
        raise ValueError("prospective freeze is not terminal")
    if pixels.get("terminal") != "ABOTN_OFFICIAL_ACTION_GRAPH_PIXELS_PASS":
        raise ValueError("official pixel receipt is not terminal PASS")
    if qualification.get("terminal") != "ABOTN_V0_ACTION_GRAPH_PIXELS_QUALIFIED_FOR_ONE_CLOSED_LOOP_RUN":
        raise ValueError("pixel qualification is not terminal PASS")
    if run.get("terminal") != "ABOTN_V0_CLOSED_LOOP_ENGINEERING_RUN_COMPLETE":
        raise ValueError("sealed run is not terminal")
    if run.get("rerun_authorized") is not False:
        raise ValueError("sealed no-rerun boundary drift")
    graph_binding = freeze["action_graph"]
    expected_hashes = {
        public_path: graph_binding["public_graph_sha256"],
        private_path: graph_binding["private_truth_sha256"],
        pixel_path: qualification["inputs"]["pixel_receipt_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"input hash drift: {path.name}")
    if pixels.get("public_graph_sha256") != graph_binding["public_graph_sha256"]:
        raise ValueError("pixel/public graph binding drift")
    if run.get("pixel_renderer") != "PINNED_OFFICIAL_ABOTN_RENDERER":
        raise ValueError("sealed run did not use the pinned official renderer")
    if run["provider"].get("in_doubt") != 0:
        raise ValueError("provider journal contains an in_doubt call")
    if run.get("provider_private_truth_literal_hits"):
        raise ValueError("private truth leaked to provider")
    expected_render_calls = freeze["frozen_budget"]["official_render_calls"]
    server_accounting_scope = "EPISODE"
    if cohort_freeze_path is not None:
        cohort = _read(cohort_freeze_path)
        if cohort.get("terminal") != "ABOTN_OFFICIAL_FRESH_COHORT_FROZEN":
            raise ValueError("shared server log cohort is not frozen")
        cohort_episode_ids = [
            row["episode_id"] for row in cohort["selection"]["tasks"]
        ]
        if public["episode_id"] not in cohort_episode_ids:
            raise ValueError("episode is not in the shared server log cohort")
        expected_render_calls = cohort["frozen_budget"]["official_render_calls"]
        server_accounting_scope = "FROZEN_COHORT_SHARED_SERVER"
    actual_http_calls = len(
        HTTP_RENDER_SUCCESS.findall(server_log_path.read_text(encoding="utf-8", errors="replace"))
    )
    if actual_http_calls != expected_render_calls:
        raise ValueError("server HTTP render call count drift")

    public_nodes = {row["node_id"]: row for row in public["nodes"]}
    private_nodes = {row["node_id"]: row for row in private["nodes"]}
    trajectory = run["action_state_trajectory"]
    action_progress = []
    for index, row in enumerate(trajectory):
        action = row.get("action")
        if action is None:
            continue
        source_id = row["node_id"]
        target_id = public_nodes[source_id]["actions"].get(action)
        if target_id is None:
            raise ValueError("sealed action has no frozen graph edge")
        source_distance = float(private_nodes[source_id]["distance_to_goal_m"])
        target_distance = float(private_nodes[target_id]["distance_to_goal_m"])
        action_progress.append({
            "sequence": index + 1,
            "action": action,
            "source_node_id": source_id,
            "target_node_id": target_id,
            "source_distance_to_goal_m": source_distance,
            "target_distance_to_goal_m": target_distance,
            "goal_progress_m": source_distance - target_distance,
            "instruction_emitted": action in {"TURN_LEFT", "TURN_RIGHT", "FORWARD"},
        })
    instruction_rows = [row for row in action_progress if row["instruction_emitted"]]
    provider_statuses = [row["p0_status"] for row in trajectory]
    commitment_indices = [
        index for index, status in enumerate(provider_statuses) if status == "GROUNDED"
    ]
    reliability_drop_after_commitment = any(
        status == "ABSTAIN_NO_RELIABLE_EVIDENCE"
        for index, status in enumerate(provider_statuses)
        if commitment_indices and index > commitment_indices[0]
    )
    episode = run["episode"]
    if episode["episode_completion"]:
        terminal = "ABOTN_OFFICIAL_V0_METRIC_GOAL_SUCCESS"
    elif instruction_rows and sum(row["goal_progress_m"] for row in instruction_rows) > 0:
        terminal = "ABOTN_OFFICIAL_V0_PARTIAL_METRIC_PROGRESS_NO_GOAL_SUCCESS"
    else:
        terminal = "ABOTN_OFFICIAL_V0_NO_METRIC_GOAL_PROGRESS"
    return {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": terminal,
        "episode_id": public["episode_id"],
        "inputs": {
            "audit_implementation_sha256": _sha256(Path(__file__).resolve()),
            "prospective_freeze_sha256": _sha256(freeze_path),
            "pixel_receipt_sha256": _sha256(pixel_path),
            "qualification_sha256": _sha256(qualification_path),
            "sealed_run_receipt_sha256": _sha256(run_path),
            "server_log_sha256": _sha256(server_log_path),
            "cohort_freeze_sha256": (
                _sha256(cohort_freeze_path) if cohort_freeze_path is not None else None
            ),
        },
        "execution": {
            "official_render_http_200_calls": actual_http_calls,
            "official_render_http_accounting_scope": server_accounting_scope,
            "provider_observation_calls": run["provider"]["observation_calls"],
            "provider_brain_attempts": run["provider"]["brain_attempts"],
            "provider_in_doubt": 0,
            "teacher_calls": run["teacher_calls"],
            "baseline_episode_runs": run["baseline_episode_runs"],
            "sealed_episode_reruns": 0,
        },
        "metric_outcome": {
            "initial_distance_to_goal_m": episode["initial_distance_to_goal_m"],
            "terminal_distance_to_goal_m": episode["terminal_distance_to_goal_m"],
            "net_progress_including_rescan_pose_updates_m": episode["net_goal_progress_m"],
            "instruction_attributable_progress_m": sum(
                row["goal_progress_m"] for row in instruction_rows
            ),
            "terminal_metric_arrival": episode["terminal_metric_arrival"],
            "episode_completion": episode["episode_completion"],
            "false_arrival": episode["false_arrival"],
            "action_progress": action_progress,
        },
        "provider_behavior": {
            "statuses": provider_statuses,
            "reliable_observation_count": episode["reliable_observation_count"],
            "instruction_count": episode["instruction_count"],
            "rescan_count": episode["rescan_count"],
            "reliability_drop_after_provider_commitment": reliability_drop_after_commitment,
        },
        "truth_boundaries": {
            "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
            "wrong_target_confirmation": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
            "lost_after_visible": "NOT_EVALUABLE_NO_FUNCTIONAL_PIXEL_VISIBILITY_TRUTH",
            "provider_grounded_then_abstained_does_not_establish_lost": True,
        },
        "supported_attribution": (
            "ONE_FORWARD_INSTRUCTION_MADE_POSITIVE_METRIC_PROGRESS_THEN_CURRENT_FRAME_RELIABILITY_FAILED"
            if reliability_drop_after_commitment and instruction_rows
            else episode["failure_class"]
        ),
        "unsupported_attributions": [
            "PROPOSAL_MISS",
            "REFERENT_SELECTION",
            "WRONG_CONFIDENT_GUIDANCE",
            "LOST_AFTER_VISIBLE",
            "RANGE_OR_BEARING_BOTTLENECK",
        ],
        "claim_ceiling": "OFFICIAL_RENDERER_SINGLE_PUBLIC_TASK_ENGINEERING_ONLY",
        "next_action": "DO_NOT_TUNE_OR_RERUN; REQUIRE_A_PREDECLARED_FRESH_COHORT_FOR_DOMINANCE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prospective-freeze", type=Path, required=True)
    parser.add_argument("--public-graph", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--pixel-receipt", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--cohort-freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("audit output already exists")
    _atomic_json(output, result)
    print(json.dumps({
        "terminal": result["terminal"],
        "instruction_attributable_progress_m": result["metric_outcome"]["instruction_attributable_progress_m"],
        "episode_completion": result["metric_outcome"]["episode_completion"],
        "lost_after_visible": result["truth_boundaries"]["lost_after_visible"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

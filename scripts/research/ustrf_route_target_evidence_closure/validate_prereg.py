from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import ContractError, load_json, validate_prereg


MATERIALIZATION_STATE = Path(
    "artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/"
    "crowdbot-holdout-materialization-state-r1.json"
)


def holdout_progress(config: dict, *, repo: Path) -> dict:
    holdout = config["sealed_holdout"]
    result = {
        "holdout_state": holdout["state"],
        "capacity_qualified_source_count": holdout["content_qualification_receipt"][
            "capacity_qualified_source_count"
        ],
        "admitted_source_count": holdout["content_qualification_receipt"]["admitted_source_count"],
        "rgbd_modality_probe_complete": holdout["modality_probe_receipt"]["rgbd_content_decoded"],
        "materialization_status": "not_started",
        "materialized_sequence_count": 0,
        "materialization_sequence_total": 16,
    }
    state_path = repo / MATERIALIZATION_STATE
    if not state_path.is_file():
        return result
    state = load_json(state_path)
    if state.get("candidate_outputs_executed") is not False:
        raise ContractError("holdout materialization state exposed candidate outputs")
    completed = state.get("sequence_completed")
    total = state.get("sequence_total")
    if not isinstance(completed, int) or not isinstance(total, int) or not (0 <= completed <= total):
        raise ContractError("holdout materialization sequence counts are invalid")
    result.update({
        "materialization_status": state.get("status", "unknown"),
        "materialized_sequence_count": completed,
        "materialization_sequence_total": total,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        config = validate_prereg(load_json(args.config), repo=args.repo.resolve())
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, ensure_ascii=False))
        return 1
    try:
        progress = holdout_progress(config, repo=args.repo.resolve())
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, ensure_ascii=False))
        return 1
    status = (
        "PREREGISTRATION_VALID_HOLDOUT_MATERIALIZATION_IN_PROGRESS"
        if progress["materialization_status"] == "running"
        else "PREREGISTRATION_VALID_INPUTS_NOT_ADMITTED"
    )
    print(json.dumps({
        "status": status,
        "oracle_arm_count": len(config["oracle_arms"]),
        "candidate_count": len(config["candidate_roster"]),
        **progress,
        "android_shadow": "closed",
        "h2": "closed",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

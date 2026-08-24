"""Run SAGE-LM V1-D active-parallax boundary-field B1 on the frozen R2 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .active_parallax_observation import ActiveParallaxBoundaryProvider, RaftSmallFlowExtractor
from .experiment import _aggregate
from .rgb_experiment import _baseline, _sage_lm, _v1_criteria
from .two_view_experiment import _arm_diagnostics, _evaluator_episode, _source_poses


SCHEMA_VERSION = "sage_lm_v1d_active_parallax_boundary_field"
ARM = "v1d"


def _available_episode_ids(report: dict, arm: str) -> set[str]:
    available = set()
    for row in report["rows"]:
        diagnostics = row[arm]["diagnostics"]
        distances = diagnostics.get("oracle_association_distances_px", [])
        if len(distances) == 4 and max(distances) <= 9.0:
            available.add(row["episode_id"])
    return available


def _direct_recall_episode_ids(report: dict, arm: str) -> set[str]:
    return {
        row["episode_id"]
        for row in report["rows"]
        if all(row[arm]["diagnostics"].get("direct_four_boundary_hits", []))
    }


def run(cohort_path: Path, r3_report_path: Path, extractor: RaftSmallFlowExtractor) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-D requires the frozen 24-episode R2 cohort")
    r3_report = json.loads(r3_report_path.read_text(encoding="utf-8"))
    r3_available = _available_episode_ids(r3_report, "b1")
    rows = []
    controls_retained = 0
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        baseline = _baseline(evaluator)
        provider = ActiveParallaxBoundaryProvider(episode_input, truth, pose_a, pose_b, extractor)
        result = _sage_lm(evaluator, provider)
        if materialized["control"] and result["true_arrival"]:
            controls_retained += 1
        rows.append(
            {
                "episode_id": evaluator.episode_id,
                "kind": evaluator.kind,
                "control": materialized["control"],
                "source": materialized["source"],
                "truth": materialized["truth"],
                "source_pose_audit": pose_audit,
                "baseline": baseline,
                ARM: result,
            }
        )

    baseline_metrics = _aggregate(row["baseline"] for row in rows)
    arm_metrics = _aggregate(row[ARM] for row in rows)
    diagnostics = _arm_diagnostics(rows, ARM)
    v1d_available = _available_episode_ids({"rows": rows}, ARM)
    direct_recall = _direct_recall_episode_ids({"rows": rows}, ARM)
    missing = sorted(set(row["episode_id"] for row in rows) - v1d_available)
    r3_missing = sorted(set(row["episode_id"] for row in rows) - r3_available)
    rescued = sorted(v1d_available - r3_available)
    lost = sorted(r3_available - v1d_available)
    diagnostics.update(
        {
            "four_boundary_recall_at_8_count": len(direct_recall),
            "true_boundary_pair_available_count": len(v1d_available),
            "boundary_candidate_missing_count": len(missing),
            "boundary_candidate_missing_episode_ids": missing,
            "r3_true_boundary_pair_available_count": len(r3_available),
            "r3_missing_episode_ids": r3_missing,
            "r3_missing_rescued_count": len(rescued),
            "r3_missing_rescued_episode_ids": rescued,
            "r3_available_lost_count": len(lost),
            "r3_available_lost_episode_ids": lost,
            "net_true_pair_delta_vs_r3": len(v1d_available) - len(r3_available),
        }
    )
    criteria = _v1_criteria(baseline_metrics, arm_metrics, controls_retained)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_DEVELOPMENT_STANDARD",
        "experiment_label": "V1_D_ACTIVE_PARALLAX_BOUNDARY_FIELD_B1",
        "question": "Can active lateral motion make weak aperture boundaries observable as pose-compensated parallax discontinuities?",
        "cohort": {
            "path": str(cohort_path.resolve()),
            "episode_count": len(rows),
            "kinds": cohort["kinds"],
            "control_count": 6,
            "source_pose_pair_gate_pass_count": sum(row["source_pose_audit"]["intended_active_pair_gate_pass"] for row in rows),
        },
        "model": extractor.identity,
        "frozen_surfaces": {
            "cohort_anchor_source_pose": "UNCHANGED_R2",
            "top_k_candidates_per_side": 8,
            "oracle_association_localization_gate_px": 9.0,
            "triangulation": "UNCHANGED_R2",
            "confidence": "UNCHANGED_R2",
            "arrival": "UNCHANGED_R2",
            "policy": "UNCHANGED_R2",
            "training": "NOT_RUN",
            "r6": "NOT_RUN",
            "b2": "NOT_RUN",
        },
        "metrics": {"bbox_center_scale": baseline_metrics, ARM: arm_metrics, "controls_retained": controls_retained},
        "observation_diagnostics": diagnostics,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "decision": {
            "standalone_exceeds_r3": len(v1d_available) > len(r3_available),
            "has_complementary_r3_missing_signal": len(rescued) > 0,
            "next_action": (
                "PROMOTE_V1_D"
                if len(v1d_available) > len(r3_available)
                else "TEST_SIMPLE_R3_PARALLAX_UNION"
                if rescued
                else "CLOSE_PARALLAX_AND_ADVANCE_V1_E_PRIVILEGED_GEOMETRY"
            ),
        },
        "rows": rows,
        "claim_ceiling": "CURATED_CONSUMED_R2_DEVELOPMENT_COHORT_ACTIVE_PARALLAX_MECHANISM_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--r3-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    extractor = RaftSmallFlowExtractor()
    report = run(args.cohort, args.r3_report, extractor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "metrics", "observation_diagnostics", "decision")}, indent=2))


if __name__ == "__main__":
    main()

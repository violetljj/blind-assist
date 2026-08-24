"""Run SAGE-LM V1-B-R4 pose-conditioned multi-view boundary accumulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dense_boundary_observation import DeepLsdDenseFieldExtractor
from .experiment import _aggregate
from .pose_accumulation_observation import PoseAccumulatedOracleBoundaryProvider
from .rgb_experiment import _baseline, _sage_lm, _v1_criteria
from .two_view_experiment import _arm_diagnostics, _evaluator_episode, _source_poses


SCHEMA_VERSION = "sage_lm_v1b_r4_pose_conditioned_multi_view_boundary_accumulation"


def run(cohort_path: Path, extractor: DeepLsdDenseFieldExtractor) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-B-R4 requires the frozen 24-episode R2 cohort")
    rows = []
    controls_retained = 0
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        provider = PoseAccumulatedOracleBoundaryProvider(episode_input, truth, pose_a, pose_b, extractor)
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
                "baseline": _baseline(evaluator),
                "b1": result,
            }
        )
    baseline_metrics = _aggregate(row["baseline"] for row in rows)
    arm_metrics = _aggregate(row["b1"] for row in rows)
    criteria = _v1_criteria(baseline_metrics, arm_metrics, controls_retained)
    diagnostics = _arm_diagnostics(rows, "b1")
    candidate_available = 0
    candidate_missing = 0
    for row in rows:
        provider_diagnostics = row["b1"]["diagnostics"]
        distances = provider_diagnostics.get("oracle_association_distances_px", [])
        if len(distances) == 4 and max(distances) <= 9.0:
            candidate_available += 1
        if provider_diagnostics.get("failure") == "BOUNDARY_HYPOTHESIS_MISSING":
            candidate_missing += 1
    diagnostics.update(
        {
            "true_boundary_pair_available_count": candidate_available,
            "boundary_hypothesis_missing_count": candidate_missing,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_DEVELOPMENT_STANDARD",
        "experiment_label": "V1_B_R4_POSE_CONDITIONED_MULTI_VIEW_BOUNDARY_ACCUMULATION_B1",
        "cohort": {
            "path": str(cohort_path.resolve()),
            "episode_count": len(rows),
            "kinds": cohort["kinds"],
            "control_count": 6,
            "source_pose_pair_gate_pass_count": sum(
                row["source_pose_audit"]["intended_active_pair_gate_pass"] for row in rows
            ),
        },
        "model": extractor.identity,
        "frozen_surfaces": {
            "source_pose": "UNCHANGED_R2",
            "interpretation_plane_geometry": "UNCHANGED_R2",
            "oracle_association_localization_gate_px": 9.0,
            "confidence": "UNCHANGED_R2",
            "arrival": "UNCHANGED_R2",
            "policy": "UNCHANGED_R2",
            "optical_flow": "NOT_RUN",
            "monocular_metric_depth": "NOT_RUN",
        },
        "metrics": {"bbox_center_scale": baseline_metrics, "b1": arm_metrics, "controls_retained": controls_retained},
        "observation_diagnostics": diagnostics,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "r4_targets": {
            "true_boundary_pair_available_at_least_18": candidate_available >= 18,
            "geometry_output_at_least_18": diagnostics["geometry_output_count"] >= 18,
            "confident_geometry_at_least_12": diagnostics["geometry_confidence_pass_count"] >= 12,
            "boundary_hypothesis_missing_at_most_6": candidate_missing <= 6,
        },
        "rows": rows,
        "claim_ceiling": "CURATED_R2_DEVELOPMENT_COHORT_POSE_CONDITIONED_BOUNDARY_ACCUMULATION_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--deeplsd-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    extractor = DeepLsdDenseFieldExtractor(args.deeplsd_root, args.runtime_root, args.checkpoint)
    report = run(args.cohort, extractor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": report["metrics"],
                "observation_diagnostics": report["observation_diagnostics"],
                "r4_targets": report["r4_targets"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

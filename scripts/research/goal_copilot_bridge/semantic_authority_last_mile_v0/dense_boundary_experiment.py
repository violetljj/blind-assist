"""Run SAGE-LM V1-B-R3 DeepLSD dense-field B1 on the frozen R2 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dense_boundary_observation import DeepLsdDenseFieldExtractor, DenseFieldBoundaryProvider
from .experiment import _aggregate
from .rgb_experiment import _baseline, _sage_lm, _v1_criteria
from .two_view_experiment import _arm_diagnostics, _evaluator_episode, _source_poses


SCHEMA_VERSION = "sage_lm_v1b_r3_deeplsd_dense_boundary_field"


def run(cohort_path: Path, extractor: DeepLsdDenseFieldExtractor, arm: str = "b1") -> dict:
    if arm not in {"b1", "b2"}:
        raise ValueError(f"unsupported R3 arm: {arm}")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-B-R3 requires the frozen 24-episode R2 cohort")
    rows = []
    controls_retained = 0
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        baseline = _baseline(evaluator)
        provider = DenseFieldBoundaryProvider(
            episode_input,
            truth if arm == "b1" else None,
            pose_a,
            pose_b,
            arm,
            extractor,
        )
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
                arm: result,
            }
        )
    baseline_metrics = _aggregate(row["baseline"] for row in rows)
    arm_metrics = _aggregate(row[arm] for row in rows)
    criteria = _v1_criteria(baseline_metrics, arm_metrics, controls_retained)
    diagnostics = _arm_diagnostics(rows, arm)
    candidate_available = 0
    candidate_missing = 0
    if arm == "b1":
        for row in rows:
            provider_diagnostics = row[arm]["diagnostics"]
            distances = provider_diagnostics.get("oracle_association_distances_px", [])
            if len(distances) == 4 and max(distances) <= 9.0:
                candidate_available += 1
            if provider_diagnostics.get("failure") == "BOUNDARY_CANDIDATE_MISSING":
                candidate_missing += 1
    diagnostics.update(
        {
            "true_boundary_pair_available_count": candidate_available if arm == "b1" else None,
            "boundary_candidate_missing_count": candidate_missing if arm == "b1" else None,
        }
    )
    source_pose_gate_count = sum(row["source_pose_audit"]["intended_active_pair_gate_pass"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_DEVELOPMENT_STANDARD",
        "experiment_label": f"V1_B_R3_DEEPLSD_DENSE_BOUNDARY_FIELD_{arm.upper()}",
        "cohort": {
            "path": str(cohort_path.resolve()),
            "episode_count": len(rows),
            "kinds": cohort["kinds"],
            "control_count": 6,
            "source_pose_pair_gate_pass_count": source_pose_gate_count,
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
        "metrics": {"bbox_center_scale": baseline_metrics, arm: arm_metrics, "controls_retained": controls_retained},
        "observation_diagnostics": diagnostics,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "r3_targets": {
            "true_boundary_pair_available_at_least_18": candidate_available >= 18 if arm == "b1" else None,
            "geometry_output_at_least_18": diagnostics["geometry_output_count"] >= 18,
            "confident_geometry_at_least_12": diagnostics["geometry_confidence_pass_count"] >= 12,
            "boundary_candidate_missing_at_most_6": candidate_missing <= 6 if arm == "b1" else None,
        },
        "rows": rows,
        "claim_ceiling": "CURATED_R2_DEVELOPMENT_COHORT_DENSE_BOUNDARY_REPRESENTATION_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--deeplsd-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("b1", "b2"), default="b1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    extractor = DeepLsdDenseFieldExtractor(args.deeplsd_root, args.runtime_root, args.checkpoint)
    report = run(args.cohort, extractor, args.arm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": report["model"],
                "metrics": report["metrics"],
                "observation_diagnostics": report["observation_diagnostics"],
                "r3_targets": report["r3_targets"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

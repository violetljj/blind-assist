"""Run and score X33 on consumed C16 as iterative synthetic Development.

The public helpers in this module are also reused by later frozen CARLA
confirmation runners.  Keeping them beside those runners makes the evidence
path reproducible without a checkout-local ``artifacts.local/work`` module.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


CARLA_DIR = Path(__file__).resolve().parent
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

import dtr_carla_x25_rigid_footprint_scorer as base
import dtr_carla_x31_source_disjoint_scorer as shared
import dtr_carla_x33_dormant_transport_reactivation_predictor as x33


ARM_X24 = shared.ARM_X24
ARM_X31 = shared.ARM_X31
ARM_X33 = x33.ARM_X33
EPISODES = shared.EPISODES
CONTACT = shared.CONTACT_EPISODES
SAFE = shared.SAFE_EPISODES
SCORE_END = shared.SCORE_WINDOW_END_SECONDS
SAFE_START = shared.SAFE_SEGMENT_START_SECONDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol = base.read_json(args.protocol.resolve(strict=True))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output_dir / "predictions.json"
    summary_path = args.output_dir / "summary.json"

    _freeze, contract, candidate_values = x33.x31.x24.require_freeze(run_root)
    started = time.perf_counter()
    cursor = 0
    episodes = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = x33.predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
        print(f"predicted {episode.episode_id}", flush=True)
    predictions = {
        "schema": "blindassist-dtr-carla-x33-dormant-transport-variant-v1",
        "status": "ITERATIVE_SAME_SOURCE_DEVELOPMENT",
        "experiment_id": x33.EXPERIMENT_ID,
        "truth_blind_prediction_inputs": True,
        "arms": [ARM_X33],
        "episodes": episodes,
        "fixed_constants": x33.fixed_constants(),
        "claim_boundary": {
            "same_source_post_score_iterative_development": True,
            "confirmation": False,
            "threshold_sweep": False,
        },
    }
    base.write_json_exclusive(predictions_path, predictions)

    x24_predictions = base.read_json(run_root / "predictions-x24.json")
    evaluator_full = {
        episode_id: base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in EPISODES
    }
    predictions_full = {
        ARM_X24: {
            episode_id: shared.c7.arm_frames_full(
                x24_predictions, episode_id, ARM_X24
            )
            for episode_id in EPISODES
        },
        ARM_X33: {
            episode_id: shared.c7.arm_frames_full(
                predictions, episode_id, ARM_X33
            )
            for episode_id in EPISODES
        },
    }
    for arm, values in predictions_full.items():
        for episode_id in EPISODES:
            base.align(
                evaluator_full[episode_id], values[episode_id], f"{arm}:{episode_id}"
            )
    evaluator = {
        episode_id: shared.c7.prefix(rows, SCORE_END[episode_id])
        for episode_id, rows in evaluator_full.items()
    }
    scored = {
        arm: {
            episode_id: shared.c7.prefix(rows, SCORE_END[episode_id])
            for episode_id, rows in values.items()
        }
        for arm, values in predictions_full.items()
    }
    aggregate = {
        arm: base.confusion(evaluator, values) for arm, values in scored.items()
    }
    contacts = {
        episode_id: {
            arm: base.contact_metrics(evaluator[episode_id], values[episode_id])
            for arm, values in scored.items()
        }
        for episode_id in CONTACT
    }
    safe = {
        episode_id: {
            arm: base.false_segments(values[episode_id], SAFE_START[episode_id])
            for arm, values in scored.items()
        }
        for episode_id in SAFE
    }
    selected = shared.validate_occlusion_reports(
        protocol,
        base.read_json(source_root / "evaluator" / "physical_occlusion_report.json"),
        evaluator_full,
    )
    for episode in predictions["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X33]
    continuity, ambiguity = shared.contact_transport_continuity(predictions, selected)
    invariants = shared.authority_invariants(predictions, SCORE_END)
    summary = {
        "status": "COMPLETE_ITERATIVE_SAME_SOURCE_DEVELOPMENT",
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "transport_continuity": continuity,
        "transport_ambiguity": ambiguity,
        "authority_invariants": invariants,
        "dormant_reactivations_by_episode": {
            episode_id: episodes[episode_id]["diagnostics"][
                "dormant_transport_reactivations"
            ]
            for episode_id in EPISODES
        },
        "maximum_surface_transport_branches_by_episode": {
            episode_id: episodes[episode_id]["diagnostics"][
                "maximum_surface_transport_branches"
            ]
            for episode_id in EPISODES
        },
        "source": {
            "x31_base_predictor_sha256": base.sha256_file(
                Path(x33.x31.__file__).resolve()
            ),
            "x32_base_predictor_sha256": base.sha256_file(
                Path(x33.x32.__file__).resolve()
            ),
            "x33_predictor_sha256": base.sha256_file(Path(x33.__file__).resolve()),
            "variant_runner_sha256": base.sha256_file(Path(__file__).resolve()),
            "predictions_sha256": base.sha256_file(predictions_path),
        },
        "claim_boundary": {
            "same_source_post_score_iterative_development": True,
            "confirmation": False,
            "risk_hold_threshold_change": False,
            "association_radius_change": False,
            "score_threshold_change": False,
        },
    }
    base.write_json_exclusive(summary_path, summary)
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "contact_recall": {
                    key: contacts[key][ARM_X33]["future_positive_recall"]
                    for key in CONTACT
                },
                "safe_segments": {
                    key: safe[key][ARM_X33]["false_alert_segment_count"]
                    for key in SAFE
                },
                "continuity": {
                    key: continuity[key]["continuous_route_risk"] for key in CONTACT
                },
                "ancestry": {
                    key: continuity[key]["parent_ancestry_status"] for key in CONTACT
                },
                "reactivations": summary["dormant_reactivations_by_episode"],
                "summary_sha256": base.sha256_file(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

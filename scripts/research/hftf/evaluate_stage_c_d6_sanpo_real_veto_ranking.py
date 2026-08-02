#!/usr/bin/env python3
"""Evaluate false-alert ranking on the human-reviewed SANPO event cohort."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from export_stage_c_d6_veto_review_candidates import (
    PublicHistoryDataset,
    build_windows,
)
from summarize_stage_c_d6_early_pair_structured_field_canary import (
    FOLDS,
    SEEDS,
    summarize_values,
)
from train_stage_c_d5_tartanground_development_student import (
    binary_metrics,
    sha256,
)
from train_stage_c_d6_veto_eligibility_ranking import (
    VetoEligibilityStudent,
    compose_ranking_logits,
    load_reference,
    reference_predictions,
)


POSITIVE_BUCKETS = {
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
}
DEFAULT_MANIFEST = Path(
    "artifacts.local/evidence/riskseg-r0/event-eval/"
    "device-view-v2/manifest.json"
)
DEFAULT_CANDIDATE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d6-veto-eligibility-confidence-residual-canary-v0"
)
DEFAULT_PRETRAINED = Path(
    "artifacts.local/models/hftf/torch/hub/checkpoints/"
    "mobilenet_v3_small-047dcff4.pth"
)


def build_event_windows(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    root = manifest_path.parent
    media_rows = []
    events = {}
    for event in manifest["events"]:
        event_id = str(event["parent_event_id"])
        events[event_id] = event
        for frame_index, frame in enumerate(event["frames"]):
            media_rows.append(
                {
                    "source_session_id": event_id,
                    "camera": "device-view",
                    "view": "rgb",
                    "frame_index": frame_index,
                    "rgb_local_path": str(
                        (root / frame["image_path"]).resolve()
                    ),
                    "timestamp_ns": None,
                    "nominal_time_ns": int(frame["timestamp_ms"])
                    * 1_000_000,
                    "time_semantics": "EVENT_RELATIVE_MILLISECONDS",
                }
            )
    windows = build_windows(media_rows)
    output = []
    for window in windows:
        event = events[window["source_session_id"]]
        anchor = int(window["anchor_frame_index"])
        if event["bucket"] not in POSITIVE_BUCKETS:
            phase = "negative_event"
            target = 1.0
        else:
            alert_start, alert_end = map(
                int, event["alertable_interval_frames"]
            )
            passed_start, passed_end = map(
                int, event["passed_interval_frames"]
            )
            if alert_start <= anchor <= alert_end:
                phase = "positive_alertable"
                target = 0.0
            elif passed_start <= anchor <= passed_end:
                phase = "positive_passed"
                target = 1.0
            else:
                phase = "unscored_transition"
                target = None
        output.append(
            {
                **window,
                "parent_event_id": event["parent_event_id"],
                "source_session_id": event["source_session_id"],
                "bucket": event["bucket"],
                "phase": phase,
                "false_alert_target": target,
            }
        )
    return output


def ranking_comparison(
    probability: np.ndarray,
    comparator: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    candidate = binary_metrics(probability, target, eligible)
    baseline = binary_metrics(comparator, target, eligible)
    return {
        "candidate": candidate,
        "baseline_inverse_risk_confidence": baseline,
        "candidate_auroc_delta": (
            candidate["auroc"] - baseline["auroc"]
            if candidate["auroc"] is not None
            and baseline["auroc"] is not None
            else None
        ),
        "candidate_average_precision_delta": (
            candidate["average_precision"]
            - baseline["average_precision"]
            if candidate["average_precision"] is not None
            and baseline["average_precision"] is not None
            else None
        ),
    }


def phase_unit_rows(
    windows: list[dict[str, Any]],
    probability: np.ndarray,
    comparator: np.ndarray,
    known_probability: np.ndarray,
    eligible: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for index, window in enumerate(windows):
        if window["false_alert_target"] is None:
            continue
        mask = eligible[index]
        if not np.any(mask):
            continue
        key = (window["parent_event_id"], window["phase"])
        group = groups.setdefault(
            key,
            {
                "candidate": [],
                "comparator": [],
                "known": [],
                "horizon_counts": np.zeros(3, dtype=np.int64),
                "height_counts": np.zeros(3, dtype=np.int64),
                "direction_counts": np.zeros(6, dtype=np.int64),
                "distance_counts": np.zeros(6, dtype=np.int64),
            },
        )
        group["candidate"].extend(
            probability[index][mask].astype(float).tolist()
        )
        group["comparator"].extend(
            comparator[index][mask].astype(float).tolist()
        )
        group["known"].extend(
            known_probability[index][mask].astype(float).tolist()
        )
        coordinates = np.argwhere(mask)
        group["horizon_counts"] += np.bincount(
            coordinates[:, 0],
            minlength=3,
        )
        group["height_counts"] += np.bincount(
            coordinates[:, 1],
            minlength=3,
        )
        group["direction_counts"] += np.bincount(
            coordinates[:, 2],
            minlength=6,
        )
        group["distance_counts"] += np.bincount(
            coordinates[:, 3],
            minlength=6,
        )
    event_lookup = {
        (window["parent_event_id"], window["phase"]): window
        for window in windows
    }
    rows = []
    for key, values in sorted(groups.items()):
        window = event_lookup[key]
        count = len(values["candidate"])
        distance_weight = np.arange(6, dtype=np.float64)
        rows.append(
            {
                "parent_event_id": key[0],
                "source_session_id": window["source_session_id"],
                "bucket": window["bucket"],
                "phase": key[1],
                "false_alert_target": window[
                    "false_alert_target"
                ],
                "eligible_cell_count": count,
                "log1p_eligible_cell_count": float(np.log1p(count)),
                "candidate_mean": float(
                    np.mean(values["candidate"])
                ),
                "candidate_p95": float(
                    np.quantile(values["candidate"], 0.95)
                ),
                "candidate_max": max(values["candidate"]),
                "comparator_mean": float(
                    np.mean(values["comparator"])
                ),
                "comparator_p95": float(
                    np.quantile(values["comparator"], 0.95)
                ),
                "comparator_max": max(values["comparator"]),
                "known_mean": float(np.mean(values["known"])),
                "known_p95": float(
                    np.quantile(values["known"], 0.95)
                ),
                "near_fraction": float(
                    values["horizon_counts"][1] / count
                ),
                "body_fraction": float(
                    values["height_counts"][1] / count
                ),
                "direction_2_fraction": float(
                    values["direction_counts"][2] / count
                ),
                "distance_mean_normalized": float(
                    np.dot(values["distance_counts"], distance_weight)
                    / (count * 5.0)
                ),
            }
        )
    return rows


def phase_ranking(
    rows: list[dict[str, Any]],
    phases: set[str],
    statistic: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["phase"] in phases]
    target = np.asarray(
        [row["false_alert_target"] for row in selected],
        dtype=np.float32,
    )
    candidate = np.asarray(
        [row[f"candidate_{statistic}"] for row in selected],
        dtype=np.float32,
    )
    comparator = np.asarray(
        [row[f"comparator_{statistic}"] for row in selected],
        dtype=np.float32,
    )
    eligible = np.ones_like(target, dtype=bool)
    return {
        "unit_count": len(selected),
        "phases": sorted(phases),
        "statistic": statistic,
        **ranking_comparison(
            candidate,
            comparator,
            target,
            eligible,
        ),
    }


def passed_alertable_pairs(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(row["parent_event_id"], {})[
            row["phase"]
        ] = row
    pairs = []
    for event_id, phases in sorted(by_event.items()):
        if not {
            "positive_alertable",
            "positive_passed",
        }.issubset(phases):
            continue
        alertable = phases["positive_alertable"]
        passed = phases["positive_passed"]
        pairs.append(
            {
                "parent_event_id": event_id,
                "candidate_p95_delta": (
                    passed["candidate_p95"]
                    - alertable["candidate_p95"]
                ),
                "comparator_p95_delta": (
                    passed["comparator_p95"]
                    - alertable["comparator_p95"]
                ),
            }
        )
    return {
        "pair_count": len(pairs),
        "candidate_passed_score_higher_count": sum(
            row["candidate_p95_delta"] > 0 for row in pairs
        ),
        "comparator_passed_score_higher_count": sum(
            row["comparator_p95_delta"] > 0 for row in pairs
        ),
        "candidate_p95_delta": summarize_values(
            [row["candidate_p95_delta"] for row in pairs]
        ),
        "comparator_p95_delta": summarize_values(
            [row["comparator_p95_delta"] for row in pairs]
        ),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite real veto ranking report")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        int(manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in manifest["events"])
        != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    windows = build_event_windows(args.manifest, manifest)
    dataset = PublicHistoryDataset(windows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    units = []
    cell_auroc_deltas = []
    event_auroc_deltas = []
    for seed in SEEDS:
        for fold in FOLDS:
            report_path = (
                args.candidate_root
                / f"seed-{seed}"
                / f"fold-{fold}"
                / "report.json"
            )
            candidate_report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            checkpoint_path = Path(
                candidate_report["checkpoint"]["path"]
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            student = VetoEligibilityStudent(zero_head=True)
            student.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            student.to(device).eval()
            reference_path = Path(
                candidate_report["reference_checkpoint_path"]
            )
            reference, _ = load_reference(
                args.pretrained,
                reference_path,
                device,
            )
            probability = np.zeros(
                (len(windows), 3, 3, 6, 6),
                dtype=np.float32,
            )
            comparator = np.zeros_like(probability)
            known_probability = np.zeros_like(probability)
            eligible = np.zeros_like(probability, dtype=bool)
            with torch.no_grad():
                for frames, indices in loader:
                    frames = frames.to(device, non_blocking=True)
                    risk_logits, known_logits = reference_predictions(
                        reference,
                        frames,
                    )
                    logits = compose_ranking_logits(
                        student(frames),
                        risk_logits,
                        "confidence_residual",
                    )
                    risk = risk_logits.sigmoid()
                    known = known_logits.sigmoid()
                    mask = (risk >= 0.5) & (known >= 0.5)
                    central = torch.zeros_like(mask)
                    central[:, 1:, 1:, 2:4, :] = True
                    mask &= central
                    index_array = indices.numpy()
                    probability[index_array] = (
                        logits.sigmoid().cpu().numpy()
                    )
                    comparator[index_array] = (
                        (1.0 - risk).cpu().numpy()
                    )
                    known_probability[index_array] = (
                        known.cpu().numpy()
                    )
                    eligible[index_array] = mask.cpu().numpy()

            scored = np.asarray(
                [
                    window["false_alert_target"] is not None
                    for window in windows
                ],
                dtype=bool,
            )
            target = np.zeros_like(probability)
            for index, window in enumerate(windows):
                if window["false_alert_target"] is not None:
                    target[index].fill(
                        float(window["false_alert_target"])
                    )
            scored_eligible = eligible & scored[
                :, None, None, None, None
            ]
            cell = ranking_comparison(
                probability,
                comparator,
                target,
                scored_eligible,
            )
            phase_rows = phase_unit_rows(
                windows,
                probability,
                comparator,
                known_probability,
                eligible,
            )
            event_p95 = phase_ranking(
                phase_rows,
                {"negative_event", "positive_alertable"},
                "p95",
            )
            event_max = phase_ranking(
                phase_rows,
                {"negative_event", "positive_alertable"},
                "max",
            )
            pairs = passed_alertable_pairs(phase_rows)
            cell_auroc_deltas.append(
                float(cell["candidate_auroc_delta"])
            )
            event_auroc_deltas.append(
                float(event_p95["candidate_auroc_delta"])
            )
            units.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "cell_ranking": cell,
                    "event_p95_ranking": event_p95,
                    "event_max_ranking": event_max,
                    "positive_passed_vs_alertable": pairs,
                    "phase_units": phase_rows,
                    "candidate_report_sha256": sha256(
                        report_path
                    ),
                    "candidate_checkpoint_sha256": sha256(
                        checkpoint_path
                    ),
                    "reference_checkpoint_sha256": sha256(
                        reference_path
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "fold": fold,
                        "cell_auroc_delta": cell[
                            "candidate_auroc_delta"
                        ],
                        "event_p95_auroc_delta": event_p95[
                            "candidate_auroc_delta"
                        ],
                    }
                ),
                flush=True,
            )
            del student
            del reference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_sanpo_real_veto_"
            "ranking_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SANPO_REAL_VETO_RANKING_DEVELOPMENT_COMPLETE",
        "policy": {
            "consumed_development": True,
            "threshold_search": False,
            "system_output_connected": False,
            "promotion_evidence": False,
        },
        "design": {
            "window_history_frames": 5,
            "central_directions": [2, 3],
            "critical_horizons": ["near", "far"],
            "critical_heights": ["body", "head"],
            "false_alert_units": (
                "all scored cells in negative events and positive "
                "passed intervals"
            ),
            "true_alert_units": (
                "scored cells in positive alertable intervals"
            ),
            "event_statistic": "p95 primary; max sensitivity",
            "comparator": "1 - frozen baseline risk probability",
        },
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "window_count": len(windows),
        "model_count": len(units),
        "summary": {
            "cell_auroc_delta": summarize_values(
                cell_auroc_deltas
            ),
            "cell_auroc_delta_positive_units": sum(
                value > 0 for value in cell_auroc_deltas
            ),
            "event_p95_auroc_delta": summarize_values(
                event_auroc_deltas
            ),
            "event_p95_auroc_delta_positive_units": sum(
                value > 0 for value in event_auroc_deltas
            ),
        },
        "units": units,
        "evidence_limit": (
            "Consumed human-reviewed SANPO Development ranking only. "
            "It distinguishes representation/calibration from "
            "execution utility; it does not authorize a veto threshold, "
            "App behavior, promotion, or safety claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

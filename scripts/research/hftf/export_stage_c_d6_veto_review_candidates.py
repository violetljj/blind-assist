#!/usr/bin/env python3
"""Export model-aware public-real hard-negative review candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from evaluate_stage_c_d6_veto_eligibility_outcome_unseen import (
    DEFAULT_CANDIDATE_ROOT,
    DEFAULT_PRETRAINED,
)
from summarize_stage_c_d6_early_pair_structured_field_canary import (
    FOLDS,
    SEEDS,
)
from train_stage_c_d5_tartanground_development_student import (
    HEIGHTS,
    HORIZONS,
    load_jsonl,
    sha256,
    transform_image,
)
from train_stage_c_d6_veto_eligibility_ranking import (
    VetoEligibilityStudent,
    compose_ranking_logits,
    load_reference,
    reference_predictions,
)


class PublicHistoryDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, windows: list[dict[str, Any]]) -> None:
        self.windows = windows
        self.cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def image(self, path: str) -> torch.Tensor:
        tensor = self.cache.get(path)
        if tensor is None:
            with Image.open(path) as source:
                tensor = transform_image(
                    source.convert("RGB").resize(
                        (224, 128),
                        Image.Resampling.BILINEAR,
                    ),
                    None,
                )
            self.cache[path] = tensor
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        paths = self.windows[index]["history_rgb_paths"]
        return torch.stack([self.image(path) for path in paths]), index


def stable_id(*parts: object) -> str:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()
    return f"veto-review-{digest[:20]}"


def build_windows(
    media_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in media_rows:
        key = (
            str(row["source_session_id"]),
            str(row["camera"]),
            str(row["view"]),
        )
        rgb_path = Path(row["rgb_local_path"])
        if not rgb_path.is_file():
            continue
        groups.setdefault(key, []).append(row)
    windows = []
    for (session, camera, view), rows in sorted(groups.items()):
        by_index = {
            int(row["frame_index"]): row for row in rows
        }
        for anchor in sorted(by_index):
            indices = list(range(anchor - 4, anchor + 1))
            if not all(index in by_index for index in indices):
                continue
            history = [by_index[index] for index in indices]
            windows.append(
                {
                    "window_id": stable_id(
                        session,
                        camera,
                        view,
                        anchor,
                    ),
                    "source_session_id": session,
                    "camera": camera,
                    "view": view,
                    "history_frame_indices": indices,
                    "anchor_frame_index": anchor,
                    "history_rgb_paths": [
                        str(Path(row["rgb_local_path"]).resolve())
                        for row in history
                    ],
                    "anchor_rgb_path": str(
                        Path(history[-1]["rgb_local_path"]).resolve()
                    ),
                    "nominal_time_ns": history[-1].get(
                        "nominal_time_ns"
                    ),
                    "timestamp_ns": history[-1].get("timestamp_ns"),
                    "time_semantics": history[-1].get(
                        "time_semantics"
                    ),
                }
            )
    return windows


def consensus_cell_rows(
    windows: list[dict[str, Any]],
    score_stack: np.ndarray,
    active_stack: np.ndarray,
    risk_stack: np.ndarray,
    known_stack: np.ndarray,
    veto_stack: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if score_stack.shape != active_stack.shape:
        raise ValueError("Score and active stacks must match")
    if score_stack.shape != risk_stack.shape:
        raise ValueError("Score and risk stacks must match")
    if score_stack.shape != known_stack.shape:
        raise ValueError("Score and known stacks must match")
    if veto_stack is not None and score_stack.shape != veto_stack.shape:
        raise ValueError("Score and veto stacks must match")
    if score_stack.shape[1] != len(windows):
        raise ValueError("Window count mismatch")
    model_count = score_stack.shape[0]
    rows = []
    for window_index, window in enumerate(windows):
        for horizon_index in (1, 2):
            for height_index in (1, 2):
                for row_index in range(6):
                    for column_index in range(6):
                        active = active_stack[
                            :,
                            window_index,
                            horizon_index,
                            height_index,
                            row_index,
                            column_index,
                        ].astype(bool)
                        active_count = int(active.sum())
                        if active_count == 0:
                            continue
                        scores = score_stack[
                            :,
                            window_index,
                            horizon_index,
                            height_index,
                            row_index,
                            column_index,
                        ]
                        risk = risk_stack[
                            :,
                            window_index,
                            horizon_index,
                            height_index,
                            row_index,
                            column_index,
                        ]
                        known = known_stack[
                            :,
                            window_index,
                            horizon_index,
                            height_index,
                            row_index,
                            column_index,
                        ]
                        active_scores = scores[active]
                        veto_count = (
                            int(
                                veto_stack[
                                    :,
                                    window_index,
                                    horizon_index,
                                    height_index,
                                    row_index,
                                    column_index,
                                ].sum()
                            )
                            if veto_stack is not None
                            else 0
                        )
                        rows.append(
                            {
                                "schema": (
                                    "blindassist_hftf_veto_review_cell_v1"
                                ),
                                "role": (
                                    "MODEL_AWARE_DEVELOPMENT_DISCOVERY"
                                ),
                                "window_id": window["window_id"],
                                "source_session_id": window[
                                    "source_session_id"
                                ],
                                "anchor_frame_index": window[
                                    "anchor_frame_index"
                                ],
                                "anchor_rgb_path": window[
                                    "anchor_rgb_path"
                                ],
                                "horizon": HORIZONS[horizon_index],
                                "height": HEIGHTS[height_index],
                                "grid_row": row_index,
                                "grid_column": column_index,
                                "active_vote_count": active_count,
                                "model_count": model_count,
                                "active_vote_fraction": (
                                    active_count / model_count
                                ),
                                "conservative_veto_vote_count": (
                                    veto_count
                                ),
                                "conservative_veto_vote_fraction": (
                                    veto_count / model_count
                                ),
                                "conservative_veto_consensus": (
                                    veto_count > model_count // 2
                                ),
                                "mean_false_eligibility_score": float(
                                    np.mean(active_scores)
                                ),
                                "minimum_false_eligibility_score": float(
                                    np.min(active_scores)
                                ),
                                "maximum_false_eligibility_score": float(
                                    np.max(active_scores)
                                ),
                                "mean_baseline_risk_probability": float(
                                    np.mean(risk)
                                ),
                                "mean_baseline_known_probability": float(
                                    np.mean(known)
                                ),
                                "consensus_review_eligible": (
                                    active_count > model_count // 2
                                ),
                                "truth_status": "NOT_EVALUABLE",
                                "system_action_authority": False,
                            }
                        )
    rows.sort(
        key=lambda row: (
            -row["active_vote_count"],
            -row["mean_false_eligibility_score"],
            row["window_id"],
            row["horizon"],
            row["height"],
            row["grid_row"],
            row["grid_column"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["discovery_rank"] = rank
    return rows


def conservative_veto_summary(
    active_stack: np.ndarray,
    veto_stack: np.ndarray,
    model_keys: list[tuple[int, int]],
) -> dict[str, Any]:
    if active_stack.shape != veto_stack.shape:
        raise ValueError("Active and veto stacks must match")
    if active_stack.shape[0] != len(model_keys):
        raise ValueError("Model key count mismatch")
    central_active_stack = np.zeros_like(active_stack)
    central_veto_stack = np.zeros_like(veto_stack)
    central_active_stack[:, :, 1:, 1:, 2:4, :] = active_stack[
        :, :, 1:, 1:, 2:4, :
    ]
    central_veto_stack[:, :, 1:, 1:, 2:4, :] = veto_stack[
        :, :, 1:, 1:, 2:4, :
    ]
    active_window = active_stack.reshape(
        active_stack.shape[0],
        active_stack.shape[1],
        -1,
    ).any(axis=2)
    veto_window = veto_stack.reshape(
        veto_stack.shape[0],
        veto_stack.shape[1],
        -1,
    ).any(axis=2)
    retained_window = (active_stack & ~veto_stack).reshape(
        active_stack.shape[0],
        active_stack.shape[1],
        -1,
    ).any(axis=2)
    cleared_window = active_window & ~retained_window
    central_active_window = central_active_stack.reshape(
        central_active_stack.shape[0],
        central_active_stack.shape[1],
        -1,
    ).any(axis=2)
    central_veto_window = central_veto_stack.reshape(
        central_veto_stack.shape[0],
        central_veto_stack.shape[1],
        -1,
    ).any(axis=2)
    central_retained_window = (
        central_active_stack & ~central_veto_stack
    ).reshape(
        central_active_stack.shape[0],
        central_active_stack.shape[1],
        -1,
    ).any(axis=2)
    central_cleared_window = (
        central_active_window & ~central_retained_window
    )
    units = []
    for model_index, (seed, fold) in enumerate(model_keys):
        active_cells = int(active_stack[model_index].sum())
        vetoed_cells = int(veto_stack[model_index].sum())
        baseline_windows = int(active_window[model_index].sum())
        cleared_windows = int(cleared_window[model_index].sum())
        central_active_cells = int(
            central_active_stack[model_index].sum()
        )
        central_vetoed_cells = int(
            central_veto_stack[model_index].sum()
        )
        central_baseline_windows = int(
            central_active_window[model_index].sum()
        )
        central_cleared_windows = int(
            central_cleared_window[model_index].sum()
        )
        units.append(
            {
                "seed": seed,
                "fold": fold,
                "baseline_active_cell_count": active_cells,
                "vetoed_active_cell_count": vetoed_cells,
                "vetoed_active_cell_fraction": (
                    vetoed_cells / active_cells
                    if active_cells
                    else None
                ),
                "baseline_active_window_count": baseline_windows,
                "veto_applied_window_count": int(
                    veto_window[model_index].sum()
                ),
                "fully_cleared_window_count": cleared_windows,
                "fully_cleared_window_fraction": (
                    cleared_windows / baseline_windows
                    if baseline_windows
                    else None
                ),
                "central_baseline_active_cell_count": (
                    central_active_cells
                ),
                "central_vetoed_active_cell_count": (
                    central_vetoed_cells
                ),
                "central_vetoed_active_cell_fraction": (
                    central_vetoed_cells / central_active_cells
                    if central_active_cells
                    else None
                ),
                "central_baseline_active_window_count": (
                    central_baseline_windows
                ),
                "central_veto_applied_window_count": int(
                    central_veto_window[model_index].sum()
                ),
                "central_fully_cleared_window_count": (
                    central_cleared_windows
                ),
                "central_fully_cleared_window_fraction": (
                    central_cleared_windows
                    / central_baseline_windows
                    if central_baseline_windows
                    else None
                ),
            }
        )
    majority = active_stack.shape[0] // 2 + 1
    return {
        "units": units,
        "total_baseline_active_model_cells": int(active_stack.sum()),
        "total_vetoed_active_model_cells": int(veto_stack.sum()),
        "total_baseline_active_model_windows": int(
            active_window.sum()
        ),
        "total_fully_cleared_model_windows": int(
            cleared_window.sum()
        ),
        "window_count_with_any_model_veto": int(
            veto_window.any(axis=0).sum()
        ),
        "window_count_with_majority_model_veto": int(
            (veto_window.sum(axis=0) >= majority).sum()
        ),
        "window_count_fully_cleared_by_any_model": int(
            cleared_window.any(axis=0).sum()
        ),
        "window_count_fully_cleared_by_majority_models": int(
            (cleared_window.sum(axis=0) >= majority).sum()
        ),
        "central_total_baseline_active_model_cells": int(
            central_active_stack.sum()
        ),
        "central_total_vetoed_active_model_cells": int(
            central_veto_stack.sum()
        ),
        "central_total_baseline_active_model_windows": int(
            central_active_window.sum()
        ),
        "central_total_fully_cleared_model_windows": int(
            central_cleared_window.sum()
        ),
        "central_window_count_with_any_model_veto": int(
            central_veto_window.any(axis=0).sum()
        ),
        "central_window_count_with_majority_model_veto": int(
            (central_veto_window.sum(axis=0) >= majority).sum()
        ),
        "central_window_count_fully_cleared_by_any_model": int(
            central_cleared_window.any(axis=0).sum()
        ),
        "central_window_count_fully_cleared_by_majority_models": int(
            (
                central_cleared_window.sum(axis=0)
                >= majority
            ).sum()
        ),
    }


def load_conservative_thresholds(
    path: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("design", {}).get("threshold_rule")
        != "nextafter(max training true-alert score, +inf)"
    ):
        raise ValueError("Unexpected conservative threshold rule")
    thresholds = {}
    for unit in report["units"]:
        key = (int(unit["seed"]), int(unit["fold"]))
        if key in thresholds:
            raise ValueError(f"Duplicate threshold unit: {key}")
        thresholds[key] = {
            "threshold": float(unit["threshold"]),
            "candidate_checkpoint_sha256": str(
                unit["candidate_checkpoint_sha256"]
            ),
        }
    return thresholds


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-manifest", type=Path, required=True)
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k-windows", type=int, default=20)
    parser.add_argument("--threshold-report", type=Path)
    args = parser.parse_args()
    if args.top_k_windows <= 0:
        raise ValueError("--top-k-windows must be positive")
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite review candidate export")
    media_rows = load_jsonl(args.media_manifest)
    scientific_labels = sorted(
        {
            str(row["scientific_label"])
            for row in media_rows
            if row.get("scientific_label")
        }
    )
    windows = build_windows(media_rows)
    if not windows:
        raise ValueError("No contiguous five-frame public windows")
    dataset = PublicHistoryDataset(windows)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    score_rows = []
    active_rows = []
    risk_rows = []
    known_rows = []
    veto_rows = []
    model_receipts = []
    thresholds = (
        load_conservative_thresholds(args.threshold_report)
        if args.threshold_report is not None
        else {}
    )
    expected_model_keys = {
        (seed, fold) for seed in SEEDS for fold in FOLDS
    }
    if thresholds and set(thresholds) != expected_model_keys:
        raise ValueError("Conservative threshold model set mismatch")
    model_keys = []
    for seed in SEEDS:
        for fold in FOLDS:
            model_keys.append((seed, fold))
            report_path = (
                args.candidate_root
                / f"seed-{seed}"
                / f"fold-{fold}"
                / "report.json"
            )
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            if report["task"]["ranking_mode"] != "confidence_residual":
                raise ValueError("Expected confidence_residual candidate")
            checkpoint_path = Path(report["checkpoint"]["path"])
            checkpoint_sha256 = sha256(checkpoint_path)
            threshold_unit = thresholds.get((seed, fold))
            if (
                threshold_unit is not None
                and threshold_unit["candidate_checkpoint_sha256"]
                != checkpoint_sha256
            ):
                raise ValueError(
                    f"Threshold checkpoint mismatch: seed={seed} "
                    f"fold={fold}"
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
                report["reference_checkpoint_path"]
            )
            reference, _ = load_reference(
                args.pretrained,
                reference_path,
                device,
            )
            scores = np.zeros(
                (len(windows), 3, 3, 6, 6),
                dtype=np.float32,
            )
            active = np.zeros_like(scores, dtype=bool)
            risks = np.zeros_like(scores)
            knowns = np.zeros_like(scores)
            vetoes = np.zeros_like(active)
            with torch.no_grad():
                for frames, indices in loader:
                    frames = frames.to(device, non_blocking=True)
                    reference_risk, reference_known = (
                        reference_predictions(reference, frames)
                    )
                    logits = compose_ranking_logits(
                        student(frames),
                        reference_risk,
                        "confidence_residual",
                    )
                    score_probability = logits.sigmoid()
                    risk_probability = reference_risk.sigmoid()
                    known_probability = reference_known.sigmoid()
                    batch_active = (
                        (risk_probability >= 0.5)
                        & (known_probability >= 0.5)
                    )
                    critical = torch.zeros_like(batch_active)
                    critical[:, 1:, 1:] = True
                    batch_active &= critical
                    batch_veto = (
                        batch_active
                        & (
                            score_probability
                            >= threshold_unit["threshold"]
                        )
                        if threshold_unit is not None
                        else torch.zeros_like(batch_active)
                    )
                    index_array = indices.numpy()
                    scores[index_array] = (
                        score_probability.cpu().numpy()
                    )
                    active[index_array] = batch_active.cpu().numpy()
                    vetoes[index_array] = batch_veto.cpu().numpy()
                    risks[index_array] = (
                        risk_probability.cpu().numpy()
                    )
                    knowns[index_array] = (
                        known_probability.cpu().numpy()
                    )
            score_rows.append(scores)
            active_rows.append(active)
            risk_rows.append(risks)
            known_rows.append(knowns)
            veto_rows.append(vetoes)
            model_receipts.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "report_path": str(report_path.resolve()),
                    "report_sha256": sha256(report_path),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "reference_checkpoint_path": str(
                        reference_path.resolve()
                    ),
                    "reference_checkpoint_sha256": sha256(
                        reference_path
                    ),
                    "conservative_veto_threshold": (
                        threshold_unit["threshold"]
                        if threshold_unit is not None
                        else None
                    ),
                }
            )
            del student
            del reference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    active_stack = np.stack(active_rows)
    veto_stack = np.stack(veto_rows)
    cell_rows = consensus_cell_rows(
        windows,
        np.stack(score_rows),
        active_stack,
        np.stack(risk_rows),
        np.stack(known_rows),
        veto_stack,
    )
    consensus_rows = [
        row for row in cell_rows if row["consensus_review_eligible"]
    ]
    by_window: dict[str, list[dict[str, Any]]] = {}
    for row in consensus_rows:
        by_window.setdefault(row["window_id"], []).append(row)
    window_lookup = {
        window["window_id"]: window for window in windows
    }
    review_queue = []
    for window_id, rows in by_window.items():
        best = rows[0]
        review_queue.append(
            {
                "schema": (
                    "blindassist_hftf_veto_review_window_v1"
                ),
                "role": "MODEL_AWARE_DEVELOPMENT_DISCOVERY",
                **window_lookup[window_id],
                "best_cell_discovery_rank": best["discovery_rank"],
                "best_active_vote_count": best[
                    "active_vote_count"
                ],
                "best_false_eligibility_score": best[
                    "mean_false_eligibility_score"
                ],
                "consensus_cell_count": len(rows),
                "conservative_veto_consensus_cell_count": sum(
                    1
                    for row in rows
                    if row["conservative_veto_consensus"]
                ),
                "top_cells": [
                    {
                        key: row[key]
                        for key in (
                            "horizon",
                            "height",
                            "grid_row",
                            "grid_column",
                            "active_vote_count",
                            "conservative_veto_vote_count",
                            "mean_false_eligibility_score",
                            "mean_baseline_risk_probability",
                        )
                    }
                    for row in rows[:5]
                ],
                "truth_status": "NOT_EVALUABLE",
                "confirmation_review_eligible": False,
                "system_action_authority": False,
            }
        )
    review_queue.sort(
        key=lambda row: (
            -row["best_active_vote_count"],
            -row["best_false_eligibility_score"],
            row["window_id"],
        )
    )
    review_queue = review_queue[: args.top_k_windows]

    args.output_root.mkdir(parents=True)
    cells_path = args.output_root / "active_cell_ranking.jsonl"
    queue_path = args.output_root / "review_queue.jsonl"
    write_jsonl(cells_path, cell_rows)
    write_jsonl(queue_path, review_queue)
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_public_real_"
            "veto_review_export_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MODEL_AWARE_DEVELOPMENT_DISCOVERY_EXPORT_COMPLETE",
        "policy": {
            "truth_authority": False,
            "confirmation_review_eligible": False,
            "system_action_authority": False,
            "d7_blind_review_files_modified": False,
            "threshold_search": False,
        },
        "media_manifest_path": str(args.media_manifest.resolve()),
        "media_manifest_sha256": sha256(args.media_manifest),
        "window_count": len(windows),
        "model_count": len(model_receipts),
        "active_cell_count": len(cell_rows),
        "consensus_active_cell_count": len(consensus_rows),
        "consensus_window_count": len(by_window),
        "review_queue_count": len(review_queue),
        "top_k_windows": args.top_k_windows,
        "ranking_rule": (
            "active vote count descending, then mean false-"
            "eligibility score descending"
        ),
        "model_receipts": model_receipts,
        "outputs": {
            "active_cell_ranking": {
                "path": str(cells_path.resolve()),
                "sha256": sha256(cells_path),
            },
            "review_queue": {
                "path": str(queue_path.resolve()),
                "sha256": sha256(queue_path),
            },
        },
        "evidence_limit": (
            (
                "Model-blind RGB review supports an actionable-negative "
                "label for each complete five-frame window. Cell "
                "localization truth, positive-event safety, system event "
                "truth, promotion, and action authority remain absent."
            )
            if scientific_labels
            == ["UNANIMOUS_MODEL_BLIND_RGB_ACTIONABLE_NEGATIVE"]
            else (
                "Unlabeled public-real model-aware Development discovery "
                "only. Scores are hypotheses, not false-alert labels."
            )
        ),
    }
    if scientific_labels:
        report["scientific_labels"] = scientific_labels
    if args.threshold_report is not None:
        report["conservative_veto_execution"] = {
            "threshold_report_path": str(
                args.threshold_report.resolve()
            ),
            "threshold_report_sha256": sha256(
                args.threshold_report
            ),
            "threshold_rule": (
                "nextafter(max training true-alert score, +inf)"
            ),
            "consensus_veto_cell_count": sum(
                1
                for row in cell_rows
                if row["conservative_veto_consensus"]
            ),
            **conservative_veto_summary(
                active_stack,
                veto_stack,
                model_keys,
            ),
        }
    report_path = args.output_root / "report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

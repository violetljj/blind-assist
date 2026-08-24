"""Sequence-disjoint small boundary head for the supervised R5 successor."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .anchor_pair_observation import AnchorPairOracleBoundaryProvider
from .dense_boundary_observation import (
    DISTANCE_THRESHOLD_PX,
    VERTICAL_ORIENTATION_TOLERANCE_DEG,
    DeepLsdDenseFieldExtractor,
    _angular_distance_mod_pi,
)
from .experiment import _aggregate
from .rgb_experiment import _baseline, _sage_lm, _v1_criteria
from .two_view_experiment import _arm_diagnostics, _evaluator_episode, _source_poses
from .two_view_observation import oracle_pixel_lines


SCHEMA_VERSION = "sage_lm_v1b_r5s_sequence_disjoint_anchor_boundary_head"
TRAINING_EPOCHS = 240
TOP_K_PER_ROLE = 8


def boundary_head_features(field: dict, anchor_bbox: tuple[int, int, int, int]) -> np.ndarray:
    distance = np.asarray(field["distance"], dtype=np.float32)
    orientation = np.asarray(field["orientation"], dtype=np.float32)
    height, width = distance.shape
    vertical_distance = _angular_distance_mod_pi(orientation, np.pi / 2).astype(np.float32)
    distance_affinity = np.exp(-distance / DISTANCE_THRESHOLD_PX)
    orientation_affinity = np.clip(
        1.0 - vertical_distance / math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG), 0.0, 1.0
    )
    support = (
        (distance <= DISTANCE_THRESHOLD_PX)
        & (vertical_distance <= math.radians(VERTICAL_ORIENTATION_TOLERANCE_DEG))
    ).astype(np.float32)
    xs = np.arange(width, dtype=np.float32)
    anchor_center = (anchor_bbox[0] + anchor_bbox[2]) * 0.5
    anchor_gaussian = np.exp(-0.5 * ((xs - anchor_center) / max(4.0, width * 0.08)) ** 2)
    anchor_mask = ((xs >= anchor_bbox[0]) & (xs <= anchor_bbox[2])).astype(np.float32)
    return np.stack(
        [
            support.mean(axis=0),
            distance_affinity.mean(axis=0),
            orientation_affinity.mean(axis=0),
            (distance_affinity * orientation_affinity).mean(axis=0),
            support.max(axis=0),
            anchor_gaussian,
            anchor_mask,
            xs / max(1.0, width - 1),
        ]
    ).astype(np.float32)


@dataclass(frozen=True)
class HeadRecord:
    episode_id: str
    sequence: str
    frame_role: str
    features: np.ndarray
    left_x: float
    right_x: float


def _make_model(torch, input_channels: int):
    class BoundaryHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Conv1d(input_channels, 24, kernel_size=9, padding=4),
                torch.nn.GELU(),
                torch.nn.Conv1d(24, 24, kernel_size=9, padding=4),
                torch.nn.GELU(),
                torch.nn.Conv1d(24, 2, kernel_size=1),
            )

        def forward(self, value):
            return self.network(value)

    return BoundaryHead()


def _train_fold(torch, records: list[HeadRecord], held_out_sequence: str):
    train = [row for row in records if row.sequence != held_out_sequence]
    width = train[0].features.shape[1]
    features = []
    targets = []
    for row in train:
        features.append(row.features)
        targets.append([round(row.left_x), round(row.right_x)])
        features.append(row.features[:, ::-1].copy())
        targets.append([width - 1 - round(row.right_x), width - 1 - round(row.left_x)])
    x = torch.from_numpy(np.stack(features)).float()
    y = torch.as_tensor(np.clip(np.asarray(targets), 0, width - 1), dtype=torch.long)
    torch.manual_seed(1701)
    random.seed(1701)
    model = _make_model(torch, x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    initial_loss = None
    model.train()
    for _ in range(TRAINING_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, width), y.reshape(-1))
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
    return model.eval(), {"train_example_count": len(train), "initial_loss": initial_loss, "final_loss": float(loss.detach())}


def _predictor(torch, model):
    def predict(field: dict, anchor_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        features = torch.from_numpy(boundary_head_features(field, anchor_bbox))[None]
        with torch.inference_mode():
            logits = model(features)[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        probabilities /= np.maximum(probabilities.max(axis=1, keepdims=True), 1e-9)
        return probabilities[0], probabilities[1]

    return predict


def _top_k_hits(probability: np.ndarray, target_x: float, top_k: int = TOP_K_PER_ROLE) -> bool:
    working = probability.copy()
    selected = []
    for _ in range(top_k):
        index = int(np.argmax(working))
        selected.append(index)
        working[max(0, index - 4) : min(len(working), index + 5)] = -math.inf
    return bool(min(abs(index - target_x) for index in selected) <= 9.0)


def run(cohort_path: Path, extractor: DeepLsdDenseFieldExtractor) -> dict:
    import torch

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-B-R5S requires the frozen 24-episode R2 cohort")
    records: list[HeadRecord] = []
    episode_cache = {}
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        visible = [row for row in episode_input.exact_anchor_observations if row.visible]
        first = next(row for row in visible if row.frame_index == 0)
        second = next(row for row in visible if row.frame_index == episode_input.active_parallax_frame_index)
        image_a = cv2.imread(str(episode_input.rgb_frames[first.frame_index]), cv2.IMREAD_COLOR)
        image_b = cv2.imread(str(episode_input.rgb_frames[second.frame_index]), cv2.IMREAD_COLOR)
        if image_a is None or image_b is None:
            raise ValueError(f"unable to decode training RGB for {evaluator.episode_id}")
        field_a = extractor.predict_field(image_a, first.bbox_xyxy)
        field_b = extractor.predict_field(image_b, second.bbox_xyxy)
        oracle_a, oracle_b = oracle_pixel_lines(episode_input, truth, pose_a, pose_b)
        mid_y = (episode_input.intrinsics.height - 1) * 0.5
        sequence = str(materialized["source"]["sequence"])
        for frame_role, field, anchor, oracle in (
            ("A", field_a, first.bbox_xyxy, oracle_a),
            ("B", field_b, second.bbox_xyxy, oracle_b),
        ):
            records.append(
                HeadRecord(
                    evaluator.episode_id,
                    sequence,
                    frame_role,
                    boundary_head_features(field, anchor),
                    oracle[0].x_at(mid_y),
                    oracle[1].x_at(mid_y),
                )
            )
        episode_cache[evaluator.episode_id] = (evaluator, episode_input, truth, pose_a, pose_b, pose_audit)

    sequences = sorted({row.sequence for row in records})
    models = {}
    fold_diagnostics = {}
    for sequence in sequences:
        models[sequence], fold_diagnostics[sequence] = _train_fold(torch, records, sequence)

    head_frame_hits = {}
    for record in records:
        predictor = _predictor(torch, models[record.sequence])
        # The cached feature is enough for evaluation; use the model directly to
        # avoid reconstructing a synthetic field.
        with torch.inference_mode():
            probability = torch.softmax(models[record.sequence](torch.from_numpy(record.features)[None])[0], dim=-1).numpy()
        head_frame_hits[(record.episode_id, record.frame_role)] = (
            _top_k_hits(probability[0], record.left_x),
            _top_k_hits(probability[1], record.right_x),
        )

    rows = []
    controls_retained = 0
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth, pose_a, pose_b, pose_audit = episode_cache[materialized["input"]["episode_id"]]
        sequence = str(materialized["source"]["sequence"])
        provider = AnchorPairOracleBoundaryProvider(
            episode_input,
            truth,
            pose_a,
            pose_b,
            extractor,
            role_predictor=_predictor(torch, models[sequence]),
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
                "head_top8_hits": {
                    "frame_a": head_frame_hits[(evaluator.episode_id, "A")],
                    "frame_b": head_frame_hits[(evaluator.episode_id, "B")],
                },
                "baseline": _baseline(evaluator),
                "b1": result,
            }
        )
    baseline_metrics = _aggregate(row["baseline"] for row in rows)
    arm_metrics = _aggregate(row["b1"] for row in rows)
    diagnostics = _arm_diagnostics(rows, "b1")
    candidate_available = sum(
        len(row["b1"]["diagnostics"].get("oracle_association_distances_px", [])) == 4
        and max(row["b1"]["diagnostics"]["oracle_association_distances_px"]) <= 9.0
        for row in rows
    )
    candidate_missing = 24 - candidate_available
    top8_pair_coverage = int(sum(
        all(row["head_top8_hits"]["frame_a"]) and all(row["head_top8_hits"]["frame_b"]) for row in rows
    ))
    diagnostics.update(
        {
            "true_boundary_pair_available_count": candidate_available,
            "aperture_pair_hypothesis_missing_count": candidate_missing,
            "head_top8_four_boundary_coverage_count": top8_pair_coverage,
        }
    )
    criteria = _v1_criteria(baseline_metrics, arm_metrics, controls_retained)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_DEVELOPMENT_SEQUENCE_DISJOINT",
        "experiment_label": "V1_B_R5S_ANCHOR_CONDITIONED_LEFT_RIGHT_BOUNDARY_HEAD_B1",
        "cohort": {"path": str(cohort_path.resolve()), "episode_count": 24, "source_sequence_count": len(sequences)},
        "model": {
            **extractor.identity,
            "head": "8ch-24-24-2 Conv1D",
            "training_epochs": TRAINING_EPOCHS,
            "evaluation": "LEAVE_ONE_SOURCE_SEQUENCE_OUT",
            "folds": fold_diagnostics,
        },
        "frozen_surfaces": {
            "deeplsd_field": "UNCHANGED_R3",
            "source_pose": "UNCHANGED_R2",
            "interpretation_plane_geometry": "UNCHANGED_R2",
            "oracle_association_localization_gate_px": 9.0,
            "confidence": "UNCHANGED_R2_REPRESENTATION_CONTRACT_NOT_REINTERPRETED",
            "arrival": "UNCHANGED_R2",
            "policy": "UNCHANGED_R2",
        },
        "metrics": {"bbox_center_scale": baseline_metrics, "b1": arm_metrics, "controls_retained": controls_retained},
        "observation_diagnostics": diagnostics,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "r5s_targets": {
            "true_boundary_pair_available_at_least_18": candidate_available >= 18,
            "geometry_output_at_least_18": diagnostics["geometry_output_count"] >= 18,
            "aperture_pair_hypothesis_missing_at_most_6": candidate_missing <= 6,
        },
        "rows": rows,
        "claim_ceiling": "CURATED_R2_DEVELOPMENT_SEQUENCE_DISJOINT_TASK_CONDITIONED_COVERAGE_ONLY",
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
    print(json.dumps({"diagnostics": report["observation_diagnostics"], "targets": report["r5s_targets"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a paired low-capacity object-slot motion-residual student."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, Dataset

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from evaluate_stage_c_d27_thor_magni_kinematic_information_ceiling import (
    DISTANCE_CAP_M,
    evaluate_scores,
)
from extract_stage_c_d29_thor_magni_object_slots import (
    DEFAULT_OUTPUT as DEFAULT_OBJECT_SLOTS,
    FEATURE_COUNT,
    LAG_COUNT,
    LAG_FEATURE_COUNT,
    MAX_SLOTS,
    SCHEMA as OBJECT_SLOT_SCHEMA,
    STATIC_FEATURE_COUNT,
)
from run_stage_c_d17_tartanground_early_temporal_onset_canary import (
    model_sha256,
    seed_everything,
)
from run_stage_c_d25_thor_magni_time_to_entry import (
    nested,
    source_weights,
    summarize_delta,
)
from run_stage_c_d26_thor_magni_counterfactual_collision_field import (
    DEFAULT_D8_SAMPLES,
    DIRECTION_NAMES,
    prepare_records,
)
from run_stage_c_d28_thor_magni_kinematic_field_distillation import (
    DEFAULT_D27_REPORT,
    DEFAULT_D27_SCORES,
    bind_teacher_scores,
    metric_paths,
    teacher_fit,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_SAMPLES,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d29_thor_magni_"
    "object_slot_motion_residual_v0"
)
FOLDS = tuple(range(5))
SEED = 17
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SMOOTH_L1_BETA = 0.05
HORIZON_COUNT = 4
HIDDEN = 64
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d29-thor-magni-object-slot-motion-residual-v0/report.json"
)


def flip_slot_features(slots: torch.Tensor) -> torch.Tensor:
    flipped = slots.clone()
    flipped[..., 0].neg_()
    for lag in range(LAG_COUNT):
        offset = STATIC_FEATURE_COUNT + lag * LAG_FEATURE_COUNT
        flipped[..., offset].neg_()
        flipped[..., offset + 2].neg_()
    return flipped


class ObjectSlotDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ]
    ]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        slots: np.ndarray,
        mask: np.ndarray,
        *,
        train: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.slots = slots
        self.mask = mask
        self.train_mode = train
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def should_flip(self, record: dict[str, Any]) -> bool:
        if not self.train_mode:
            return False
        import hashlib

        payload = (
            f"{self.seed}:{self.epoch}:{record['sample_id']}:horizontal"
        )
        return bool(
            hashlib.sha256(payload.encode("utf-8")).digest()[0] & 1
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        record = self.records[index]
        cache_index = int(record["_d29_slot_index"])
        slots = torch.from_numpy(
            np.array(self.slots[cache_index], copy=True)
        ).float()
        mask = torch.from_numpy(
            np.array(self.mask[cache_index], copy=True)
        ).bool()
        current = torch.from_numpy(
            np.asarray(
                record["_d28_current_teacher"],
                dtype=np.float32,
            )
        ).div(DISTANCE_CAP_M)
        history = torch.from_numpy(
            np.asarray(
                record["_d28_history_teacher"],
                dtype=np.float32,
            )
        ).div(DISTANCE_CAP_M)
        if self.should_flip(record):
            slots = flip_slot_features(slots)
            current = current.flip(0)
            history = history.flip(0)
        return slots, mask, current, history, index


class ObjectSlotMotionResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.motion_encoder = nn.Sequential(
            nn.Linear(FEATURE_COUNT, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.static_head = nn.Linear(
            HIDDEN * 2,
            len(DIRECTION_NAMES) * HORIZON_COUNT,
        )
        self.motion_head = nn.Linear(
            HIDDEN * 2,
            len(DIRECTION_NAMES) * HORIZON_COUNT,
        )
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)

    @staticmethod
    def pool(
        embedding: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim != 3 or mask.shape != embedding.shape[:2]:
            raise ValueError("D29 slot embedding/mask shape mismatch")
        float_mask = mask.unsqueeze(-1).to(embedding.dtype)
        count = torch.clamp(float_mask.sum(dim=1), min=1.0)
        mean = (embedding * float_mask).sum(dim=1) / count
        masked = embedding.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        maximum = torch.amax(masked, dim=1)
        any_slot = mask.any(dim=1, keepdim=True)
        maximum = torch.where(any_slot, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=1)

    @staticmethod
    def field(logits: torch.Tensor) -> torch.Tensor:
        raw = -torch.sigmoid(logits).reshape(
            -1,
            len(DIRECTION_NAMES),
            HORIZON_COUNT,
        )
        return torch.cummax(raw, dim=2).values

    def forward(
        self,
        slots: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if slots.shape[1:] != (MAX_SLOTS, FEATURE_COUNT):
            raise ValueError("D29 slot tensor shape mismatch")
        static_pool = self.pool(
            self.static_encoder(slots[..., :STATIC_FEATURE_COUNT]),
            mask,
        )
        motion_pool = self.pool(self.motion_encoder(slots), mask)
        static_logits = self.static_head(static_pool)
        any_slot = mask.any(dim=1, keepdim=True).to(slots.dtype)
        history_logits = (
            static_logits + self.motion_head(motion_pool) * any_slot
        )
        return self.field(static_logits), self.field(history_logits)


def load_object_slots(
    path: Path,
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    report_path = path.with_suffix(path.suffix + ".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report["schema"] != OBJECT_SLOT_SCHEMA
        or report["status"] != "D29_THOR_MAGNI_OBJECT_SLOTS_MATERIALIZED"
        or report["output"]["sha256"] != sha256(path)
    ):
        raise ValueError("D29 object-slot report binding mismatch")
    payload = np.load(path)
    sample_ids = [str(value) for value in payload["sample_ids"]]
    expected = [str(record["sample_id"]) for record in records]
    if sample_ids != expected:
        raise ValueError("D29 object-slot sample ordering mismatch")
    slots = np.asarray(payload["slots"], dtype=np.float32)
    mask = np.asarray(payload["mask"], dtype=bool)
    feature_names = [str(value) for value in payload["feature_names"]]
    if (
        slots.shape != (len(records), MAX_SLOTS, FEATURE_COUNT)
        or mask.shape != (len(records), MAX_SLOTS)
        or feature_names != report["design"]["feature_names"]
        or not np.isfinite(slots).all()
    ):
        raise ValueError("D29 object-slot cache shape/content mismatch")
    for index, record in enumerate(records):
        record["_d29_slot_index"] = index
    return slots, mask, report


@torch.inference_mode()
def predict(
    model: nn.Module,
    records: list[dict[str, Any]],
    slots: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ObjectSlotDataset(
        records,
        slots,
        mask,
        train=False,
        seed=SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    current_rows = []
    history_rows = []
    model.eval()
    for batch_slots, batch_mask, _, _, _ in loader:
        current, history = model(
            batch_slots.to(device, non_blocking=True),
            batch_mask.to(device, non_blocking=True),
        )
        current_rows.append(current.cpu().numpy())
        history_rows.append(history.cpu().numpy())
    return (
        np.concatenate(current_rows) * DISTANCE_CAP_M,
        np.concatenate(history_rows) * DISTANCE_CAP_M,
    )


def evaluate(
    records: list[dict[str, Any]],
    prediction_m: np.ndarray,
    arm: str,
) -> dict[str, Any]:
    result = evaluate_scores(records, prediction_m)
    result["teacher_fit"] = teacher_fit(records, prediction_m, arm)
    return result


def train_fold(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    slots: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    nn.Module,
]:
    seed_everything(SEED)
    model = ObjectSlotMotionResidual().to(device)
    initial_sha256 = model_sha256(model)
    dataset = ObjectSlotDataset(
        train_records,
        slots,
        mask,
        train=True,
        seed=SEED,
    )
    sample_weight = source_weights(train_records)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    epoch_rows = []
    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(SEED * 1000 + epoch)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        model.train()
        total = 0.0
        batches = 0
        for (
            batch_slots,
            batch_mask,
            current_target,
            history_target,
            indices,
        ) in loader:
            batch_slots = batch_slots.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            current_target = current_target.to(device, non_blocking=True)
            history_target = history_target.to(device, non_blocking=True)
            weights = sample_weight[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            current, history = model(batch_slots, batch_mask)
            current_loss = nnf.smooth_l1_loss(
                current,
                current_target,
                beta=SMOOTH_L1_BETA,
                reduction="none",
            ).mean(dim=(1, 2))
            history_loss = nnf.smooth_l1_loss(
                history,
                history_target,
                beta=SMOOTH_L1_BETA,
                reduction="none",
            ).mean(dim=(1, 2))
            paired_loss = 0.5 * (current_loss + history_loss)
            loss = torch.sum(paired_loss * weights) / torch.sum(weights)
            if not torch.isfinite(loss):
                raise RuntimeError("D29 encountered non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        mean_loss = total / batches
        epoch_rows.append(mean_loss)
        if epoch in {1, EPOCHS}:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "mean_train_loss": mean_loss,
                    }
                ),
                flush=True,
            )
    current_m, history_m = predict(
        model,
        test_records,
        slots,
        mask,
        device,
    )
    return (
        evaluate(test_records, current_m, "current"),
        evaluate(test_records, history_m, "history"),
        {
            "initial_model_sha256": initial_sha256,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "fixed_final_epoch": EPOCHS,
            "first_epoch_loss": epoch_rows[0],
            "final_epoch_loss": epoch_rows[-1],
        },
        model,
    )


def build_gate(
    aggregate: dict[str, dict[str, Any]],
    monotonicity_violations: int,
    detector_coverage: float,
) -> dict[str, Any]:
    source_auroc = aggregate[
        "source_macro_direction_horizon_macro.auroc"
    ]
    source_ap = aggregate[
        "source_macro_direction_horizon_macro.average_precision"
    ]
    pooled_auroc = aggregate[
        "pooled_direction_horizon_macro.auroc"
    ]
    pooled_ap = aggregate[
        "pooled_direction_horizon_macro.average_precision"
    ]
    safe_choice = aggregate["safe_choice.source_macro_accuracy"]
    teacher_mae = aggregate["teacher_fit.source_macro_mae_m"]
    auroc_positive_directions = sum(
        aggregate[
            f"by_direction.{name}."
            "source_macro_horizon_macro.auroc"
        ]["mean"]
        > 0
        for name in DIRECTION_NAMES
    )
    ap_positive_directions = sum(
        aggregate[
            f"by_direction.{name}."
            "source_macro_horizon_macro.average_precision"
        ]["mean"]
        > 0
        for name in DIRECTION_NAMES
    )
    checks = {
        "detector_anchor_coverage": detector_coverage >= 0.80,
        "source_macro_auroc_effect": source_auroc["mean"] >= 0.010,
        "source_macro_ap_effect": source_ap["mean"] >= 0.005,
        "source_macro_auroc_positive_folds": (
            source_auroc["positive_folds"] >= 3
        ),
        "source_macro_ap_positive_folds": (
            source_ap["positive_folds"] >= 3
        ),
        "auroc_positive_directions": auroc_positive_directions >= 2,
        "ap_positive_directions": ap_positive_directions >= 2,
        "safe_choice_effect": safe_choice["mean"] >= 0.020,
        "safe_choice_positive_folds": (
            safe_choice["positive_folds"] >= 3
        ),
        "pooled_auroc_noninferiority": pooled_auroc["mean"] >= -0.005,
        "pooled_ap_noninferiority": pooled_ap["mean"] >= -0.005,
        "teacher_mae_noninferiority": teacher_mae["mean"] <= 0.25,
        "monotonicity_exact": monotonicity_violations == 0,
    }
    return {
        "frozen_thresholds": {
            "detector_anchor_coverage_floor": 0.80,
            "source_macro_auroc_mean_floor": 0.010,
            "source_macro_ap_mean_floor": 0.005,
            "positive_folds": 3,
            "positive_directions": 2,
            "safe_choice_mean_delta_floor": 0.020,
            "safe_choice_positive_folds": 3,
            "pooled_noninferiority_floor": -0.005,
            "history_minus_current_teacher_mae_ceiling_m": 0.25,
            "monotonicity_violations": 0,
        },
        "direction_breadth": {
            "auroc_positive_directions": auroc_positive_directions,
            "ap_positive_directions": ap_positive_directions,
        },
        "checks": checks,
        "supported": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--d8-samples", type=Path, default=DEFAULT_D8_SAMPLES)
    parser.add_argument(
        "--object-slots",
        type=Path,
        default=DEFAULT_OBJECT_SLOTS,
    )
    parser.add_argument(
        "--d27-report",
        type=Path,
        default=DEFAULT_D27_REPORT,
    )
    parser.add_argument(
        "--teacher-scores",
        type=Path,
        default=DEFAULT_D27_SCORES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("D29 student output is non-overwriting")
    if not torch.cuda.is_available():
        raise RuntimeError("D29 student requires CUDA")
    d27 = json.loads(args.d27_report.read_text(encoding="utf-8"))
    if d27["status"] != (
        "D27_THOR_MAGNI_HISTORY_KINEMATIC_INFORMATION_CEILING_SUPPORTED"
    ):
        raise ValueError("D29 requires supported D27")
    if d27["inputs"]["samples_sha256"] != sha256(args.samples):
        raise ValueError("D29 D27 sample binding mismatch")
    if d27["inputs"]["d8_samples_sha256"] != sha256(args.d8_samples):
        raise ValueError("D29 D27 D8 binding mismatch")
    if d27["inputs"]["oracle_scores_sha256"] != sha256(
        args.teacher_scores
    ):
        raise ValueError("D29 D27 score binding mismatch")
    records = prepare_records(
        load_jsonl(args.samples),
        load_jsonl(args.d8_samples),
    )
    bind_teacher_scores(records, args.teacher_scores)
    slots, mask, slot_report = load_object_slots(
        args.object_slots,
        records,
    )
    device = torch.device("cuda")
    paths = metric_paths()
    units = []
    checkpoints = []
    for fold in FOLDS:
        train_records = [
            record for record in records if int(record["fold"]) != fold
        ]
        test_records = [
            record for record in records if int(record["fold"]) == fold
        ]
        current, history, training, model = train_fold(
            train_records,
            test_records,
            slots,
            mask,
            device,
        )
        delta: dict[str, Any] = {}
        for path in paths:
            cursor = delta
            parts = path.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = (
                nested(history, path) - nested(current, path)
            )
        heldout_sources = sorted(
            {
                str(record["source_session_id"])
                for record in test_records
            }
        )
        checkpoint_path = (
            args.output.parent
            / "checkpoints"
            / f"fold-{fold}"
            / f"seed-{SEED}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.cpu()
        torch.save(
            {
                "schema": SCHEMA,
                "fold": fold,
                "seed": SEED,
                "heldout_source_sessions": heldout_sources,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        checkpoints.append(
            {
                "fold": fold,
                "seed": SEED,
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256(checkpoint_path),
            }
        )
        units.append(
            {
                "fold": fold,
                "seed": SEED,
                "heldout_source_sessions": heldout_sources,
                "current": current,
                "history": history,
                "history_minus_current": delta,
                "training": training,
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "source_macro_auroc_delta": delta[
                        "source_macro_direction_horizon_macro"
                    ]["auroc"],
                    "source_macro_ap_delta": delta[
                        "source_macro_direction_horizon_macro"
                    ]["average_precision"],
                    "safe_choice_delta": delta["safe_choice"][
                        "source_macro_accuracy"
                    ],
                    "teacher_mae_delta_m": delta["teacher_fit"][
                        "source_macro_mae_m"
                    ],
                }
            ),
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    aggregate = {
        path: summarize_delta(units, path) for path in paths
    }
    monotonicity_violations = sum(
        int(unit[arm]["monotonicity_violations"])
        for unit in units
        for arm in ("current", "history")
    )
    detector_coverage = float(slot_report["counts"]["anchor_coverage"])
    gate = build_gate(
        aggregate,
        monotonicity_violations,
        detector_coverage,
    )
    status = (
        "D29_THOR_MAGNI_OBJECT_SLOT_MOTION_RESIDUAL_INCREMENT_SUPPORTED"
        if gate["supported"]
        else (
            "D29_THOR_MAGNI_OBJECT_SLOT_MOTION_RESIDUAL_"
            "INCREMENT_NOT_SUPPORTED"
        )
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development source-heldout explicit object-motion "
                "bottleneck canary"
            ),
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "d8_samples_path": str(args.d8_samples.resolve()),
            "d8_samples_sha256": sha256(args.d8_samples),
            "object_slots_path": str(args.object_slots.resolve()),
            "object_slots_sha256": sha256(args.object_slots),
            "object_slots_report_sha256": sha256(
                args.object_slots.with_suffix(
                    args.object_slots.suffix + ".json"
                )
            ),
            "d27_report_path": str(args.d27_report.resolve()),
            "d27_report_sha256": sha256(args.d27_report),
            "teacher_scores_path": str(args.teacher_scores.resolve()),
            "teacher_scores_sha256": sha256(args.teacher_scores),
        },
        "design": {
            "folds": len(FOLDS),
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "source-balanced paired Smooth L1",
            "smooth_l1_beta_normalized": SMOOTH_L1_BETA,
            "smooth_l1_beta_m": SMOOTH_L1_BETA * DISTANCE_CAP_M,
            "selection": "fixed final epoch",
            "representation": (
                "masked mean/max DeepSets static base plus "
                "zero-initialized motion residual"
            ),
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(
                {str(record["source_session_id"]) for record in records}
            ),
            "paired_units": len(units),
            "training_runs": len(units),
            "monotonicity_violations": monotonicity_violations,
            "detector_anchor_coverage": detector_coverage,
        },
        "object_slot_opportunity": slot_report["counts"],
        "checkpoints": checkpoints,
        "units": units,
        "aggregate_fold_mean_history_minus_current": aggregate,
        "gate": gate,
        "next_action": (
            "replicate the explicit object-slot mechanism"
            if gate["supported"]
            else (
                "retain D27 information ceiling and stop the frozen "
                "YOLO-box plus within-box-flow residual recipe"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_hash = sha256(args.output)
    sidecar.write_text(
        f"{report_hash}  {args.output.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": status,
                "gate": gate,
                "aggregate": aggregate,
                "report_sha256": report_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

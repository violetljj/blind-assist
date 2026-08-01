from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.research.riskseg_r0_pidnet_preflight.modeling import (
    CLASS_ORDER,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    build_pidnet_s,
    load_imagenet_backbone,
    official_repo_commit,
    set_deterministic_seed,
    sha256_file,
)


ALLOWED_SEEDS = (20260801, 20260802, 20260803)
BATCH_SIZE = 4
MAX_EPOCHS = 200
MIN_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 30
BASE_LR = 0.005
POLY_POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
OHEM_THRESHOLD = 0.9
OHEM_MIN_KEPT = 131072
SEMANTIC_BALANCE_WEIGHTS = (0.4, 1.0)
BOUNDARY_BCE_COEFFICIENT = 20.0
SEMANTIC_BOUNDARY_THRESHOLD = 0.8
IGNORE_LABEL = 255
BASE_SIZE = 512
SCALE_FACTOR = 16
EDGE_SIZE = 4
EDGE_PAD = 6
NUM_CLASSES = 4


@dataclass(frozen=True)
class Row:
    row_id: str
    role: str
    session_id: str
    image_path: Path
    mask_path: Path
    image_sha256: str
    mask_sha256: str


def load_rows(
    *,
    manifest_path: Path,
    view_root: Path,
    repo_root: Path,
    role: str,
) -> list[Row]:
    rows: list[Row] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload["role"] != role:
            continue
        rows.append(
            Row(
                row_id=payload["id"],
                role=payload["role"],
                session_id=payload["source_session_id"],
                image_path=repo_root / payload["source_image_path"],
                mask_path=view_root / payload["output_mask_path"],
                image_sha256=payload["source_image_sha256"],
                mask_sha256=payload["output_mask_sha256"],
            )
        )
    expected = 320 if role == "train" else 200
    if len(rows) != expected:
        raise ValueError(f"{role} row count {len(rows)} != {expected}")
    for row in rows:
        if not row.image_path.is_file() or not row.mask_path.is_file():
            raise FileNotFoundError(f"missing image/mask for {row.row_id}")
    return rows


class RisksegDataset(Dataset):
    def __init__(self, rows: list[Row], *, training: bool) -> None:
        self.rows = rows
        self.training = training
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row.image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
        with Image.open(row.mask_path) as mask_file:
            mask = np.asarray(mask_file.convert("L"), dtype=np.uint8)
        unique = np.unique(mask)
        if np.any(unique >= NUM_CLASSES):
            raise ValueError(f"{row.row_id} contains mask ids {unique.tolist()}")
        edge = make_boundary(mask)
        if self.training:
            image, mask, edge = self._augment(image, mask, edge)
        else:
            image = cv2.resize(
                image,
                (INPUT_WIDTH, INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask,
                (INPUT_WIDTH, INPUT_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
            edge = make_boundary(mask)
        normalized = image.astype(np.float32) / np.float32(255.0)
        normalized = (normalized - self.mean) / self.std
        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(normalized.transpose(2, 0, 1))
            ),
            "mask": torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64))),
            "boundary": torch.from_numpy(
                np.ascontiguousarray(edge.astype(np.float32))
            ),
            "row_id": row.row_id,
            "session_id": row.session_id,
        }

    def _augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        edge: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        random_scale = 0.5 + random.randint(0, SCALE_FACTOR) / 10.0
        long_size = int(BASE_SIZE * random_scale + 0.5)
        height, width = image.shape[:2]
        if height > width:
            new_height = long_size
            new_width = int(width * long_size / height + 0.5)
        else:
            new_width = long_size
            new_height = int(height * long_size / width + 0.5)
        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )
        edge = cv2.resize(
            edge,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )
        pad_height = max(INPUT_HEIGHT - new_height, 0)
        pad_width = max(INPUT_WIDTH - new_width, 0)
        if pad_height or pad_width:
            image = cv2.copyMakeBorder(
                image,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            mask = cv2.copyMakeBorder(
                mask,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=IGNORE_LABEL,
            )
            edge = cv2.copyMakeBorder(
                edge,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=0,
            )
        max_y = mask.shape[0] - INPUT_HEIGHT
        max_x = mask.shape[1] - INPUT_WIDTH
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        image = image[y : y + INPUT_HEIGHT, x : x + INPUT_WIDTH]
        mask = mask[y : y + INPUT_HEIGHT, x : x + INPUT_WIDTH]
        edge = edge[y : y + INPUT_HEIGHT, x : x + INPUT_WIDTH]
        if np.random.choice(2) == 1:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
            edge = edge[:, ::-1]
        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
            np.ascontiguousarray(edge),
        )


def make_boundary(mask: np.ndarray) -> np.ndarray:
    edge = cv2.Canny(mask, 0.1, 0.2)
    kernel = np.ones((EDGE_SIZE, EDGE_SIZE), np.uint8)
    if mask.shape[0] > EDGE_PAD * 2 and mask.shape[1] > EDGE_PAD * 2:
        edge = edge[EDGE_PAD:-EDGE_PAD, EDGE_PAD:-EDGE_PAD]
        edge = np.pad(
            edge,
            ((EDGE_PAD, EDGE_PAD), (EDGE_PAD, EDGE_PAD)),
            mode="constant",
        )
    return (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)


class OhemCrossEntropy(nn.Module):
    """The official PIDNet OHEM semantic loss with frozen RISKSEG-R0 constants."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=IGNORE_LABEL,
            reduction="none",
        )

    def ordinary(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(score, target).mean()

    def ohem(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        valid = target.contiguous().view(-1) != IGNORE_LABEL
        safe_target = target.clone()
        safe_target[safe_target == IGNORE_LABEL] = 0
        selected_probabilities = probabilities.gather(1, safe_target.unsqueeze(1))
        selected_probabilities, indices = (
            selected_probabilities.contiguous().view(-1)[valid].contiguous().sort()
        )
        if selected_probabilities.numel() == 0:
            raise ValueError("OHEM received no valid pixels")
        min_value = selected_probabilities[
            min(OHEM_MIN_KEPT, selected_probabilities.numel() - 1)
        ]
        threshold = max(float(min_value.detach().cpu()), OHEM_THRESHOLD)
        selected_losses = pixel_losses[valid][indices]
        selected_losses = selected_losses[selected_probabilities < threshold]
        if selected_losses.numel() == 0:
            raise ValueError("OHEM selected no pixels")
        return selected_losses.mean()

    def semantic_outputs(
        self,
        outputs: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if len(outputs) != 2:
            raise ValueError(f"expected two semantic outputs, got {len(outputs)}")
        return (
            SEMANTIC_BALANCE_WEIGHTS[0] * self.ordinary(outputs[0], target)
            + SEMANTIC_BALANCE_WEIGHTS[1] * self.ohem(outputs[1], target)
        )


def boundary_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = prediction.permute(0, 2, 3, 1).contiguous().view(1, -1)
    flat_target = target.view(1, -1)
    positive = flat_target == 1
    negative = flat_target == 0
    positive_count = positive.sum()
    negative_count = negative.sum()
    total = positive_count + negative_count
    if total == 0:
        raise ValueError("boundary target contains no binary pixels")
    weights = torch.zeros_like(logits)
    weights[positive] = negative_count.float() / total.float()
    weights[negative] = positive_count.float() / total.float()
    return BOUNDARY_BCE_COEFFICIENT * F.binary_cross_entropy_with_logits(
        logits,
        flat_target,
        weights,
        reduction="mean",
    )


def resize_outputs(
    outputs: list[torch.Tensor],
    target: torch.Tensor,
) -> list[torch.Tensor]:
    size = target.shape[-2:]
    return [
        output
        if output.shape[-2:] == size
        else F.interpolate(
            output,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        for output in outputs
    ]


def compute_loss(
    outputs: list[torch.Tensor],
    target: torch.Tensor,
    boundary_target: torch.Tensor,
    semantic_loss: OhemCrossEntropy,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(outputs) != 3:
        raise ValueError(f"expected aux/final/boundary outputs, got {len(outputs)}")
    aux, final, boundary = resize_outputs(outputs, target)
    semantic = semantic_loss.semantic_outputs([aux, final], target)
    boundary_value = boundary_loss(boundary, boundary_target)
    filler = torch.full_like(target, IGNORE_LABEL)
    boundary_semantic_target = torch.where(
        torch.sigmoid(boundary[:, 0]) > SEMANTIC_BOUNDARY_THRESHOLD,
        target,
        filler,
    )
    semantic_boundary = semantic_loss.ohem(final, boundary_semantic_target)
    total = semantic + boundary_value + semantic_boundary
    return total, {
        "semantic": float(semantic.detach().cpu()),
        "boundary": float(boundary_value.detach().cpu()),
        "semantic_boundary": float(semantic_boundary.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def confusion_update(
    confusion: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    valid = target != IGNORE_LABEL
    encoded = NUM_CLASSES * target[valid].astype(np.int64) + prediction[valid]
    confusion += np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    true_positive = np.diag(confusion)
    ground_truth = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=union > 0,
    )
    return {
        "mean_iou": float(iou.mean()),
        "per_class_iou": {
            name: float(iou[index]) for index, name in enumerate(CLASS_ORDER)
        },
        "pixel_accuracy": float(
            true_positive.sum() / max(1, ground_truth.sum())
        ),
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def boundary_f1(
    truth_masks: list[np.ndarray],
    predicted_masks: list[np.ndarray],
) -> float:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    kernel = np.ones((3, 3), np.uint8)
    for truth, predicted in zip(truth_masks, predicted_masks, strict=True):
        truth_edge = make_boundary(truth).astype(np.uint8)
        predicted_edge = make_boundary(predicted).astype(np.uint8)
        truth_tolerance = cv2.dilate(truth_edge, kernel, iterations=1) > 0
        predicted_tolerance = cv2.dilate(predicted_edge, kernel, iterations=1) > 0
        predicted_positive = predicted_edge > 0
        truth_positive = truth_edge > 0
        true_positive += int(np.logical_and(predicted_positive, truth_tolerance).sum())
        false_positive += int(
            np.logical_and(predicted_positive, ~truth_tolerance).sum()
        )
        false_negative += int(
            np.logical_and(truth_positive, ~predicted_tolerance).sum()
        )
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    by_session: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    )
    truth_masks: list[np.ndarray] = []
    predicted_masks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].numpy()
            outputs = model(images)
            final = outputs[-2]
            if final.shape[-2:] != (INPUT_HEIGHT, INPUT_WIDTH):
                final = F.interpolate(
                    final,
                    size=(INPUT_HEIGHT, INPUT_WIDTH),
                    mode="bilinear",
                    align_corners=False,
                )
            prediction = final.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for index, session_id in enumerate(batch["session_id"]):
                confusion_update(confusion, target[index], prediction[index])
                confusion_update(
                    by_session[session_id],
                    target[index],
                    prediction[index],
                )
                truth_masks.append(target[index].astype(np.uint8))
                predicted_masks.append(prediction[index])
    metrics = metrics_from_confusion(confusion)
    metrics["boundary_f1_tolerance_1px"] = boundary_f1(
        truth_masks,
        predicted_masks,
    )
    metrics["per_session"] = {
        session_id: metrics_from_confusion(matrix)
        for session_id, matrix in sorted(by_session.items())
    }
    metrics["worst_session_mean_iou"] = min(
        item["mean_iou"] for item in metrics["per_session"].values()
    )
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}")
    set_deterministic_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("formal RISKSEG-R0 training requires CUDA")

    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest.resolve()
    view_root = args.view_root.resolve()
    official_repo = args.official_repo.resolve()
    pretrained = args.pretrained.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(
        manifest_path=manifest_path,
        view_root=view_root,
        repo_root=repo_root,
        role="train",
    )
    dev_rows = load_rows(
        manifest_path=manifest_path,
        view_root=view_root,
        repo_root=repo_root,
        role="dev",
    )
    train_sessions = sorted({row.session_id for row in train_rows})
    dev_sessions = sorted({row.session_id for row in dev_rows})
    if set(train_sessions) & set(dev_sessions):
        raise ValueError("train/dev source sessions overlap")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RisksegDataset(train_rows, training=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    dev_loader = DataLoader(
        RisksegDataset(dev_rows, training=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = build_pidnet_s(official_repo=official_repo, augment=True)
    pretrained_load = load_imagenet_backbone(
        model=model,
        checkpoint_path=pretrained,
    )
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=False,
    )
    semantic_loss = OhemCrossEntropy()
    total_steps = MAX_EPOCHS * len(train_loader)
    global_step = 0
    best_epoch = 0
    best_mean_iou = -math.inf
    best_dev_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history_path = output_dir / "epoch_metrics.jsonl"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    started_at = datetime.now(timezone.utc)
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            losses: list[dict[str, float]] = []
            for batch in train_loader:
                lr = BASE_LR * (1.0 - global_step / total_steps) ** POLY_POWER
                for group in optimizer.param_groups:
                    group["lr"] = lr
                images = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                boundary_target = batch["boundary"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss, loss_parts = compute_loss(
                    outputs,
                    target,
                    boundary_target,
                    semantic_loss,
                )
                if not loss.isfinite():
                    raise ValueError(f"non-finite training loss at step {global_step}")
                loss.backward()
                optimizer.step()
                losses.append(loss_parts)
                global_step += 1

            dev_metrics = evaluate(model, dev_loader, device)
            mean_iou = dev_metrics["mean_iou"]
            improved = mean_iou > best_mean_iou + 1e-6
            if improved:
                best_mean_iou = mean_iou
                best_epoch = epoch
                best_dev_metrics = dev_metrics
                epochs_without_improvement = 0
                torch.save(
                    {
                        "schema_version": "blindassist.riskseg_r0.pidnet_checkpoint.v1",
                        "protocol_id": "RISKSEG-R0",
                        "seed": args.seed,
                        "epoch": epoch,
                        "class_order": list(CLASS_ORDER),
                        "model_state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "dev_metrics": dev_metrics,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate_last_step": lr,
                "train_loss": {
                    name: float(np.mean([item[name] for item in losses]))
                    for name in ("semantic", "boundary", "semantic_boundary", "total")
                },
                "dev": dev_metrics,
                "best_epoch": best_epoch,
                "best_mean_iou": best_mean_iou,
                "improved": improved,
            }
            history.write(json.dumps(epoch_row, sort_keys=True) + "\n")
            history.flush()
            print(
                json.dumps(
                    {
                        "seed": args.seed,
                        "epoch": epoch,
                        "train_total_loss": epoch_row["train_loss"]["total"],
                        "dev_mean_iou": mean_iou,
                        "dev_boundary_f1": dev_metrics["boundary_f1_tolerance_1px"],
                        "best_epoch": best_epoch,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                epoch >= MIN_EPOCHS
                and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
            ):
                break

    if best_dev_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("training completed without a best checkpoint")
    finished_at = datetime.now(timezone.utc)
    report = {
        "schema_version": "blindassist.riskseg_r0.pidnet_training.v1",
        "protocol_id": "RISKSEG-R0",
        "status": "TRAINING_COMPLETE_DEV_SELECTED",
        "seed": args.seed,
        "decision_seed": args.seed == ALLOWED_SEEDS[0],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "official_repo_commit": official_repo_commit(official_repo),
        "pretrained_sha256": sha256_file(pretrained),
        "data": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "train_frames": len(train_rows),
            "dev_frames": len(dev_rows),
            "train_sessions": train_sessions,
            "dev_sessions": dev_sessions,
            "session_overlap": [],
        },
        "recipe": recipe_json(),
        "epochs_completed": epoch,
        "global_steps": global_step,
        "stop_reason": (
            "DEV_MIOU_EARLY_STOPPING"
            if epoch < MAX_EPOCHS
            else "MAX_EPOCHS_REACHED"
        ),
        "best_epoch": best_epoch,
        "best_dev_metrics": best_dev_metrics,
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_path": history_path.name,
        "history_sha256": sha256_file(history_path),
        "pretrained_load": pretrained_load,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "automatic_mixed_precision": False,
        },
        "event_eval_outcome_accessed_by_trainer": False,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def recipe_json() -> dict[str, Any]:
    return {
        "architecture": "official_PIDNet-S",
        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
        "class_order": list(CLASS_ORDER),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "minimum_epochs": MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_rule": "maximum_dev_mean_iou_then_earliest_epoch",
        "optimizer": "SGD",
        "base_learning_rate": BASE_LR,
        "schedule": f"poly_power_{POLY_POWER}",
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": False,
        "augmentation": {
            "official_random_scale": "0.5_to_2.1_in_0.1_steps",
            "base_size": BASE_SIZE,
            "random_crop": [INPUT_HEIGHT, INPUT_WIDTH],
            "horizontal_flip_probability": 0.5,
            "color_jitter": False,
            "rotation": False,
        },
        "loss": {
            "semantic_outputs": ["auxiliary", "final"],
            "semantic_balance_weights": list(SEMANTIC_BALANCE_WEIGHTS),
            "final_ohem_threshold": OHEM_THRESHOLD,
            "final_ohem_min_kept": OHEM_MIN_KEPT,
            "boundary_bce_coefficient": BOUNDARY_BCE_COEFFICIENT,
            "semantic_boundary_threshold": SEMANTIC_BOUNDARY_THRESHOLD,
            "class_balance": False,
        },
        "boundary_target": {
            "source": "class_transition_edges",
            "official_canny_then_dilate": True,
            "dilation_kernel": [EDGE_SIZE, EDGE_SIZE],
            "edge_pad_pixels": EDGE_PAD,
        },
        "selection_metrics": [
            "dev_mean_iou",
            "per_class_iou",
            "boundary_f1_tolerance_1px",
            "worst_session_mean_iou",
        ],
        "forbidden": [
            "fp_sampler",
            "class_weight_tuning",
            "manual_gate",
            "component_classifier",
            "event_eval_feedback",
            "seed_selection_on_event_eval",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

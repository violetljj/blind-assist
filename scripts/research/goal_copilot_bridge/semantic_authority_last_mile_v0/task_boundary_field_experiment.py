"""Train and evaluate the SAGE-LM V1-C task-specific aperture boundary field.

Training uses public TartanAir source-native door masks.  The already-opened
24-episode ARKitScenes cohort is evaluation-only.  C0 and C1 share the same
RGB encoder/decoder; only C1 receives a target-anchor heatmap in the decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .experiment import _aggregate
from .rgb_experiment import _baseline, _sage_lm
from .two_view_experiment import _arm_diagnostics, _evaluator_episode, _source_poses
from .two_view_observation import (
    ImageLine,
    SourcePoseTwoViewBoundaryProvider,
    _image_line_from_points,
    _intrinsic_matrix,
    _line_distance,
    oracle_pixel_lines,
    triangulate_aperture,
)


SCHEMA_VERSION = "sage_lm_v1c_task_specific_aperture_boundary_field_v1"
MODEL_HEIGHT = 192
MODEL_WIDTH = 256
TOP_K_PER_ROLE = 8
LOCALIZATION_GATE_PX = 9.0
NMS_RADIUS_MODEL_PX = 5
DEFAULT_EPOCHS = 24
DEFAULT_BATCH_SIZE = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _component(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    choices = [(int(stats[index, cv2.CC_STAT_AREA]), index) for index in range(1, count)]
    if not choices:
        return None
    area, index = max(choices)
    if area < 64:
        return None
    x, y, width, height = (int(value) for value in stats[index, :4])
    selected = (labels == index).astype(np.uint8)
    return selected, (x, y, x + width - 1, y + height - 1)


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_NEAREST)


def boundary_targets(mask: np.ndarray, sigma_px: float = 2.5) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Return two soft heatmaps and the selected component's column targets."""

    resized = _resize_mask(mask)
    selected = _component(resized)
    targets = np.zeros((2, MODEL_HEIGHT, MODEL_WIDTH), dtype=np.float32)
    if selected is None:
        return targets, None
    component, bbox = selected
    xs = np.arange(MODEL_WIDTH, dtype=np.float32)
    left_values = []
    right_values = []
    for y in range(MODEL_HEIGHT):
        columns = np.flatnonzero(component[y])
        if columns.size == 0:
            continue
        left = float(columns[0])
        right = float(columns[-1])
        left_values.append(left)
        right_values.append(right)
        targets[0, y] = np.maximum(targets[0, y], np.exp(-0.5 * ((xs - left) / sigma_px) ** 2))
        targets[1, y] = np.maximum(targets[1, y], np.exp(-0.5 * ((xs - right) / sigma_px) ** 2))
    if not left_values:
        return np.zeros_like(targets), None
    return targets, (float(np.median(left_values)), float(np.median(right_values)))


def synthetic_anchor(bbox: tuple[int, int, int, int], key: str) -> np.ndarray:
    """Create a deterministic controlled target anchor near the selected door."""

    x1, y1, x2, y2 = bbox
    width = max(8, x2 - x1 + 1)
    digest = hashlib.sha256(key.encode()).digest()
    relation = digest[0] % 3
    if relation == 0:
        center_x = (x1 + x2) * 0.5
    elif relation == 1:
        center_x = x1 - width * 0.28
    else:
        center_x = x2 + width * 0.28
    center_y = y1 + max(6.0, (y2 - y1) * 0.30)
    half = max(3, int(round(MODEL_WIDTH * 0.018)))
    xa = int(np.clip(round(center_x) - half, 0, MODEL_WIDTH - 1))
    xb = int(np.clip(round(center_x) + half, 0, MODEL_WIDTH - 1))
    ya = int(np.clip(round(center_y) - half, 0, MODEL_HEIGHT - 1))
    yb = int(np.clip(round(center_y) + half, 0, MODEL_HEIGHT - 1))
    heatmap = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.float32)
    heatmap[ya : yb + 1, xa : xb + 1] = 1.0
    return heatmap


@dataclass(frozen=True)
class DoorRecord:
    image: Path
    mask: Path
    kind: str
    split: str


class DoorDataset:
    def __init__(self, torch, records: list[DoorRecord], conditioned: bool, augment: bool) -> None:
        self.torch = torch
        self.records = records
        self.conditioned = conditioned
        self.augment = augment
        self.cache = []
        for record in records:
            bgr = cv2.imread(str(record.image), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(record.mask), cv2.IMREAD_UNCHANGED)
            if bgr is None or mask is None:
                raise ValueError(f"unable to decode training pair: {record.image}")
            bgr = cv2.resize(bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA)
            mask = _resize_mask(mask)
            targets, columns = boundary_targets(mask)
            selected = _component(mask)
            anchor = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.uint8)
            if self.conditioned and selected is not None:
                anchor = (synthetic_anchor(selected[1], record.image.name) * 255).astype(np.uint8)
            column_target = (-1, -1) if columns is None else tuple(int(round(value)) for value in columns)
            self.cache.append((bgr, anchor, targets.astype(np.float16), column_target))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        bgr, anchor_u8, targets_f16, column_target = self.cache[index]
        bgr = bgr.copy()
        anchor = anchor_u8.astype(np.float32) / 255.0
        targets = targets_f16.astype(np.float32)
        columns = None if column_target[0] < 0 else column_target
        if self.augment and random.random() < 0.5:
            bgr = bgr[:, ::-1].copy()
            anchor = anchor[:, ::-1].copy()
            targets = targets[::-1, :, ::-1].copy()
            if columns is not None:
                columns = (MODEL_WIDTH - 1 - columns[1], MODEL_WIDTH - 1 - columns[0])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray([0.229, 0.224, 0.225], np.float32)
        column_target = (-1, -1) if columns is None else tuple(int(round(value)) for value in columns)
        return (
            self.torch.from_numpy(rgb.transpose(2, 0, 1)),
            self.torch.from_numpy(anchor[None]),
            self.torch.from_numpy(targets),
            self.torch.as_tensor(column_target, dtype=self.torch.long),
        )


def make_model(torch):
    class Block(torch.nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.GELU(),
                torch.nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.GELU(),
            )

        def forward(self, value):
            return self.layers(value)

    class BoundaryField(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e1, self.e2, self.e3, self.e4 = Block(3, 24), Block(24, 40), Block(40, 64), Block(64, 96)
            self.pool = torch.nn.MaxPool2d(2)
            self.d3 = Block(96 + 64 + 1, 64)
            self.d2 = Block(64 + 40 + 1, 40)
            self.d1 = Block(40 + 24 + 1, 24)
            self.output = torch.nn.Conv2d(24, 2, 1)

        def forward(self, rgb, anchor):
            e1 = self.e1(rgb)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))

            def up(value, skip, block):
                value = torch.nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                anchor_at_scale = torch.nn.functional.interpolate(anchor, size=skip.shape[-2:], mode="nearest")
                return block(torch.cat([value, skip, anchor_at_scale], dim=1))

            return self.output(up(up(up(e4, e3, self.d3), e2, self.d2), e1, self.d1))

    return BoundaryField()


def _records(dataset_root: Path) -> tuple[list[DoorRecord], list[DoorRecord], dict]:
    receipt_path = dataset_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = [DoorRecord(Path(row["image_path"]), Path(row["mask_path"]), row["kind"], row["split"]) for row in receipt["cases"]]
    return (
        [row for row in rows if row.split == "train"],
        [row for row in rows if row.split == "val"],
        {"receipt": str(receipt_path.resolve()), "receipt_sha256": _sha256(receipt_path), "case_count": len(rows), "train_count": receipt["train_count"], "val_count": receipt["val_count"], "source_environments": receipt["source_environments"], "excluded_formal_environments": receipt["excluded_formal_environments"], "label_provenance": receipt.get("label_provenance", "SOURCE_NATIVE_SEMANTIC_MASK")},
    )


def _loss(torch, logits, targets, columns):
    weight = 1.0 + 10.0 * targets
    heatmap = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weight)
    profile = logits.amax(dim=2)
    valid = columns >= 0
    if bool(valid.any()):
        column = torch.nn.functional.cross_entropy(profile[valid], columns[valid])
    else:
        column = logits.sum() * 0.0
    return heatmap + 0.35 * column


def _select_peaks(profile: np.ndarray, top_k: int = TOP_K_PER_ROLE) -> list[int]:
    working = profile.astype(np.float64).copy()
    selected = []
    for _ in range(top_k):
        index = int(np.argmax(working))
        selected.append(index)
        working[max(0, index - NMS_RADIUS_MODEL_PX) : min(len(working), index + NMS_RADIUS_MODEL_PX + 1)] = -math.inf
    return selected


def validate_model(torch, model, loader, device) -> dict:
    model.eval()
    hits = np.zeros(2, dtype=np.int64)
    both_hits = 0
    positives = 0
    losses = []
    with torch.inference_mode():
        for rgb, anchor, targets, columns in loader:
            rgb, anchor, targets, columns = rgb.to(device), anchor.to(device), targets.to(device), columns.to(device)
            logits = model(rgb, anchor)
            losses.append(float(_loss(torch, logits, targets, columns)))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for batch_index, target in enumerate(columns.cpu().numpy()):
                if target[0] < 0:
                    continue
                positives += 1
                row_hits = []
                for role in range(2):
                    profile = 0.7 * probabilities[batch_index, role].max(axis=0) + 0.3 * probabilities[batch_index, role].mean(axis=0)
                    hit = min(abs(index - target[role]) for index in _select_peaks(profile)) <= 9
                    hits[role] += hit
                    row_hits.append(hit)
                both_hits += all(row_hits)
    return {"loss": float(np.mean(losses)), "positive_count": positives, "left_recall_at_8": float(hits[0] / positives), "right_recall_at_8": float(hits[1] / positives), "both_recall_at_8": float(both_hits / positives)}


def train_arm(torch, dataset_root: Path, conditioned: bool, output: Path, epochs: int, batch_size: int) -> dict:
    train_records, val_records, source = _records(dataset_root)
    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(torch).to(device)
    train_loader = torch.utils.data.DataLoader(DoorDataset(torch, train_records, conditioned, True), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(DoorDataset(torch, val_records, conditioned, False), batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best = None
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for rgb, anchor, targets, columns in train_loader:
            rgb, anchor, targets, columns = rgb.to(device, non_blocking=True), anchor.to(device, non_blocking=True), targets.to(device, non_blocking=True), columns.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(torch, model(rgb, anchor), targets, columns)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = validate_model(torch, model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **validation}
        history.append(row)
        print(json.dumps({"arm": "V1-C1" if conditioned else "V1-C0", **row}), flush=True)
        if best is None or validation["loss"] < best[0]:
            best = (validation["loss"], {key: value.detach().cpu() for key, value in model.state_dict().items()}, row)
    assert best is not None
    model.load_state_dict(best[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "conditioned": conditioned, "schema_version": SCHEMA_VERSION, "model_size": [MODEL_HEIGHT, MODEL_WIDTH]}, output)
    return {"checkpoint": str(output.resolve()), "checkpoint_sha256": _sha256(output), "conditioned": conditioned, "epochs": epochs, "batch_size": batch_size, "best_epoch": best[2], "history": history, "training_source": source, "device": str(device)}


class BoundaryFieldInference:
    def __init__(self, torch, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.torch = torch
        self.conditioned = bool(payload["conditioned"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = make_model(torch).to(self.device).eval()
        self.model.load_state_dict(payload["state_dict"])

    def predict(self, bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        image = bgr.copy()
        x1, y1, x2, y2 = bbox
        image[max(0, y1 - 3) : min(image.shape[0], y2 + 4), max(0, x1 - 3) : min(image.shape[1], x2 + 4)] = 0
        resized = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray([0.229, 0.224, 0.225], np.float32)
        anchor = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.float32)
        if self.conditioned:
            sx, sy = MODEL_WIDTH / bgr.shape[1], MODEL_HEIGHT / bgr.shape[0]
            xa, xb = int(np.clip(round(x1 * sx), 0, MODEL_WIDTH - 1)), int(np.clip(round(x2 * sx), 0, MODEL_WIDTH - 1))
            ya, yb = int(np.clip(round(y1 * sy), 0, MODEL_HEIGHT - 1)), int(np.clip(round(y2 * sy), 0, MODEL_HEIGHT - 1))
            anchor[min(ya, yb) : max(ya, yb) + 1, min(xa, xb) : max(xa, xb) + 1] = 1.0
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(self.device), self.torch.from_numpy(anchor)[None, None].to(self.device))[0]
        return self.torch.sigmoid(logits).cpu().numpy()


def _line_candidates(heatmap: np.ndarray, image_width: int, image_height: int) -> list[ImageLine]:
    profile = 0.7 * heatmap.max(axis=0) + 0.3 * heatmap.mean(axis=0)
    candidates = []
    for peak in _select_peaks(profile):
        points = []
        weights = []
        for y in range(MODEL_HEIGHT):
            xa, xb = max(0, peak - 7), min(MODEL_WIDTH, peak + 8)
            local = heatmap[y, xa:xb]
            if local.size == 0 or float(local.max()) < 0.05:
                continue
            x = xa + int(np.argmax(local))
            points.append((x * image_width / MODEL_WIDTH, y * image_height / MODEL_HEIGHT))
            weights.append(float(local.max()))
        if len(points) >= 8:
            fitted = _image_line_from_points(points, sum(weights), len(points))
        else:
            x = peak * image_width / MODEL_WIDTH
            fitted = _image_line_from_points([(x, 0.0), (x, image_height - 1.0)], float(profile[peak]), 1)
        candidates.append(ImageLine(fitted.coefficients, float(profile[peak] * image_height), len(points)))
    return candidates


class TaskBoundaryProvider(SourcePoseTwoViewBoundaryProvider):
    def __init__(self, episode_input, truth, pose_a, pose_b, inference: BoundaryFieldInference, arm: str) -> None:
        super().__init__(episode_input, truth, pose_a, pose_b, "b1")
        self.inference = inference
        self.arm_name = arm

    def observe(self):
        visible = [row for row in self.input.exact_anchor_observations if row.visible]
        first = next(row for row in visible if row.frame_index == 0)
        second = next(row for row in visible if row.frame_index == self.input.active_parallax_frame_index)
        images = [self._load(self.input.rgb_frames[first.frame_index]), self._load(self.input.rgb_frames[second.frame_index])]
        heatmaps = [self.inference.predict(images[0], first.bbox_xyxy), self.inference.predict(images[1], second.bbox_xyxy)]
        roles = [[_line_candidates(heatmaps[frame][role], self.input.intrinsics.width, self.input.intrinsics.height) for role in range(2)] for frame in range(2)]
        oracle_a, oracle_b = oracle_pixel_lines(self.input, self.truth, self.pose_a, self.pose_b)
        intrinsic = _intrinsic_matrix(self.input)
        height, width = self.input.intrinsics.height, self.input.intrinsics.width
        candidates = []
        for left_a in roles[0][0]:
            for right_a in roles[0][1]:
                span_a = right_a.x_at(height * 0.5) - left_a.x_at(height * 0.5)
                if not width * 0.10 <= span_a <= width * 0.70:
                    continue
                for left_b in roles[1][0]:
                    for right_b in roles[1][1]:
                        span_b = right_b.x_at(height * 0.5) - left_b.x_at(height * 0.5)
                        if not width * 0.10 <= span_b <= width * 0.70:
                            continue
                        geometry = triangulate_aperture(left_a, right_a, left_b, right_b, self.pose_a, self.pose_b, intrinsic, height * 0.55)
                        if geometry is None:
                            continue
                        distances = [_line_distance(left_a, oracle_a[0], height), _line_distance(right_a, oracle_a[1], height), _line_distance(left_b, oracle_b[0], height), _line_distance(right_b, oracle_b[1], height)]
                        candidates.append((sum(distances), distances, geometry, (left_a, right_a, left_b, right_b)))
        selected = min(candidates, key=lambda row: row[0], default=None)
        distances = [] if selected is None else selected[1]
        direct_hits = [
            min(_line_distance(line, oracle, height) for line in pool) <= LOCALIZATION_GATE_PX
            for pool, oracle in zip((roles[0][0], roles[0][1], roles[1][0], roles[1][1]), (*oracle_a, *oracle_b))
        ]
        self.diagnostics.update({"top_k_per_role": TOP_K_PER_ROLE, "role_candidate_counts": [len(pool) for frame in roles for pool in frame], "direct_four_boundary_hits": direct_hits, "oracle_association_distances_px": distances, "valid_geometry_combination_count": len(candidates)})
        if selected is None or max(distances, default=math.inf) > LOCALIZATION_GATE_PX:
            self.diagnostics["failure"] = "TASK_BOUNDARY_PAIR_MISSING"
            return self._observation(None)
        self.diagnostics["geometry"] = selected[2].__dict__
        return self._observation(selected[2], math.exp(-float(np.mean(distances)) / 7.0))


def evaluate_arm(torch, cohort_path: Path, checkpoint: Path, arm: str) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-C evaluation requires the frozen 24-episode R2 cohort")
    inference = BoundaryFieldInference(torch, checkpoint)
    rows = []
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        provider = TaskBoundaryProvider(episode_input, truth, pose_a, pose_b, inference, arm)
        result = _sage_lm(evaluator, provider)
        rows.append({"episode_id": evaluator.episode_id, "kind": evaluator.kind, "control": materialized["control"], "source": materialized["source"], "truth": materialized["truth"], "source_pose_audit": pose_audit, "baseline": _baseline(evaluator), "b1": result})
    diagnostics = _arm_diagnostics(rows, "b1")
    candidate_available = sum(len(row["b1"]["diagnostics"].get("oracle_association_distances_px", [])) == 4 and max(row["b1"]["diagnostics"]["oracle_association_distances_px"]) <= LOCALIZATION_GATE_PX for row in rows)
    direct_coverage = sum(all(row["b1"]["diagnostics"]["direct_four_boundary_hits"]) for row in rows)
    diagnostics.update({"four_boundary_recall_at_8_count": direct_coverage, "true_boundary_pair_available_count": candidate_available, "aperture_pair_hypothesis_missing_count": 24 - candidate_available})
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint), "conditioned": inference.conditioned, "metrics": _aggregate(row["b1"] for row in rows), "observation_diagnostics": diagnostics, "target_20_of_24": candidate_available >= 20 and diagnostics["geometry_output_count"] >= 20, "rows": rows}


def run(args) -> dict:
    import torch

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_receipt = json.loads((args.training_dataset / "receipt.json").read_text(encoding="utf-8"))
    same_domain = training_receipt.get("source_dataset") == "ARKitScenes Training"
    training = {}
    evaluation = {}
    for arm, conditioned in (("V1-C0", False), ("V1-C1", True)):
        checkpoint = output_dir / f"{arm.lower()}-boundary-field.pt"
        training[arm] = train_arm(torch, args.training_dataset, conditioned, checkpoint, args.epochs, args.batch_size)
        evaluation[arm] = evaluate_arm(torch, args.cohort, checkpoint, arm)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_SOURCE_DISJOINT_SAME_DOMAIN_DEVELOPMENT" if same_domain else "REVERSIBLE_EXPLORATION_SYNTHETIC_TO_REAL_DEVELOPMENT",
        "experiment_label": "SAGE_LM_V1_C_TASK_SPECIFIC_APERTURE_BOUNDARY_FIELD",
        "cohort": {"path": str(args.cohort.resolve()), "episode_count": 24, "usage": "EVALUATION_ONLY_OPENED_DEVELOPMENT"},
        "training": training,
        "evaluation": evaluation,
        "frozen_surfaces": {"source_pose": "UNCHANGED_R2", "oracle_localization_gate_px": LOCALIZATION_GATE_PX, "triangulation": "UNCHANGED_R2", "confidence_arrival_b2_r6": "NOT_ADJUDICATED"},
        "claim_ceiling": "CURATED_ARKITSCENES_R2_DEVELOPMENT_SYNTHETIC_TARTANAIR_TRAINING_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    report = run(args)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: value["observation_diagnostics"] for arm, value in report["evaluation"].items()}, indent=2))


if __name__ == "__main__":
    main()

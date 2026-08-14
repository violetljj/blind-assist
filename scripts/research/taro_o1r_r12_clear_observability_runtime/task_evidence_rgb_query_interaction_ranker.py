#!/usr/bin/env python3
"""R25 TARO ranker with source-time reference/candidate RGB interaction.

R23 showed that query-conditioned geometry can improve held-source macro gain
but not broad opportunity-parent coverage.  R24's extra teacher supervision did
not transfer.  R25 therefore changes the observable signal, not the target or
gate: it augments each of the nine task/candidate token pairs with normalized
RGB appearance, gradient, texture, and reference/candidate disagreement at the
same projected query cells.

Both RGB frames are already present in the historical candidate buffer.  No
candidate depth, teacher target, passive coverage, held-source outcome, model
output, or network request enters the student feature path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from scripts.research.taro_o1r_r12_clear_observability_runtime import arkitscenes_balanced_pose_source_frontdoor as arkit
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_query_conditioned_no_regret_ranker as r23
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_rgb_query_interaction_ranker.v1"
RGB_CHANNELS = (
    "candidate_in_frame",
    "reference_in_frame",
    "candidate_luma_z",
    "candidate_gradient",
    "candidate_texture",
    "reference_luma_z",
    "reference_gradient",
    "reference_texture",
    "absolute_luma_delta",
    "positive_texture_delta",
)
RGB_TOKEN_WIDTH = r23.STATIC_TOKEN_WIDTH * len(RGB_CHANNELS)
CANDIDATE_TOKEN_WIDTH = r23.GEOMETRY_TOKEN_WIDTH + RGB_TOKEN_WIDTH
RGB_FEATURE_COUNT = r23.QUERY_COUNT * RGB_TOKEN_WIDTH
TOTAL_FEATURE_COUNT = r23.TOTAL_FEATURE_COUNT + RGB_FEATURE_COUNT

SEEDS = (25013, 25031, 25049)
EPOCHS = 220
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
REGRESSION_LOSS_WEIGHT = 0.25
GRADIENT_CLIP_NORM = 5.0
RGB_CANDIDATE_HIDDEN = 192


@dataclass(frozen=True)
class RgbAsset:
    storage_kind: str
    source_path: Path
    identity: str
    orientation_index: int = 0
    expected_bytes: int | None = None
    expected_sha256: str | None = None


def _rgb_planes(rgb: np.ndarray) -> np.ndarray:
    shared.require(rgb.ndim == 3 and rgb.shape[2] == 3, "R25 RGB shape drift")
    if (rgb.shape[1], rgb.shape[0]) != tum.LOW_SIZE_WH:
        rgb = cv2.resize(rgb, tum.LOW_SIZE_WH, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = float(np.mean(gray))
    std = max(float(np.std(gray)), 1.0 / 255.0)
    luma_z = np.clip((gray - mean) / (3.0 * std), -1.0, 1.0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(dx * dx + dy * dy)
    gradient_scale = max(float(np.quantile(gradient, 0.95)), 1e-6)
    gradient = np.clip(gradient / gradient_scale, 0.0, 1.0)
    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (5, 5), normalize=True)
    local_square = cv2.boxFilter(gray * gray, cv2.CV_32F, (5, 5), normalize=True)
    texture = np.sqrt(np.maximum(local_square - local_mean * local_mean, 0.0))
    texture_scale = max(float(np.quantile(texture, 0.95)), 1e-6)
    texture = np.clip(texture / texture_scale, 0.0, 1.0)
    output = np.stack((luma_z, gradient, texture)).astype(np.float16)
    shared.require(output.shape == (3, tum.LOW_SIZE_WH[1], tum.LOW_SIZE_WH[0]), "R25 RGB plane shape drift")
    shared.require(np.all(np.isfinite(output)), "R25 RGB plane non-finite")
    return output


class RgbStore:
    def __init__(self, assets: Mapping[str, RgbAsset], authority: str):
        self.assets = dict(assets)
        self.authority = authority
        self._rgb: dict[str, np.ndarray] = {}
        self._planes: dict[str, np.ndarray] = {}
        self._archives: dict[Path, tarfile.TarFile] = {}
        self._payload_receipts: dict[str, str] = {}

    def _payload(self, asset: RgbAsset) -> bytes:
        if asset.storage_kind == "tgz":
            archive = self._archives.get(asset.source_path)
            if archive is None:
                archive = tarfile.open(asset.source_path, "r:*")
                self._archives[asset.source_path] = archive
            stream = archive.extractfile(asset.identity)
            shared.require(stream is not None, f"R25 RGB archive member unreadable: {asset.identity}")
            return stream.read()
        shared.require(asset.storage_kind == "file", f"R25 unsupported RGB storage: {asset.storage_kind}")
        path = Path(asset.identity)
        shared.require(path.is_file(), f"R25 RGB file absent: {path}")
        if asset.expected_bytes is not None:
            shared.require(path.stat().st_size == asset.expected_bytes, f"R25 RGB byte drift: {path}")
        return path.read_bytes()

    def _decode_rgb_payload(self, frame_id: str, asset: RgbAsset, payload: bytes) -> np.ndarray:
        digest = hashlib.sha256(payload).hexdigest().upper()
        if asset.expected_sha256 is not None:
            shared.require(digest == asset.expected_sha256, f"R25 RGB hash drift: {frame_id}")
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if asset.orientation_index == 2:
            rgb = np.ascontiguousarray(np.rot90(rgb, 2))
        self._rgb[frame_id] = rgb
        self._payload_receipts[frame_id] = digest
        return rgb

    def preload(self, frame_ids: Sequence[str]) -> None:
        """Decode requested archive members in physical offset order."""
        grouped: dict[Path, list[tuple[str, RgbAsset]]] = defaultdict(list)
        for frame_id in sorted(set(frame_ids)):
            if frame_id in self._rgb:
                continue
            asset = self.assets.get(frame_id)
            if asset is not None and asset.storage_kind == "tgz":
                grouped[asset.source_path].append((frame_id, asset))
        for source_path, rows in grouped.items():
            archive = self._archives.get(source_path)
            if archive is None:
                archive = tarfile.open(source_path, "r:*")
                self._archives[source_path] = archive
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            ordered = sorted(rows, key=lambda row: members[row[1].identity].offset_data)
            for frame_id, asset in ordered:
                stream = archive.extractfile(members[asset.identity])
                shared.require(stream is not None, f"R25 RGB archive member unreadable: {asset.identity}")
                self._decode_rgb_payload(frame_id, asset, stream.read())

    def rgb(self, frame: Any) -> np.ndarray:
        frame_id = str(frame.frame_id)
        cached = self._rgb.get(frame_id)
        if cached is not None:
            return cached
        asset = self.assets.get(frame_id)
        if asset is None:
            path = Path(frame.rgb_path)
            asset = RgbAsset("file", path.parent, str(path))
        return self._decode_rgb_payload(frame_id, asset, self._payload(asset))

    def planes(self, frame: Any) -> np.ndarray:
        frame_id = str(frame.frame_id)
        cached = self._planes.get(frame_id)
        if cached is not None:
            return cached
        planes = _rgb_planes(self.rgb(frame))
        self._planes[frame_id] = planes
        return planes

    def receipt(self) -> dict[str, Any]:
        rows = [{"frame_id": key, "payload_sha256": value} for key, value in sorted(self._payload_receipts.items())]
        return {
            "authority": self.authority,
            "available_asset_count": len(self.assets),
            "unique_rgb_payload_decode_count": len(rows),
            "decoded_payload_receipt_sha256": hashlib.sha256(shared.canonical_json_bytes(rows)).hexdigest().upper(),
            "rgb_model_runs": 0,
        }

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()


def _tum_rgb_assets(manifest_paths: Sequence[Path]) -> dict[str, RgbAsset]:
    source_rows, _receipts = tum._manifest_rows(manifest_paths)
    output: dict[str, RgbAsset] = {}
    for source in source_rows:
        parent_id = str(source["parent_id"])
        storage_kind = str(source["storage_kind"])
        source_path = (tum.REPO_ROOT / str(source["source_path"])).resolve()
        if storage_kind == "tgz":
            controls, members = tum._tar_controls(source_path)
        else:
            shared.require(storage_kind == "directory", f"R25 TUM RGB storage drift: {storage_kind}")
            controls, members = tum._directory_controls(source_path)
        for row in tum.tum.parse_tum_index(controls["rgb.txt"]):
            relative = row.relative_path.replace("\\", "/")
            frame_id = f"{parent_id}:{row.timestamp_seconds:.5f}"
            if storage_kind == "tgz":
                candidates = [name for name in members if name == relative or name.endswith("/" + relative)]
                shared.require(len(candidates) == 1, f"R25 TUM RGB member ambiguity: {frame_id}")
                asset = RgbAsset("tgz", source_path, candidates[0])
            else:
                path = source_path / relative
                shared.require(path.is_file(), f"R25 TUM RGB file absent: {path}")
                asset = RgbAsset("file", source_path, str(path))
            shared.require(frame_id not in output, f"R25 duplicate TUM RGB frame: {frame_id}")
            output[frame_id] = asset
    return output


def _arkit_rgb_assets(dataset_root: Path) -> dict[str, RgbAsset]:
    _frames, frame_assets, _source = arkit.load_outcome_blind_roster(dataset_root)
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    output: dict[str, RgbAsset] = {}
    for video in manifest["videos"]:
        parent_id = str(video["video_id"])
        selected = [str(value) for value in video["selected_frame_stems"]]
        rgb_rows = list(video["extracted"]["lowres_wide"])
        shared.require(len(selected) == len(rgb_rows), f"R25 ARKit RGB roster count drift: {parent_id}")
        for stem, entry in zip(selected, rgb_rows, strict=True):
            shared.require(Path(str(entry["path"])).stem == stem, f"R25 ARKit RGB stem drift: {stem}")
            timestamp = float(stem.rsplit("_", 1)[1])
            frame_id = f"{parent_id}:{timestamp:.5f}"
            if frame_id not in frame_assets:
                continue
            output[frame_id] = RgbAsset(
                "file",
                Path(str(entry["path"])).parent,
                str(entry["path"]),
                frame_assets[frame_id].orientation_index,
                int(entry["bytes"]),
                str(entry["sha256"]),
            )
    shared.require(output, "R25 ARKit RGB asset roster empty")
    return output


def _sample_planes(planes: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    width, height = tum.LOW_SIZE_WH
    columns = np.clip(np.rint(u).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
    sampled = np.moveaxis(planes[:, rows, columns], 0, -1).astype(np.float32)
    return sampled * valid[..., None]


def rgb_query_candidate_features(
    context: scorer.ReferenceContext,
    pair: Any,
    reference_planes: np.ndarray,
    candidate_planes: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    geometry_features, analytic = r23.structured_candidate_features(context, pair)
    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    inverse = np.linalg.inv(relative)
    points_rows: list[np.ndarray] = []
    for query in context.queries:
        centers, _along = scorer._cell_centers(query)
        points_rows.append(centers)
    points_ref = np.stack(points_rows, axis=0)
    points_candidate = points_ref @ inverse[:3, :3].T + inverse[:3, 3]
    ref_z = points_ref[..., 2]
    candidate_z = points_candidate[..., 2]
    k = context.intrinsics
    ref_u = k[0, 0] * points_ref[..., 0] / np.maximum(ref_z, 1e-9) + k[0, 2]
    ref_v = k[1, 1] * points_ref[..., 1] / np.maximum(ref_z, 1e-9) + k[1, 2]
    candidate_u = k[0, 0] * points_candidate[..., 0] / np.maximum(candidate_z, 1e-9) + k[0, 2]
    candidate_v = k[1, 1] * points_candidate[..., 1] / np.maximum(candidate_z, 1e-9) + k[1, 2]
    width, height = tum.LOW_SIZE_WH
    unknown = ~context.static.reshape(r23.QUERY_COUNT, -1)
    reference_in = unknown & (ref_z > 0.0) & (ref_u >= 0.0) & (ref_u < width) & (ref_v >= 0.0) & (ref_v < height)
    candidate_in = unknown & (candidate_z > 0.0) & (candidate_u >= 0.0) & (candidate_u < width) & (candidate_v >= 0.0) & (candidate_v < height)
    reference_values = _sample_planes(reference_planes, ref_u, ref_v, reference_in)
    candidate_values = _sample_planes(candidate_planes, candidate_u, candidate_v, candidate_in)
    both = reference_in & candidate_in
    luma_delta = np.abs(candidate_values[..., 0] - reference_values[..., 0]) * both
    positive_texture_delta = np.maximum(candidate_values[..., 2] - reference_values[..., 2], 0.0) * both
    rgb_channels = np.concatenate(
        (
            candidate_in[..., None].astype(np.float32),
            reference_in[..., None].astype(np.float32),
            candidate_values,
            reference_values,
            luma_delta[..., None],
            positive_texture_delta[..., None],
        ),
        axis=-1,
    )
    shared.require(
        rgb_channels.shape == (r23.QUERY_COUNT, r23.STATIC_TOKEN_WIDTH, len(RGB_CHANNELS)),
        "R25 RGB query token shape drift",
    )
    result = np.concatenate((geometry_features, rgb_channels.reshape(-1))).astype(np.float32)
    shared.require(result.shape == (TOTAL_FEATURE_COUNT,), "R25 total feature width drift")
    shared.require(np.all(np.isfinite(result)), "R25 feature non-finite")
    return result, analytic


class RgbStructuredFeatureTransform(r23.StructuredFeatureTransform):
    @classmethod
    def fit(cls, records: Sequence[scorer.CandidateRecord]) -> "RgbStructuredFeatureTransform":
        shared.require(bool(records), "R25 transform fit records empty")
        base = np.stack([record.features[: r23.BASE_FEATURE_COUNT] for record in records]).astype(np.float32)
        pose = base[:, r23.POSE_FEATURE_INDICES]
        mean = np.mean(pose, axis=0)
        scale = np.std(pose, axis=0)
        scale[scale < 1e-9] = 1.0
        return cls(mean, scale)

    def apply(self, records: Sequence[scorer.CandidateRecord]) -> r23.ModelInputs:
        raw = np.stack([record.features for record in records]).astype(np.float32)
        shared.require(raw.shape[1] == TOTAL_FEATURE_COUNT, "R25 transform feature width drift")
        base = raw[:, : r23.BASE_FEATURE_COUNT]
        pose_raw = base[:, r23.POSE_FEATURE_INDICES]
        pose_z, pose_unit = r23._reference_blocks(pose_raw, records)
        pose = np.concatenate(((pose_raw - self.pose_mean) / self.pose_scale, pose_z, pose_unit), axis=1)
        static_start = r23.BASE_FEATURE_COUNT
        geometry_start = static_start + r23.STATIC_FEATURE_COUNT
        rgb_start = geometry_start + r23.GEOMETRY_FEATURE_COUNT
        task = raw[:, static_start:geometry_start].reshape(-1, r23.QUERY_COUNT, r23.STATIC_TOKEN_WIDTH)
        geometry = raw[:, geometry_start:rgb_start].reshape(-1, r23.QUERY_COUNT, r23.GEOMETRY_TOKEN_WIDTH)
        rgb = raw[:, rgb_start:].reshape(-1, r23.QUERY_COUNT, RGB_TOKEN_WIDTH)
        candidate = np.concatenate((geometry, rgb), axis=2)
        translation_position = r23.POSE_FEATURE_INDICES.index(scorer.FEATURE_NAMES.index("translation_m"))
        rotation_position = r23.POSE_FEATURE_INDICES.index(scorer.FEATURE_NAMES.index("rotation_deg"))
        generic_base = pose_unit[:, translation_position] + r21.BASE_ROTATION_WEIGHT * pose_unit[:, rotation_position]
        shared.require(candidate.shape[2] == CANDIDATE_TOKEN_WIDTH, "R25 candidate token width drift")
        shared.require(
            np.all(np.isfinite(pose))
            and np.all(np.isfinite(task))
            and np.all(np.isfinite(candidate))
            and np.all(np.isfinite(generic_base)),
            "R25 transformed feature non-finite",
        )
        return r23.ModelInputs(
            pose.astype(np.float32),
            task.astype(np.float32),
            candidate.astype(np.float32),
            generic_base.astype(np.float32),
        )


class RgbQueryInteractionRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(r23.STATIC_TOKEN_WIDTH, r23.STATIC_HIDDEN),
            nn.GELU(),
            nn.Linear(r23.STATIC_HIDDEN, r23.EMBED_DIM),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_TOKEN_WIDTH, RGB_CANDIDATE_HIDDEN),
            nn.GELU(),
            nn.Linear(RGB_CANDIDATE_HIDDEN, r23.EMBED_DIM),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(r23.POSE_TRANSFORMED_WIDTH, r23.POSE_HIDDEN),
            nn.GELU(),
            nn.Linear(r23.POSE_HIDDEN, r23.EMBED_DIM),
        )
        self.query_embedding = nn.Parameter(torch.empty(r23.QUERY_COUNT, r23.EMBED_DIM))
        nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(r23.EMBED_DIM, r23.ATTENTION_HEADS, batch_first=True)
        self.task_norm = nn.LayerNorm(r23.EMBED_DIM)
        self.candidate_norm = nn.LayerNorm(r23.EMBED_DIM)
        self.fusion = nn.Sequential(
            nn.Linear(r23.EMBED_DIM * 5, r23.FUSION_HIDDEN[0]),
            nn.GELU(),
            nn.Linear(r23.FUSION_HIDDEN[0], r23.FUSION_HIDDEN[1]),
            nn.GELU(),
            nn.Linear(r23.FUSION_HIDDEN[1], 2),
        )

    def forward(
        self,
        pose: torch.Tensor,
        task_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        generic_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_embedding.unsqueeze(0)
        pose_token = self.pose_encoder(pose)
        task = self.task_norm(self.task_encoder(task_tokens) + query)
        candidate = self.candidate_norm(self.candidate_encoder(candidate_tokens) + query + pose_token.unsqueeze(1))
        attended_task, _weights = self.cross_attention(candidate, task, task, need_weights=False)
        pooled = torch.cat(
            (
                candidate.mean(dim=1),
                attended_task.mean(dim=1),
                (candidate * attended_task).mean(dim=1),
                (candidate * task).mean(dim=1),
                pose_token,
            ),
            dim=1,
        )
        output = self.fusion(pooled)
        mean = generic_base + r23.RESIDUAL_SCALE * torch.tanh(output[:, 0])
        log_variance = torch.clamp(output[:, 1], min=-5.0, max=1.0)
        return mean, log_variance


def train_ranker(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
    transform: RgbStructuredFeatureTransform,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[RgbQueryInteractionRanker, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    inputs = r23._torch_inputs(transform.apply(records))
    high_np, low_np, pair_weights_np = r21._pair_indices(records, sources)
    targets_np, candidate_weights_np = r23._balanced_candidate_targets(records, sources)
    high = torch.from_numpy(high_np)
    low = torch.from_numpy(low_np)
    pair_weights = torch.from_numpy(pair_weights_np)
    targets = torch.from_numpy(targets_np)
    candidate_weights = torch.from_numpy(candidate_weights_np)
    model = RgbQueryInteractionRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_rank = float("nan")
    final_regression = float("nan")
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mean, log_variance = model(*inputs)
        rank_loss = torch.mean(F.softplus(-(mean[high] - mean[low])) * pair_weights)
        regression_rows = 0.5 * (torch.exp(-log_variance) * (mean - targets) ** 2 + log_variance)
        regression_loss = torch.mean(regression_rows * candidate_weights)
        loss = rank_loss + REGRESSION_LOSS_WEIGHT * regression_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        final_rank = float(rank_loss.detach().cpu())
        final_regression = float(regression_loss.detach().cpu())
    state_bytes = b"".join(value.detach().cpu().numpy().tobytes() for _name, value in sorted(model.state_dict().items()))
    return model, {
        "seed": seed,
        "epochs": epochs,
        "pair_count": len(high_np),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_pairwise_loss": final_rank,
        "final_heteroscedastic_regression_loss": final_regression,
        "model_state_sha256": hashlib.sha256(state_bytes).hexdigest().upper(),
    }


def ensemble_predictions(
    records: Sequence[scorer.CandidateRecord],
    transform: RgbStructuredFeatureTransform,
    models: Sequence[RgbQueryInteractionRanker],
) -> tuple[np.ndarray, np.ndarray]:
    inputs = r23._torch_inputs(transform.apply(records))
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            mean, log_variance = model(*inputs)
            means.append(mean.cpu().numpy().astype(np.float64))
            log_variances.append(log_variance.cpu().numpy().astype(np.float64))
    return np.stack(means), np.stack(log_variances)


def _build_tum_records() -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = RgbStore(_tum_rgb_assets(r21.TUM_MANIFESTS), "frozen TUM rgb.txt identities and source manifests")
    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return rgb_query_candidate_features(context, pair, store.planes(pair.reference), store.planes(pair.neighbor))
    try:
        records, source, abstained = r21._build_tum_records(feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def _build_bonn_records(root: Path) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = RgbStore({}, "Bonn rgb.txt-associated direct RGB paths")
    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return rgb_query_candidate_features(context, pair, store.planes(pair.reference), store.planes(pair.neighbor))
    try:
        records, source, abstained = r21._build_bonn_records(root, feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def _build_arkit_records(root: Path) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = RgbStore(_arkit_rgb_assets(root), "ARKitScenes manifest lowres_wide entries with per-file bytes and SHA-256")
    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return rgb_query_candidate_features(context, pair, store.planes(pair.reference), store.planes(pair.neighbor))
    try:
        records, source, abstained = r21._build_arkit_records(root, feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def run_lofo(
    datasets: Mapping[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]],
) -> dict[str, Any]:
    audit = r23.dataset_audit(datasets)
    folds: dict[str, Any] = {}
    parameter_count: int | None = None
    for held_source in r21.SOURCE_NAMES:
        train_sources = [source for source in r21.SOURCE_NAMES if source != held_source]
        train_records: list[scorer.CandidateRecord] = []
        train_source_rows: list[str] = []
        for source in train_sources:
            records = datasets[source][0]
            train_records.extend(records)
            train_source_rows.extend([source] * len(records))
        transform = RgbStructuredFeatureTransform.fit(train_records)
        models: list[RgbQueryInteractionRanker] = []
        training_receipts: list[dict[str, Any]] = []
        for seed in SEEDS:
            model, receipt = train_ranker(train_records, train_source_rows, transform, seed)
            models.append(model)
            training_receipts.append(receipt)
        parameter_count = training_receipts[0]["parameter_count"]
        held_records = datasets[held_source][0]
        means, log_variances = ensemble_predictions(held_records, transform, models)
        ungated_scores = np.mean(means, axis=0)
        gated_scores, gate = r23.no_regret_gate(held_records, means, log_variances)
        folds[held_source] = {
            "held_source_excluded_from_normalizer_fit": True,
            "held_source_excluded_from_model_fit": True,
            "held_source_target_excluded_from_stopping_uncertainty_and_selection": True,
            "source_qualified_parent_sets_disjoint": True,
            "train_sources": train_sources,
            "training_candidate_count": len(train_records),
            "training_parent_count": len(
                {(source, record.parent_id) for source, record in zip(train_source_rows, train_records, strict=True)}
            ),
            "training_receipts": training_receipts,
            "normalizer_sha256": transform.receipt_sha256(),
            "ungated_metrics": r21.fold_metrics(held_records, ungated_scores),
            "no_regret_gate": gate,
            "metrics": r21.fold_metrics(held_records, gated_scores),
        }
    all_checks = all(all(fold["metrics"]["checks"].values()) for fold in folds.values())
    terminal = (
        "TASK_EVIDENCE_RGB_QUERY_INTERACTION_LOFO_PASS"
        if all_checks
        else "STOP_TASK_EVIDENCE_RGB_QUERY_INTERACTION_LOFO_FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "successor_of": r23.SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_LEAVE_ONE_SOURCE_FAMILY_OUT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside frozen body/path capsules; UNKNOWN remains unknown.",
        "candidate_table_audit": audit,
        "frozen_model_family": {
            "architecture": "R23 query-conditioned utility distribution with projected reference/candidate RGB interaction channels",
            "parameter_count": parameter_count,
            "parameter_target_range": [100000, 1000000],
            "geometry_and_task_contract_identical_to_r23": True,
            "new_source_time_signal": {
                "rgb_channels": list(RGB_CHANNELS),
                "per_query_rgb_shape": [r23.STATIC_TOKEN_WIDTH, len(RGB_CHANNELS)],
                "candidate_token_width": CANDIDATE_TOKEN_WIDTH,
                "raw_feature_count": TOTAL_FEATURE_COUNT,
                "candidate_rgb_available_in_historical_buffer": True,
                "candidate_depth_in_student_input": False,
            },
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss_identical_to_r23": "source-parent-reference-balanced pairwise softplus plus heteroscedastic normalized-utility regression",
            "no_regret_gate_identical_to_r23": {
                "lcb_z": r23.LCB_Z,
                "aleatoric_gate_weight": r23.ALEATORIC_GATE_WEIGHT,
                "minimum_normalized_advantage": r23.MIN_NORMALIZED_ADVANTAGE,
                "fallback": "generic pose-diversity candidate",
            },
        },
        "sources": {
            source: {
                "disposition": "CONSUMED_DEVELOPMENT",
                "candidate_count": len(rows[0]),
                "parent_count": len({record.parent_id for record in rows[0]}),
                "geometry_abstention_count": rows[2],
                "source_receipt": rows[1],
            }
            for source, rows in datasets.items()
        },
        "folds": folds,
        "terminal": terminal,
        "fresh_confirmation_source_lock_authorized": terminal == "TASK_EVIDENCE_RGB_QUERY_INTERACTION_LOFO_PASS",
        "android_candidate_authorized": False,
        "read_boundary": {
            "reference_and_candidate_rgb_payload_decodes": sum(
                rows[1]["rgb_signal_receipt"]["unique_rgb_payload_decode_count"] for rows in datasets.values()
            ),
            "neighbor_depth_in_student_input": False,
            "held_source_target_in_fit_or_selection": False,
            "network_requests": 0,
            "r11_reads": 0,
        },
        "claim_ceiling": "Consumed three-source Development and source-family holdout evidence only; not fresh Confirmation, collision correctness, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": _build_tum_records(),
        "BONN_RGBD_DYNAMIC": _build_bonn_records(bonn_root),
        "ARKITSCENES": _build_arkit_records(arkit_root),
    }
    return run_lofo(datasets)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonn-root", type=Path, default=shared.DEFAULT_BONN_ROOT)
    parser.add_argument("--arkit-root", type=Path, default=r21.DEFAULT_ARKIT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve(), args.arkit_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V3-A typed semantic-referent graph scorer with frozen V2 belief.

The train split is domain-randomized synthetic graph data.  Evaluation reuses
the recorded V2.1 RapidOCR rows so substring, handcrafted V2, a learned
no-relation ablation, and the full typed graph network see identical OCR text,
confidence, polygons, candidate geometry, quality, and episode order.

Only the per-candidate graph score is learned.  V2 observability, novelty,
belief update, NONE construction, and terminal thresholds remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .semantic_anchor_graph_and_belief_v2 import (
    Box,
    Candidate,
    Frame,
    ReferentBelief,
    TargetGraph,
    Token,
    _edit_similarity,
    _metrics,
    _row_correct,
    frame_observability,
    normalize_text,
)


EXPERIMENT_ID = "TYPED_SEMANTIC_REFERENT_GRAPH_V3_A"
SCHEMA_VERSION = "blindassist_typed_semantic_referent_graph_v3_a"
CLAIM_CEILING = (
    "SYNTHETIC_GRAPH_TRAIN_GENERATED_PIXEL_RAPIDOCR_TUNED_ON_DEVELOPMENT_"
    "NO_NATURAL_PHOTO_GENERALIZATION_LEARNED_BELIEF_ACTIVE_PERCEPTION_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM"
)
NODE_FEATURES = 24
EDGE_FEATURES = 8
NODE_TYPES = {"TARGET": 0, "OCR": 1, "CANDIDATE": 2}
RELATIONS = (
    "SELF",
    "TARGET_TO_OCR",
    "OCR_TO_TARGET",
    "OCR_SAME_LINE",
    "OCR_LEFT",
    "OCR_RIGHT",
    "OCR_TO_CANDIDATE_ABOVE",
    "CANDIDATE_TO_OCR_ABOVE",
    "OCR_TO_CANDIDATE_INSIDE",
    "CANDIDATE_TO_OCR_INSIDE",
    "OCR_TO_CANDIDATE_NEAR",
    "CANDIDATE_TO_OCR_NEAR",
    "CANDIDATE_LEFT",
    "CANDIDATE_RIGHT",
)
RELATION_INDEX = {name: index for index, name in enumerate(RELATIONS)}
TARGET = TargetGraph(("ROOM", "302"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_features(text: str) -> list[float]:
    normalized = normalize_text(text)
    length = max(1, len(normalized))
    bins = [0.0] * 8
    for character in normalized:
        bins[(ord(character) * 17 + 11) % len(bins)] += 1 / length
    return [
        min(1.0, len(normalized) / 12),
        sum(character.isalpha() for character in normalized) / length,
        sum(character.isdigit() for character in normalized) / length,
        normalized.count("?") / length,
        *bins,
    ]


def _node_features(kind: str, text: str = "", box: Box | None = None, confidence: float = 0.0) -> list[float]:
    box = box or Box(0.0, 0.0, 0.0, 0.0)
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    base = [
        box.x0,
        box.y0,
        box.x1,
        box.y1,
        box.cx,
        box.cy,
        width,
        height,
        confidence,
        1.0 if kind == "TARGET" else 0.0,
        1.0 if kind == "OCR" else 0.0,
        1.0 if kind == "CANDIDATE" else 0.0,
        *_text_features(text),
    ]
    if len(base) != NODE_FEATURES:
        raise AssertionError(f"node feature width drifted: {len(base)}")
    return base


def _confusion_similarity(left: str, right: str) -> float:
    table = str.maketrans({"O": "0", "I": "1", "L": "1", "B": "8", "S": "5"})
    return _edit_similarity(normalize_text(left).translate(table), normalize_text(right).translate(table))


def _lexical_edge(expected: str, observed: Token, distinctiveness: float) -> list[float]:
    left, right = normalize_text(expected), normalize_text(observed.text)
    exact = float(left == right)
    prefix = float(bool(left and right) and (left.startswith(right.replace("?", "")) or right.startswith(left)))
    return [
        _edit_similarity(left, right),
        exact,
        prefix,
        _confusion_similarity(left, right),
        min(len(left), len(right)) / max(1, max(len(left), len(right))),
        observed.confidence,
        distinctiveness,
        1.0,
    ]


def _horizontal_overlap(left: Box, right: Box) -> float:
    overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    return overlap / max(1e-6, min(left.x1 - left.x0, right.x1 - right.x0))


def _spatial_edge(source: Box, target: Box, extra: float = 0.0) -> list[float]:
    dx = target.cx - source.cx
    dy = target.cy - source.cy
    distance = math.sqrt(dx * dx + dy * dy)
    inside = float(target.x0 <= source.cx <= target.x1 and target.y0 <= source.cy <= target.y1)
    return [
        float(np.clip(dx, -1.0, 1.0)),
        float(np.clip(dy, -1.0, 1.0)),
        math.exp(-distance / 0.25),
        _horizontal_overlap(source, target),
        inside,
        max(0.0, target.y0 - source.y1),
        extra,
        1.0,
    ]


@dataclass
class GraphRecord:
    node_features: torch.Tensor
    node_types: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    edge_features: torch.Tensor
    target_mask: torch.Tensor
    ocr_mask: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_ids: tuple[str, ...]
    candidate_labels: torch.Tensor
    reliability_target: float
    none_target: float


def build_typed_graph(target: TargetGraph, frame: Frame, *, include_spatial_relations: bool) -> GraphRecord:
    nodes: list[list[float]] = []
    node_types: list[int] = []
    target_indices: list[int] = []
    ocr_indices: list[int] = []
    candidate_indices: list[int] = []

    for text in target.tokens:
        target_indices.append(len(nodes))
        nodes.append(_node_features("TARGET", text=text, confidence=1.0))
        node_types.append(NODE_TYPES["TARGET"])
    for token in frame.tokens:
        ocr_indices.append(len(nodes))
        nodes.append(_node_features("OCR", text=token.text, box=token.box, confidence=token.confidence))
        node_types.append(NODE_TYPES["OCR"])
    for candidate in frame.candidates:
        candidate_indices.append(len(nodes))
        nodes.append(_node_features("CANDIDATE", box=candidate.box, confidence=1.0))
        node_types.append(NODE_TYPES["CANDIDATE"])

    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    edge_features: list[list[float]] = []

    def edge(source: int, destination: int, relation: str, features: Sequence[float]) -> None:
        edges.append((source, destination))
        edge_types.append(RELATION_INDEX[relation])
        edge_features.append(list(features))

    for index in range(len(nodes)):
        edge(index, index, "SELF", [0.0] * (EDGE_FEATURES - 1) + [1.0])

    token_matches: dict[str, int] = {}
    for expected in target.tokens:
        token_matches[expected] = sum(_edit_similarity(expected, token.text) >= 0.72 for token in frame.tokens)
    for target_offset, target_index in enumerate(target_indices):
        expected = target.tokens[target_offset]
        distinctiveness = 1 / max(1, token_matches[expected])
        for token_offset, ocr_index in enumerate(ocr_indices):
            features = _lexical_edge(expected, frame.tokens[token_offset], distinctiveness)
            edge(target_index, ocr_index, "TARGET_TO_OCR", features)
            edge(ocr_index, target_index, "OCR_TO_TARGET", features)

    if include_spatial_relations:
        for left_offset, left_index in enumerate(ocr_indices):
            left = frame.tokens[left_offset]
            for right_offset, right_index in enumerate(ocr_indices):
                if left_offset == right_offset:
                    continue
                right = frame.tokens[right_offset]
                features = _spatial_edge(left.box, right.box)
                same_line = abs(left.box.cy - right.box.cy) <= max(0.018, max(left.box.height, right.box.height) * 0.8)
                relation = "OCR_SAME_LINE" if same_line else "OCR_LEFT" if left.box.cx < right.box.cx else "OCR_RIGHT"
                edge(left_index, right_index, relation, features)

        for token_offset, ocr_index in enumerate(ocr_indices):
            token = frame.tokens[token_offset]
            for candidate_offset, candidate_index in enumerate(candidate_indices):
                candidate = frame.candidates[candidate_offset]
                inside = candidate.box.x0 <= token.box.cx <= candidate.box.x1 and candidate.box.y0 <= token.box.cy <= candidate.box.y1
                above = token.box.y1 <= candidate.box.y0 + 0.03 and abs(token.box.cx - candidate.box.cx) <= 0.24
                suffix = "INSIDE" if inside else "ABOVE" if above else "NEAR"
                features = _spatial_edge(token.box, candidate.box, token.confidence)
                edge(ocr_index, candidate_index, f"OCR_TO_CANDIDATE_{suffix}", features)
                edge(candidate_index, ocr_index, f"CANDIDATE_TO_OCR_{suffix}", features)

    for left_offset, left_index in enumerate(candidate_indices):
        for right_offset, right_index in enumerate(candidate_indices):
            if left_offset == right_offset:
                continue
            left, right = frame.candidates[left_offset], frame.candidates[right_offset]
            relation = "CANDIDATE_LEFT" if left.box.cx < right.box.cx else "CANDIDATE_RIGHT"
            edge(left_index, right_index, relation, _spatial_edge(left.box, right.box))

    target_mask = torch.zeros(len(nodes), dtype=torch.bool)
    ocr_mask = torch.zeros(len(nodes), dtype=torch.bool)
    candidate_mask = torch.zeros(len(nodes), dtype=torch.bool)
    target_mask[target_indices] = True
    ocr_mask[ocr_indices] = True
    candidate_mask[candidate_indices] = True
    candidate_labels = torch.tensor(
        [float(frame.truth == candidate.candidate_id) for candidate in frame.candidates],
        dtype=torch.float32,
    )
    return GraphRecord(
        node_features=torch.tensor(nodes, dtype=torch.float32),
        node_types=torch.tensor(node_types, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_types=torch.tensor(edge_types, dtype=torch.long),
        edge_features=torch.tensor(edge_features, dtype=torch.float32),
        target_mask=target_mask,
        ocr_mask=ocr_mask,
        candidate_mask=candidate_mask,
        candidate_ids=tuple(candidate.candidate_id for candidate in frame.candidates),
        candidate_labels=candidate_labels,
        reliability_target=frame_observability(frame),
        none_target=float(frame.truth == "NONE"),
    )


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    node_types: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    edge_features: torch.Tensor
    node_graph: torch.Tensor
    target_mask: torch.Tensor
    ocr_mask: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_graph: torch.Tensor
    candidate_labels: torch.Tensor
    reliability_targets: torch.Tensor
    none_targets: torch.Tensor
    graph_count: int

    @classmethod
    def from_records(cls, records: Sequence[GraphRecord], device: torch.device) -> "GraphBatch":
        offsets = []
        total = 0
        for record in records:
            offsets.append(total)
            total += record.node_features.shape[0]
        edge_index = torch.cat(
            [record.edge_index + offset for record, offset in zip(records, offsets, strict=True)], dim=1
        )
        node_graph = torch.cat(
            [torch.full((record.node_features.shape[0],), index, dtype=torch.long) for index, record in enumerate(records)]
        )
        candidate_graph = torch.cat(
            [torch.full((len(record.candidate_ids),), index, dtype=torch.long) for index, record in enumerate(records)]
        )
        return cls(
            node_features=torch.cat([record.node_features for record in records]).to(device),
            node_types=torch.cat([record.node_types for record in records]).to(device),
            edge_index=edge_index.to(device),
            edge_types=torch.cat([record.edge_types for record in records]).to(device),
            edge_features=torch.cat([record.edge_features for record in records]).to(device),
            node_graph=node_graph.to(device),
            target_mask=torch.cat([record.target_mask for record in records]).to(device),
            ocr_mask=torch.cat([record.ocr_mask for record in records]).to(device),
            candidate_mask=torch.cat([record.candidate_mask for record in records]).to(device),
            candidate_graph=candidate_graph.to(device),
            candidate_labels=torch.cat([record.candidate_labels for record in records]).to(device),
            reliability_targets=torch.tensor([record.reliability_target for record in records], dtype=torch.float32, device=device),
            none_targets=torch.tensor([record.none_target for record in records], dtype=torch.float32, device=device),
            graph_count=len(records),
        )


def _mean_pool(values: torch.Tensor, graph_index: torch.Tensor, count: int) -> torch.Tensor:
    output = values.new_zeros((count, values.shape[1]))
    denominator = values.new_zeros((count, 1))
    output.index_add_(0, graph_index, values)
    denominator.index_add_(0, graph_index, values.new_ones((values.shape[0], 1)))
    return output / denominator.clamp_min(1.0)


class TypedRelationalLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in RELATIONS)
        self.edge_projection = nn.Linear(EDGE_FEATURES, hidden)
        self.relation_embedding = nn.Embedding(len(RELATIONS), hidden)
        self.attention = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        source, destination = edge_index
        projected_edges = self.edge_projection(edge_features)
        relation = self.relation_embedding(edge_types)
        messages = hidden.new_zeros((edge_types.shape[0], hidden.shape[1]))
        for relation_index, transform in enumerate(self.messages):
            mask = edge_types == relation_index
            if mask.any():
                messages[mask] = transform(hidden[source[mask]])
        attention = torch.sigmoid(
            self.attention(torch.cat([hidden[source], hidden[destination], projected_edges, relation], dim=1))
        )
        messages = (messages + projected_edges + relation) * attention
        aggregate = hidden.new_zeros(hidden.shape)
        weights = hidden.new_zeros((hidden.shape[0], 1))
        aggregate.index_add_(0, destination, messages)
        weights.index_add_(0, destination, attention)
        aggregate = aggregate / weights.clamp_min(1e-6)
        return self.norm(hidden + self.update(torch.cat([hidden, aggregate], dim=1)))


class TypedSemanticReferentGraphNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(NODE_FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden)
        self.layers = nn.ModuleList(TypedRelationalLayer(hidden) for _ in range(layers))
        self.identity_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.reliability_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.none_head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        hidden = self.input_projection(batch.node_features) + self.type_embedding(batch.node_types)
        for layer in self.layers:
            hidden = layer(hidden, batch.edge_index, batch.edge_types, batch.edge_features)
        target_pool = _mean_pool(hidden[batch.target_mask], batch.node_graph[batch.target_mask], batch.graph_count)
        ocr_pool = _mean_pool(hidden[batch.ocr_mask], batch.node_graph[batch.ocr_mask], batch.graph_count)
        candidate_pool = _mean_pool(hidden[batch.candidate_mask], batch.candidate_graph, batch.graph_count)
        candidate_hidden = hidden[batch.candidate_mask]
        identity_logits = self.identity_head(
            torch.cat([candidate_hidden, target_pool[batch.candidate_graph]], dim=1)
        ).squeeze(1)
        reliability = torch.sigmoid(self.reliability_head(torch.cat([target_pool, ocr_pool], dim=1)).squeeze(1))
        none_logits = self.none_head(torch.cat([target_pool, ocr_pool, candidate_pool], dim=1)).squeeze(1)
        return {"identity_logits": identity_logits, "reliability": reliability, "none_logits": none_logits}


def _candidate_boxes(rng: random.Random, *, directory_layout: bool) -> tuple[Candidate, ...]:
    nominal = (0.39, 0.63, 0.87) if directory_layout else (0.18, 0.50, 0.82)
    half_width = 0.105 if directory_layout else 0.12
    centers = [center + rng.uniform(-0.025, 0.025) for center in nominal]
    ids = ["A", "B", "C"]
    rng.shuffle(ids)
    return tuple(
        Candidate(candidate_id, Box(center - half_width, 0.34, center + half_width, 0.96))
        for candidate_id, center in zip(ids, centers, strict=True)
    )


def _sign_tokens(prefix: str, number: str, candidate: Candidate, rng: random.Random, confidence: float) -> list[Token]:
    y = 0.245 + rng.uniform(-0.012, 0.012)
    height = rng.uniform(0.032, 0.058)
    if rng.random() < 0.16:
        return [Token(f"{prefix} {number}", Box(candidate.box.cx - 0.09, y, candidate.box.cx + 0.09, y + height), confidence)]
    return [
        Token(prefix, Box(candidate.box.cx - 0.10, y, candidate.box.cx - 0.018, y + height), confidence),
        Token(number, Box(candidate.box.cx + 0.005, y, candidate.box.cx + 0.075, y + height), confidence),
    ]


def _near_number(target: str, rng: random.Random) -> str:
    value = int(target)
    choices = [value - 1, value + 1, int(str(value)[:-1] + str((value + 3) % 10)), int(str(value)[::-1])]
    return str(rng.choice(choices)).zfill(len(target))


def generate_synthetic_frame(seed: int) -> tuple[TargetGraph, Frame, str]:
    rng = random.Random(seed)
    prefix = rng.choice(("ROOM", "ROOM", "GATE", "OFFICE", "AREA"))
    number = str(rng.randint(101, 989))
    target = TargetGraph((prefix, number))
    scenario = rng.choice(("clean", "directory_partial", "suffix", "absent", "detached_absent", "confusion", "duplicate"))
    directory_layout = scenario in {"directory_partial", "detached_absent", "duplicate"} and rng.random() < 0.65
    candidates = _candidate_boxes(rng, directory_layout=directory_layout)
    truth_candidate = rng.choice(candidates)
    truth = truth_candidate.candidate_id if scenario not in {"absent", "detached_absent"} else "NONE"
    tokens: list[Token] = []
    confidence = rng.uniform(0.70, 0.99)
    blur = rng.uniform(0.0, 0.32)
    perspective = rng.uniform(0.0, 0.35)

    for candidate in candidates:
        candidate_number = _near_number(number, rng)
        if candidate.candidate_id == truth:
            candidate_number = number
            if scenario == "directory_partial":
                candidate_number = number[:-1] + "?"
            elif scenario == "confusion":
                candidate_number = number.replace("0", "O").replace("1", "I").replace("8", "B")
        elif scenario == "suffix" and candidate is candidates[0]:
            candidate_number = number + rng.choice(("A", "B"))
        tokens.extend(_sign_tokens(prefix, candidate_number, candidate, rng, confidence))

    if scenario in {"directory_partial", "detached_absent", "duplicate"}:
        x = rng.uniform(0.015, 0.11) if directory_layout else rng.choice((rng.uniform(0.03, 0.10), rng.uniform(0.82, 0.90)))
        y = rng.uniform(0.40, 0.62)
        tokens.extend(
            [
                Token(prefix, Box(x, y, x + 0.09, y + 0.04), rng.uniform(0.82, 0.99)),
                Token(number, Box(x + 0.10, y, x + 0.17, y + 0.04), rng.uniform(0.82, 0.99)),
            ]
        )

    if scenario in {"absent", "detached_absent"}:
        tokens = [token for token in tokens if normalize_text(token.text) != number or token.box.y0 >= 0.34]

    for _ in range(rng.randint(0, 3)):
        text = rng.choice(("EXIT", "LEVEL", "LOBBY", "STAIRS", "INFO"))
        x, y = rng.uniform(0.02, 0.88), rng.uniform(0.05, 0.68)
        tokens.append(Token(text, Box(x, y, x + rng.uniform(0.05, 0.12), y + rng.uniform(0.025, 0.05)), rng.uniform(0.50, 0.95)))

    rng.shuffle(tokens)
    frame = Frame(
        f"synthetic-{seed}",
        0,
        f"domain-randomized-{seed}",
        candidates,
        tuple(tokens),
        blur,
        perspective,
        truth,
        "TARGET" if truth != "NONE" else "NONE",
        scenario,
    )
    return target, frame, scenario


def _training_records(seeds: Iterable[int], include_relations: bool) -> list[GraphRecord]:
    return [
        build_typed_graph(target, frame, include_spatial_relations=include_relations)
        for target, frame, _ in (generate_synthetic_frame(seed) for seed in seeds)
    ]


def _loss(outputs: Mapping[str, torch.Tensor], batch: GraphBatch) -> torch.Tensor:
    candidate_loss = F.binary_cross_entropy_with_logits(
        outputs["identity_logits"],
        batch.candidate_labels,
        pos_weight=outputs["identity_logits"].new_tensor(3.0),
    )
    ranking_losses = []
    for graph_index in range(batch.graph_count):
        if batch.none_targets[graph_index] >= 0.5:
            continue
        mask = batch.candidate_graph == graph_index
        target_index = batch.candidate_labels[mask].argmax().view(1)
        ranking_losses.append(F.cross_entropy(outputs["identity_logits"][mask].view(1, -1), target_index))
    ranking_loss = torch.stack(ranking_losses).mean() if ranking_losses else candidate_loss.new_zeros(())
    none_loss = F.binary_cross_entropy_with_logits(outputs["none_logits"], batch.none_targets)
    reliability_loss = F.mse_loss(outputs["reliability"], batch.reliability_targets)
    return candidate_loss + 0.80 * ranking_loss + 0.35 * none_loss + 0.20 * reliability_loss


def train_model(
    *,
    include_relations: bool,
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[TypedSemanticReferentGraphNet, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_records = _training_records(train_seeds, include_relations)
    validation_records = _training_records(validation_seeds, include_relations)
    model = TypedSemanticReferentGraphNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = random.Random(seed)
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_records)))
        generator.shuffle(order)
        train_losses = []
        for start in range(0, len(order), batch_size):
            records = [train_records[index] for index in order[start : start + batch_size]]
            batch = GraphBatch.from_records(records, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(batch), batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        candidate_correct = 0
        candidate_total = 0
        present_top1_correct = 0
        present_graphs = 0
        with torch.inference_mode():
            for start in range(0, len(validation_records), batch_size):
                records = validation_records[start : start + batch_size]
                batch = GraphBatch.from_records(records, device)
                outputs = model(batch)
                validation_losses.append(float(_loss(outputs, batch).cpu()))
                predictions = torch.sigmoid(outputs["identity_logits"]) >= 0.5
                candidate_correct += int((predictions == (batch.candidate_labels >= 0.5)).sum().cpu())
                candidate_total += batch.candidate_labels.numel()
                for graph_index in range(batch.graph_count):
                    if batch.none_targets[graph_index] >= 0.5:
                        continue
                    mask = batch.candidate_graph == graph_index
                    present_top1_correct += int(
                        outputs["identity_logits"][mask].argmax().item()
                        == batch.candidate_labels[mask].argmax().item()
                    )
                    present_graphs += 1
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "validation_candidate_binary_accuracy": candidate_correct / candidate_total,
                "validation_present_top1": present_top1_correct / max(1, present_graphs),
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise AssertionError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "include_relations": include_relations,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_validation_loss": best_validation,
        "history": history,
    }


def _box(value: Mapping[str, Any]) -> Box:
    return Box(float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))


def frame_from_record(value: Mapping[str, Any]) -> Frame:
    return Frame(
        str(value["episode_id"]),
        int(value["frame_index"]),
        str(value["viewpoint"]),
        tuple(Candidate(str(item["candidate_id"]), _box(item["box"])) for item in value["candidates"]),
        tuple(Token(str(item["text"]), _box(item["box"]), float(item["confidence"])) for item in value["tokens"]),
        float(value["blur"]),
        float(value["perspective"]),
        str(value["truth"]),
        str(value["expected_state"]),
        str(value["note"]),
    )


def predict_scores(
    model: TypedSemanticReferentGraphNet,
    target: TargetGraph,
    frame: Frame,
    *,
    include_relations: bool,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    record = build_typed_graph(target, frame, include_spatial_relations=include_relations)
    batch = GraphBatch.from_records([record], device)
    with torch.inference_mode():
        outputs = model(batch)
        probabilities = torch.sigmoid(outputs["identity_logits"]).cpu().tolist()
        diagnostics = {
            "learned_reliability_diagnostic_only": float(outputs["reliability"][0].cpu()),
            "learned_none_diagnostic_only": float(torch.sigmoid(outputs["none_logits"])[0].cpu()),
        }
    return {
        candidate_id: {"score": float(score), "identity_probability": float(score)}
        for candidate_id, score in zip(record.candidate_ids, probabilities, strict=True)
    }, diagnostics


def evaluate_model_arm(
    source_rows: Sequence[Mapping[str, Any]],
    model: TypedSemanticReferentGraphNet,
    *,
    include_relations: bool,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invariance_errors = []
    episode_ids = list(dict.fromkeys(str(row["episode_id"]) for row in source_rows))
    for episode_id in episode_ids:
        episode = [row for row in source_rows if row["episode_id"] == episode_id]
        first_frame = frame_from_record(episode[0]["frame"])
        belief = ReferentBelief(candidate.candidate_id for candidate in first_frame.candidates)
        for source_row in episode:
            frame = frame_from_record(source_row["frame"])
            scores, diagnostics = predict_scores(
                model, TARGET, frame, include_relations=include_relations, device=device
            )
            decision = belief.update(frame, scores)
            permuted = Frame(
                frame.episode_id,
                frame.frame_index,
                frame.viewpoint,
                tuple(reversed(frame.candidates)),
                frame.tokens,
                frame.blur,
                frame.perspective,
                frame.truth,
                frame.expected_state,
                frame.note,
            )
            permuted_scores, _ = predict_scores(
                model, TARGET, permuted, include_relations=include_relations, device=device
            )
            invariance_errors.append(max(abs(scores[key]["score"] - permuted_scores[key]["score"]) for key in scores))
            rows.append(
                {
                    "episode_id": frame.episode_id,
                    "frame_index": frame.frame_index,
                    "truth": frame.truth,
                    "expected_state": frame.expected_state,
                    "frame": asdict(frame),
                    "decision": decision,
                    "diagnostic_heads": diagnostics,
                }
            )
    adapted = [
        {**row, "arm": row.pop("decision")}
        for row in rows
    ]
    metrics = _metrics(adapted, "arm")
    evaluable_target_frames = sum(
        row["truth"] != "NONE" and row["expected_state"] != "UNKNOWN" for row in adapted
    )
    metrics.update(
        {
            "evaluable_target_frames": evaluable_target_frames,
            "target_recall_evaluable": metrics["target_correct_locks"] / max(1, evaluable_target_frames),
            "none_accuracy": metrics["none_correct"] / max(1, metrics["absent_frames"]),
            "unknown_preservation_rate": metrics["unknown_preserved"] / max(1, metrics["unknown_frames"]),
            "directory_false_bindings": sum(
                row["episode_id"] in {"directory_binding", "directory_absence_burst"}
                and row["arm"]["state"] == "TARGET"
                and row["arm"]["selected_candidate"] != row["truth"]
                for row in adapted
            ),
            "hard_neighbor_wrong_locks": sum(
                row["episode_id"] in {"adjacent_301_302_320", "suffix_302a"}
                and row["arm"]["state"] == "TARGET"
                and row["arm"]["selected_candidate"] != row["truth"]
                for row in adapted
            ),
            "candidate_permutation_max_abs_error": max(invariance_errors, default=0.0),
            "candidate_permutation_invariant_frames": sum(error <= 1e-5 for error in invariance_errors),
        }
    )
    return adapted, metrics


def _baseline_extended_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    metrics = _metrics(rows, arm)
    evaluable_target_frames = sum(
        row["truth"] != "NONE" and row["expected_state"] != "UNKNOWN" for row in rows
    )
    metrics.update(
        {
            "evaluable_target_frames": evaluable_target_frames,
            "target_recall_evaluable": metrics["target_correct_locks"] / max(1, evaluable_target_frames),
            "none_accuracy": metrics["none_correct"] / max(1, metrics["absent_frames"]),
            "unknown_preservation_rate": metrics["unknown_preserved"] / max(1, metrics["unknown_frames"]),
            "directory_false_bindings": sum(
                row["episode_id"] in {"directory_binding", "directory_absence_burst"}
                and row[arm]["state"] == "TARGET"
                and row[arm]["selected_candidate"] != row["truth"]
                for row in rows
            ),
            "hard_neighbor_wrong_locks": sum(
                row["episode_id"] in {"adjacent_301_302_320", "suffix_302a"}
                and row[arm]["state"] == "TARGET"
                and row[arm]["selected_candidate"] != row["truth"]
                for row in rows
            ),
            "candidate_permutation_max_abs_error": None,
            "candidate_permutation_invariant_frames": None,
        }
    )
    return metrics


def run(
    run_dir: Path,
    v2_1_raw_path: Path,
    *,
    train_count: int,
    validation_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite run: {run_dir}")
    run_dir.mkdir(parents=True)
    source = json.loads(v2_1_raw_path.read_text(encoding="utf-8"))
    source_rows = source["rows"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_seeds = list(range(seed, seed + train_count))
    validation_seeds = list(range(seed + 100_000, seed + 100_000 + validation_count))
    arms = {}
    arm_rows = {}
    training = {}
    for arm_name, include_relations, arm_seed in (
        ("v3_no_relation", False, seed + 1),
        ("v3_full", True, seed + 2),
    ):
        model, training_report = train_model(
            include_relations=include_relations,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=arm_seed,
        )
        model_path = run_dir / f"{arm_name}.pt"
        torch.save({"state_dict": model.state_dict(), "hidden": 64, "layers": 3}, model_path)
        rows, metrics = evaluate_model_arm(
            source_rows, model, include_relations=include_relations, device=device
        )
        arms[arm_name] = metrics
        arm_rows[arm_name] = rows
        training[arm_name] = {
            **training_report,
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256_file(model_path),
        }

    metrics = {
        "substring_fsm": _baseline_extended_metrics(source_rows, "baseline"),
        "v2_heuristic": _baseline_extended_metrics(source_rows, "v2"),
        **arms,
    }
    raw = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_v2_1_raw_path": str(v2_1_raw_path.resolve()),
        "source_v2_1_raw_sha256": _sha256_file(v2_1_raw_path),
        "rows": arm_rows,
    }
    _atomic_json(run_dir / "raw-decisions.json", raw)
    frozen_v2_source = Path(inspect.getsourcefile(ReferentBelief) or "")
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "SYNTHETIC_GRAPH_TRAIN_GENERATED_PIXEL_RAPIDOCR_TUNED_ON_DEVELOPMENT",
        "device": str(device),
        "torch": torch.__version__,
        "question": "Does typed relational message passing improve candidate evidence over V2 and a no-relation learned ablation while V2 belief remains frozen?",
        "architecture": {
            "node_types": NODE_TYPES,
            "relations": RELATIONS,
            "message_passing_layers": 3,
            "hidden_width": 64,
            "heads": ["candidate_identity_used", "reliability_diagnostic_only", "NONE_diagnostic_only"],
        },
        "frozen_v2_belief": {
            "source_path": str(frozen_v2_source.resolve()),
            "source_sha256": _sha256_file(frozen_v2_source),
            "observability_novelty_belief_none_and_terminal_thresholds": "unchanged",
        },
        "training": training,
        "metrics": metrics,
        "decision": {
            "v3_full_vs_v2_correct_terminal_delta": metrics["v3_full"]["correct_terminal_frames"] - metrics["v2_heuristic"]["correct_terminal_frames"],
            "v3_full_wrong_lock_delta": metrics["v3_full"]["wrong_locks"] - metrics["v2_heuristic"]["wrong_locks"],
            "relation_ablation_correct_terminal_delta": metrics["v3_full"]["correct_terminal_frames"] - metrics["v3_no_relation"]["correct_terminal_frames"],
            "relation_ablation_wrong_lock_delta": metrics["v3_full"]["wrong_locks"] - metrics["v3_no_relation"]["wrong_locks"],
        },
        "interpretation_boundary": [
            "Gradient training uses only deterministic domain-randomized synthetic graphs.",
            "Earlier V3 iterations inspected these V2.1 RapidOCR rows and expanded directory-layout randomization after a false binding; the final numbers are therefore tuned-on Development, not held-out test evidence.",
            "The V2.1 test images are generated pixels, not natural photographs; this is Development, not source-disjoint natural-photo validation.",
            "Reliability and NONE heads are trained and reported only as diagnostics; the used sequential belief, observability, novelty, NONE construction, and terminal thresholds are frozen V2 code.",
            "No V3-B learned belief, V4 active acquisition, Android, navigation, safety, or product behavior is evaluated.",
        ],
        "claim_ceiling": CLAIM_CEILING,
        "raw_decisions_sha256": _sha256_file(run_dir / "raw-decisions.json"),
    }
    _atomic_json(run_dir / "final-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--v2-1-raw", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2400)
    parser.add_argument("--validation-count", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=302)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(
        args.run_dir.resolve(),
        args.v2_1_raw.resolve(),
        train_count=args.train_count,
        validation_count=args.validation_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps({"metrics": report["metrics"], "decision": report["decision"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

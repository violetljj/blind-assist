"""SAGE-R V3-C authority-typed sign-to-destination graph.

The scorer deliberately separates OCR tokens from physical candidates.  OCR
tokens first belong to a frozen sign-carrier proposal; optional frozen visual
cue proposals are decoded from pixels; a small synthetic multi-task classifier
then predicts the carrier-to-candidate relation.  Only LABELS and POINTS
relations may create identity evidence in the full arm.  LISTS, NEAR and
UNRELATED remain diagnostic/search evidence.

This module is Development code.  Carrier/cue/candidate proposals are inputs,
not detector claims, and the frozen V2 sequential belief remains the terminal
decision rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic_anchor_graph_and_belief_v2 import (
    Box,
    Candidate,
    Frame,
    TargetGraph,
    Token,
    _edit_similarity,
    normalize_text,
)


SCHEMA_VERSION = "blindassist_sage_r_v3_c_authority_graph_v1"
RELATION_TYPES = ("LABELS", "POINTS", "LISTS", "NEAR", "UNRELATED")
RELATION_INDEX = {name: index for index, name in enumerate(RELATION_TYPES)}
GENERIC_TOKENS = {
    "AREA", "CLASSROOM", "EMERGENCY", "ENTRANCE", "EXIT", "GATE", "LAB",
    "LABORATORY", "LEVEL", "OFFICE", "PLATFORM", "ROOM", "STATION",
}
FEATURE_NAMES = (
    "generic_coverage", "decisive_coverage", "all_token_coverage",
    "carrier_token_count", "carrier_line_count", "carrier_area",
    "candidate_overlap", "candidate_horizontal_overlap", "candidate_inside",
    "candidate_above_alignment", "candidate_center_alignment",
    "cue_present", "cue_alignment", "cue_confidence", "spatial_near",
)


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


def token_roles(target: TargetGraph) -> dict[str, tuple[str, ...]]:
    normalized = tuple(normalize_text(value) for value in target.tokens)
    explicit = tuple(value for value in normalized if any(ch.isdigit() for ch in value))
    if explicit:
        decisive = explicit
    else:
        non_generic = tuple(value for value in normalized if value not in GENERIC_TOKENS)
        decisive = non_generic or normalized
    generic = tuple(value for value in normalized if value not in decisive)
    return {"generic": generic, "decisive": decisive, "all": normalized}


@dataclass(frozen=True)
class SignCarrier:
    carrier_id: str
    box: Box


@dataclass(frozen=True)
class DirectionCue:
    cue_id: str
    carrier_id: str
    box: Box
    orientation: str
    confidence: float


def _box_iou(a: Box, b: Box) -> float:
    width = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    intersection = width * height
    a_area = max(0.0, a.x1 - a.x0) * max(0.0, a.y1 - a.y0)
    b_area = max(0.0, b.x1 - b.x0) * max(0.0, b.y1 - b.y0)
    union = max(1e-8, a_area + b_area - intersection)
    return intersection / union


def _contains(outer: Box, inner: Box, margin: float = 0.015) -> bool:
    return (
        outer.x0 - margin <= inner.cx <= outer.x1 + margin
        and outer.y0 - margin <= inner.cy <= outer.y1 + margin
    )


def _crop(image: np.ndarray, box: Box) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, int(round(box.x0 * width))))
    y0 = max(0, min(height - 1, int(round(box.y0 * height))))
    x1 = max(x0 + 1, min(width, int(round(box.x1 * width))))
    y1 = max(y0 + 1, min(height, int(round(box.y1 * height))))
    return image[y0:y1, x0:x1]


def _arrow_template(orientation: str, thickness: int, tip_length: float) -> np.ndarray:
    canvas = np.zeros((64, 64), dtype=np.uint8)
    endpoints = {
        "RIGHT": ((9, 32), (55, 32)), "LEFT": ((55, 32), (9, 32)),
        "DOWN": ((32, 9), (32, 55)), "UP": ((32, 55), (32, 9)),
    }
    start, end = endpoints[orientation]
    cv2.arrowedLine(canvas, start, end, 255, thickness, cv2.LINE_AA, tipLength=tip_length)
    return canvas > 96


_ARROW_TEMPLATES = {
    orientation: [
        _arrow_template(orientation, thickness, tip)
        for thickness in (7, 11, 15)
        for tip in (0.32, 0.45, 0.58)
    ]
    for orientation in ("LEFT", "RIGHT", "UP", "DOWN")
}


def infer_direction_cue(image: np.ndarray, box: Box, cue_id: str, carrier_id: str) -> DirectionCue:
    crop = _crop(image, box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    side = max(gray.shape[:2])
    padded = np.full((side, side), int(np.median(gray)), dtype=np.uint8)
    offset_y = (side - gray.shape[0]) // 2
    offset_x = (side - gray.shape[1]) // 2
    padded[offset_y:offset_y + gray.shape[0], offset_x:offset_x + gray.shape[1]] = gray
    resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [binary > 0, binary == 0]
    best_orientation, best_score = "UNKNOWN", -1.0
    second_score = -1.0
    for orientation, templates in _ARROW_TEMPLATES.items():
        orientation_score = -1.0
        for mask in masks:
            for template in templates:
                intersection = np.logical_and(mask, template).sum()
                union = np.logical_or(mask, template).sum()
                score = float(intersection / max(1, union))
                orientation_score = max(orientation_score, score)
        if orientation_score > best_score:
            second_score = best_score
            best_orientation, best_score = orientation, orientation_score
        else:
            second_score = max(second_score, orientation_score)
    margin = max(0.0, best_score - second_score)
    confidence = float(np.clip((best_score - 0.16) / 0.34 + 1.5 * margin, 0.0, 1.0))
    if confidence < 0.18:
        best_orientation = "UNKNOWN"
    return DirectionCue(cue_id, carrier_id, box, best_orientation, confidence)


def _coverage(expected: Sequence[str], tokens: Sequence[Token]) -> float:
    if not expected:
        return 1.0
    return float(np.mean([
        max((_edit_similarity(value, token.text) for token in tokens), default=0.0)
        for value in expected
    ]))


def _line_count(tokens: Sequence[Token]) -> int:
    centers = sorted(token.box.cy for token in tokens)
    lines: list[float] = []
    for center in centers:
        match = next((index for index, value in enumerate(lines) if abs(center - value) <= 0.035), None)
        if match is None:
            lines.append(center)
        else:
            lines[match] = (lines[match] + center) / 2
    return len(lines)


def _cue_alignment(cue: DirectionCue | None, carrier: SignCarrier, candidate: Candidate) -> float:
    if cue is None or cue.orientation == "UNKNOWN":
        return 0.0
    dx = candidate.box.cx - carrier.box.cx
    dy = candidate.box.cy - carrier.box.cy
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0
    unit = {"LEFT": (-1.0, 0.0), "RIGHT": (1.0, 0.0), "UP": (0.0, -1.0), "DOWN": (0.0, 1.0)}[cue.orientation]
    cosine = (dx * unit[0] + dy * unit[1]) / distance
    return max(0.0, cosine) * math.exp(-0.45 * distance)


def relation_features(
    target: TargetGraph,
    tokens: Sequence[Token],
    carrier: SignCarrier,
    candidate: Candidate,
    cue: DirectionCue | None,
) -> np.ndarray:
    roles = token_roles(target)
    carrier_tokens = tuple(token for token in tokens if _contains(carrier.box, token.box, 0.025))
    overlap = _box_iou(carrier.box, candidate.box)
    horizontal_overlap = carrier.box.horizontal_overlap(candidate.box)
    inside = float(_contains(candidate.box, carrier.box, 0.0))
    vertical_gap = candidate.box.y0 - carrier.box.y1
    above = math.exp(-abs(vertical_gap - 0.025) / 0.12) if vertical_gap >= -0.04 else 0.0
    center_alignment = math.exp(-abs(carrier.box.cx - candidate.box.cx) / 0.18)
    center_distance = math.hypot(carrier.box.cx - candidate.box.cx, carrier.box.cy - candidate.box.cy)
    return np.asarray(
        [
            _coverage(roles["generic"], carrier_tokens),
            _coverage(roles["decisive"], carrier_tokens),
            _coverage(roles["all"], carrier_tokens),
            min(1.0, len(carrier_tokens) / 12.0),
            min(1.0, _line_count(carrier_tokens) / 6.0),
            min(1.0, (carrier.box.x1 - carrier.box.x0) * (carrier.box.y1 - carrier.box.y0) / 0.22),
            overlap,
            horizontal_overlap,
            inside,
            above,
            center_alignment,
            float(cue is not None and cue.orientation != "UNKNOWN"),
            _cue_alignment(cue, carrier, candidate),
            0.0 if cue is None else cue.confidence,
            math.exp(-center_distance / 0.30),
        ],
        dtype=np.float32,
    )


class AuthorityRelationNet(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.relation_head = nn.Linear(hidden, len(RELATION_TYPES))
        self.decisive_head = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "relation_logits": self.relation_head(hidden),
            "decisive_logit": self.decisive_head(hidden).squeeze(-1),
        }


def _synthetic_row(rng: random.Random, relation: str) -> tuple[np.ndarray, int, float]:
    def clipped(mean: float, spread: float) -> float:
        return float(np.clip(rng.gauss(mean, spread), 0.0, 1.0))
    full = clipped(0.91 if relation in {"LABELS", "POINTS", "LISTS"} else 0.38, 0.13)
    decisive = clipped(full - rng.uniform(-0.05, 0.12), 0.08)
    generic = clipped(full + rng.uniform(-0.08, 0.08), 0.07)
    if relation == "LABELS":
        values = [generic, decisive, full, clipped(.28,.16), clipped(.28,.14), clipped(.36,.30), clipped(.18,.13), clipped(.86,.12), clipped(.50,.30), clipped(.78,.17), clipped(.88,.10), 0.0, 0.0, 0.0, clipped(.74,.18)]
    elif relation == "POINTS":
        values = [generic, decisive, full, clipped(.28,.15), clipped(.28,.14), clipped(.30,.24), clipped(.02,.03), clipped(.20,.18), 0.0, clipped(.16,.13), clipped(.24,.15), 1.0, clipped(.88,.10), clipped(.82,.12), clipped(.35,.16)]
    elif relation == "LISTS":
        values = [generic, decisive, full, clipped(.82,.10), clipped(.82,.10), clipped(.68,.20), clipped(.06,.06), clipped(.45,.25), clipped(.12,.15), clipped(.35,.22), clipped(.48,.20), clipped(.05,.10), clipped(.04,.08), clipped(.08,.12), clipped(.55,.20)]
    elif relation == "NEAR":
        values = [generic, decisive, full, clipped(.28,.16), clipped(.32,.18), clipped(.22,.13), clipped(.03,.05), clipped(.65,.18), clipped(.20,.20), clipped(.58,.20), clipped(.72,.16), 0.0, 0.0, 0.0, clipped(.82,.12)]
    else:
        values = [generic, decisive, full, clipped(.25,.16), clipped(.26,.17), clipped(.25,.17), clipped(.01,.02), clipped(.12,.14), 0.0, clipped(.12,.14), clipped(.22,.18), clipped(.10,.20), clipped(.08,.15), clipped(.12,.18), clipped(.25,.18)]
    decisive_target = float(decisive >= 0.78)
    return np.asarray(values, dtype=np.float32), RELATION_INDEX[relation], decisive_target


def train_relation_model(
    *, seed: int = 403, examples_per_type: int = 1800, epochs: int = 32,
    device: torch.device | None = None,
) -> tuple[AuthorityRelationNet, dict[str, Any]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = [_synthetic_row(rng, relation) for relation in RELATION_TYPES for _ in range(examples_per_type)]
    rng.shuffle(rows)
    features = torch.from_numpy(np.stack([row[0] for row in rows])).to(device)
    labels = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    decisive = torch.tensor([row[2] for row in rows], dtype=torch.float32, device=device)
    split = int(len(rows) * 0.82)
    model = AuthorityRelationNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(split, device=device)
        train_losses = []
        for start in range(0, split, 256):
            index = permutation[start:start + 256]
            output = model(features[index])
            loss = F.cross_entropy(output["relation_logits"], labels[index]) + 0.45 * F.binary_cross_entropy_with_logits(output["decisive_logit"], decisive[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            output = model(features[split:])
            validation = F.cross_entropy(output["relation_logits"], labels[split:]) + 0.45 * F.binary_cross_entropy_with_logits(output["decisive_logit"], decisive[split:])
            accuracy = float((output["relation_logits"].argmax(-1) == labels[split:]).float().mean().cpu())
        validation_value = float(validation.cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_value, "relation_accuracy": accuracy})
        if validation_value < best_validation:
            best_validation = validation_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("relation training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "seed": seed, "examples": len(rows), "examples_per_type": examples_per_type,
        "epochs": epochs, "best_validation_loss": best_validation,
        "final_validation_relation_accuracy": history[-1]["relation_accuracy"], "history": history,
    }


def score_candidates(
    model: AuthorityRelationNet,
    target: TargetGraph,
    frame: Frame,
    image: np.ndarray,
    carriers: Sequence[SignCarrier],
    cue_boxes: Sequence[Mapping[str, Any]],
    *, authority_typing: bool,
    device: torch.device | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    device = device or next(model.parameters()).device
    cues = [
        DirectionCue(
            str(item["cue_id"]), str(item["carrier_id"]), Box(*item["box"]),
            str(item["orientation"]), float(item.get("confidence", 1.0)),
        )
        if item.get("orientation")
        else infer_direction_cue(image, Box(*item["box"]), str(item["cue_id"]), str(item["carrier_id"]))
        for item in cue_boxes
    ]
    cue_by_carrier = {cue.carrier_id: cue for cue in cues}
    records: list[tuple[str, str, np.ndarray]] = []
    for carrier in carriers:
        for candidate in frame.candidates:
            records.append((carrier.carrier_id, candidate.candidate_id, relation_features(target, frame.tokens, carrier, candidate, cue_by_carrier.get(carrier.carrier_id))))
    tensor = torch.from_numpy(np.stack([row[2] for row in records])).to(device) if records else torch.empty((0, len(FEATURE_NAMES)), device=device)
    with torch.inference_mode():
        output = model(tensor)
        relation_probabilities = torch.softmax(output["relation_logits"], -1).cpu().numpy()
        decisive_probabilities = torch.sigmoid(output["decisive_logit"]).cpu().numpy()
    scores = {candidate.candidate_id: {"score": 0.04} for candidate in frame.candidates}
    diagnostics = []
    for (carrier_id, candidate_id, features), probabilities, decisive_probability in zip(records, relation_probabilities, decisive_probabilities, strict=True):
        relation = {name: float(probabilities[index]) for index, name in enumerate(RELATION_TYPES)}
        if authority_typing:
            authority = relation["LABELS"] + relation["POINTS"]
        else:
            authority = 1.0 - relation["UNRELATED"]
        decisive_gate = float(features[1])
        learned_gate = float(decisive_probability)
        completeness = min(decisive_gate, learned_gate)
        score = 0.04 + 0.93 * authority * completeness
        if score > scores[candidate_id]["score"]:
            scores[candidate_id] = {
                "score": float(score), "authority_support": float(authority),
                "decisive_coverage": decisive_gate, "learned_decisive_completeness": learned_gate,
                "carrier_id": carrier_id, "top_relation": max(relation, key=relation.get),
            }
        diagnostics.append({
            "carrier_id": carrier_id, "candidate_id": candidate_id,
            "features": dict(zip(FEATURE_NAMES, map(float, features), strict=True)),
            "relation_probabilities": relation, "learned_decisive_completeness": learned_gate,
        })
    return scores, {
        "token_roles": token_roles(target),
        "cues": [asdict(cue) for cue in cues],
        "relations": diagnostics,
    }


def save_checkpoint(path: Path, model: AuthorityRelationNet, training: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "state_dict": model.state_dict(), "hidden": 48, "training": dict(training)}, path)
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[AuthorityRelationNet, dict[str, Any]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("V3-C checkpoint schema mismatch")
    model = AuthorityRelationNet(hidden=int(checkpoint["hidden"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["training"]

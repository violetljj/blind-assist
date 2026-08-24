"""Frozen spatial-layout identity verification over fresh Washington RGB-D objects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    dinov2_local_appearance_probe as dino,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    visible_identity_probe as base,
)


PROTOCOL_ID = "SPATIAL_LAYOUT_IDENTITY_VERIFICATION_V0"
SCHEMA_VERSION = "blindassist_spatial_layout_identity_verification_v0"
ARCHIVE_URL = "https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/rgbd-dataset_eval.zip"
ARCHIVE_EXPECTED_BYTES = 673_456_874
RGB_MEMBER_RE = re.compile(
    r"^(?P<category>.+)_(?P<instance>\d+)_(?P<video>\d+)_(?P<frame>\d+)_crop\.png$",
    re.IGNORECASE,
)
REFERENCE_VIDEO = 1
CANDIDATE_VIDEO = 3
REFERENCE_QUANTILE = 0.50
CANDIDATE_QUANTILES = (0.25, 0.50, 0.75)
TOP_MUTUAL_K = 64
COARSE_GRID = 4
LOCAL_SOURCE_K = 4
LOCAL_TARGET_K = 8
PROCRUSTES_SCALE = 0.5
CONTROL_RETENTION_GATE = 0.80
STABLE_MIN_INSTANCES = 5
STABLE_RECALL_GATE = 0.50
DIRECTION_TOLERANCE = 1e-9
CLAIM_CEILING = (
    "DEVELOPMENT_CONTROLLED_PUBLIC_OBJECT_IDENTITY_RANKING_SIGNAL_ONLY_NO_OPEN_SET_CALIBRATION_"
    "NONE_PHYSICAL_INSTANCE_AUTHORITY_P1_CONTROL_SAFETY_OR_PRODUCT_CLAIM"
)


class SpatialLayoutExperimentError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpatialLayoutExperimentError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _body_hash(value: Mapping[str, Any]) -> str:
    return base._body_hash(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_body(value: Mapping[str, Any], label: str) -> None:
    _require(value.get("body_sha256") == _body_hash(value), f"{label} body SHA mismatch")


def protocol_payload(protocol_doc: Path) -> dict[str, Any]:
    _require(protocol_doc.is_file(), f"missing protocol document: {protocol_doc}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": _utc_now(),
        "protocol_document": {"path": str(protocol_doc.resolve()), "sha256": _sha256_file(protocol_doc)},
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__))},
        "dataset": {
            "name": "Washington_RGBD_Object_Dataset_Evaluation_Set",
            "archive_url": ARCHIVE_URL,
            "expected_bytes": ARCHIVE_EXPECTED_BYTES,
            "license": "NON_COMMERCIAL_RESEARCH_EDUCATIONAL",
            "rgb_member_pattern": RGB_MEMBER_RE.pattern,
            "reference_video": REFERENCE_VIDEO,
            "candidate_video": CANDIDATE_VIDEO,
            "reference_quantile": REFERENCE_QUANTILE,
            "candidate_quantiles": list(CANDIDATE_QUANTILES),
            "hard_negative_rule": "next_numeric_instance_within_lexicographic_category_cycle",
            "frame_rule": "numeric_frame_sort_then_floor(q*(n-1))",
        },
        "arms": {
            "baseline": {
                "name": "FROZEN_DINOV2_S_SYMMETRIC_MEAN_NEAREST_PATCH",
                "repository": dino.MODEL_REPOSITORY,
                "revision": dino.MODEL_REVISION,
                "model_files": dino.MODEL_FILES,
            },
            "layout": {
                "name": "ANALYTIC_RECIPROCAL_SPATIAL_LAYOUT",
                "top_mutual_k": TOP_MUTUAL_K,
                "coarse_grid": COARSE_GRID,
                "local_source_k": LOCAL_SOURCE_K,
                "local_target_k": LOCAL_TARGET_K,
                "procrustes_scale": PROCRUSTES_SCALE,
                "aggregation": "equal_weight_geometric_mean_then_bidirectional_min",
                "trained_parameters": 0,
            },
        },
        "gates": {
            "rescue_gt_collateral": True,
            "control_retention_min": CONTROL_RETENTION_GATE,
            "direction_invariance": 1.0,
            "direction_absolute_tolerance": DIRECTION_TOLERANCE,
            "candidate_permutation_invariance": 1.0,
            "stable_baseline_wrong_all_three_quantiles_min_instances": STABLE_MIN_INSTANCES,
            "stable_layout_target_outrank_min": STABLE_RECALL_GATE,
        },
        "forbidden": [
            "TRAINING", "CALIBRATION", "TARGET_ABSENT", "NONE", "NEARID", "PDM", "FUSION",
            "DEEP_SETS", "MULTIPLE_REFERENCES", "THRESHOLD_SWEEP", "ACTIVE_SEARCH", "TRACKER",
            "P1", "DEFAULT_APP",
        ],
        "recovery": {
            "before_final_report": "resume_only_under_identical_protocol_roster_archive_model_and_code_hashes",
            "after_final_report": "refuse_overwrite_or_second_adjudication",
            "external_model_calls": 0,
        },
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "SPATIAL_LAYOUT_PROTOCOL_FROZEN_NO_OUTCOME",
    }
    payload["body_sha256"] = _body_hash(payload)
    return payload


def freeze_protocol(protocol_doc: Path, output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite protocol freeze: {output}")
    payload = protocol_payload(protocol_doc)
    _atomic_json(output, payload)
    return payload


def _parse_rgb_members(archive: Path) -> dict[tuple[str, int, int], list[tuple[int, str]]]:
    _require(archive.is_file(), f"missing archive: {archive}")
    _require(archive.stat().st_size == ARCHIVE_EXPECTED_BYTES, "archive byte count drifted")
    indexed: dict[tuple[str, int, int], list[tuple[int, str]]] = {}
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename.replace("\\", "/")).name
            match = RGB_MEMBER_RE.fullmatch(name)
            if match is None:
                continue
            _require(info.filename not in seen, "duplicate archive member")
            seen.add(info.filename)
            key = (match.group("category").lower(), int(match.group("instance")), int(match.group("video")))
            indexed.setdefault(key, []).append((int(match.group("frame")), info.filename))
    for key, rows in indexed.items():
        rows.sort(key=lambda item: (item[0], item[1]))
        _require(len({frame for frame, _ in rows}) == len(rows), f"duplicate numeric frame for {key}")
    return indexed


def _choose_member(rows: Sequence[tuple[int, str]], quantile: float) -> str:
    _require(bool(rows), "cannot choose from empty frame sequence")
    return str(rows[math.floor(quantile * (len(rows) - 1))][1])


def _slot_for_pair(pair_id: str) -> str:
    return "A" if hashlib.sha256(pair_id.encode("utf-8")).digest()[0] % 2 == 0 else "B"


def build_roster(protocol_path: Path, archive: Path) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    _verify_body(protocol, "protocol")
    _require(protocol["protocol_id"] == PROTOCOL_ID, "protocol id drifted")
    indexed = _parse_rgb_members(archive)
    by_category: dict[str, list[int]] = {}
    for category, instance, video in indexed:
        if video != REFERENCE_VIDEO:
            continue
        if (category, instance, CANDIDATE_VIDEO) not in indexed:
            continue
        by_category.setdefault(category, []).append(instance)
    by_category = {
        category: sorted(set(instances))
        for category, instances in sorted(by_category.items())
        if len(set(instances)) >= 2
    }
    _require(len(by_category) >= 40, "fewer than 40 eligible categories")
    samples: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []

    def add_sample(category: str, instance: int, video: int, quantile: float) -> str:
        sample_id = f"{category}-i{instance:02d}-v{video}-q{int(round(quantile*100)):03d}"
        rows = indexed[(category, instance, video)]
        samples.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "category": category,
                "physical_instance": f"{category}_{instance}",
                "instance_number": instance,
                "video": video,
                "quantile": quantile,
                "archive_member": _choose_member(rows, quantile),
            },
        )
        return sample_id

    for category, instances in by_category.items():
        for position, instance in enumerate(instances):
            hard = instances[(position + 1) % len(instances)]
            reference = add_sample(category, instance, REFERENCE_VIDEO, REFERENCE_QUANTILE)
            for quantile in CANDIDATE_QUANTILES:
                pair_id = f"{category}-i{instance:02d}-v{CANDIDATE_VIDEO}-q{int(quantile*100):03d}"
                target = add_sample(category, instance, CANDIDATE_VIDEO, quantile)
                distractor = add_sample(category, hard, CANDIDATE_VIDEO, quantile)
                target_slot = _slot_for_pair(pair_id)
                hard_slot = "B" if target_slot == "A" else "A"
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "category": category,
                        "reference_instance": f"{category}_{instance}",
                        "hard_instance": f"{category}_{hard}",
                        "candidate_quantile": quantile,
                        "reference": reference,
                        "candidate_slots": {target_slot: target, hard_slot: distractor},
                        "target_slot": target_slot,
                        "hard_slot": hard_slot,
                    }
                )
    roster = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "protocol_body_sha256": protocol["body_sha256"],
        "archive": {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": _sha256_file(archive),
        },
        "eligible_categories": by_category,
        "counts": {
            "categories": len(by_category),
            "physical_instances": sum(len(values) for values in by_category.values()),
            "samples": len(samples),
            "pairs": len(pairs),
        },
        "samples": [samples[key] for key in sorted(samples)],
        "pairs": pairs,
        "terminal": "FRESH_ROSTER_FROZEN_NO_PIXELS_DECODED",
    }
    validate_roster(roster)
    roster["body_sha256"] = _body_hash(roster)
    return roster


def validate_roster(roster: Mapping[str, Any]) -> None:
    if "body_sha256" in roster:
        _verify_body(roster, "roster")
    samples = roster["samples"]
    pairs = roster["pairs"]
    by_id = {row["sample_id"]: row for row in samples}
    _require(len(by_id) == len(samples), "duplicate sample id")
    _require(len(pairs) == roster["counts"]["physical_instances"] * len(CANDIDATE_QUANTILES), "pair count drift")
    for pair in pairs:
        reference = by_id[pair["reference"]]
        target = by_id[pair["candidate_slots"][pair["target_slot"]]]
        hard = by_id[pair["candidate_slots"][pair["hard_slot"]]]
        _require(reference["physical_instance"] == target["physical_instance"], "target identity mismatch")
        _require(reference["physical_instance"] != hard["physical_instance"], "hard identity collapsed")
        _require(reference["category"] == target["category"] == hard["category"], "category mismatch")
        _require(reference["video"] == REFERENCE_VIDEO, "reference video drift")
        _require(target["video"] == hard["video"] == CANDIDATE_VIDEO, "candidate video drift")
        _require(target["quantile"] == hard["quantile"] == pair["candidate_quantile"], "candidate context drift")


def freeze_roster(protocol_path: Path, archive: Path, output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite roster: {output}")
    roster = build_roster(protocol_path, archive)
    _atomic_json(output, roster)
    return roster


def _patch_coordinates() -> np.ndarray:
    centers = (np.arange(dino.PATCH_SIDE, dtype=np.float64) + 0.5) / dino.PATCH_SIDE
    xs, ys = np.meshgrid(centers, centers)
    return np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)


PATCH_COORDINATES = _patch_coordinates()


def _convex_hull_fraction(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    area = float(cv2.contourArea(hull))
    maximum = ((dino.PATCH_SIDE - 1) / dino.PATCH_SIDE) ** 2
    return min(1.0, max(0.0, area / maximum))


def _cell_coverage(points: np.ndarray) -> float:
    cells = np.minimum(COARSE_GRID - 1, np.floor(points * COARSE_GRID).astype(np.int64))
    occupied = len({(int(x), int(y)) for x, y in cells})
    return occupied / float(COARSE_GRID * COARSE_GRID)


def _local_preservation(source: np.ndarray, target: np.ndarray) -> float:
    count = len(source)
    if count <= LOCAL_SOURCE_K:
        return 0.0
    source_dist = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    target_dist = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
    np.fill_diagonal(source_dist, np.inf)
    np.fill_diagonal(target_dist, np.inf)
    source_neighbors = np.argsort(source_dist, axis=1)[:, : min(LOCAL_SOURCE_K, count - 1)]
    target_neighbors = np.argsort(target_dist, axis=1)[:, : min(LOCAL_TARGET_K, count - 1)]
    values = []
    for index in range(count):
        values.append(len(set(source_neighbors[index]) & set(target_neighbors[index])) / source_neighbors.shape[1])
    return float(np.mean(values))


def _procrustes_consistency(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    if len(reference) < 3:
        return 0.0, float("inf")
    ref = reference - reference.mean(axis=0, keepdims=True)
    cand = candidate - candidate.mean(axis=0, keepdims=True)
    ref_scale = math.sqrt(float(np.mean(np.sum(ref * ref, axis=1))))
    cand_scale = math.sqrt(float(np.mean(np.sum(cand * cand, axis=1))))
    if ref_scale <= 1e-12 or cand_scale <= 1e-12:
        return 0.0, float("inf")
    ref /= ref_scale
    cand /= cand_scale
    u, _, vt = np.linalg.svd(ref.T @ cand, full_matrices=False)
    rotation = u @ vt
    aligned = ref @ rotation
    residual = float(np.median(np.linalg.norm(aligned - cand, axis=1)))
    return float(math.exp(-residual / PROCRUSTES_SCALE)), residual


def _geometric_mean(values: Sequence[float]) -> float:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, 1.0)
    return float(np.exp(np.log(clipped).mean()))


def spatial_layout_score(reference_patches: np.ndarray, candidate_patches: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference_patches, dtype=np.float64)
    candidate = np.asarray(candidate_patches, dtype=np.float64)
    _require(reference.shape == (dino.PATCH_COUNT, dino.FEATURE_DIM), "reference patch shape drifted")
    _require(candidate.shape == (dino.PATCH_COUNT, dino.FEATURE_DIM), "candidate patch shape drifted")
    similarities = reference @ candidate.T
    ref_best = similarities.argmax(axis=1)
    cand_best = similarities.argmax(axis=0)
    mutual = [(i, int(j), float(similarities[i, j])) for i, j in enumerate(ref_best) if cand_best[j] == i]
    mutual.sort(key=lambda row: (-row[2], row[0], row[1]))
    selected = mutual[:TOP_MUTUAL_K]
    if len(selected) < 3:
        return {
            "score": 0.0,
            "forward_score": 0.0,
            "reverse_score": 0.0,
            "mutual_count": len(mutual),
            "selected_count": len(selected),
            "direction_delta": 0.0,
            "components": {},
        }
    ref_indices = np.asarray([row[0] for row in selected], dtype=np.int64)
    cand_indices = np.asarray([row[1] for row in selected], dtype=np.int64)
    ref_xy = PATCH_COORDINATES[ref_indices]
    cand_xy = PATCH_COORDINATES[cand_indices]
    ref_unique = len(set(int(value) for value in cand_best)) / dino.PATCH_COUNT
    cand_unique = len(set(int(value) for value in ref_best)) / dino.PATCH_COUNT
    mutual_support = min(1.0, len(mutual) / TOP_MUTUAL_K)
    procrustes, residual = _procrustes_consistency(ref_xy, cand_xy)

    def directed(source: np.ndarray, target: np.ndarray, uniqueness: float) -> tuple[float, dict[str, float]]:
        components = {
            "cell_coverage": _cell_coverage(source),
            "dispersion": _convex_hull_fraction(source),
            "local_preservation": _local_preservation(source, target),
            "procrustes_consistency": procrustes,
            "conflict_consistency": uniqueness,
            "mutual_support": mutual_support,
        }
        return _geometric_mean(list(components.values())), components

    forward, forward_components = directed(ref_xy, cand_xy, ref_unique)
    reverse, reverse_components = directed(cand_xy, ref_xy, cand_unique)
    score = min(forward, reverse)
    return {
        "score": score,
        "forward_score": forward,
        "reverse_score": reverse,
        "mutual_count": len(mutual),
        "selected_count": len(selected),
        "procrustes_median_residual": residual,
        "direction_delta": abs(forward - reverse),
        "components": {"forward": forward_components, "reverse": reverse_components},
    }


def _decode_rgb(bundle: zipfile.ZipFile, member: str) -> np.ndarray:
    encoded = np.frombuffer(bundle.read(member), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    _require(image is not None, f"cannot decode RGB member: {member}")
    resized = cv2.resize(image, (dino.INPUT_SIZE, dino.INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
    std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
    return np.transpose((rgb - mean) / std, (2, 0, 1))


def extract_features(roster: Mapping[str, Any], archive: Path, model_dir: Path, device: str) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    model = dino._validate_model(model_dir, device)
    encoder = dino.DenseEncoder(model_dir, device)
    sample_ids: list[str] = []
    features: list[np.ndarray] = []
    with zipfile.ZipFile(archive) as bundle:
        for start in range(0, len(roster["samples"]), dino.BATCH_SIZE):
            rows = roster["samples"][start : start + dino.BATCH_SIZE]
            tensors = [_decode_rgb(bundle, row["archive_member"]) for row in rows]
            encoded = encoder.encode(tensors)
            sample_ids.extend(str(row["sample_id"]) for row in rows)
            features.extend(encoded)
    return sample_ids, np.asarray(features, dtype=np.float32), {
        **model,
        "forward_batches": encoder.forward_batches,
        "encoded_crops": encoder.encoded_crops,
    }


def _winner(score_a: float, score_b: float) -> str:
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "TIE"


def score_pairs(roster: Mapping[str, Any], sample_ids: Sequence[str], features: np.ndarray) -> dict[str, Any]:
    lookup = {sample_id: features[index] for index, sample_id in enumerate(sample_ids)}
    rows = []
    all_mask = np.ones(dino.PATCH_COUNT, dtype=bool)
    direction_ok = 0
    direction_total = 0
    permutation_ok = 0
    for pair in roster["pairs"]:
        reference = lookup[pair["reference"]]
        arm_scores: dict[str, dict[str, Any]] = {"baseline": {}, "layout": {}}
        for slot in ("A", "B"):
            candidate = lookup[pair["candidate_slots"][slot]]
            baseline = dino.symmetric_local_score(reference, candidate, all_mask, all_mask)
            layout = spatial_layout_score(reference, candidate)
            reverse = spatial_layout_score(candidate, reference)
            delta = abs(float(layout["score"]) - float(reverse["score"]))
            direction_total += 1
            direction_ok += int(delta <= DIRECTION_TOLERANCE)
            layout["swapped_score"] = reverse["score"]
            layout["swap_delta"] = delta
            arm_scores["baseline"][slot] = baseline
            arm_scores["layout"][slot] = layout
        baseline_winner = _winner(
            float(arm_scores["baseline"]["A"]["symmetric_score"]),
            float(arm_scores["baseline"]["B"]["symmetric_score"]),
        )
        layout_winner = _winner(
            float(arm_scores["layout"]["A"]["score"]), float(arm_scores["layout"]["B"]["score"])
        )
        swapped_layout_winner = _winner(
            float(arm_scores["layout"]["B"]["score"]), float(arm_scores["layout"]["A"]["score"])
        )
        expected_swapped = "B" if layout_winner == "A" else "A" if layout_winner == "B" else "TIE"
        permutation_ok += int(swapped_layout_winner == expected_swapped)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "reference_instance": pair["reference_instance"],
                "candidate_quantile": pair["candidate_quantile"],
                "candidate_slots": pair["candidate_slots"],
                "baseline": {
                    "scores": arm_scores["baseline"],
                    "winner": baseline_winner,
                    "margin_a_minus_b": float(arm_scores["baseline"]["A"]["symmetric_score"])
                    - float(arm_scores["baseline"]["B"]["symmetric_score"]),
                },
                "layout": {
                    "scores": arm_scores["layout"],
                    "winner": layout_winner,
                    "margin_a_minus_b": float(arm_scores["layout"]["A"]["score"])
                    - float(arm_scores["layout"]["B"]["score"]),
                },
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "roster_body_sha256": roster["body_sha256"],
        "direction_invariance": {"passed": direction_ok, "total": direction_total},
        "candidate_permutation_invariance": {"passed": permutation_ok, "total": len(rows)},
        "pairs": rows,
    }
    payload["body_sha256"] = _body_hash(payload)
    return payload


def adjudicate(raw: Mapping[str, Any], roster: Mapping[str, Any]) -> dict[str, Any]:
    _verify_body(raw, "raw scores")
    by_pair = {row["pair_id"]: pair for row, pair in zip(raw["pairs"], roster["pairs"])}
    _require(len(by_pair) == len(roster["pairs"]), "pair alignment drifted")
    baseline_correct = 0
    layout_correct = 0
    rescue = 0
    collateral = 0
    category_metrics: dict[str, dict[str, int]] = {}
    quantile_metrics: dict[str, dict[str, int]] = {}
    per_instance: dict[str, list[tuple[bool, bool]]] = {}
    evaluated_rows = []
    for row, private in zip(raw["pairs"], roster["pairs"]):
        _require(row["pair_id"] == private["pair_id"], "private pair order drifted")
        target_slot = private["target_slot"]
        baseline_ok = row["baseline"]["winner"] == target_slot
        layout_ok = row["layout"]["winner"] == target_slot
        baseline_correct += int(baseline_ok)
        layout_correct += int(layout_ok)
        rescue += int((not baseline_ok) and layout_ok)
        collateral += int(baseline_ok and (not layout_ok))
        for bucket, key in (
            (category_metrics, str(private["category"])),
            (quantile_metrics, f"{float(private['candidate_quantile']):.2f}"),
        ):
            metric = bucket.setdefault(key, {"total": 0, "baseline_correct": 0, "layout_correct": 0})
            metric["total"] += 1
            metric["baseline_correct"] += int(baseline_ok)
            metric["layout_correct"] += int(layout_ok)
        per_instance.setdefault(str(private["reference_instance"]), []).append((baseline_ok, layout_ok))
        evaluated_rows.append(
            {
                "pair_id": row["pair_id"],
                "target_slot": target_slot,
                "baseline_target_outranks": baseline_ok,
                "layout_target_outranks": layout_ok,
                "transition": "RESCUE" if (not baseline_ok and layout_ok) else "COLLATERAL" if (baseline_ok and not layout_ok) else "UNCHANGED",
            }
        )
    stable = {key: values for key, values in per_instance.items() if len(values) == 3 and not any(v[0] for v in values)}
    stable_pairs = sum(len(values) for values in stable.values())
    stable_layout_correct = sum(int(layout_ok) for values in stable.values() for _, layout_ok in values)
    retention = layout_correct - rescue
    retention_rate = retention / baseline_correct if baseline_correct else 0.0
    stable_rate = stable_layout_correct / stable_pairs if stable_pairs else 0.0
    direction = raw["direction_invariance"]
    permutation = raw["candidate_permutation_invariance"]
    gates = {
        "rescue_gt_collateral": rescue > collateral,
        "control_retention_min_0p80": retention_rate >= CONTROL_RETENTION_GATE,
        "direction_invariance_100pct": direction["passed"] == direction["total"],
        "candidate_permutation_invariance_100pct": permutation["passed"] == permutation["total"],
        "stable_instances_min_5": len(stable) >= STABLE_MIN_INSTANCES,
        "stable_layout_target_outrank_min_0p50": stable_rate >= STABLE_RECALL_GATE,
    }
    if not gates["stable_instances_min_5"]:
        terminal = "SPATIAL_LAYOUT_IDENTITY_NOT_EVALUABLE"
    elif all(gates.values()):
        terminal = "SPATIAL_LAYOUT_IDENTITY_SIGNAL_SUPPORTED_DEVELOPMENT"
    elif rescue > 0:
        terminal = "SPATIAL_LAYOUT_IDENTITY_MIXED_WITH_COLLATERAL_DEVELOPMENT"
    else:
        terminal = "SPATIAL_LAYOUT_IDENTITY_NOT_SUPPORTED_DEVELOPMENT"
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "terminal": terminal,
        "claim_ceiling": CLAIM_CEILING,
        "counts": roster["counts"],
        "metrics": {
            "baseline_target_outrank": {"count": baseline_correct, "total": len(raw["pairs"])},
            "layout_target_outrank": {"count": layout_correct, "total": len(raw["pairs"])},
            "rescue": rescue,
            "collateral": collateral,
            "net": rescue - collateral,
            "control_retention": {"count": retention, "total": baseline_correct, "rate": retention_rate},
            "stable_same_class_distractors": {
                "instances": len(stable),
                "pairs": stable_pairs,
                "layout_correct": stable_layout_correct,
                "layout_rate": stable_rate,
            },
            "direction_invariance": direction,
            "candidate_permutation_invariance": permutation,
            "by_category": category_metrics,
            "by_candidate_quantile": quantile_metrics,
        },
        "gates": gates,
        "pair_adjudication": evaluated_rows,
        "scientific_boundaries": [
            "OPEN_SET_CALIBRATION_NOT_RUN", "TARGET_ABSENT_NOT_RUN", "NONE_NOT_RUN",
            "RELIABLE_VERIFIER_NOT_ESTABLISHED", "NO_P1", "DEFAULT_APP_UNCHANGED",
        ],
    }
    report["body_sha256"] = _body_hash(report)
    return report


def run(protocol_path: Path, roster_path: Path, archive: Path, model_dir: Path, run_dir: Path, device: str) -> dict[str, Any]:
    _require(not (run_dir / "final-report.json").exists(), "refusing to overwrite completed run")
    protocol = _load_json(protocol_path)
    roster = _load_json(roster_path)
    _verify_body(protocol, "protocol")
    _verify_body(roster, "roster")
    validate_roster(roster)
    for existing_path in run_dir.parent.glob("run-*/final-report.json"):
        if existing_path == run_dir / "final-report.json":
            continue
        existing = _load_json(existing_path)
        _require(
            not (
                existing.get("protocol_id") == PROTOCOL_ID
                and existing.get("evidence", {}).get("roster_body_sha256") == roster["body_sha256"]
            ),
            f"roster already adjudicated in {existing_path}",
        )
    _require(protocol["body_sha256"] == roster["protocol_body_sha256"], "protocol/roster mismatch")
    _require(archive.stat().st_size == roster["archive"]["bytes"], "archive byte drift")
    _require(_sha256_file(archive) == roster["archive"]["sha256"], "archive SHA drift")
    run_dir.mkdir(parents=True, exist_ok=True)
    features_path = run_dir / "features.npz"
    if features_path.exists():
        cached = np.load(features_path, allow_pickle=False)
        sample_ids = [str(value) for value in cached["sample_ids"]]
        features = np.asarray(cached["features"], dtype=np.float32)
        model = _load_json(run_dir / "feature-receipt.json")["model"]
    else:
        sample_ids, features, model = extract_features(roster, archive, model_dir, device)
        _atomic_npz(features_path, sample_ids=np.asarray(sample_ids), features=features.astype(np.float16))
        features = features.astype(np.float16).astype(np.float32)
        feature_receipt = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "roster_body_sha256": roster["body_sha256"],
            "features_sha256": _sha256_file(features_path),
            "model": model,
        }
        feature_receipt["body_sha256"] = _body_hash(feature_receipt)
        _atomic_json(run_dir / "feature-receipt.json", feature_receipt)
    raw_path = run_dir / "raw-scores.json"
    if raw_path.exists():
        raw = _load_json(raw_path)
    else:
        raw = score_pairs(roster, sample_ids, features)
        _atomic_json(raw_path, raw)
    report = adjudicate(raw, roster)
    report["evidence"] = {
        "protocol_body_sha256": protocol["body_sha256"],
        "roster_body_sha256": roster["body_sha256"],
        "archive_sha256": roster["archive"]["sha256"],
        "features_sha256": _sha256_file(features_path),
        "raw_scores_sha256": _sha256_file(raw_path),
        "model": model,
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(run_dir / "final-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-protocol")
    freeze.add_argument("--protocol-doc", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    roster = sub.add_parser("freeze-roster")
    roster.add_argument("--protocol", type=Path, required=True)
    roster.add_argument("--archive", type=Path, required=True)
    roster.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--roster", type=Path, required=True)
    execute.add_argument("--archive", type=Path, required=True)
    execute.add_argument("--model-dir", type=Path, required=True)
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.command == "freeze-protocol":
        result = freeze_protocol(args.protocol_doc, args.output)
    elif args.command == "freeze-roster":
        result = freeze_roster(args.protocol, args.archive, args.output)
    else:
        result = run(args.protocol, args.roster, args.archive, args.model_dir, args.run_dir, args.device)
    print(json.dumps({"terminal": result["terminal"], "body_sha256": result["body_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

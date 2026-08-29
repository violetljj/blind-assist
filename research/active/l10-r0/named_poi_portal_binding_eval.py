from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import named_poi_facade_fingerprint as facade  # noqa: E402
import named_poi_portal_lattice as portal  # noqa: E402
from named_poi_entrance_binding import expanded_context_box  # noqa: E402
from named_poi_native_support_ray import (  # noqa: E402
    RayBindingState,
    SupportRayConfig,
    bind_support_rays_to_entrances,
)
from named_poi_portal_binding import (  # noqa: E402
    BindingDecision,
    BindingState,
    Episode,
    HeadConfig,
    fit_head,
    predict_head,
    reduce_logits,
    self_test,
)
from named_poi_target_support_field import (  # noqa: E402
    SupportFieldConfig,
    SupportProposal,
    build_target_support_field,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label}_HASH_MISMATCH:expected={expected}:observed={observed}")


def _candidate_member(box: Sequence[float], truth_boxes: Sequence[Sequence[float]]) -> bool:
    return any(portal._portal_set_member(box, truth) for truth in truth_boxes)


def _encode_pil(
    rows: list[tuple[str, Image.Image]],
    models: dict[str, Any],
    clip_processor: Any,
    dino_processor: Any,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    encoded: dict[str, dict[str, np.ndarray]] = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [image for _, image in batch]
        clip_pixels = clip_processor(images=images, return_tensors="pt")["pixel_values"]
        dino_pixels = dino_processor(images=images, return_tensors="pt")["pixel_values"]
        clip, dino_hidden = facade._forward(models, clip_pixels, dino_pixels, device)
        dino = facade._normalized(dino_hidden[:, 1:].mean(dim=1))
        for (key, _), clip_value, dino_value in zip(batch, clip, dino, strict=True):
            encoded[key] = {
                "clip": clip_value.detach().cpu().numpy().astype(np.float32),
                "dino": dino_value.detach().cpu().numpy().astype(np.float32),
            }
    return encoded


def _clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    return left, top, right, bottom


def _upper_box(
    box: Sequence[float], width: int, height: int, height_multiplier: float, candidate_fraction: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    candidate_height = max(1.0, y2 - y1)
    candidate_width = max(1.0, x2 - x1)
    upper = (
        x1 - 0.25 * candidate_width,
        y1 - height_multiplier * candidate_height,
        x2 + 0.25 * candidate_width,
        y1 + candidate_fraction * candidate_height,
    )
    return _clip_box(upper, width, height)


def _score_vector(
    vector: np.ndarray,
    split_targets: list[str],
    reference_keys: dict[str, list[str]],
    encoded: dict[str, dict[str, np.ndarray]],
    feature: str,
) -> dict[str, float]:
    return {
        target: max(float(vector @ encoded[key][feature]) for key in reference_keys[target])
        for target in split_targets
    }


def _target_and_margin(scores: dict[str, float], target: str) -> tuple[float, float]:
    target_value = float(scores[target])
    other = max(value for key, value in scores.items() if key != target)
    return target_value, target_value - float(other)


def _geometry_scores(
    query_patches: np.ndarray,
    split_targets: list[str],
    reference_keys: dict[str, list[str]],
    encoded: dict[str, dict[str, np.ndarray]],
    patch_grid: int,
) -> dict[str, float]:
    return {
        target: max(
            facade._patch_geometry(query_patches, encoded[key]["patches"], patch_grid)["score"]
            for key in reference_keys[target]
        )
        for target in split_targets
    }


def _head_config(protocol: dict[str, Any]) -> HeadConfig:
    row = protocol["training"]
    return HeadConfig(
        hidden_size=int(row["hidden_size"]),
        embedding_size=int(row["embedding_size"]),
        learning_rate=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
        maximum_epochs=int(row["maximum_epochs"]),
        early_stopping_patience=int(row["early_stopping_patience"]),
        minimum_delta=float(row["minimum_delta"]),
        candidate_set_logit_gap=float(row["candidate_set_logit_gap"]),
        seed=int(row["seed"]),
    )


def _decision_from_ray(decision: Any, proposal_ids: list[str]) -> BindingDecision:
    by_id = {proposal_id: index for index, proposal_id in enumerate(proposal_ids)}
    edge_by_id = {edge.proposal_id: edge for edge in decision.edges}
    logits = tuple(float(edge_by_id[proposal_id].edge_score) for proposal_id in proposal_ids)
    if decision.state == RayBindingState.SEARCH:
        return BindingDecision(BindingState.NONE, None, (), logits, 1.0)
    if decision.state == RayBindingState.COMMIT:
        selected = by_id[decision.selected_proposal_id]
        return BindingDecision(BindingState.COMMIT, selected, (selected,), logits, -1.0)
    selected = tuple(by_id[proposal_id] for proposal_id in decision.candidate_set)
    return BindingDecision(BindingState.SET_VALUED, None, selected, logits, -1.0)


def _structure_decision(scores: Sequence[float]) -> BindingDecision:
    values = tuple(float(value) for value in scores)
    selected = int(np.argmax(np.asarray(values)))
    return BindingDecision(BindingState.COMMIT, selected, (selected,), values, -float("inf"))


def _empty_decision() -> BindingDecision:
    return BindingDecision(BindingState.NONE, None, (), (), 1.0)


def _metric(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    positives = [row for row in records if row["correct_target"] and row["truth_present"]]
    none_expected = [row for row in records if row["expected_none"]]
    wrong_targets = [row for row in records if not row["correct_target"]]
    top1 = 0
    top3 = 0
    correct_commit = 0
    wrong_portal_commit = 0
    set_coverage = 0
    commit_or_set_coverage = 0
    false_none_available = 0
    state_counts = {state.value: 0 for state in BindingState}
    for row in records:
        decision: BindingDecision = row["decisions"][arm]
        state_counts[decision.state.value] += 1
        if row in positives and decision.candidate_logits:
            order = np.argsort(-np.asarray(decision.candidate_logits), kind="stable")
            top1 += bool(row["positive_mask"][int(order[0])])
            top3 += any(row["positive_mask"][int(index)] for index in order[:3])
            if decision.state == BindingState.COMMIT:
                if row["positive_mask"][int(decision.selected_index)]:
                    correct_commit += 1
                    commit_or_set_coverage += 1
                else:
                    wrong_portal_commit += 1
            elif decision.state == BindingState.SET_VALUED:
                covered = any(row["positive_mask"][index] for index in decision.candidate_indices)
                set_coverage += covered
                commit_or_set_coverage += covered
            elif bool(row["positive_mask"].any()):
                false_none_available += 1
        elif row in positives and not decision.candidate_logits:
            if decision.state == BindingState.NONE and bool(row["positive_mask"].any()):
                false_none_available += 1
    none_false_accept = sum(
        row["decisions"][arm].state != BindingState.NONE for row in none_expected
    )
    wrong_building = sum(
        row["decisions"][arm].state != BindingState.NONE for row in wrong_targets
    )
    available_positive = sum(bool(row["positive_mask"].any()) for row in positives)
    return {
        "positive_episodes": len(positives),
        "proposer_available_positive_episodes": available_positive,
        "top1_truth_retention": int(top1),
        "top1_truth_recall": top1 / len(positives) if positives else 0.0,
        "top3_truth_retention": int(top3),
        "top3_truth_recall": top3 / len(positives) if positives else 0.0,
        "correct_portal_commits": correct_commit,
        "wrong_portal_commits": wrong_portal_commit,
        "set_valued_truth_coverage": set_coverage,
        "commit_or_set_valued_truth_coverage": commit_or_set_coverage,
        "none_expected_episodes": len(none_expected),
        "none_false_accepts": none_false_accept,
        "none_false_accept_rate": none_false_accept / len(none_expected) if none_expected else 0.0,
        "wrong_target_episodes": len(wrong_targets),
        "wrong_building_confirmations": wrong_building,
        "wrong_building_confirmation_rate": wrong_building / len(wrong_targets) if wrong_targets else 0.0,
        "false_none_on_proposer_available_positive": false_none_available,
        "state_counts": state_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("named_poi_portal_binding_protocol_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-portal-binding-v1",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for label, source in protocol["sources"].items():
        path = ROOT / source["path"]
        _require_hash(path, source["sha256"], label.upper())
    manifest_path = ROOT / protocol["sources"]["dataset_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["inventory"]["inventory_sha256"] != protocol["sources"]["dataset_manifest"]["inventory_sha256"]:
        raise ValueError("DATASET_INVENTORY_HASH_MISMATCH")
    proposer_path = ROOT / protocol["frozen_components"]["portal_proposer"]["path"]
    _require_hash(
        proposer_path,
        protocol["frozen_components"]["portal_proposer"]["sha256"],
        "PORTAL_PROPOSER",
    )
    clip_path = ROOT / protocol["frozen_components"]["clip"]["path"]
    dino_path = ROOT / protocol["frozen_components"]["dinov2"]["path"]
    _require_hash(
        clip_path / "pytorch_model.bin",
        protocol["frozen_components"]["clip"]["weights_sha256"],
        "CLIP_WEIGHTS",
    )
    _require_hash(
        dino_path / "model.safetensors",
        protocol["frozen_components"]["dinov2"]["weights_sha256"],
        "DINO_WEIGHTS",
    )
    split_ids = {key: set(value) for key, value in manifest["split_entity_ids"].items()}
    for left, right in (("train", "development"), ("train", "test"), ("development", "test")):
        if split_ids[left] & split_ids[right]:
            raise ValueError(f"BUILDING_SPLIT_OVERLAP:{left}:{right}")

    entities = manifest["entities"]
    image_rows: list[facade.ImageRow] = []
    reference_keys: dict[str, dict[str, list[str]]] = {split: {} for split in split_ids}
    query_payloads: list[dict[str, Any]] = []
    for entity in entities:
        split = entity["split"]
        reference_keys[split][entity["id"]] = []
        for index, row in enumerate(entity["references"], start=1):
            key = f"ref:{split}:{entity['id']}:{index:02d}"
            path = Path(row["local_path"])
            _require_hash(path, row["sha256"], f"REFERENCE:{key}")
            image_rows.append(facade.ImageRow(key, entity["id"], "reference", path, row["sha256"], row.get("commons_file", "")))
            reference_keys[split][entity["id"]].append(key)
        for row in entity["queries"]:
            path = Path(row["local_path"])
            _require_hash(path, row["sha256"], f"QUERY:{row['key']}")
            key = f"query:{row['key']}"
            image_rows.append(facade.ImageRow(key, entity["id"], "query", path, row["sha256"], ""))
            query_payloads.append({**row, "encoded_key": key, "target_id": entity["id"], "split": split})

    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = facade.AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_images = [Image.open(row.path).convert("RGB") for row in image_rows[:8]]
    representative = (
        clip_processor(images=representative_images, return_tensors="pt")["pixel_values"],
        dino_processor(images=representative_images, return_tensors="pt")["pixel_values"],
    )
    encoder_backend, models = facade._select_backend(
        clip_path,
        dino_path,
        representative,
        output_root / "encoder_backend_receipt.json",
    )
    encoder_device = str(encoder_backend["selected_device_type"])
    if encoder_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    patch_grid = int(protocol["frozen_components"]["dinov2"]["patch_grid"])
    encoded = facade._encode_images(
        image_rows,
        models,
        clip_processor,
        dino_processor,
        encoder_device,
        patch_grid,
        args.batch_size,
    )

    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    crops: list[tuple[str, Image.Image]] = []
    context_scale = float(protocol["candidate_context"]["context_expansion_scale"])
    upper_multiplier = float(protocol["candidate_context"]["upper_facade_height_multiplier"])
    upper_fraction = float(protocol["candidate_context"]["upper_facade_candidate_fraction"])
    maximum_candidates = int(protocol["frozen_components"]["portal_proposer"]["maximum_candidates"])
    for query in query_payloads:
        image = Image.open(query["local_path"]).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        proposals, diagnostics = portal.propose_portal_sets(bgr)
        rows = []
        for index, proposal in enumerate(proposals[:maximum_candidates]):
            proposal_id = f"{query['key']}:portal:{index + 1:02d}"
            local_box = _clip_box(proposal.box_xyxy, image.width, image.height)
            context_box = _clip_box(
                expanded_context_box(proposal.box_xyxy, image.width, image.height, context_scale),
                image.width,
                image.height,
            )
            upper_box = _upper_box(
                proposal.box_xyxy,
                image.width,
                image.height,
                upper_multiplier,
                upper_fraction,
            )
            crop_keys = {
                "local": f"crop:{proposal_id}:local",
                "context": f"crop:{proposal_id}:context",
                "upper": f"crop:{proposal_id}:upper",
            }
            crops.extend(
                [
                    (crop_keys["local"], image.crop(local_box)),
                    (crop_keys["context"], image.crop(context_box)),
                    (crop_keys["upper"], image.crop(upper_box)),
                ]
            )
            rows.append(
                {
                    "proposal": proposal,
                    "proposal_id": proposal_id,
                    "crop_keys": crop_keys,
                    "local_box_xyxy": local_box,
                    "context_box_xyxy": context_box,
                    "upper_box_xyxy": upper_box,
                }
            )
        candidate_rows[query["key"]] = rows
        query["proposal_diagnostics"] = diagnostics
    crop_encoded = _encode_pil(
        crops,
        models,
        clip_processor,
        dino_processor,
        encoder_device,
        args.batch_size,
    )
    accelerator_observation = {
        "encoder_selected_device": encoder_device,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if encoder_device == "cuda" else 0,
    }

    feature_names = list(protocol["feature_names"])
    field_config = SupportFieldConfig()
    ray_config = SupportRayConfig()
    records: list[dict[str, Any]] = []
    for query in query_payloads:
        split = query["split"]
        targets = sorted(split_ids[split])
        candidates = candidate_rows[query["key"]]
        proposal_ids = [row["proposal_id"] for row in candidates]
        support_proposals = [
            SupportProposal(
                row["proposal_id"],
                float(row["proposal"].proposal_score),
                tuple(float(value) for value in row["proposal"].box_xyxy),
            )
            for row in candidates
        ]
        query_features = encoded[query["encoded_key"]]
        reference_patches = {
            target: [encoded[key]["patches"] for key in reference_keys[split][target]]
            for target in targets
        }
        global_clip_scores = _score_vector(query_features["clip"], targets, reference_keys[split], encoded, "clip")
        global_dino_scores = _score_vector(query_features["dino"], targets, reference_keys[split], encoded, "dino")
        global_geometry_scores = _geometry_scores(
            query_features["patches"], targets, reference_keys[split], encoded, patch_grid
        )
        per_view_scores: dict[str, list[dict[str, dict[str, float]]]] = {view: [] for view in ("local", "context", "upper")}
        for candidate in candidates:
            for view in per_view_scores:
                crop = crop_encoded[candidate["crop_keys"][view]]
                per_view_scores[view].append(
                    {
                        "clip": _score_vector(crop["clip"], targets, reference_keys[split], encoded, "clip"),
                        "dino": _score_vector(crop["dino"], targets, reference_keys[split], encoded, "dino"),
                    }
                )
        image_width, image_height = [int(value) for value in query["image_size"]]
        truth_boxes = query["truth_boxes_xyxy"]
        truth_present = bool(truth_boxes)
        positive_mask = np.asarray(
            [_candidate_member(row["proposal"].box_xyxy, truth_boxes) for row in candidates],
            dtype=np.bool_,
        )
        for target in targets:
            if candidates:
                field, field_diagnostics = build_target_support_field(
                    target,
                    query_features["patches"],
                    reference_patches,
                    patch_grid,
                    field_config,
                )
                ray = bind_support_rays_to_entrances(
                    field,
                    support_proposals,
                    image_width,
                    image_height,
                    ray_config,
                )
                ray_edges = {edge.proposal_id: edge for edge in ray.edges}
            else:
                field_diagnostics = {"active_patches": 0}
                ray = None
                ray_edges = {}
            target_global_clip, margin_global_clip = _target_and_margin(global_clip_scores, target)
            target_global_dino, margin_global_dino = _target_and_margin(global_dino_scores, target)
            target_geometry, margin_geometry = _target_and_margin(global_geometry_scores, target)
            feature_rows = []
            for index, candidate in enumerate(candidates):
                proposal = candidate["proposal"]
                local_clip, local_clip_margin = _target_and_margin(per_view_scores["local"][index]["clip"], target)
                local_dino, local_dino_margin = _target_and_margin(per_view_scores["local"][index]["dino"], target)
                context_clip, context_clip_margin = _target_and_margin(per_view_scores["context"][index]["clip"], target)
                context_dino, context_dino_margin = _target_and_margin(per_view_scores["context"][index]["dino"], target)
                upper_clip, upper_clip_margin = _target_and_margin(per_view_scores["upper"][index]["clip"], target)
                upper_dino, upper_dino_margin = _target_and_margin(per_view_scores["upper"][index]["dino"], target)
                x1, y1, x2, y2 = proposal.box_xyxy
                edge = ray_edges[candidate["proposal_id"]]
                feature_rows.append(
                    [
                        proposal.proposal_score,
                        float(proposal.family == "REPEATED_POST_LATTICE"),
                        min(1.0, proposal.post_count / 12.0),
                        proposal.normalized_span,
                        proposal.normalized_height,
                        proposal.spacing_regularity,
                        proposal.horizontal_boundary_support,
                        max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, image_width * image_height),
                        y2 / max(1.0, image_height),
                        local_clip,
                        local_clip_margin,
                        local_dino,
                        local_dino_margin,
                        context_clip,
                        context_clip_margin,
                        context_dino,
                        context_dino_margin,
                        upper_clip,
                        upper_clip_margin,
                        upper_dino,
                        upper_dino_margin,
                        target_global_clip,
                        margin_global_clip,
                        target_global_dino,
                        margin_global_dino,
                        target_geometry,
                        margin_geometry,
                        edge.ray_score,
                        edge.ray_mass_fraction,
                        edge.ray_peak,
                        edge.ray_patch_count / float(patch_grid * patch_grid),
                    ]
                )
            features = np.asarray(feature_rows, dtype=np.float32).reshape(len(feature_rows), len(feature_names))
            correct_target = target == query["target_id"]
            episode_mask = positive_mask.copy() if correct_target else np.zeros(len(candidates), dtype=np.bool_)
            expected_none = not (correct_target and truth_present and bool(episode_mask.any()))
            episode = Episode(
                f"{query['key']}|goal={target}",
                features,
                episode_mask,
                expected_none,
            )
            if candidates:
                structure = _structure_decision([row["proposal"].proposal_score for row in candidates])
                local_scores = 0.5 * (features[:, feature_names.index("local_clip_margin")] + features[:, feature_names.index("local_dino_margin")])
                local = reduce_logits(
                    local_scores,
                    float(protocol["baselines"]["candidate_local_clip_dino"]["none_threshold"]),
                    float(protocol["baselines"]["candidate_local_clip_dino"]["set_margin"]),
                )
                ray_decision = _decision_from_ray(ray, proposal_ids)
            else:
                structure = local = ray_decision = _empty_decision()
            records.append(
                {
                    "episode": episode,
                    "split": split,
                    "query": query["key"],
                    "query_target": query["target_id"],
                    "goal_target": target,
                    "correct_target": correct_target,
                    "truth_present": truth_present,
                    "expected_none": expected_none,
                    "positive_mask": episode_mask,
                    "proposal_ids": proposal_ids,
                    "proposal_boxes_xyxy": [list(row["proposal"].box_xyxy) for row in candidates],
                    "field_diagnostics": field_diagnostics,
                    "decisions": {
                        "structure_rank": structure,
                        "candidate_local_clip_dino": local,
                        "target_support_ray": ray_decision,
                    },
                }
            )

    train_episodes = [row["episode"] for row in records if row["split"] == "train" and len(row["episode"].features)]
    development_episodes = [row["episode"] for row in records if row["split"] == "development" and len(row["episode"].features)]
    model, training_receipt, mean, scale = fit_head(train_episodes, development_episodes, _head_config(protocol))
    for row in records:
        row["decisions"]["portal_binding_head"] = (
            predict_head(model, row["episode"], mean, scale)
            if len(row["episode"].features)
            else _empty_decision()
        )

    model_path = output_root / "portal_binding_head.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "normalization_mean": mean,
            "normalization_scale": scale,
            "config": asdict(model.config),
        },
        model_path,
    )
    arms = ["structure_rank", "candidate_local_clip_dino", "target_support_ray", "portal_binding_head"]
    metrics = {
        split: {
            arm: _metric([row for row in records if row["split"] == split], arm)
            for arm in arms
        }
        for split in ("train", "development", "test")
    }
    baseline_names = ["structure_rank", "candidate_local_clip_dino", "target_support_ray"]
    development_baseline = max(
        baseline_names,
        key=lambda arm: (
            metrics["development"][arm]["top1_truth_recall"],
            metrics["development"][arm]["top3_truth_recall"],
            -metrics["development"][arm]["none_false_accept_rate"],
            -baseline_names.index(arm),
        ),
    )
    baseline_test = metrics["test"][development_baseline]
    head_test = metrics["test"]["portal_binding_head"]
    no_top3_loss = head_test["top3_truth_retention"] >= baseline_test["top3_truth_retention"]
    top1_effect = head_test["top1_truth_retention"] >= baseline_test["top1_truth_retention"] + 1
    baseline_wrong = baseline_test["wrong_portal_commits"]
    commit_effect = (
        baseline_wrong > 0
        and head_test["wrong_portal_commits"] <= 0.5 * baseline_wrong
    )
    positive_admission_not_lower = (
        head_test["commit_or_set_valued_truth_coverage"]
        >= baseline_test["commit_or_set_valued_truth_coverage"]
    )
    non_degenerate_positive_admission = head_test["commit_or_set_valued_truth_coverage"] > 0
    passed = (
        no_top3_loss
        and positive_admission_not_lower
        and non_degenerate_positive_admission
        and (top1_effect or commit_effect)
    )
    result = {
        "schema": "l10-named-poi-portal-binding-result-v1",
        "protocol_sha256": _sha256(protocol_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "dataset_inventory": manifest["inventory"],
        "split_entity_ids": manifest["split_entity_ids"],
        "feature_names": feature_names,
        "frozen_portal_proposer_sha256": _sha256(proposer_path),
        "execution_backend": encoder_backend,
        "accelerator_observation": accelerator_observation,
        "head_training": training_receipt,
        "head_artifact": {"path": str(model_path), "sha256": _sha256(model_path)},
        "mechanism_self_test": self_test(),
        "metrics": metrics,
        "development_selected_baseline": development_baseline,
        "promotion_gate": {
            "no_top3_loss": no_top3_loss,
            "top1_gain_at_least_one_building": top1_effect,
            "wrong_portal_commit_reduction_at_least_50_percent": commit_effect,
            "positive_admission_not_lower_than_baseline": positive_admission_not_lower,
            "non_degenerate_positive_admission": non_degenerate_positive_admission,
            "passed": passed,
            "decision": (
                "L10_PB1_TARGET_CONDITIONED_PORTAL_BINDING_EFFECT"
                if passed
                else "L10_PB1_FRESH_BUILDING_GATE_NOT_MET_STOP_EMBEDDING_FUSION"
            ),
        },
        "ocr_calls": 0,
        "test_rows": [
            {
                "episode": row["episode"].key,
                "query_target": row["query_target"],
                "goal_target": row["goal_target"],
                "correct_target": row["correct_target"],
                "truth_present": row["truth_present"],
                "positive_candidate_count": int(row["positive_mask"].sum()),
                "proposal_ids": row["proposal_ids"],
                "proposal_boxes_xyxy": row["proposal_boxes_xyxy"],
                "decisions": {
                    arm: asdict(row["decisions"][arm]) for arm in arms
                },
            }
            for row in records
            if row["split"] == "test"
        ],
        "claim_scope": protocol["claim_boundary"],
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(output_path),
                "execution_backend": encoder_backend,
                "accelerator_observation": accelerator_observation,
                "development_selected_baseline": development_baseline,
                "test_baseline": baseline_test,
                "test_head": head_test,
                "promotion_gate": result["promotion_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

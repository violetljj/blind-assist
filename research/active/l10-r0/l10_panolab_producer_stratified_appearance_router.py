#!/usr/bin/env python3
"""Evaluate the frozen temporal appearance router on three pixel-unseen producer strata."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

import l10_panolab_federated_router_confirmation_appearance as prior
import l10_panolab_ordered_appearance_bank as appearance
import named_poi_facade_fingerprint as fingerprint


PROTOCOL_SCHEMA = "blindassist-l10-panolab-producer-stratified-appearance-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-producer-stratified-appearance-router-result-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    appearance.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = appearance.load(protocol_path)
    appearance.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    appearance.require(
        appearance.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"],
        "EVALUATOR_HASH_MISMATCH",
    )

    inputs = protocol["inputs"]
    base_protocol = appearance.load(appearance.verify(inputs["base_component_protocol"]))
    prior_six = appearance.load(appearance.verify(inputs["prior_six_target_result"]))
    prior_eight = appearance.load(appearance.verify(inputs["prior_eight_target_result"]))
    selection_path = appearance.verify(inputs["stratified_selection"])
    selection = appearance.load(selection_path)
    manifest_path = appearance.verify(inputs["stratified_materialization"])
    manifest = appearance.load(manifest_path)
    truth = appearance.load(appearance.verify(inputs["stratified_truth"]))
    orientation = appearance.load(appearance.verify(inputs["orientation_projection_protocol"]))
    appearance.require(
        manifest["selection_sha256"] == appearance.sha256(selection_path),
        "STRATIFIED_SELECTION_LINK_MISMATCH",
    )
    appearance.require(
        truth["inputs"]["selection"]["sha256"] == appearance.sha256(selection_path)
        and truth["inputs"]["materialization"]["sha256"] == appearance.sha256(manifest_path),
        "STRATIFIED_TRUTH_LINK_MISMATCH",
    )
    appearance.require(
        selection["selected_pixel_views_before_freeze"] == 0
        and selection["selected_appearance_calls_before_freeze"] == 0
        and truth["appearance_calls_on_selected_images_before_truth_freeze"] == 0,
        "SELECTED_PIXEL_NOT_APPEARANCE_UNSEEN_BEFORE_TRUTH",
    )

    thresholds = protocol["accept_contract"]
    prior_thresholds = prior_eight["frozen_thresholds"]
    appearance.require(
        abs(float(thresholds["minimum_top1_score"]) - float(prior_thresholds["minimum_top1_score"]))
        < 1e-12,
        "SCORE_THRESHOLD_CHANGED",
    )
    appearance.require(
        abs(float(thresholds["minimum_top1_margin"]) - float(prior_thresholds["minimum_top1_margin"]))
        < 1e-12,
        "MARGIN_THRESHOLD_CHANGED",
    )

    target_ids = protocol["target_roster"]
    evaluated_ids = protocol["evaluated_episodes"]
    appearance.require(len(target_ids) == len(set(target_ids)) == 11, "TARGET_ROSTER_NOT_ELEVEN")
    appearance.require(len(evaluated_ids) == len(set(evaluated_ids)) == 3, "EVALUATED_EPISODES_NOT_THREE")
    appearance.require(target_ids[-3:] == evaluated_ids, "STRATIFIED_TARGET_ORDER_MISMATCH")
    reference_rows: dict[str, list[fingerprint.ImageRow]] = {target_id: [] for target_id in target_ids}
    all_rows: list[fingerprint.ImageRow] = []
    old_receipts = [row for row in prior_six["crop_receipts"] if row["role"] == "reference"]
    confirmation_receipts = [row for row in prior_eight["crop_receipts"] if row["role"] == "reference"]
    appearance.require(len(old_receipts) == 12, "PRIOR_SIX_REFERENCE_COUNT_NOT_TWELVE")
    appearance.require(len(confirmation_receipts) == 4, "PRIOR_EIGHT_EXTENSION_REFERENCE_COUNT_NOT_FOUR")
    for receipt in [*old_receipts, *confirmation_receipts]:
        appearance.require(receipt["target_id"] in target_ids[:-3], f"UNKNOWN_PRIOR_TARGET:{receipt['target_id']}")
        row = prior.prior_crop_row(receipt)
        reference_rows[receipt["target_id"]].append(row)
        all_rows.append(row)
    appearance.require(
        all(len(reference_rows[target_id]) == 2 for target_id in target_ids[:-3]),
        "PRIOR_REFERENCE_BANK_NOT_TWO_PER_TARGET",
    )

    episode_index = {row["episode_id"]: row for row in selection["episodes"]}
    truth_index = {row["episode_id"]: row for row in truth["episodes"]}
    appearance.require(
        set(evaluated_ids) == set(episode_index) == set(truth_index),
        "STRATIFIED_EPISODE_MISMATCH",
    )
    appearance.require(all(row["valid_for_fixed_replay"] for row in truth_index.values()), "INVALID_TRUTH_ROW")
    image_index = {row["item_id"]: row for row in manifest["images"]}
    appearance.require(len(image_index) == 15, "STRATIFIED_IMAGE_COUNT_NOT_FIFTEEN")
    output_root = appearance.resolve(protocol["output_root"])
    appearance.require(not output_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True)
    query_rows: dict[str, list[fingerprint.ImageRow]] = {}
    crop_receipts: list[dict[str, Any]] = []
    reference_metadata: list[dict[str, Any]] = []
    for episode_id in evaluated_ids:
        episode = episode_index[episode_id]
        references = sorted(episode["references"], key=lambda row: int(row["sequence_index"]))
        queries = sorted(episode["queries"], key=lambda row: int(row["sequence_index"]))
        appearance.require(
            [row["relation_to_anchor"] for row in references] == ["anchor", "next"]
            and all(row["reciprocal_anchor_link"] for row in references),
            f"REFERENCE_SEQUENCE_INVALID:{episode_id}",
        )
        appearance.require(
            [row["relation_to_anchor"] for row in queries] == ["prev", "anchor", "next"]
            and all(row["reciprocal_anchor_link"] for row in queries),
            f"QUERY_SEQUENCE_INVALID:{episode_id}",
        )
        for index, member in enumerate(references, start=1):
            receipt = image_index[member["item_id"]]
            ray = appearance.strict_ray(prior.frame(episode, member, receipt), orientation)
            row, crop_receipt = appearance.write_crop(
                f"{episode_id}_ref{index}",
                episode_id,
                "reference",
                member["item_id"],
                prior.image_spec(receipt),
                ray,
                base_protocol["observation_window"],
                crop_root,
            )
            reference_rows[episode_id].append(row)
            all_rows.append(row)
            crop_receipts.append(crop_receipt)
            reference_metadata.append(
                {
                    "target_id": episode_id,
                    "item_id": member["item_id"],
                    "producer_stratum": episode["producer_stratum"],
                    "reference_collection": episode["reference_collection"],
                    "query_collection": episode["query_collection"],
                    "collection_disjoint": episode["reference_collection"] != episode["query_collection"],
                }
            )
        query_rows[episode_id] = []
        for member in queries:
            receipt = image_index[member["item_id"]]
            ray = appearance.strict_ray(prior.frame(episode, member, receipt), orientation)
            row, crop_receipt = appearance.write_crop(
                f"{episode_id}_{member['relation_to_anchor']}",
                episode_id,
                member["relation_to_anchor"],
                member["item_id"],
                prior.image_spec(receipt),
                ray,
                base_protocol["observation_window"],
                crop_root,
            )
            query_rows[episode_id].append(row)
            all_rows.append(row)
            crop_receipts.append(crop_receipt)

    model_spec = base_protocol["models"]
    clip_path = appearance.resolve(model_spec["clip"]["path"])
    dino_path = appearance.resolve(model_spec["dinov2"]["path"])
    appearance.require(
        appearance.sha256(clip_path / "pytorch_model.bin") == model_spec["clip"]["weights_sha256"],
        "CLIP_MODEL_HASH_MISMATCH",
    )
    appearance.require(
        appearance.sha256(dino_path / "model.safetensors") == model_spec["dinov2"]["weights_sha256"],
        "DINO_MODEL_HASH_MISMATCH",
    )
    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_image = Image.open(all_rows[0].path).convert("RGB")
    representative = (
        clip_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
        dino_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
    )
    backend_path = output_root / "backend_receipt.json"
    backend, models = fingerprint._select_backend(clip_path, dino_path, representative, backend_path)
    patch_grid = int(model_spec["dinov2"]["patch_grid"])
    started = time.perf_counter()
    encoded = fingerprint._encode_images(
        all_rows,
        models,
        clip_processor,
        dino_processor,
        str(backend["selected_device_type"]),
        patch_grid,
        int(base_protocol["execution"]["batch_size"]),
    )

    rows = []
    accepted_correct = 0
    accepted_wrong = 0
    unknown = 0
    wrong_goal_candidates = 0
    for episode_id in evaluated_ids:
        per_frame = []
        for query in query_rows[episode_id]:
            scores = {
                candidate_id: appearance.appearance_score(
                    query,
                    reference_rows[candidate_id],
                    encoded,
                    patch_grid,
                    base_protocol["appearance_score"]["weights"],
                )
                for candidate_id in target_ids
            }
            per_frame.append(
                {"relation": query.role, "scores": scores, "ranking": appearance.prediction(scores, episode_id)}
            )
        aggregate_scores = {}
        for candidate_id in target_ids:
            ranked_frames = sorted(
                (
                    {"relation": row["relation"], "score": float(row["scores"][candidate_id]["score"])}
                    for row in per_frame
                ),
                key=lambda row: (-row["score"], row["relation"]),
            )
            aggregate_scores[candidate_id] = {
                "score": sum(row["score"] for row in ranked_frames[:2]) / 2.0,
                "top_two_frames": ranked_frames[:2],
                "discarded_frame": ranked_frames[2],
            }
        ranked = appearance.prediction(aggregate_scores, episode_id)
        predicted = ranked["prediction"]
        top_score = float(aggregate_scores[predicted]["score"])
        margin = top_score - max(
            float(aggregate_scores[target_id]["score"])
            for target_id in target_ids
            if target_id != predicted
        )
        score_gate = top_score >= float(thresholds["minimum_top1_score"])
        margin_gate = margin >= float(thresholds["minimum_top1_margin"])
        candidate = predicted if score_gate and margin_gate else None
        if candidate == episode_id:
            accepted_correct += 1
            route = "TEMPORAL_APPEARANCE_SEARCH_PRIORITY_CANDIDATE"
        elif candidate is None:
            unknown += 1
            route = "UNKNOWN_KEEP_SEARCHING"
        else:
            accepted_wrong += 1
            route = "WRONG_TEMPORAL_APPEARANCE_SEARCH_PRIORITY_CANDIDATE"
        controls = []
        for wrong_target in target_ids:
            if wrong_target == episode_id:
                continue
            emitted = candidate == wrong_target
            wrong_goal_candidates += int(emitted)
            controls.append(
                {
                    "wrong_target": wrong_target,
                    "search_priority_candidate": emitted,
                    "aggregate_score": aggregate_scores[wrong_target]["score"],
                }
            )
        episode = episode_index[episode_id]
        rows.append(
            {
                "episode_id": episode_id,
                "producer_stratum": episode["producer_stratum"],
                "target_name": episode["target_name"],
                "target_way_id": episode["target_way_id"],
                "source_city": episode["source_city"],
                "human_truth": truth_index[episode_id]["human_truth"],
                "per_frame": per_frame,
                "temporal_aggregate": {"scores": aggregate_scores, **ranked},
                "acceptance": {
                    "score_gate": score_gate,
                    "margin_gate": margin_gate,
                    "top1_score": top_score,
                    "top1_margin": margin,
                    "route": route,
                    "candidate": candidate,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
                "wrong_goal_controls": controls,
            }
        )

    metrics = {
        "pixel_unseen_episodes": len(evaluated_ids),
        "producer_strata": len({episode_index[row]["producer_stratum"] for row in evaluated_ids}),
        "query_frames": sum(len(value) for value in query_rows.values()),
        "combined_target_roster": len(target_ids),
        "combined_reference_views": sum(len(value) for value in reference_rows.values()),
        "stratified_reference_views": sum(len(reference_rows[row]) for row in evaluated_ids),
        "collection_disjoint_stratified_reference_views": sum(row["collection_disjoint"] for row in reference_metadata),
        "prior_router_target_way_overlap": selection["prior_router_target_way_overlap"],
        "prior_router_item_overlap": selection["prior_router_item_overlap"],
        "accepted_correct_wrong_unknown": [accepted_correct, accepted_wrong, unknown],
        "positive_coverage": accepted_correct / len(evaluated_ids),
        "wrong_goal_trials": len(evaluated_ids) * (len(target_ids) - 1),
        "wrong_goal_search_priority_candidates": wrong_goal_candidates,
        "portal_ownership_bindings_emitted": 0,
        "model_seconds": time.perf_counter() - started,
    }
    gate = {
        "three_valid_pixel_unseen_truth_rows": len(evaluated_ids) == 3,
        "three_producer_strata": metrics["producer_strata"] == 3,
        "six_cross_collection_reference_views": metrics["collection_disjoint_stratified_reference_views"] == 6,
        "nine_reciprocal_query_frames": metrics["query_frames"] == 9,
        "zero_prior_router_target_way_overlap": metrics["prior_router_target_way_overlap"] == 0,
        "zero_prior_router_item_overlap": metrics["prior_router_item_overlap"] == 0,
        "three_of_three_accepted_correct": accepted_correct == 3,
        "zero_accepted_wrong": accepted_wrong == 0,
        "zero_wrong_goal_search_priority_candidates": wrong_goal_candidates == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "PRODUCER_STRATIFIED_PIXEL_UNSEEN_TARGET_WAY_ITEM_DISJOINT_SAME_PROVIDER_DEVELOPMENT_ROUTING_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": appearance.sha256(protocol_path),
        "evaluator_sha256": appearance.sha256(Path(__file__).resolve()),
        "execution_backend": backend,
        "backend_receipt_sha256": appearance.sha256(backend_path),
        "frozen_thresholds": thresholds,
        "reference_metadata": reference_metadata,
        "crop_receipts": crop_receipts,
        "metrics": metrics,
        "gate": gate,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), "decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

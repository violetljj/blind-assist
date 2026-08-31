#!/usr/bin/env python3
"""Evaluate a frozen top-two-of-three temporal query appearance router."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

import l10_panolab_ordered_appearance_bank as appearance
import named_poi_facade_fingerprint as fingerprint


PROTOCOL_SCHEMA = "blindassist-l10-panolab-temporal-query-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-temporal-query-router-result-v1"


def image_spec(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": receipt["path"],
        "sha256": receipt["sha256"],
        "bytes": receipt["bytes"],
        "image_size": receipt["image_size"],
    }


def frame(episode: dict[str, Any], member: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    entrance = episode["main_entrance_node"]
    return {
        "target": {
            "entrance_node": {
                "id": entrance["id"],
                "lon_lat": [entrance["lon"], entrance["lat"]],
            }
        },
        "panorama": {
            "image_size": receipt["image_size"],
            "provider_item": member["provider_item"],
        },
    }


def prior_crop_row(receipt: dict[str, Any]) -> fingerprint.ImageRow:
    path = appearance.resolve(receipt["crop_path"])
    appearance.require(path.is_file(), f"PRIOR_CROP_MISSING:{path}")
    appearance.require(appearance.sha256(path) == receipt["crop_sha256"], f"PRIOR_CROP_HASH:{path}")
    appearance.require(path.stat().st_size == int(receipt["crop_bytes"]), f"PRIOR_CROP_BYTES:{path}")
    return fingerprint.ImageRow(
        key=receipt["key"],
        target_id=receipt["target_id"],
        role="reference",
        path=path,
        sha256=receipt["crop_sha256"],
        commons_file=receipt["item_id"],
    )


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
    prior_result = appearance.load(appearance.verify(inputs["single_frame_result"]))
    selection_path = appearance.verify(inputs["temporal_selection"])
    selection = appearance.load(selection_path)
    manifest_path = appearance.verify(inputs["temporal_materialization"])
    manifest = appearance.load(manifest_path)
    truth = appearance.load(appearance.verify(inputs["temporal_truth"]))
    orientation = appearance.load(appearance.verify(inputs["orientation_projection_protocol"]))
    appearance.require(
        manifest["selection_sha256"] == appearance.sha256(selection_path),
        "TEMPORAL_SELECTION_LINK_MISMATCH",
    )
    appearance.require(
        truth["inputs"]["selection"]["sha256"] == appearance.sha256(selection_path)
        and truth["inputs"]["materialization"]["sha256"] == appearance.sha256(manifest_path),
        "TEMPORAL_TRUTH_LINK_MISMATCH",
    )
    appearance.require(
        selection["new_neighbor_pixel_views_before_freeze"] == 0
        and selection["new_neighbor_model_calls_before_freeze"] == 0
        and truth["new_neighbor_model_calls_before_truth_freeze"] == 0,
        "NEIGHBOR_NOT_MODEL_UNSEEN_BEFORE_TRUTH",
    )

    target_ids = protocol["target_roster"]
    appearance.require(len(target_ids) == len(set(target_ids)) == 6, "TARGET_ROSTER_NOT_SIX")
    reference_rows: dict[str, list[fingerprint.ImageRow]] = {target_id: [] for target_id in target_ids}
    reference_receipts = [row for row in prior_result["crop_receipts"] if row["role"] == "reference"]
    appearance.require(len(reference_receipts) == 12, "PRIOR_REFERENCE_CROP_COUNT_NOT_TWELVE")
    all_rows: list[fingerprint.ImageRow] = []
    for receipt in reference_receipts:
        appearance.require(receipt["target_id"] in reference_rows, f"UNKNOWN_REFERENCE_TARGET:{receipt['target_id']}")
        row = prior_crop_row(receipt)
        reference_rows[receipt["target_id"]].append(row)
        all_rows.append(row)
    appearance.require(
        all(len(reference_rows[target_id]) == 2 for target_id in target_ids),
        "REFERENCE_BANK_NOT_TWO_PER_TARGET",
    )

    truth_index = {row["episode_id"]: row for row in truth["episodes"]}
    episode_index = {row["episode_id"]: row for row in selection["episodes"]}
    positive_ids = protocol["evaluated_episodes"]
    appearance.require(len(positive_ids) == len(set(positive_ids)) == 2, "EVALUATED_EPISODES_NOT_TWO")
    appearance.require(set(positive_ids) == set(truth_index) == set(episode_index), "EPISODE_MISMATCH")
    appearance.require(
        all(row["valid_for_fixed_replay"] for row in truth_index.values()),
        "INVALID_TEMPORAL_TRUTH_ROW",
    )
    image_index = {row["item_id"]: row for row in manifest["images"]}
    appearance.require(len(image_index) == 6, "TEMPORAL_IMAGE_COUNT_NOT_SIX")
    output_root = appearance.resolve(protocol["output_root"])
    appearance.require(not output_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True)
    query_rows: dict[str, list[fingerprint.ImageRow]] = {}
    crop_receipts: list[dict[str, Any]] = []
    for episode_id in positive_ids:
        episode = episode_index[episode_id]
        members = sorted(episode["members"], key=lambda row: int(row["sequence_index"]))
        appearance.require(
            [row["relation_to_anchor"] for row in members] == ["prev", "anchor", "next"],
            f"TEMPORAL_ORDER_MISMATCH:{episode_id}",
        )
        appearance.require(
            all(row["reciprocal_anchor_link"] for row in members),
            f"NON_RECIPROCAL_SEQUENCE:{episode_id}",
        )
        query_rows[episode_id] = []
        for member in members:
            receipt = image_index[member["item_id"]]
            ray = appearance.strict_ray(frame(episode, member, receipt), orientation)
            row, crop_receipt = appearance.write_crop(
                f"{episode_id}_{member['relation_to_anchor']}",
                episode_id,
                member["relation_to_anchor"],
                member["item_id"],
                image_spec(receipt),
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
    backend, models = fingerprint._select_backend(
        clip_path, dino_path, representative, backend_path
    )
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

    thresholds = protocol["accept_contract"]
    prior_rows = {row["episode_id"]: row for row in prior_result["rows"]}
    rows: list[dict[str, Any]] = []
    accepted_correct = 0
    accepted_wrong = 0
    unknown = 0
    wrong_goal_candidates = 0
    for episode_id in positive_ids:
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
                {
                    "relation": query.role,
                    "scores": scores,
                    "ranking": appearance.prediction(scores, episode_id),
                }
            )
        aggregate_scores = {}
        for candidate_id in target_ids:
            ranked_frames = sorted(
                (
                    {
                        "relation": row["relation"],
                        "score": float(row["scores"][candidate_id]["score"]),
                    }
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
        score_gate = float(aggregate_scores[predicted]["score"]) >= float(
            thresholds["minimum_top1_score"]
        )
        predicted_margin = float(aggregate_scores[predicted]["score"]) - max(
            float(aggregate_scores[candidate_id]["score"])
            for candidate_id in target_ids
            if candidate_id != predicted
        )
        margin_gate = predicted_margin >= float(thresholds["minimum_top1_margin"])
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
        rows.append(
            {
                "episode_id": episode_id,
                "target_name": episode_index[episode_id]["target_way"]["tags"]["name"],
                "single_frame_baseline": {
                    "prediction": prior_rows[episode_id]["appearance"]["prediction"],
                    "truth_score": prior_rows[episode_id]["appearance"]["truth_score"],
                    "truth_margin": prior_rows[episode_id]["appearance"]["truth_margin"],
                    "route": prior_rows[episode_id]["acceptance"]["route"],
                },
                "per_frame": per_frame,
                "temporal_aggregate": {"scores": aggregate_scores, **ranked},
                "acceptance": {
                    "score_gate": score_gate,
                    "margin_gate": margin_gate,
                    "route": route,
                    "candidate": candidate,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
                "wrong_goal_controls": controls,
            }
        )

    baseline_correct = sum(
        prior_rows[episode_id]["acceptance"]["candidate"] == episode_id
        for episode_id in positive_ids
    )
    metrics = {
        "consumed_posthoc_episodes": len(positive_ids),
        "query_frames": sum(len(value) for value in query_rows.values()),
        "new_neighbor_frames": sum(
            row["relation_to_anchor"] != "anchor"
            for episode in selection["episodes"]
            for row in episode["members"]
        ),
        "combined_target_roster": len(target_ids),
        "reference_views": sum(len(value) for value in reference_rows.values()),
        "single_frame_baseline_correct_wrong_unknown": [baseline_correct, 0, len(positive_ids) - baseline_correct],
        "temporal_router_correct_wrong_unknown": [accepted_correct, accepted_wrong, unknown],
        "temporal_correct_gain": accepted_correct - baseline_correct,
        "wrong_goal_trials": len(positive_ids) * (len(target_ids) - 1),
        "wrong_goal_search_priority_candidates": wrong_goal_candidates,
        "portal_ownership_bindings_emitted": 0,
        "model_seconds": time.perf_counter() - started,
    }
    gate = {
        "two_valid_reciprocal_sequences": len(positive_ids) == 2,
        "six_query_frames": metrics["query_frames"] == 6,
        "two_of_two_temporal_router_correct": accepted_correct == 2,
        "minimum_one_correct_gain": accepted_correct - baseline_correct >= 1,
        "zero_temporal_router_wrong": accepted_wrong == 0,
        "zero_wrong_goal_search_priority_candidates": wrong_goal_candidates == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "CONSUMED_POSTHOC_TEMPORAL_MECHANISM_DEVELOPMENT_CANDIDATE_ROUTING_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": appearance.sha256(protocol_path),
        "evaluator_sha256": appearance.sha256(Path(__file__).resolve()),
        "execution_backend": backend,
        "backend_receipt_sha256": appearance.sha256(backend_path),
        "frozen_thresholds": thresholds,
        "crop_receipts": crop_receipts,
        "metrics": metrics,
        "gate": gate,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"result": str(output_path), "decision": result["decision"], "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

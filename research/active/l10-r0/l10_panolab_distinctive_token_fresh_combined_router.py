#!/usr/bin/env python3
"""Evaluate the frozen appearance fallback only for fresh lexical NO_MATCH rows."""

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


PROTOCOL_SCHEMA = "blindassist-l10-panolab-distinctive-token-fresh-combined-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-distinctive-token-fresh-combined-router-result-v1"


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
    appearance.require(appearance.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")

    inputs = protocol["inputs"]
    base_protocol = appearance.load(appearance.verify(inputs["base_component_protocol"]))
    prior_six = appearance.load(appearance.verify(inputs["prior_six_target_result"]))
    prior_eight = appearance.load(appearance.verify(inputs["prior_eight_target_result"]))
    prior_eleven = appearance.load(appearance.verify(inputs["prior_eleven_target_result"]))
    selection_path = appearance.verify(inputs["fresh_selection"])
    selection = appearance.load(selection_path)
    manifest_path = appearance.verify(inputs["fresh_materialization"])
    manifest = appearance.load(manifest_path)
    truth = appearance.load(appearance.verify(inputs["fresh_truth"]))
    lexical_result = appearance.load(appearance.verify(inputs["fresh_lexical_result"]))
    orientation = appearance.load(appearance.verify(inputs["orientation_projection_protocol"]))
    appearance.require(manifest["selection_sha256"] == appearance.sha256(selection_path), "SELECTION_LINK_MISMATCH")
    appearance.require(
        truth["inputs"]["selection"]["sha256"] == appearance.sha256(selection_path)
        and truth["inputs"]["materialization"]["sha256"] == appearance.sha256(manifest_path),
        "TRUTH_LINK_MISMATCH",
    )
    appearance.require(
        selection["selected_appearance_calls_before_freeze"] == 0
        and truth["appearance_calls_on_selected_images_before_truth_freeze"] == 0
        and protocol["appearance_calls_on_selected_images_before_protocol_freeze"] == 0,
        "SELECTED_APPEARANCE_PREEXPOSED",
    )
    appearance.require(lexical_result["metrics"]["lexical_correct_wrong_no_match_ambiguous"] == [1, 0, 1, 0], "UNEXPECTED_FRESH_LEXICAL_RESULT")

    thresholds = protocol["accept_contract"]
    appearance.require(thresholds == prior_eleven["frozen_thresholds"], "APPEARANCE_THRESHOLDS_CHANGED")
    target_ids = protocol["target_roster"]
    evaluated_ids = protocol["evaluated_episodes"]
    appearance.require(len(target_ids) == len(set(target_ids)) == 13, "TARGET_ROSTER_NOT_THIRTEEN")
    appearance.require(target_ids[-2:] == evaluated_ids, "FRESH_TARGET_ORDER_MISMATCH")
    lexical_index = {row["episode_id"]: row for row in lexical_result["rows"]}
    fallback_ids = [episode_id for episode_id in evaluated_ids if lexical_index[episode_id]["lexical"]["state"] == "NO_MATCH"]
    appearance.require(fallback_ids == protocol["appearance_fallback_episodes"], "FALLBACK_EPISODES_CHANGED")

    reference_rows: dict[str, list[fingerprint.ImageRow]] = {target_id: [] for target_id in target_ids}
    all_rows: list[fingerprint.ImageRow] = []
    prior_receipt_groups = [
        [row for row in prior_six["crop_receipts"] if row["role"] == "reference"],
        [row for row in prior_eight["crop_receipts"] if row["role"] == "reference"],
        [row for row in prior_eleven["crop_receipts"] if row["role"] == "reference"],
    ]
    appearance.require([len(rows) for rows in prior_receipt_groups] == [12, 4, 6], "PRIOR_REFERENCE_COUNTS_CHANGED")
    for receipt in [row for group in prior_receipt_groups for row in group]:
        appearance.require(receipt["target_id"] in target_ids[:-2], f"UNKNOWN_PRIOR_TARGET:{receipt['target_id']}")
        image_row = prior.prior_crop_row(receipt)
        reference_rows[receipt["target_id"]].append(image_row)
        all_rows.append(image_row)
    appearance.require(all(len(reference_rows[target_id]) == 2 for target_id in target_ids[:-2]), "PRIOR_REFERENCE_BANK_NOT_TWO_PER_TARGET")

    episode_index = {row["episode_id"]: row for row in selection["episodes"]}
    truth_index = {row["episode_id"]: row for row in truth["episodes"]}
    image_index = {row["item_id"]: row for row in manifest["images"]}
    appearance.require(set(evaluated_ids) == set(episode_index) == set(truth_index), "FRESH_EPISODE_MISMATCH")
    appearance.require(all(row["valid_for_fixed_replay"] for row in truth_index.values()), "INVALID_TRUTH_ROW")
    output_root = appearance.resolve(protocol["output_root"])
    appearance.require(not output_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True)
    crop_receipts: list[dict[str, Any]] = []
    query_rows: dict[str, list[fingerprint.ImageRow]] = {}
    for episode_id in evaluated_ids:
        episode = episode_index[episode_id]
        references = sorted(episode["references"], key=lambda row: int(row["sequence_index"]))
        appearance.require([row["relation_to_anchor"] for row in references] == ["anchor", "next"], f"REFERENCE_SEQUENCE_INVALID:{episode_id}")
        for index, member in enumerate(references, start=1):
            receipt = image_index[member["item_id"]]
            ray = appearance.strict_ray(prior.frame(episode, member, receipt), orientation)
            image_row, crop_receipt = appearance.write_crop(
                f"{episode_id}_ref{index}", episode_id, "reference", member["item_id"],
                prior.image_spec(receipt), ray, base_protocol["observation_window"], crop_root,
            )
            reference_rows[episode_id].append(image_row)
            all_rows.append(image_row)
            crop_receipts.append(crop_receipt)
        if episode_id not in fallback_ids:
            continue
        members = sorted(episode["queries"], key=lambda row: int(row["sequence_index"]))
        appearance.require([row["relation_to_anchor"] for row in members] == ["prev", "anchor", "next"], f"QUERY_SEQUENCE_INVALID:{episode_id}")
        query_rows[episode_id] = []
        for member in members:
            receipt = image_index[member["item_id"]]
            ray = appearance.strict_ray(prior.frame(episode, member, receipt), orientation)
            image_row, crop_receipt = appearance.write_crop(
                f"{episode_id}_{member['relation_to_anchor']}", episode_id, member["relation_to_anchor"],
                member["item_id"], prior.image_spec(receipt), ray, base_protocol["observation_window"], crop_root,
            )
            query_rows[episode_id].append(image_row)
            all_rows.append(image_row)
            crop_receipts.append(crop_receipt)

    model_spec = base_protocol["models"]
    clip_path = appearance.resolve(model_spec["clip"]["path"])
    dino_path = appearance.resolve(model_spec["dinov2"]["path"])
    appearance.require(appearance.sha256(clip_path / "pytorch_model.bin") == model_spec["clip"]["weights_sha256"], "CLIP_MODEL_HASH_MISMATCH")
    appearance.require(appearance.sha256(dino_path / "model.safetensors") == model_spec["dinov2"]["weights_sha256"], "DINO_MODEL_HASH_MISMATCH")
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
        all_rows, models, clip_processor, dino_processor, str(backend["selected_device_type"]),
        patch_grid, int(base_protocol["execution"]["batch_size"]),
    )

    appearance_results = {}
    wrong_goal_candidates = 0
    for episode_id in fallback_ids:
        per_frame = []
        for query in query_rows[episode_id]:
            scores = {
                candidate_id: appearance.appearance_score(
                    query, reference_rows[candidate_id], encoded, patch_grid,
                    base_protocol["appearance_score"]["weights"],
                )
                for candidate_id in target_ids
            }
            per_frame.append({"relation": query.role, "scores": scores})
        aggregate_scores = {}
        for candidate_id in target_ids:
            ranked_frames = sorted(
                ({"relation": row["relation"], "score": float(row["scores"][candidate_id]["score"])} for row in per_frame),
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
        margin = top_score - max(float(aggregate_scores[target_id]["score"]) for target_id in target_ids if target_id != predicted)
        accepted = top_score >= float(thresholds["minimum_top1_score"]) and margin >= float(thresholds["minimum_top1_margin"])
        candidate = predicted if accepted else None
        wrong_goal_candidates += int(candidate is not None and candidate != episode_id)
        appearance_results[episode_id] = {
            "per_frame": per_frame,
            "temporal_aggregate": {"scores": aggregate_scores, **ranked},
            "acceptance": {
                "top1_score": top_score,
                "top1_margin": margin,
                "score_gate": top_score >= float(thresholds["minimum_top1_score"]),
                "margin_gate": margin >= float(thresholds["minimum_top1_margin"]),
                "candidate": candidate,
                "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                "portal_ownership_binding": None,
            },
        }

    rows = []
    combined_correct = combined_wrong = combined_unknown = 0
    lexical_route_count = appearance_route_count = 0
    for episode_id in evaluated_ids:
        lexical_candidate = lexical_index[episode_id]["lexical"]["candidate"]
        if lexical_candidate is not None:
            candidate = lexical_candidate
            route = "FRESH_DISTINCTIVE_EDIT_TOKEN_SEARCH_PRIORITY_CANDIDATE"
            lexical_route_count += 1
        else:
            candidate = appearance_results[episode_id]["acceptance"]["candidate"]
            route = "TEMPORAL_APPEARANCE_SEARCH_PRIORITY_CANDIDATE" if candidate else "UNKNOWN_KEEP_SEARCHING"
            appearance_route_count += int(candidate is not None)
        if candidate == episode_id:
            combined_correct += 1
        elif candidate is None:
            combined_unknown += 1
        else:
            combined_wrong += 1
        rows.append(
            {
                "episode_id": episode_id,
                "target_name": episode_index[episode_id]["target_name"],
                "producer_stratum": episode_index[episode_id]["producer_stratum"],
                "human_truth": truth_index[episode_id]["human_truth"],
                "lexical": lexical_index[episode_id]["lexical"],
                "appearance_fallback": appearance_results.get(episode_id),
                "combined_router": {
                    "route": route,
                    "candidate": candidate,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
            }
        )

    metrics = {
        "fresh_episodes": len(evaluated_ids),
        "lexical_routes": lexical_route_count,
        "appearance_fallback_trials": len(fallback_ids),
        "appearance_fallback_routes": appearance_route_count,
        "appearance_wrong_goal_trials": len(fallback_ids) * (len(target_ids) - 1),
        "appearance_wrong_goal_candidates": wrong_goal_candidates,
        "combined_correct_wrong_unknown": [combined_correct, combined_wrong, combined_unknown],
        "portal_ownership_bindings_emitted": 0,
        "model_seconds": round(time.perf_counter() - started, 6),
    }
    gate = {
        "two_fresh_sequences_routed": len(evaluated_ids) == 2,
        "exactly_one_lexical_route": lexical_route_count == 1,
        "exactly_one_conditional_appearance_trial": len(fallback_ids) == 1,
        "two_of_two_combined_correct": combined_correct == 2,
        "zero_combined_wrong": combined_wrong == 0,
        "zero_combined_unknown": combined_unknown == 0,
        "zero_appearance_wrong_goal_candidates": wrong_goal_candidates == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "PIXEL_AND_OCR_UNSEEN_TARGET_WAY_ITEM_DISJOINT_SAME_PROVIDER_DEVELOPMENT_ROUTING_ONLY",
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
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

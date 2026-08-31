#!/usr/bin/env python3
"""Replay the frozen appearance router on target- and collection-unseen positives."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

import l10_panolab_component_router as component
import l10_panolab_ordered_appearance_bank as appearance
import named_poi_facade_fingerprint as fingerprint


PROTOCOL_SCHEMA = "blindassist-l10-panolab-cross-collection-positive-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-cross-collection-positive-router-result-v1"


def image_spec(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": receipt["path"],
        "sha256": receipt["sha256"],
        "bytes": receipt["bytes"],
        "image_size": receipt["image_size"],
    }


def selected_frame(
    episode: dict[str, Any], item: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
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
            "provider_item": item["provider_item"],
        },
    }


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
    calibration = appearance.load(appearance.verify(inputs["positive_calibration_result"]))
    positive_selection_path = appearance.verify(inputs["positive_selection"])
    positive_selection = appearance.load(positive_selection_path)
    positive_materialization_path = appearance.verify(inputs["positive_materialization"])
    positive_materialization = appearance.load(positive_materialization_path)
    positive_truth = appearance.load(appearance.verify(inputs["positive_truth"]))

    appearance.require(
        calibration["protocol_sha256"] == inputs["base_component_protocol"]["sha256"],
        "CALIBRATION_PROTOCOL_LINK_MISMATCH",
    )
    appearance.require(
        positive_materialization["selection_sha256"] == appearance.sha256(positive_selection_path),
        "POSITIVE_SELECTION_LINK_MISMATCH",
    )
    appearance.require(
        positive_truth["inputs"]["selection"]["sha256"] == appearance.sha256(positive_selection_path)
        and positive_truth["inputs"]["materialization"]["sha256"]
        == appearance.sha256(positive_materialization_path),
        "POSITIVE_TRUTH_LINK_MISMATCH",
    )
    appearance.require(
        positive_selection["selected_pixel_views_before_freeze"] == 0
        and positive_selection["selected_human_pixel_reviews_before_freeze"] == 0
        and positive_selection["selected_model_calls_before_freeze"] == 0,
        "POSITIVE_SELECTION_NOT_PIXEL_UNSEEN",
    )
    appearance.require(
        positive_truth["model_calls_on_selected_images_before_truth_freeze"] == 0,
        "POSITIVE_TRUTH_NOT_PRE_MODEL",
    )

    thresholds = protocol["accept_contract"]
    calibration_scores = [float(row["appearance"]["truth_score"]) for row in calibration["rows"]]
    calibration_margins = [float(row["appearance"]["truth_margin"]) for row in calibration["rows"]]
    appearance.require(
        abs(min(calibration_scores) - float(thresholds["minimum_top1_score"])) < 1e-12,
        "SCORE_THRESHOLD_NOT_CALIBRATION_MINIMUM",
    )
    appearance.require(
        abs(min(calibration_margins) - float(thresholds["minimum_top1_margin"])) < 1e-12,
        "MARGIN_THRESHOLD_NOT_CALIBRATION_MINIMUM",
    )

    base_inputs = base_protocol["inputs"]
    base_selection = appearance.load(appearance.verify(base_inputs["fresh_selection"]))
    base_materialization = appearance.load(appearance.verify(base_inputs["fresh_materialization"]))
    portal_source = appearance.load(appearance.verify(base_inputs["reference_portal_source"]))
    node_source = appearance.load(appearance.verify(base_inputs["node_credential_source"]))
    viviani_selection_path = appearance.verify(base_inputs["viviani_reference_selection"])
    viviani_selection = appearance.load(viviani_selection_path)
    viviani_materialization = appearance.load(
        appearance.verify(base_inputs["viviani_reference_materialization"])
    )
    orientation = appearance.load(appearance.verify(base_inputs["orientation_projection_protocol"]))
    appearance.require(
        viviani_materialization["selection_sha256"] == appearance.sha256(viviani_selection_path),
        "VIVIANI_SELECTION_LINK_MISMATCH",
    )

    base_selected = {row["episode_id"]: row for row in base_selection["episodes"]}
    base_target_specs = base_protocol["target_roster"]
    base_target_ids = [row["episode_id"] for row in base_target_specs]
    appearance.require(len(base_target_ids) == len(set(base_target_ids)) == 4, "BASE_ROSTER_INVALID")
    positive_episodes = {row["episode_id"]: row for row in positive_selection["episodes"]}
    positive_target_ids = sorted(positive_episodes)
    appearance.require(
        len(positive_target_ids) == len(set(positive_target_ids)) == 2,
        "POSITIVE_ROSTER_NOT_TWO_UNIQUE_EPISODES",
    )
    target_ids = base_target_ids + positive_target_ids
    appearance.require(len(target_ids) == len(set(target_ids)) == 6, "COMBINED_ROSTER_NOT_SIX_UNIQUE")

    truth_index = {row["episode_id"]: row for row in positive_truth["episodes"]}
    appearance.require(set(truth_index) == set(positive_target_ids), "TRUTH_EPISODE_MISMATCH")
    appearance.require(
        all(row["valid_for_fixed_replay"] for row in truth_index.values()),
        "INVALID_POSITIVE_TRUTH_ROW",
    )
    positive_image_index = {row["item_id"]: row for row in positive_materialization["images"]}
    appearance.require(len(positive_image_index) == 6, "POSITIVE_IMAGE_COUNT_NOT_SIX")

    output_root = appearance.resolve(protocol["output_root"])
    appearance.require(not output_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True)
    reference_rows: dict[str, list[fingerprint.ImageRow]] = {}
    query_rows: dict[str, fingerprint.ImageRow] = {}
    all_rows: list[fingerprint.ImageRow] = []
    crop_receipts: list[dict[str, Any]] = []
    reference_metadata: list[dict[str, Any]] = []

    base_query_item_ids = {row["item_id"] for row in base_materialization["images"]}
    positive_item_ids = set(positive_image_index)
    base_reference_item_ids: set[str] = set()

    for target_spec in base_target_specs:
        target_id = target_spec["episode_id"]
        episode = base_selected[target_id]
        reference_rows[target_id] = []
        for index, reference_spec in enumerate(target_spec["references"], start=1):
            if reference_spec["source"] == "reference_portal":
                reference = appearance.portal_reference(
                    portal_source, int(target_spec["target_way_id"]), reference_spec["item_id"]
                )
                ray = reference["ray"]
            elif reference_spec["source"] == "node_credential":
                reference = appearance.node_reference(
                    node_source, int(target_spec["target_way_id"]), reference_spec["item_id"]
                )
                ray = reference["ray"]
            elif reference_spec["source"] == "viviani_unseen":
                reference = component.viviani_reference(
                    viviani_selection, viviani_materialization, reference_spec["item_id"]
                )
                ray = appearance.strict_ray(reference["ray_frame"], orientation)
            else:
                raise ValueError(f"UNKNOWN_REFERENCE_SOURCE:{reference_spec['source']}")
            base_reference_item_ids.add(reference["item_id"])
            row, receipt = appearance.write_crop(
                f"{target_id}_ref{index}",
                target_id,
                "reference",
                reference["item_id"],
                reference["image"],
                ray,
                base_protocol["observation_window"],
                crop_root,
            )
            reference_rows[target_id].append(row)
            all_rows.append(row)
            crop_receipts.append(receipt)
            reference_metadata.append(
                {
                    "target_id": target_id,
                    "item_id": reference["item_id"],
                    "source": reference_spec["source"],
                    "collection": reference["collection"],
                    "query_collection": episode["sequence_id"],
                    "collection_disjoint": reference["collection"] != episode["sequence_id"],
                }
            )
        appearance.require(len(reference_rows[target_id]) == 2, f"BASE_REFERENCE_COUNT:{target_id}")

    appearance.require(not (positive_item_ids & base_query_item_ids), "POSITIVE_BASE_QUERY_ITEM_OVERLAP")
    appearance.require(not (positive_item_ids & base_reference_item_ids), "POSITIVE_BASE_REFERENCE_ITEM_OVERLAP")

    for target_id in positive_target_ids:
        episode = positive_episodes[target_id]
        query = episode["query"]
        query_receipt = positive_image_index[query["item_id"]]
        query_ray = appearance.strict_ray(selected_frame(episode, query, query_receipt), orientation)
        query_row, query_crop_receipt = appearance.write_crop(
            f"{target_id}_query",
            target_id,
            "query",
            query["item_id"],
            image_spec(query_receipt),
            query_ray,
            base_protocol["observation_window"],
            crop_root,
        )
        query_rows[target_id] = query_row
        all_rows.append(query_row)
        crop_receipts.append(query_crop_receipt)

        reference_rows[target_id] = []
        for index, reference in enumerate(episode["references"], start=1):
            receipt = positive_image_index[reference["item_id"]]
            ray = appearance.strict_ray(selected_frame(episode, reference, receipt), orientation)
            row, crop_receipt = appearance.write_crop(
                f"{target_id}_ref{index}",
                target_id,
                "reference",
                reference["item_id"],
                image_spec(receipt),
                ray,
                base_protocol["observation_window"],
                crop_root,
            )
            reference_rows[target_id].append(row)
            all_rows.append(row)
            crop_receipts.append(crop_receipt)
            reference_metadata.append(
                {
                    "target_id": target_id,
                    "item_id": reference["item_id"],
                    "source": "cross_collection_positive",
                    "collection": reference["collection"],
                    "query_collection": query["collection"],
                    "collection_disjoint": reference["collection"] != query["collection"],
                }
            )
        appearance.require(len(reference_rows[target_id]) == 2, f"POSITIVE_REFERENCE_COUNT:{target_id}")

    model_spec = base_protocol["models"]
    clip_path = appearance.resolve(model_spec["clip"]["path"])
    dino_path = appearance.resolve(model_spec["dinov2"]["path"])
    appearance.require(
        appearance.sha256(clip_path / "pytorch_model.bin") == model_spec["clip"]["weights_sha256"],
        "CLIP_MODEL_HASH_MISMATCH",
    )
    appearance.require(
        appearance.sha256(dino_path / "model.safetensors")
        == model_spec["dinov2"]["weights_sha256"],
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

    rows: list[dict[str, Any]] = []
    accepted_correct = 0
    accepted_wrong = 0
    unknown = 0
    wrong_goal_candidates = 0
    for target_id in positive_target_ids:
        scores = {
            candidate_id: appearance.appearance_score(
                query_rows[target_id],
                reference_rows[candidate_id],
                encoded,
                patch_grid,
                base_protocol["appearance_score"]["weights"],
            )
            for candidate_id in target_ids
        }
        ranked = appearance.prediction(scores, target_id)
        score_gate = float(scores[ranked["prediction"]]["score"]) >= float(
            thresholds["minimum_top1_score"]
        )
        margin_gate = (
            float(scores[ranked["prediction"]]["score"])
            - max(
                float(scores[candidate_id]["score"])
                for candidate_id in target_ids
                if candidate_id != ranked["prediction"]
            )
            >= float(thresholds["minimum_top1_margin"])
        )
        candidate = ranked["prediction"] if score_gate and margin_gate else None
        if candidate == target_id:
            accepted_correct += 1
            route = "APPEARANCE_SEARCH_PRIORITY_CANDIDATE"
        elif candidate is None:
            unknown += 1
            route = "UNKNOWN_KEEP_SEARCHING"
        else:
            accepted_wrong += 1
            route = "WRONG_APPEARANCE_SEARCH_PRIORITY_CANDIDATE"

        controls = []
        for wrong_target in target_ids:
            if wrong_target == target_id:
                continue
            emitted = candidate == wrong_target
            wrong_goal_candidates += int(emitted)
            controls.append(
                {
                    "wrong_target": wrong_target,
                    "search_priority_candidate": emitted,
                    "score": scores[wrong_target]["score"],
                }
            )
        episode = positive_episodes[target_id]
        rows.append(
            {
                "episode_id": target_id,
                "target_name": episode["target_way"]["tags"]["name"],
                "target_way_id": episode["target_way"]["id"],
                "reference_collection": episode["reference_collection"],
                "query_collection": episode["query_collection"],
                "human_truth": truth_index[target_id]["human_truth"],
                "appearance": {"scores": scores, **ranked},
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

    metrics = {
        "fresh_positive_episodes": len(positive_target_ids),
        "fresh_positive_cities": len(
            {positive_episodes[target_id]["source_city"] for target_id in positive_target_ids}
        ),
        "fresh_positive_reference_query_collections": len(
            {
                collection
                for target_id in positive_target_ids
                for collection in (
                    positive_episodes[target_id]["reference_collection"],
                    positive_episodes[target_id]["query_collection"],
                )
            }
        ),
        "combined_target_roster": len(target_ids),
        "combined_reference_views": sum(len(value) for value in reference_rows.values()),
        "fresh_positive_reference_views": sum(
            len(reference_rows[target_id]) for target_id in positive_target_ids
        ),
        "fresh_positive_collection_disjoint_reference_views": sum(
            row["collection_disjoint"]
            for row in reference_metadata
            if row["source"] == "cross_collection_positive"
        ),
        "fresh_positive_exact_item_overlap_with_prior_router": len(
            positive_item_ids & (base_query_item_ids | base_reference_item_ids)
        ),
        "accepted_correct_wrong_unknown": [accepted_correct, accepted_wrong, unknown],
        "positive_coverage": accepted_correct / len(positive_target_ids),
        "wrong_goal_trials": len(positive_target_ids) * (len(target_ids) - 1),
        "wrong_goal_search_priority_candidates": wrong_goal_candidates,
        "portal_ownership_bindings_emitted": 0,
        "model_seconds": time.perf_counter() - started,
    }
    gate = {
        "two_valid_fresh_positive_truth_rows": len(positive_target_ids) == 2,
        "four_cross_collection_reference_views": metrics[
            "fresh_positive_collection_disjoint_reference_views"
        ]
        == 4,
        "zero_prior_item_overlap": metrics["fresh_positive_exact_item_overlap_with_prior_router"]
        == 0,
        "two_of_two_accepted_correct": accepted_correct == 2,
        "zero_accepted_wrong": accepted_wrong == 0,
        "zero_wrong_goal_search_priority_candidates": wrong_goal_candidates == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "FRESH_POSITIVE_TARGET_WAY_ITEM_COLLECTION_DISJOINT_SAME_PROVIDER_DEVELOPMENT_CANDIDATE_ROUTING_ONLY",
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
    print(
        json.dumps(
            {"result": str(output_path), "decision": result["decision"], "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

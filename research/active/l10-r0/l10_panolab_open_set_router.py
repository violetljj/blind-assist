#!/usr/bin/env python3
"""Evaluate a frozen positive-calibrated abstention gate on unseen negatives."""

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


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-open-set-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-open-set-router-result-v1"


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
    negative_source_protocol = appearance.load(appearance.verify(inputs["negative_source_protocol"]))
    negative_selection = appearance.load(appearance.verify(inputs["negative_selection"]))
    negative_materialization = appearance.load(appearance.verify(inputs["negative_materialization"]))
    negative_truth = appearance.load(appearance.verify(inputs["negative_truth"]))
    component_protocol = appearance.load(appearance.verify(inputs["component_router_protocol"]))
    component_result = appearance.load(appearance.verify(inputs["component_router_result"]))
    portal_source = appearance.load(appearance.verify(inputs["reference_portal_source"]))
    node_source = appearance.load(appearance.verify(inputs["node_credential_source"]))
    viviani_selection_path = appearance.verify(inputs["viviani_reference_selection"])
    viviani_selection = appearance.load(viviani_selection_path)
    viviani_materialization = appearance.load(
        appearance.verify(inputs["viviani_reference_materialization"])
    )
    orientation = appearance.load(appearance.verify(inputs["orientation_projection_protocol"]))

    appearance.require(
        negative_selection["protocol_sha256"] == inputs["negative_source_protocol"]["sha256"],
        "NEGATIVE_SELECTION_PROTOCOL_LINK_MISMATCH",
    )
    appearance.require(
        negative_materialization["selection_sha256"] == inputs["negative_selection"]["sha256"],
        "NEGATIVE_MATERIALIZATION_SELECTION_LINK_MISMATCH",
    )
    appearance.require(
        negative_truth["selection_sha256"] == inputs["negative_selection"]["sha256"]
        and negative_truth["materialization_sha256"] == inputs["negative_materialization"]["sha256"],
        "NEGATIVE_TRUTH_SOURCE_LINK_MISMATCH",
    )
    appearance.require(
        all(not row["exact_positive_roster_entity_present"] for row in negative_truth["episodes"]),
        "NEGATIVE_TRUTH_NOT_ALL_ABSENT",
    )
    appearance.require(
        negative_truth["review_contract"]["model_calls_before_truth_freeze"] == 0,
        "MODEL_CALLED_BEFORE_NEGATIVE_TRUTH",
    )
    appearance.require(
        negative_source_protocol["open_set_contract"] == protocol["open_set_contract"],
        "OPEN_SET_CONTRACT_DRIFT",
    )
    positive_scores = [float(row["appearance"]["truth_score"]) for row in component_result["rows"]]
    positive_margins = [float(row["appearance"]["truth_margin"]) for row in component_result["rows"]]
    appearance.require(
        min(positive_scores) == float(protocol["open_set_contract"]["minimum_top1_score"]),
        "POSITIVE_SCORE_THRESHOLD_DRIFT",
    )
    appearance.require(
        min(positive_margins) == float(protocol["open_set_contract"]["minimum_top1_margin"]),
        "POSITIVE_MARGIN_THRESHOLD_DRIFT",
    )

    target_specs = component_protocol["target_roster"]
    target_ids = [row["episode_id"] for row in target_specs]
    target_names = {row["episode_id"]: row["target_name"] for row in target_specs}
    appearance.require(len(target_ids) == 4, "POSITIVE_ROSTER_NOT_FOUR")
    query_item_ids = {row["item_id"] for row in negative_materialization["images"]}
    output_root = appearance.resolve(protocol["output_root"])
    appearance.require(not output_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True)
    all_rows: list[fingerprint.ImageRow] = []
    reference_rows: dict[str, list[fingerprint.ImageRow]] = {}
    negative_rows: dict[str, fingerprint.ImageRow] = {}
    crop_receipts = []

    for target_spec in target_specs:
        target_id = target_spec["episode_id"]
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
            appearance.require(
                reference["item_id"] not in query_item_ids,
                f"NEGATIVE_REFERENCE_ITEM_OVERLAP:{reference['item_id']}",
            )
            row, receipt = appearance.write_crop(
                f"{target_id}_ref{index}",
                target_id,
                "reference",
                reference["item_id"],
                reference["image"],
                ray,
                protocol["observation_window"],
                crop_root,
            )
            reference_rows[target_id].append(row)
            all_rows.append(row)
            crop_receipts.append(receipt)

    image_index = {row["episode_id"]: row for row in negative_materialization["images"]}
    truth_index = {row["episode_id"]: row for row in negative_truth["episodes"]}
    for episode in negative_selection["episodes"]:
        episode_id = episode["episode_id"]
        image_receipt = image_index[episode_id]
        truth = truth_index[episode_id]
        appearance.require(
            image_receipt["sha256"] == truth["image_sha256"],
            f"NEGATIVE_TRUTH_IMAGE_HASH_MISMATCH:{episode_id}",
        )
        frame = {
            "target": {
                "entrance_node": {
                    "id": episode["main_entrance_node"]["id"],
                    "lon_lat": [
                        episode["main_entrance_node"]["lon"],
                        episode["main_entrance_node"]["lat"],
                    ],
                }
            },
            "panorama": {
                "image_size": image_receipt["image_size"],
                "provider_item": episode["provider_item"],
            },
        }
        ray = appearance.strict_ray(frame, orientation)
        row, receipt = appearance.write_crop(
            f"{episode_id}_query",
            episode_id,
            "negative_query",
            episode["item_id"],
            image_receipt,
            ray,
            protocol["observation_window"],
            crop_root,
        )
        negative_rows[episode_id] = row
        all_rows.append(row)
        crop_receipts.append(receipt)

    model_spec = component_protocol["models"]
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
    device = str(backend["selected_device_type"])
    patch_grid = int(model_spec["dinov2"]["patch_grid"])
    started = time.perf_counter()
    encoded = fingerprint._encode_images(
        all_rows,
        models,
        clip_processor,
        dino_processor,
        device,
        patch_grid,
        int(protocol["execution"]["batch_size"]),
    )

    score_threshold = float(protocol["open_set_contract"]["minimum_top1_score"])
    margin_threshold = float(protocol["open_set_contract"]["minimum_top1_margin"])
    rows = []
    false_accepts = 0
    for episode in negative_selection["episodes"]:
        episode_id = episode["episode_id"]
        scores = {
            target_id: appearance.appearance_score(
                negative_rows[episode_id],
                reference_rows[target_id],
                encoded,
                patch_grid,
                component_protocol["appearance_score"]["weights"],
            )
            for target_id in target_ids
        }
        ranking = sorted(target_ids, key=lambda value: (-float(scores[value]["score"]), value))
        top1 = ranking[0]
        top1_score = float(scores[top1]["score"])
        runner_up_score = float(scores[ranking[1]]["score"])
        margin = top1_score - runner_up_score
        accepted = top1_score >= score_threshold and margin >= margin_threshold
        false_accepts += int(accepted)
        rows.append(
            {
                "episode_id": episode_id,
                "negative_target_way_id": episode["target_way"]["id"],
                "negative_target_name": episode["target_way"]["tags"]["name"],
                "minimum_camera_to_positive_roster_entrance_m": episode[
                    "minimum_camera_to_positive_roster_entrance_m"
                ],
                "truth": truth_index[episode_id]["truth"],
                "scores": scores,
                "ranking": ranking,
                "top1_target_id": top1,
                "top1_target_name": target_names[top1],
                "top1_score": top1_score,
                "runner_up_score": runner_up_score,
                "top1_margin": margin,
                "score_gate_met": top1_score >= score_threshold,
                "margin_gate_met": margin >= margin_threshold,
                "decision": "APPEARANCE_SEARCH_PRIORITY_CANDIDATE" if accepted else "UNKNOWN_KEEP_SEARCHING",
                "false_accept": accepted,
                "portal_ownership_binding": None,
            }
        )

    negatives = len(rows)
    rejected = negatives - false_accepts
    positive_accepted = sum(
        float(row["appearance"]["truth_score"]) >= score_threshold
        and float(row["appearance"]["truth_margin"]) >= margin_threshold
        for row in component_result["rows"]
    )
    metrics = {
        "positive_calibration_queries": len(component_result["rows"]),
        "positive_calibration_accepted": positive_accepted,
        "pixel_unseen_negative_queries": negatives,
        "negative_source_cities": len({row["source_city"] for row in negative_selection["episodes"]}),
        "negative_distinct_target_ways": len({int(row["target_way"]["id"]) for row in negative_selection["episodes"]}),
        "negative_distinct_items": len({row["item_id"] for row in negative_selection["episodes"]}),
        "negative_distinct_collections": len({row["collection"] for row in negative_selection["episodes"]}),
        "negative_rejected_unknown": rejected,
        "negative_false_accepts": false_accepts,
        "negative_rejection_rate": rejected / negatives,
        "combined_accepted_correct": positive_accepted,
        "combined_accepted_wrong": false_accepts,
        "combined_accepted_precision": positive_accepted / (positive_accepted + false_accepts),
        "combined_positive_coverage": positive_accepted / len(component_result["rows"]),
        "portal_ownership_bindings_emitted": 0,
        "model_seconds": time.perf_counter() - started,
    }
    gate = {
        "four_negative_queries": negatives == 4,
        "three_negative_source_cities": metrics["negative_source_cities"] == 3,
        "truth_frozen_before_model": negative_truth["review_contract"]["model_calls_before_truth_freeze"] == 0,
        "positive_threshold_retention_four_of_four": positive_accepted == 4,
        "negative_rejection_four_of_four": rejected == 4,
        "zero_negative_false_accepts": false_accepts == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "FRESH_NEGATIVE_OPEN_SET_DEVELOPMENT_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": appearance.sha256(protocol_path),
        "evaluator_sha256": appearance.sha256(Path(__file__).resolve()),
        "execution_backend": backend,
        "backend_receipt_sha256": appearance.sha256(backend_path),
        "open_set_contract": protocol["open_set_contract"],
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

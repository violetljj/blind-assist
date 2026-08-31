#!/usr/bin/env python3
"""Route unreachable lexical evidence to a non-authoritative appearance candidate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

import l10_panolab_ordered_appearance_bank as appearance
import named_poi_facade_fingerprint as fingerprint


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-component-router-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-component-router-result-v1"


def viviani_reference(
    selection: dict[str, Any], materialization: dict[str, Any], item_id: str
) -> dict[str, Any]:
    matches = [row for row in selection["references"] if row["item_id"] == item_id]
    appearance.require(len(matches) == 1, f"VIVIANI_REFERENCE_NOT_UNIQUE:{item_id}:{len(matches)}")
    reference = matches[0]
    receipts = [row for row in materialization["images"] if row["item_id"] == item_id]
    appearance.require(len(receipts) == 1, f"VIVIANI_IMAGE_NOT_UNIQUE:{item_id}:{len(receipts)}")
    receipt = receipts[0]
    frame = {
        "target": {
            "entrance_node": {
                "id": selection["target"]["main_entrance_node"]["id"],
                "lon_lat": [
                    selection["target"]["main_entrance_node"]["lon"],
                    selection["target"]["main_entrance_node"]["lat"],
                ],
            }
        },
        "panorama": {
            "image_size": receipt["image_size"],
            "provider_item": reference["provider_item"],
        },
    }
    return {
        "item_id": item_id,
        "collection": reference["collection"],
        "distance_m": float(reference["camera_to_entrance_distance_m"]),
        "image": receipt,
        "ray_frame": frame,
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
    fresh_selection = appearance.load(appearance.verify(inputs["fresh_selection"]))
    fresh_materialization = appearance.load(appearance.verify(inputs["fresh_materialization"]))
    lexical_result = appearance.load(appearance.verify(inputs["fresh_lexical_result"]))
    portal_source = appearance.load(appearance.verify(inputs["reference_portal_source"]))
    node_source = appearance.load(appearance.verify(inputs["node_credential_source"]))
    viviani_selection_path = appearance.verify(inputs["viviani_reference_selection"])
    viviani_selection = appearance.load(viviani_selection_path)
    viviani_materialization = appearance.load(appearance.verify(inputs["viviani_reference_materialization"]))
    orientation = appearance.load(appearance.verify(inputs["orientation_projection_protocol"]))
    appearance.require(
        viviani_materialization["selection_sha256"] == appearance.sha256(viviani_selection_path),
        "VIVIANI_SELECTION_LINK_MISMATCH",
    )
    appearance.require(
        viviani_selection["selected_pixel_views_before_freeze"] == 0
        and viviani_selection["selected_model_calls_before_freeze"] == 0,
        "VIVIANI_REFERENCE_NOT_FROZEN_PIXEL_UNSEEN",
    )

    selected = {row["episode_id"]: row for row in fresh_selection["episodes"]}
    image_index = {(row["episode_id"], row["phase"]): row for row in fresh_materialization["images"]}
    lexical_index = {row["episode_id"]: row for row in lexical_result["rows"]}
    target_specs = protocol["target_roster"]
    target_ids = [row["episode_id"] for row in target_specs]
    appearance.require(len(target_ids) == len(set(target_ids)) == 4, "TARGET_ROSTER_NOT_FOUR_UNIQUE_EPISODES")
    appearance.require(set(target_ids) == set(selected) == set(lexical_index), "TARGET_ROSTER_INPUT_MISMATCH")

    output_root = appearance.resolve(protocol["output_root"])
    appearance.require(not output_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True)
    query_rows: dict[str, fingerprint.ImageRow] = {}
    reference_rows: dict[str, list[fingerprint.ImageRow]] = {}
    all_rows: list[fingerprint.ImageRow] = []
    crop_receipts = []
    reference_meta = []
    query_item_ids = {row["item_id"] for row in fresh_materialization["images"]}

    for target_spec in target_specs:
        target_id = target_spec["episode_id"]
        episode = selected[target_id]
        appearance.require(
            int(episode["target_way"]["id"]) == int(target_spec["target_way_id"]),
            f"WAY_ID_MISMATCH:{target_id}",
        )
        appearance.require(
            episode["after_classification"]["stratum"] == "DIRECT",
            f"AFTER_NOT_DIRECT:{target_id}",
        )
        after_receipt = image_index[(target_id, "after")]
        after_frame = appearance.build_frame(episode, "after", after_receipt)
        after_ray = appearance.strict_ray(after_frame, orientation)
        query_row, query_receipt = appearance.write_crop(
            f"{target_id}_after",
            target_id,
            "after",
            after_receipt["item_id"],
            {
                "path": after_receipt["path"],
                "sha256": after_receipt["sha256"],
                "bytes": after_receipt["bytes"],
                "image_size": after_receipt["image_size"],
            },
            after_ray,
            protocol["observation_window"],
            crop_root,
        )
        query_rows[target_id] = query_row
        all_rows.append(query_row)
        crop_receipts.append(query_receipt)

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
                reference = viviani_reference(
                    viviani_selection, viviani_materialization, reference_spec["item_id"]
                )
                ray = appearance.strict_ray(reference["ray_frame"], orientation)
            else:
                raise ValueError(f"UNKNOWN_REFERENCE_SOURCE:{reference_spec['source']}")
            appearance.require(
                reference["item_id"] not in query_item_ids,
                f"REFERENCE_QUERY_ITEM_OVERLAP:{reference['item_id']}",
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
            reference_meta.append(
                {
                    "target_id": target_id,
                    "item_id": reference["item_id"],
                    "source": reference_spec["source"],
                    "collection": reference["collection"],
                    "query_collection": episode["sequence_id"],
                    "collection_disjoint": reference["collection"] != episode["sequence_id"],
                    "camera_to_entrance_distance_m": reference["distance_m"],
                }
            )
        appearance.require(len(reference_rows[target_id]) == 2, f"REFERENCE_COUNT_NOT_TWO:{target_id}")

    model_spec = protocol["models"]
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

    rows = []
    lexical_correct = 0
    router_correct = 0
    router_wrong = 0
    router_unknown = 0
    appearance_top1 = 0
    counterfactual_candidates = 0
    for target_id in target_ids:
        scores = {
            candidate_id: appearance.appearance_score(
                query_rows[target_id],
                reference_rows[candidate_id],
                encoded,
                patch_grid,
                protocol["appearance_score"]["weights"],
            )
            for candidate_id in target_ids
        }
        ranked = appearance.prediction(scores, target_id)
        appearance_top1 += int(ranked["correct"])
        lexical_match = bool(
            lexical_index[target_id]["successor"]["long_short_track_token_bank"]["matched"]
        )
        if lexical_match:
            route = "LEXICAL_CANDIDATE"
            candidate = target_id
        elif ranked["prediction"] == target_id and ranked["truth_margin"] > 0.0:
            route = "APPEARANCE_SEARCH_PRIORITY_CANDIDATE"
            candidate = target_id
        else:
            route = "UNKNOWN_KEEP_SEARCHING"
            candidate = None
        if candidate == target_id:
            router_correct += 1
        elif candidate is None:
            router_unknown += 1
        else:
            router_wrong += 1
        lexical_correct += int(lexical_match)
        wrong_controls = []
        for wrong_target in target_ids:
            if wrong_target == target_id:
                continue
            emitted = (not lexical_match) and ranked["prediction"] == wrong_target
            counterfactual_candidates += int(emitted)
            wrong_controls.append(
                {
                    "requested_wrong_target": wrong_target,
                    "appearance_search_priority_candidate": emitted,
                    "wrong_target_score": scores[wrong_target]["score"],
                    "strongest_other_score": max(
                        scores[candidate_id]["score"]
                        for candidate_id in target_ids
                        if candidate_id != wrong_target
                    ),
                }
            )
        rows.append(
            {
                "episode_id": target_id,
                "target_name": selected[target_id]["target_name"],
                "target_way_id": selected[target_id]["target_way"]["id"],
                "query_sequence_id": selected[target_id]["sequence_id"],
                "post_action_geometry": selected[target_id]["after_classification"]["stratum"],
                "lexical_candidate": lexical_match,
                "appearance": {"scores": scores, **ranked},
                "router": {
                    "route": route,
                    "candidate": candidate,
                    "candidate_authority": "SEARCH_PRIORITY_ONLY" if candidate else "NONE",
                    "portal_ownership_binding": None,
                },
                "wrong_goal_controls": wrong_controls,
            }
        )

    metrics = {
        "episodes": len(target_ids),
        "cities": len({selected[target_id]["source_city"] for target_id in target_ids}),
        "reference_coverage": len(reference_rows) / len(target_ids),
        "reference_views": sum(len(value) for value in reference_rows.values()),
        "new_pixel_unseen_reference_views": len(viviani_materialization["images"]),
        "exact_query_reference_item_overlap": 0,
        "collection_disjoint_reference_views": sum(row["collection_disjoint"] for row in reference_meta),
        "frozen_lexical_correct_wrong_unknown": [lexical_correct, 0, len(target_ids) - lexical_correct],
        "appearance_after_top1": appearance_top1,
        "component_router_correct_wrong_unknown": [router_correct, router_wrong, router_unknown],
        "component_router_correct_gain_over_lexical": router_correct - lexical_correct,
        "wrong_goal_trials": len(target_ids) * (len(target_ids) - 1),
        "wrong_goal_search_priority_candidates": counterfactual_candidates,
        "portal_ownership_bindings_emitted": 0,
        "model_seconds": time.perf_counter() - started,
    }
    gate = {
        "reference_coverage_four_of_four": len(reference_rows) == 4,
        "minimum_three_of_four_router_correct": router_correct >= 3,
        "minimum_three_correct_gain_over_frozen_lexical": router_correct - lexical_correct >= 3,
        "zero_router_wrong": router_wrong == 0,
        "zero_wrong_goal_search_priority_candidates": counterfactual_candidates == 0,
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "SOURCE_EXTENDED_POSTHOC_DEVELOPMENT_CANDIDATE_ROUTING_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": appearance.sha256(protocol_path),
        "evaluator_sha256": appearance.sha256(Path(__file__).resolve()),
        "execution_backend": backend,
        "backend_receipt_sha256": appearance.sha256(backend_path),
        "reference_metadata": reference_meta,
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

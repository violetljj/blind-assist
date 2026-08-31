#!/usr/bin/env python3
"""Run the frozen SEVN V5 portal witness on source-disjoint LSAA facades."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_panolab import require, sha256_file
import l10_sevn_pixel_replay as v1
import l10_sevn_pixel_topology_replay as v2
import l10_sevn_ppocrv6_medium_fallback as base
import l10_sevn_ppocrv6_medium_portal_witness as v5


SCHEMA = "blindassist-l10-lsaa-ppocrv6-portal-witness-transfer-protocol-v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def verify_sha256(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"missing frozen input: {path}")
    require(sha256_file(path) == spec["sha256"], f"SHA-256 mismatch: {path}")
    return path


def verify_materialized_image(spec: dict[str, Any], materialization_dir: Path) -> Path:
    path = materialization_dir / f"{spec['item_id']}.jpg"
    require(path.is_file(), f"missing materialized image: {path}")
    require(sha256_file(path) == spec["member_sha256"], f"image SHA-256 mismatch: {path}")
    return path


def score_output(
    output: dict[str, Any],
    masks: dict[str, np.ndarray],
    truth_box: list[float],
    positive: bool,
    evaluation_contract: dict[str, Any],
) -> dict[str, Any]:
    selected = output["selected_binding"]
    exact = bool(output["exact_target_text_candidates"])
    mask_metrics = None
    if selected is not None:
        candidate_id = selected["portal"]["candidate_id"]
        require(candidate_id in masks, f"selected portal mask missing: {candidate_id}")
        mask_metrics = v2.truth_mask_metrics(masks[candidate_id], truth_box, evaluation_contract)
    if positive:
        if selected is None:
            outcome = "UNKNOWN"
        elif mask_metrics["correct"]:
            outcome = "CORRECT_TARGET_DOOR"
        else:
            outcome = "WRONG_DOOR"
    else:
        outcome = "CORRECT_ABSTENTION" if selected is None else "FALSE_BINDING_CONTROL"
    return {
        "outcome": outcome,
        "exact_target_text": exact,
        "binding_emitted": selected is not None,
        "selected_mask_truth_metrics": mask_metrics,
    }


def oracle_portal_coverage(
    masks: dict[str, np.ndarray],
    truth_box: list[float],
    evaluation_contract: dict[str, Any],
) -> dict[str, Any]:
    rows = {
        candidate_id: v2.truth_mask_metrics(mask, truth_box, evaluation_contract)
        for candidate_id, mask in masks.items()
    }
    correct_ids = sorted(candidate_id for candidate_id, metrics in rows.items() if metrics["correct"])
    return {
        "portal_candidate_count": len(rows),
        "correct_truth_portal_candidate_ids": correct_ids,
        "truth_portal_present": bool(correct_ids),
    }


def summarize(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    positive = [row["positive"]["evaluation"][stage] for row in rows]
    controls = [row["counterfactual"]["evaluation"][stage] for row in rows]
    visible_rows = [row for row in rows if row["truth"]["target_house_number_visible"]]
    exact_visible = sum(row["positive"]["evaluation"][stage]["exact_target_text"] for row in visible_rows)
    positive_correct = sum(row["outcome"] == "CORRECT_TARGET_DOOR" for row in positive)
    positive_wrong = sum(row["outcome"] == "WRONG_DOOR" for row in positive)
    positive_unknown = sum(row["outcome"] == "UNKNOWN" for row in positive)
    control_false = sum(row["outcome"] == "FALSE_BINDING_CONTROL" for row in controls)
    control_abstention = sum(row["outcome"] == "CORRECT_ABSTENTION" for row in controls)
    positive_emissions = positive_correct + positive_wrong
    all_emissions = positive_emissions + control_false
    baseline_correct_ids = {
        row["item_id"]
        for row in rows
        if row["positive"]["evaluation"]["baseline"]["outcome"] == "CORRECT_TARGET_DOOR"
    }
    retained = sum(
        row["item_id"] in baseline_correct_ids
        and row["positive"]["evaluation"][stage]["outcome"] == "CORRECT_TARGET_DOOR"
        for row in rows
    )
    return {
        "positive_facades": len(positive),
        "counterfactual_controls": len(controls),
        "truth_visible_house_number_opportunities": len(visible_rows),
        "exact_target_text_when_truth_visible": exact_visible,
        "exact_target_text_when_truth_visible_rate": round(exact_visible / len(visible_rows), 6),
        "positive_correct_target_door_bindings": positive_correct,
        "positive_wrong_door_bindings": positive_wrong,
        "positive_unknown": positive_unknown,
        "positive_binding_rate": round(positive_emissions / len(positive), 6),
        "positive_binding_precision": round(positive_correct / positive_emissions, 6) if positive_emissions else 0.0,
        "counterfactual_false_bindings": control_false,
        "counterfactual_correct_abstentions": control_abstention,
        "counterfactual_false_binding_rate": round(control_false / len(controls), 6),
        "all_emissions": all_emissions,
        "control_adjusted_binding_precision": round(positive_correct / all_emissions, 6) if all_emissions else 0.0,
        "joint_correct_positive_or_control_abstention": positive_correct + control_abstention,
        "joint_decision_accuracy": round((positive_correct + control_abstention) / (len(positive) + len(controls)), 6),
        "baseline_correct_positive_bindings_retained": retained,
        "baseline_correct_positive_bindings_total": len(baseline_correct_ids),
        "positive_truth_portal_candidate_present": sum(row["positive"]["portal_oracle"]["truth_portal_present"] for row in rows),
        "portal_private_medium_branch_executed": sum(
            row["positive"]["runtime_output"][stage].get("medium_receipt", {}).get("executed", False)
            for row in rows
        ) if stage == "successor" else 0,
        "unique_portal_witness_proposals": sum(
            row["positive"]["runtime_output"][stage].get("observation_branch")
            == "PPOCRV6_MEDIUM_UNIQUE_PORTAL_WITNESS"
            for row in rows
        ) if stage == "successor" else 0,
    }


def evaluate_gates(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    successor: dict[str, Any],
    protocol: dict[str, Any],
    source_checks: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    gate = protocol["gate"]
    delta = {
        "visible_exact_ocr_rate_gain": round(
            successor["exact_target_text_when_truth_visible_rate"]
            - baseline["exact_target_text_when_truth_visible_rate"],
            6,
        ),
        "positive_correct_binding_gain": (
            successor["positive_correct_target_door_bindings"]
            - baseline["positive_correct_target_door_bindings"]
        ),
        "positive_wrong_binding_delta": (
            successor["positive_wrong_door_bindings"] - baseline["positive_wrong_door_bindings"]
        ),
        "counterfactual_false_binding_delta": (
            successor["counterfactual_false_bindings"] - baseline["counterfactual_false_bindings"]
        ),
        "joint_decision_accuracy_gain": round(
            successor["joint_decision_accuracy"] - baseline["joint_decision_accuracy"], 6
        ),
    }
    gates = {
        "all_positive_and_counterfactual_rows_evaluated": len(rows) == int(gate["positive_facades"]),
        "frozen_visible_opportunity_count": successor["truth_visible_house_number_opportunities"] == int(gate["visible_opportunities"]),
        "unique_buildings_across_full_source_panel": source_checks["unique_buildings"] == source_checks["source_rows"],
        "zero_development_confirmation_building_overlap": source_checks["cross_split_building_overlap"] == 0,
        "minimum_successor_visible_exact_ocr_rate": successor["exact_target_text_when_truth_visible_rate"] >= float(gate["minimum_successor_visible_exact_ocr_rate"]),
        "minimum_visible_exact_ocr_rate_gain": delta["visible_exact_ocr_rate_gain"] >= float(gate["minimum_visible_exact_ocr_rate_gain"]),
        "minimum_successor_positive_correct_bindings": successor["positive_correct_target_door_bindings"] >= int(gate["minimum_successor_positive_correct_bindings"]),
        "minimum_positive_correct_binding_gain": delta["positive_correct_binding_gain"] >= int(gate["minimum_positive_correct_binding_gain"]),
        "minimum_successor_positive_binding_precision": successor["positive_binding_precision"] >= float(gate["minimum_successor_positive_binding_precision"]),
        "minimum_control_adjusted_binding_precision": successor["control_adjusted_binding_precision"] >= float(gate["minimum_control_adjusted_binding_precision"]),
        "maximum_successor_positive_wrong_bindings": successor["positive_wrong_door_bindings"] <= int(gate["maximum_successor_positive_wrong_bindings"]),
        "maximum_successor_counterfactual_false_bindings": successor["counterfactual_false_bindings"] <= int(gate["maximum_successor_counterfactual_false_bindings"]),
        "counterfactual_false_bindings_not_increased": successor["counterfactual_false_bindings"] <= baseline["counterfactual_false_bindings"],
        "retain_all_baseline_correct_positive_bindings": successor["baseline_correct_positive_bindings_retained"] == successor["baseline_correct_positive_bindings_total"],
    }
    return delta, gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite result: {output_path}")
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    require(protocol.get("schema") == SCHEMA, "protocol schema mismatch")
    require(sha256_file(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "evaluator hash mismatch")

    frozen = protocol["frozen_inputs"]
    source_path = verify_sha256(frozen["public_source"])
    truth_draft_path = verify_sha256(frozen["official_door_truth_draft"])
    pixel_truth_path = verify_sha256(frozen["development_pixel_truth"])
    materialization_path = verify_sha256(frozen["materialization_manifest"])
    selection_receipt_path = verify_sha256(frozen["selection_receipt"])
    source_protocol_path = verify_sha256(frozen["source_protocol"])
    v5_protocol_path = verify_sha256(frozen["v5_protocol"])
    v5_evaluator_path = verify_sha256(frozen["v5_evaluator"])
    require(Path(v5.__file__).resolve() == v5_evaluator_path, "loaded V5 evaluator mismatch")

    source = read_json(source_path)
    truth_draft = read_json(truth_draft_path)
    pixel_truth = read_json(pixel_truth_path)
    materialization = read_json(materialization_path)
    selection_receipt = read_json(selection_receipt_path)
    source_protocol = read_json(source_protocol_path)
    v5_protocol = read_json(v5_protocol_path)
    v2_protocol_path = base.verify_sha256(v5_protocol["frozen_inputs"]["v2_protocol"])
    v2_evaluator_path = base.verify_sha256(v5_protocol["frozen_inputs"]["v2_evaluator"])
    require(Path(v2.__file__).resolve() == v2_evaluator_path, "loaded V2 evaluator mismatch")
    v2_protocol = read_json(v2_protocol_path)

    dev_rows = [row for row in source["rows"] if row["split"] == "DEVELOPMENT"]
    holdout_rows = [row for row in source["rows"] if row["split"] == "CONFIRMATION_HOLDOUT"]
    require(len(dev_rows) == int(protocol["cohort"]["development_facades"]), "development row count mismatch")
    require(len(holdout_rows) == int(protocol["cohort"]["confirmation_holdout_facades"]), "holdout row count mismatch")
    require({row["item_id"] for row in dev_rows} == set(pixel_truth["annotations"]), "pixel truth row mismatch")
    require(pixel_truth["confirmation_holdout_state"] == "PIXELS_NOT_MATERIALIZED_OR_REVIEWED", "holdout state mismatch")
    require(materialization["split"] == "DEVELOPMENT", "materialization split mismatch")
    materialized_by_id = {row["item_id"]: row for row in materialization["rows"]}
    require(set(materialized_by_id) == {row["item_id"] for row in dev_rows}, "materialization row mismatch")
    all_buildings = [row["building"] for row in source["rows"]]
    dev_buildings = {row["building"] for row in dev_rows}
    holdout_buildings = {row["building"] for row in holdout_rows}
    source_checks = {
        "source_rows": len(source["rows"]),
        "unique_buildings": len(set(all_buildings)),
        "unique_facades": len({row["facade_name"] for row in source["rows"]}),
        "unique_panoramas": len({row["panorama_id"] for row in source["rows"]}),
        "cross_split_building_overlap": len(dev_buildings & holdout_buildings),
        "selection_pixel_access_count": selection_receipt["selection_pixel_access_count"],
        "selected_duplicate_building_groups": selection_receipt["selected_duplicate_building_groups"],
    }
    require(source_protocol["selection"]["unique_buildings_across_panel"], "source protocol did not require unique buildings")
    require(source_checks["selection_pixel_access_count"] == 0, "source selection accessed pixels")
    require(not source_checks["selected_duplicate_building_groups"], "duplicate source buildings")

    versions = {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlex": importlib.metadata.version("paddlex"),
    }
    require(versions == v5_protocol["runtime"]["versions"], f"runtime version mismatch: {versions}")
    require(torch.cuda.is_available(), "CUDA unavailable for portal model")

    model_path = base.verify_sha256(v2_protocol["models"]["portal_model"]["weights"])
    postprocessor_path = base.verify_sha256(v2_protocol["models"]["portal_model"]["postprocessor"])
    base.verify_sha256(v2_protocol["models"]["portal_model"]["model_card"])
    primary_root = base.resolve(v2_protocol["models"]["ocr"]["model_root"])
    for filename, expected in v2_protocol["models"]["ocr"]["sha256"].items():
        require(sha256_file(primary_root / filename) == expected, f"primary OCR model mismatch: {filename}")
    medium_det_root = base.verify_model_directory(v5_protocol["models"]["medium_detection"])
    medium_rec_root = base.verify_model_directory(v5_protocol["models"]["medium_recognition"])

    postprocessor = v1.load_module(postprocessor_path)
    portal_model = YOLO(str(model_path))
    require(portal_model.task == "segment", "portal model task mismatch")
    primary_ocr = RapidOCR(params={
        "Global.model_root_dir": str(primary_root),
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": int(v2_protocol["pixel_contract"]["ocr"]["intra_op_threads"]),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })
    medium_ocr = PaddleOCR(
        text_detection_model_dir=str(medium_det_root),
        text_recognition_model_dir=str(medium_rec_root),
        engine="onnxruntime",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    v5.PORTAL_CONTRACT = copy.deepcopy(v5_protocol["portal_medium_observation"])

    materialization_dir = materialization_path.parent
    evaluation_contract = v2_protocol["pixel_contract"]["evaluation"]
    rows = []
    started = time.perf_counter()
    for sequence, row in enumerate(dev_rows, start=1):
        item_id = row["item_id"]
        image_path = verify_materialized_image(materialized_by_id[item_id], materialization_dir)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        require(image is not None, f"image decode failed: {image_path}")
        require(list(image.shape[:2]) == materialized_by_id[item_id]["image_shape_hw"], f"image shape mismatch: {item_id}")
        truth_box = truth_draft["truth"][item_id]["official_single_door_box_xyxy"]
        annotation = pixel_truth["annotations"][item_id]

        positive_baseline, positive_successor, positive_masks = v5.infer_successor(
            image,
            row["mission"],
            primary_ocr,
            medium_ocr,
            portal_model,
            postprocessor,
            v2_protocol["pixel_contract"],
        )
        control_baseline, control_successor, control_masks = v5.infer_successor(
            image,
            row["counterfactual_mission"],
            primary_ocr,
            medium_ocr,
            portal_model,
            postprocessor,
            v2_protocol["pixel_contract"],
        )
        positive_evaluation = {
            "baseline": score_output(positive_baseline, positive_masks, truth_box, True, evaluation_contract),
            "successor": score_output(positive_successor, positive_masks, truth_box, True, evaluation_contract),
        }
        control_evaluation = {
            "baseline": score_output(control_baseline, control_masks, truth_box, False, evaluation_contract),
            "successor": score_output(control_successor, control_masks, truth_box, False, evaluation_contract),
        }
        rows.append(
            {
                "item_id": item_id,
                "building": row["building"],
                "panorama_id": row["panorama_id"],
                "image_receipt": materialized_by_id[item_id],
                "truth": {
                    "official_single_door_box_xyxy": truth_box,
                    "target_house_number_visible": annotation["target_house_number_visible"],
                    "target_credential_box_xyxy": annotation["target_credential_box_xyxy"],
                    "review_note": annotation["review_note"],
                },
                "positive": {
                    "mission": row["mission"],
                    "runtime_output": {"baseline": positive_baseline, "successor": positive_successor},
                    "evaluation": positive_evaluation,
                    "portal_oracle": oracle_portal_coverage(positive_masks, truth_box, evaluation_contract),
                },
                "counterfactual": {
                    "mission": row["counterfactual_mission"],
                    "runtime_output": {"baseline": control_baseline, "successor": control_successor},
                    "evaluation": control_evaluation,
                },
            }
        )
        print(
            json.dumps(
                {
                    "item": item_id,
                    "progress": f"{sequence}/{len(dev_rows)}",
                    "visible": annotation["target_house_number_visible"],
                    "baseline_positive": positive_evaluation["baseline"]["outcome"],
                    "successor_positive": positive_evaluation["successor"]["outcome"],
                    "successor_branch": positive_successor["observation_branch"],
                    "successor_control": control_evaluation["successor"]["outcome"],
                }
            ),
            flush=True,
        )

    actual_device = str(next(portal_model.model.parameters()).device)
    require(actual_device.startswith("cuda"), f"silent portal fallback: {actual_device}")
    baseline_metrics = summarize(rows, "baseline")
    successor_metrics = summarize(rows, "successor")
    delta, gates = evaluate_gates(rows, baseline_metrics, successor_metrics, protocol, source_checks)
    gate_met = all(gates.values())
    result = {
        "schema": "blindassist-l10-lsaa-ppocrv6-portal-witness-transfer-result-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision": (
            "L10_LSAA_PPOCRV6_PORTAL_WITNESS_SOURCE_DISJOINT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_LSAA_PPOCRV6_PORTAL_WITNESS_SOURCE_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "claim_scope": "PROVIDER_CITY_AND_SELECTED_BUILDING_DISJOINT_LSAA_DEVELOPMENT",
        "question": protocol["question"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "official_door_truth_draft": {"path": str(truth_draft_path), "sha256": sha256_file(truth_draft_path)},
            "development_pixel_truth": {"path": str(pixel_truth_path), "sha256": sha256_file(pixel_truth_path)},
            "materialization_manifest": {"path": str(materialization_path), "sha256": sha256_file(materialization_path)},
            "selection_receipt": {"path": str(selection_receipt_path), "sha256": sha256_file(selection_receipt_path)},
            "source_protocol": {"path": str(source_protocol_path), "sha256": sha256_file(source_protocol_path)},
            "v5_protocol": {"path": str(v5_protocol_path), "sha256": sha256_file(v5_protocol_path)},
            "v5_evaluator": {"path": str(v5_evaluator_path), "sha256": sha256_file(v5_evaluator_path)},
            "v2_protocol": {"path": str(v2_protocol_path), "sha256": sha256_file(v2_protocol_path)},
            "v2_evaluator": {"path": str(v2_evaluator_path), "sha256": sha256_file(v2_evaluator_path)},
        },
        "runtime": {
            "versions": versions,
            "torch_cuda": torch.version.cuda,
            "portal_device": actual_device,
            "device_name": torch.cuda.get_device_name(0),
            "medium_ocr_device": "cpu",
        },
        "source_checks": source_checks,
        "baseline_metrics": baseline_metrics,
        "successor_metrics": successor_metrics,
        "delta": delta,
        "gates": gates,
        "gate_met": gate_met,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
        "non_claims": protocol["non_claims"],
    }
    output_path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "baseline_metrics": result["baseline_metrics"],
                "successor_metrics": result["successor_metrics"],
                "delta": result["delta"],
                "gates": result["gates"],
                "wall_seconds": result["wall_seconds"],
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

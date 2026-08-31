#!/usr/bin/env python3
"""Evaluate high-resolution portal-relative credential fields on LSAA Vienna.

The mechanism expands each strong portal into a high-resolution credential
observation field.  Exact text is retained as lexical evidence, while the
portal-relative edge remains explicitly non-authoritative for door ownership.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from ultralytics import YOLO

from l10_panolab import require, sha256_file
import l10_sevn_pixel_replay as v1
import l10_sevn_pixel_topology_replay as v2
import l10_sevn_ppocrv6_medium_fallback as medium
import l10_sevn_ppocrv6_medium_portal_witness as v5
import l10_lsaa_ppocrv6_portal_witness_transfer as transfer


SCHEMA = "blindassist-l10-lsaa-adaptive-credential-fields-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-lsaa-adaptive-credential-fields-result-v1"
ADAPTIVE_CONTRACT: dict[str, Any] = {}
_FIELD_CACHE: dict[str, tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...]]] = {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def adaptive_credential_specs(
    portals: list[dict[str, Any]],
    image: np.ndarray,
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    ranked = sorted(
        portals,
        key=lambda row: (-row["summary"]["confidence"], row["summary"]["candidate_id"]),
    )[: int(ADAPTIVE_CONTRACT["maximum_portals"])]
    specs: list[dict[str, Any]] = []
    for portal in ranked:
        summary = portal["summary"]
        x1, y1, x2, y2 = [float(value) for value in summary["mask_box_xyxy"]]
        portal_width = max(1.0, x2 - x1)
        portal_height = max(1.0, y2 - y1)
        crop_x1 = max(
            0,
            math.floor(x1 - float(ADAPTIVE_CONTRACT["lateral_portal_widths"]) * portal_width),
        )
        crop_x2 = min(
            width,
            math.ceil(x2 + float(ADAPTIVE_CONTRACT["lateral_portal_widths"]) * portal_width),
        )
        crop_y1 = max(
            0,
            math.floor(y1 - float(ADAPTIVE_CONTRACT["upward_portal_heights"]) * portal_height),
        )
        crop_y2 = min(
            height,
            math.ceil(y2 + float(ADAPTIVE_CONTRACT["downward_portal_heights"]) * portal_height),
        )
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        if min(crop_width, crop_height) < int(ADAPTIVE_CONTRACT["minimum_crop_side_pixels"]):
            continue
        scale = min(
            float(ADAPTIVE_CONTRACT["maximum_magnification"]),
            float(ADAPTIVE_CONTRACT["canonical_long_side_pixels"]) / max(crop_width, crop_height),
        )
        scale = max(1.0, scale)
        specs.append(
            {
                "pass_id": f"PORTAL_CREDENTIAL_FIELD_{summary['candidate_id']}",
                "source_portal_id": summary["candidate_id"],
                "x": crop_x1,
                "y": crop_y1,
                "width": crop_width,
                "height": crop_height,
                "output_width": max(1, round(crop_width * scale)),
                "output_height": max(1, round(crop_height * scale)),
                "magnification": round(scale, 6),
                "credential_region": "PORTAL_RELATIVE_HIGH_RESOLUTION_FIELD",
                "association_authority": "PROXIMITY_ONLY_NOT_PORTAL_OWNERSHIP",
            }
        )
    return specs


def _global_box(local_box: list[float], spec: dict[str, Any]) -> list[float]:
    scale = float(spec["magnification"])
    return [
        round(float(spec["x"]) + float(local_box[0]) / scale, 6),
        round(float(spec["y"]) + float(local_box[1]) / scale, 6),
        round(float(spec["x"]) + float(local_box[2]) / scale, 6),
        round(float(spec["y"]) + float(local_box[3]) / scale, 6),
    ]


def adaptive_credential_rows(
    pipeline: PaddleOCR,
    image: np.ndarray,
    portals: list[dict[str, Any]],
    v2_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = adaptive_credential_specs(portals, image)
    signature = tuple(
        f"{row['source_portal_id']}:{row['x']}:{row['y']}:{row['width']}:{row['height']}:{row['magnification']}"
        for row in specs
    )
    image_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
    cached = _FIELD_CACHE.get(image_sha256)
    if cached is not None and cached[2] == signature:
        rows, receipt, _ = cached
        reused = copy.deepcopy(receipt)
        reused["cache_reused_for_same_pixels_different_mission"] = True
        reused["seconds"] = 0.0
        return copy.deepcopy(rows), reused

    rows: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []
    started = time.perf_counter()
    for spec in specs:
        crop = image[
            spec["y"] : spec["y"] + spec["height"],
            spec["x"] : spec["x"] + spec["width"],
        ]
        require(crop.size > 0, f"empty credential field: {spec['source_portal_id']}")
        if (spec["output_width"], spec["output_height"]) != (spec["width"], spec["height"]):
            crop = cv2.resize(
                crop,
                (spec["output_width"], spec["output_height"]),
                interpolation=cv2.INTER_CUBIC,
            )
        current, receipt = medium.medium_full_view(pipeline, crop, v2_contract)
        for row in current:
            local_box = list(row["box_xyxy"])
            row["credential_field_local_box_xyxy"] = local_box
            row["box_xyxy"] = _global_box(local_box, spec)
            row["source_portal_id"] = spec["source_portal_id"]
            row["source_pass"] = spec["pass_id"]
            row["observed_passes"] = [spec["pass_id"]]
            row["association_authority"] = "PROXIMITY_ONLY_NOT_PORTAL_OWNERSHIP"
        rows.extend(current)
        passes.append(
            {
                **spec,
                "raw_word_count": receipt["raw_word_count"],
                "deduplicated_word_count": receipt["deduplicated_word_count"],
                "seconds": receipt["seconds"],
            }
        )
    result_receipt = {
        "executed": True,
        "branch": "PPOCRV6_MEDIUM_ADAPTIVE_PORTAL_CREDENTIAL_FIELDS",
        "passes": passes,
        "raw_word_count": len(rows),
        "seconds": round(time.perf_counter() - started, 6),
        "cache_reused_for_same_pixels_different_mission": False,
        "binding_policy": "LEXICAL_EVIDENCE_ONLY_PORTAL_OWNERSHIP_REMAINS_UNKNOWN",
    }
    _FIELD_CACHE[image_sha256] = (copy.deepcopy(rows), copy.deepcopy(result_receipt), signature)
    return rows, result_receipt


def infer_successor(
    image: np.ndarray,
    mission: dict[str, Any],
    primary_ocr: RapidOCR,
    medium_ocr: PaddleOCR,
    portal_model: YOLO,
    postprocessor: Any,
    v2_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    baseline, masks = v2.infer_binding(
        image,
        mission,
        primary_ocr,
        portal_model,
        postprocessor,
        v2_contract,
    )
    successor = copy.deepcopy(baseline)
    successor["observation_branch"] = "V2_RESULT_RETAINED"
    successor["portal_witness_set"] = []
    successor["medium_exact_target_candidates"] = []
    successor["medium_receipt"] = {
        "executed": False,
        "reason": "V2_ALREADY_EMITTED_A_PROPOSAL",
        "passes": [],
        "raw_word_count": 0,
        "seconds": 0.0,
    }
    successor["portal_medium_receipt"] = copy.deepcopy(successor["medium_receipt"])
    if baseline["selected_binding"] is not None:
        return baseline, successor, masks
    if baseline["state"] not in {
        "UNKNOWN_TARGET_TEXT",
        "UNKNOWN_NO_ADMISSIBLE_MASK_TOPOLOGY",
    }:
        successor["medium_receipt"]["reason"] = "BASELINE_DEFICIT_NOT_ELIGIBLE"
        successor["portal_medium_receipt"] = copy.deepcopy(successor["medium_receipt"])
        return baseline, successor, masks

    portals = [
        {"summary": summary, "mask": masks[summary["candidate_id"]]}
        for summary in baseline["portal_mask_candidates"]
    ]
    if not portals:
        successor["medium_receipt"]["reason"] = "NO_SURVIVING_PORTAL_MASK"
        successor["portal_medium_receipt"] = copy.deepcopy(successor["medium_receipt"])
        return baseline, successor, masks

    rows, receipt = adaptive_credential_rows(medium_ocr, image, portals, v2_contract)
    target = v1.canonical_token(mission["house_number"])
    exact = [
        row
        for row in rows
        if row["canonical"] == target
        and row["score"] >= float(v2_contract["ocr"]["minimum_score"])
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in exact:
        grouped.setdefault(row["source_portal_id"], []).append(row)
    witness_set = [
        {
            "portal_id": portal_id,
            "supporting_exact_rows": len(grouped[portal_id]),
            "best_text": sorted(
                grouped[portal_id],
                key=lambda row: (-row["score"], tuple(row["box_xyxy"])),
            )[0],
            "authority": "PORTAL_RELATIVE_OBSERVATION_NOT_PORTAL_OWNERSHIP",
        }
        for portal_id in sorted(grouped)
    ]
    successor["medium_receipt"] = receipt
    successor["portal_medium_receipt"] = receipt
    successor["medium_ocr_words"] = rows
    successor["medium_exact_target_candidates"] = exact
    successor["portal_witness_set"] = witness_set
    successor["candidate_portal_observation_edges"] = witness_set
    if exact:
        successor["exact_target_text_candidates"] = exact
        successor["exact_target_text_full_pass"] = False
        successor["exact_target_text_tile_pass"] = False
        successor["state"] = "LEXICAL_OBSERVED_ADAPTIVE_CREDENTIAL_FIELD_PORTAL_OWNERSHIP_UNKNOWN"
        successor["selected_binding"] = None
        successor["observation_branch"] = "PPOCRV6_MEDIUM_ADAPTIVE_CREDENTIAL_FIELD_EXACT_LEXICAL_ONLY"
    else:
        successor["observation_branch"] = "PPOCRV6_MEDIUM_ADAPTIVE_CREDENTIAL_FIELD_NO_EXACT_TOKEN"
    return baseline, successor, masks


def evaluate_gates(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    successor: dict[str, Any],
    protocol: dict[str, Any],
    source_checks: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    gate = protocol["gate"]
    visible_gain = round(
        successor["exact_target_text_when_truth_visible_rate"]
        - baseline["exact_target_text_when_truth_visible_rate"],
        6,
    )
    new_bindings = 0
    lexical_rows = 0
    for row in rows:
        for side in ("positive", "counterfactual"):
            prior = row[side]["runtime_output"]["baseline"]
            current = row[side]["runtime_output"]["successor"]
            new_bindings += int(prior["selected_binding"] is None and current["selected_binding"] is not None)
            lexical_rows += int(
                current.get("observation_branch")
                == "PPOCRV6_MEDIUM_ADAPTIVE_CREDENTIAL_FIELD_EXACT_LEXICAL_ONLY"
            )
    delta = {
        "visible_exact_ocr_rate_gain": visible_gain,
        "visible_exact_ocr_count_gain": (
            successor["exact_target_text_when_truth_visible"]
            - baseline["exact_target_text_when_truth_visible"]
        ),
        "new_portal_bindings_from_lexical_only_branch": new_bindings,
        "adaptive_credential_field_exact_lexical_rows": lexical_rows,
        "counterfactual_false_binding_delta": (
            successor["counterfactual_false_bindings"] - baseline["counterfactual_false_bindings"]
        ),
    }
    gates = {
        "all_positive_and_counterfactual_rows_evaluated": len(rows) == int(gate["positive_facades"]),
        "frozen_visible_opportunity_count": successor["truth_visible_house_number_opportunities"] == int(gate["visible_opportunities"]),
        "unique_buildings_across_full_source_panel": source_checks["unique_buildings"] == source_checks["source_rows"],
        "zero_development_confirmation_building_overlap": source_checks["cross_split_building_overlap"] == 0,
        "minimum_successor_visible_exact_ocr_rate": successor["exact_target_text_when_truth_visible_rate"] >= float(gate["minimum_successor_visible_exact_ocr_rate"]),
        "minimum_visible_exact_ocr_rate_gain": visible_gain >= float(gate["minimum_visible_exact_ocr_rate_gain"]),
        "minimum_visible_exact_ocr_count": successor["exact_target_text_when_truth_visible"] >= int(gate["minimum_visible_exact_ocr_count"]),
        "no_new_portal_binding_from_lexical_only_branch": new_bindings == 0,
        "counterfactual_false_bindings_not_increased": successor["counterfactual_false_bindings"] <= baseline["counterfactual_false_bindings"],
        "retain_all_baseline_correct_positive_bindings": successor["baseline_correct_positive_bindings_retained"] == successor["baseline_correct_positive_bindings_total"],
    }
    return delta, gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    global ADAPTIVE_CONTRACT
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite result: {output_path}")
    protocol = read_json(args.protocol.resolve())
    require(protocol.get("schema") == SCHEMA, "protocol schema mismatch")
    base_runner = transfer.verify_sha256(protocol["frozen_inputs"]["base_transfer_runner"])
    require(base_runner == Path(transfer.__file__).resolve(), "loaded transfer runner mismatch")
    ADAPTIVE_CONTRACT = copy.deepcopy(protocol["adaptive_credential_observation"])
    _FIELD_CACHE.clear()

    temporary = output_path.with_name(output_path.name + ".internal-runner.tmp")
    require(not temporary.exists(), f"stale internal runner result: {temporary}")
    original_schema = transfer.SCHEMA
    original_file = transfer.__file__
    original_infer = v5.infer_successor
    original_gates = transfer.evaluate_gates
    transfer.SCHEMA = SCHEMA
    transfer.__file__ = str(Path(__file__).resolve())
    v5.infer_successor = infer_successor
    transfer.evaluate_gates = evaluate_gates
    try:
        result = transfer.run(argparse.Namespace(protocol=args.protocol, output=temporary))
        result["schema"] = RESULT_SCHEMA
        result["decision"] = (
            "L10_LSAA_ADAPTIVE_CREDENTIAL_FIELDS_LEXICAL_SOURCE_DISJOINT_DEVELOPMENT_GATE_MET"
            if result["gate_met"]
            else "L10_LSAA_ADAPTIVE_CREDENTIAL_FIELDS_LEXICAL_SOURCE_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
        )
        result["question"] = protocol["question"]
        result["inputs"]["base_transfer_runner"] = {
            "path": str(base_runner),
            "sha256": sha256_file(base_runner),
            "reuse": "HASH_BOUND_SOURCE_LOAD_SCORE_AND_RECEIPT_RUNNER_WITH_FROZEN_INFERENCE_HOOK",
        }
        result["adaptive_credential_observation"] = ADAPTIVE_CONTRACT
        result["research_basis"] = protocol["research_basis"]
        result["retention"] = {
            "durable_result": str(output_path),
            "internal_runner_temporary_result_removed": True,
            "confirmation_holdout": "PIXELS_NOT_MATERIALIZED_OR_REVIEWED",
        }
        output_path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        return result
    finally:
        transfer.SCHEMA = original_schema
        transfer.__file__ = original_file
        v5.infer_successor = original_infer
        transfer.evaluate_gates = original_gates
        _FIELD_CACHE.clear()
        if temporary.exists():
            temporary.unlink()


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
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

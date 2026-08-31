#!/usr/bin/env python3
"""Evaluate portal-private PP-OCRv6 medium lexical witnesses on fresh SEVN PAN views.

This reuses the hash-bound medium-expert evaluator's frozen loading, rendering,
scoring, and receipt runner.  Only its abstention-branch inference and gates are
replaced: every portal gets its own upper-mask observation, and exactly one
portal-local exact-token witness is required before a proposal is emitted.
"""

from __future__ import annotations

import argparse
import copy
import json
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
import l10_sevn_portal_conditioned_ocr as portal_crop
import l10_sevn_ppocrv6_medium_fallback as base


SCHEMA = "blindassist-l10-sevn-ppocrv6-medium-portal-witness-protocol-v1"
PORTAL_CONTRACT: dict[str, Any] = {}


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


def portal_private_rows(
    pipeline: PaddleOCR,
    image: np.ndarray,
    portals: list[dict[str, Any]],
    v2_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = portal_crop.portal_conditioned_specs(
        portals,
        image,
        PORTAL_CONTRACT,
        v2_contract["association"],
    )
    rows: list[dict[str, Any]] = []
    passes = []
    started = time.perf_counter()
    for spec in specs:
        crop = image[
            spec["y"] : spec["y"] + spec["height"],
            spec["x"] : spec["x"] + spec["width"],
        ]
        require(crop.size > 0, f"empty portal crop: {spec['source_portal_id']}")
        if (spec["output_width"], spec["output_height"]) != (spec["width"], spec["height"]):
            crop = cv2.resize(
                crop,
                (spec["output_width"], spec["output_height"]),
                interpolation=cv2.INTER_CUBIC,
            )
        current, receipt = base.medium_full_view(pipeline, crop, v2_contract)
        for row in current:
            row["source_portal_id"] = spec["source_portal_id"]
            row["portal_local_box_xyxy"] = row["box_xyxy"]
        rows.extend(current)
        passes.append(
            {
                **spec,
                "raw_word_count": receipt["raw_word_count"],
                "deduplicated_word_count": receipt["deduplicated_word_count"],
                "seconds": receipt["seconds"],
            }
        )
    return rows, {
        "executed": True,
        "branch": "PPOCRV6_MEDIUM_PORTAL_PRIVATE_OBSERVATIONS",
        "passes": passes,
        "raw_word_count": len(rows),
        "seconds": round(time.perf_counter() - started, 6),
    }


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

    rows, receipt = portal_private_rows(medium_ocr, image, portals, v2_contract)
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
    witness_set = []
    for portal_id in sorted(grouped):
        best = sorted(
            grouped[portal_id],
            key=lambda row: (-row["score"], tuple(row["portal_local_box_xyxy"])),
        )[0]
        witness_set.append(
            {
                "portal_id": portal_id,
                "text": best,
                "supporting_exact_rows": len(grouped[portal_id]),
            }
        )

    successor["medium_receipt"] = receipt
    successor["portal_medium_receipt"] = receipt
    successor["medium_ocr_words"] = rows
    successor["medium_exact_target_candidates"] = [row["text"] for row in witness_set]
    successor["portal_witness_set"] = witness_set
    if witness_set:
        successor["exact_target_text_candidates"] = [row["text"] for row in witness_set]
        successor["exact_target_text_full_pass"] = False
        successor["exact_target_text_tile_pass"] = False
    if len(witness_set) == 1:
        witness = witness_set[0]
        portal = next(row for row in portals if row["summary"]["candidate_id"] == witness["portal_id"])
        successor["state"] = "PIXEL_BOUND_PPOCRV6_MEDIUM_UNIQUE_PORTAL_WITNESS"
        successor["selected_binding"] = {
            "text": witness["text"],
            "portal": portal["summary"],
            "authority_edge": {
                "type": "PORTAL_TO_PRIVATE_MEDIUM_OBSERVATION_TO_EXACT_TOKEN",
                "source_portal_id": witness["portal_id"],
                "unique_portal_witness_count": 1,
            },
        }
        successor["observation_branch"] = "PPOCRV6_MEDIUM_UNIQUE_PORTAL_WITNESS"
    elif len(witness_set) > 1:
        successor["state"] = "UNKNOWN_AMBIGUOUS_PPOCRV6_MEDIUM_PORTAL_WITNESS_SET"
        successor["selected_binding"] = None
        successor["observation_branch"] = "PPOCRV6_MEDIUM_AMBIGUOUS_PORTAL_WITNESS_SET"
    else:
        successor["observation_branch"] = "PPOCRV6_MEDIUM_NO_PORTAL_WITNESS"
    return baseline, successor, masks


def evaluate_gates(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    successor: dict[str, Any],
    disjointness: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    exact_gain = successor["exact_target_text_when_truth_visible"] - baseline["exact_target_text_when_truth_visible"]
    exact_rate_gain = successor["exact_target_text_when_truth_visible_rate"] - baseline["exact_target_text_when_truth_visible_rate"]
    correct_gain = successor["correct_target_door_bindings"] - baseline["correct_target_door_bindings"]
    wrong_delta = successor["wrong_door_bindings"] - baseline["wrong_door_bindings"]
    baseline_correct_ids = {
        row["episode_id"]
        for row in rows
        if row["evaluation"]["baseline"]["outcome"] == "CORRECT_TARGET_DOOR"
    }
    retained = sum(
        row["episode_id"] in baseline_correct_ids
        and row["evaluation"]["successor"]["outcome"] == "CORRECT_TARGET_DOOR"
        for row in rows
    )
    emitted = successor["correct_target_door_bindings"] + successor["wrong_door_bindings"]
    precision = round(successor["correct_target_door_bindings"] / emitted, 6) if emitted else 0.0
    unique_rows = [
        row for row in rows
        if row["runtime_output"]["successor"]["observation_branch"]
        == "PPOCRV6_MEDIUM_UNIQUE_PORTAL_WITNESS"
    ]
    unique_correct = sum(row["evaluation"]["successor"]["outcome"] == "CORRECT_TARGET_DOOR" for row in unique_rows)
    delta = {
        "exact_target_text_gain": exact_gain,
        "exact_target_text_rate_gain": round(exact_rate_gain, 6),
        "correct_target_door_binding_gain": correct_gain,
        "correct_target_door_binding_rate_gain": round(successor["correct_target_door_binding_rate"] - baseline["correct_target_door_binding_rate"], 6),
        "wrong_door_binding_delta": wrong_delta,
        "successor_binding_precision": precision,
        "baseline_correct_bindings_retained": retained,
        "baseline_correct_bindings_total": len(baseline_correct_ids),
        "portal_medium_branch_executed": sum(row["runtime_output"]["successor"]["medium_receipt"]["executed"] for row in rows),
        "unique_portal_witness_proposals": len(unique_rows),
        "unique_portal_witness_correct": unique_correct,
        "unique_portal_witness_wrong": len(unique_rows) - unique_correct,
        "ambiguous_portal_witness_sets": sum(
            row["runtime_output"]["successor"]["observation_branch"]
            == "PPOCRV6_MEDIUM_AMBIGUOUS_PORTAL_WITNESS_SET"
            for row in rows
        ),
    }
    gates = {
        "all_episodes_rendered": len(rows) == int(gate["episode_count"]),
        "minimum_truth_visible_house_number_opportunities": successor["truth_visible_house_number_opportunities"] >= int(gate["minimum_truth_visible_house_number_opportunities"]),
        "minimum_successor_visible_text_exact_ocr_rate": successor["exact_target_text_when_truth_visible_rate"] >= float(gate["minimum_successor_visible_text_exact_ocr_rate"]),
        "minimum_exact_ocr_rate_gain": exact_rate_gain >= float(gate["minimum_exact_ocr_rate_gain"]),
        "minimum_successor_correct_binding_rate": successor["correct_target_door_binding_rate"] >= float(gate["minimum_successor_correct_binding_rate"]),
        "minimum_correct_binding_gain": correct_gain >= int(gate["minimum_correct_binding_gain"]),
        "minimum_successor_binding_precision": precision >= float(gate["minimum_successor_binding_precision"]),
        "maximum_successor_wrong_bindings": successor["wrong_door_bindings"] <= int(gate["maximum_successor_wrong_bindings"]),
        "maximum_wrong_binding_delta": wrong_delta <= int(gate["maximum_wrong_binding_delta"]),
        "retain_all_baseline_correct_bindings": retained == len(baseline_correct_ids),
        "cumulative_reference_address_disjoint": disjointness["reference_address_overlap_count"] == 0,
        "cumulative_reference_panorama_frame_disjoint": disjointness["reference_frame_overlap_count"] == 0,
        "within_panel_cross_episode_frame_unique": disjointness["within_panel_cross_episode_reused_frames"] == {},
    }
    return delta, gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    global PORTAL_CONTRACT
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite result: {output_path}")
    protocol = read_json(args.protocol.resolve())
    require(protocol.get("schema") == SCHEMA, "protocol schema mismatch")
    base_runner = verify_sha256(protocol["frozen_inputs"]["base_runner"])
    require(base_runner == Path(base.__file__).resolve(), "loaded base runner mismatch")
    PORTAL_CONTRACT = protocol["portal_medium_observation"]

    temporary = output_path.with_name(output_path.name + ".internal-runner.tmp")
    require(not temporary.exists(), f"stale internal runner result: {temporary}")
    original_schema = base.SCHEMA
    original_file = base.__file__
    original_infer = base.infer_successor
    original_gates = base.evaluate_gates
    base.SCHEMA = SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    base.infer_successor = infer_successor
    base.evaluate_gates = evaluate_gates
    try:
        base_args = argparse.Namespace(
            protocol=args.protocol,
            highres_archive=args.highres_archive,
            output=temporary,
        )
        result = base.run(base_args)
        result["schema"] = "blindassist-l10-sevn-ppocrv6-medium-portal-witness-result-v1"
        result["decision"] = (
            "L10_SEVN_PPOCRV6_MEDIUM_PORTAL_WITNESS_FRESH_PAN_DEVELOPMENT_GATE_MET"
            if result["gate_met"]
            else "L10_SEVN_PPOCRV6_MEDIUM_PORTAL_WITNESS_FRESH_PAN_DEVELOPMENT_GATE_NOT_MET"
        )
        result["inputs"]["base_runner"] = {
            "path": str(base_runner),
            "sha256": sha256_file(base_runner),
            "reuse": "HASH_BOUND_RENDER_LOAD_SCORE_AND_RECEIPT_RUNNER_WITH_FROZEN_INFERENCE_HOOK",
        }
        result["retention"]["durable_result"] = str(output_path)
        result["retention"]["internal_runner_temporary_result_removed"] = True
        output_path.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        return result
    finally:
        base.SCHEMA = original_schema
        base.__file__ = original_file
        base.infer_successor = original_infer
        base.evaluate_gates = original_gates
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--highres-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "decision": result["decision"],
        "baseline_metrics": result["baseline_metrics"],
        "successor_metrics": result["successor_metrics"],
        "delta": result["delta"],
        "gates": result["gates"],
        "wall_seconds": result["wall_seconds"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

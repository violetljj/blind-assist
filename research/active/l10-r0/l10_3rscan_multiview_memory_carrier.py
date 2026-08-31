#!/usr/bin/env python3
"""Run the frozen coherent-cycle carrier with an arbitrary frozen image cardinality."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import zipfile
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cycle_component_open_set_posthoc as carrier  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402
import l10_3rscan_roma_cycle_prompt_dual_surface_posthoc as dual  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


def load_images_unbounded(
    protocol: dict[str, Any], cohort: dict[str, Any]
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    receipts: dict[str, Any] = {}
    for row in cohort["images"].values():
        episode_id, role = str(row["episode_id"]), str(row["role"])
        key = f"{episode_id}:{role}"
        manifest_key = f"{row['scan_id']}/sequence.zip"
        archive_path = artifact_root / cohort["source_manifest"][manifest_key]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        base.require(list(image.size) == row["color_size"], f"IMAGE_SIZE:{key}")
        images[key] = image
        receipts[key] = {
            "scan_id": row["scan_id"],
            "frame": int(row["frame"]),
            "zip_member": row["zip_member"],
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "image_bytes": len(payload),
            "target_bbox_xyxy_evaluation_only": row["bbox_xyxy"],
        }
    base.require(len(images) == len(cohort["images"]), "IMAGE_COUNT_MISMATCH")
    return images, receipts


def replay(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    revision = protocol["execution_revision"]
    pixel.require(pixel.sha256(Path(__file__)) == revision["loader_sha256"], "LOADER_HASH")
    pixel.require(pixel.sha256(HERE / revision["failure_receipt_path"]) == revision["failure_receipt_sha256"], "FAILURE_RECEIPT_HASH")
    resume = protocol["resume_revision"]
    pixel.require(pixel.sha256(HERE / resume["failure_receipt_path"]) == resume["failure_receipt_sha256"], "RESUME_RECEIPT_HASH")
    consumed = pixel.load_json(HERE / resume["failure_receipt_path"])["consumed_pair"]
    consumed_id = str(consumed["id"])

    original_loader = base.load_images
    original_protocol_loader = base.load_protocol
    original_dual = dual.dual_surface_cycle_affine_prompt
    original_iou = base.bbox_iou

    def load_resumed_protocol(path: Path) -> dict[str, Any]:
        loaded = deepcopy(original_protocol_loader(path))
        pairs = loaded["evaluation"]["pairs"]
        base.require(str(pairs[0]["id"]) == consumed_id, "CONSUMED_PAIR_ORDER")
        loaded["evaluation"]["pairs"] = pairs[1:]
        loaded["decision_gate"]["required_positive_pairs"] -= 1
        return loaded

    def zero_safe_dual(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        try:
            return original_dual(*args, **kwargs)
        except ValueError as exc:
            if str(exc) != "NO_REFERENCE_CYCLES":
                raise
            return None, {
                "selection_authority": "ZERO_REFERENCE_CYCLES_DETERMINISTIC_NON_COMMIT",
                "all_cycle_pixels": 0,
                "all_cycle_fraction": 0.0,
                "selected_component_fraction_of_cycles": 0.0,
                "prompt_box_xyxy": None,
            }

    def none_safe_iou(first: Any, second: Any) -> tuple[float, float, float]:
        if first is None:
            return 0.0, 0.0, 0.0
        return original_iou(first, second)

    base.load_images = load_images_unbounded
    base.load_protocol = load_resumed_protocol
    dual.dual_surface_cycle_affine_prompt = zero_safe_dual
    base.bbox_iou = none_safe_iou
    try:
        carrier.replay(protocol_path, output_path)
    finally:
        base.load_images = original_loader
        base.load_protocol = original_protocol_loader
        dual.dual_surface_cycle_affine_prompt = original_dual
        base.bbox_iou = original_iou
    result = pixel.load_json(output_path)
    consumed_decision = {
        **next(row for row in protocol["evaluation"]["pairs"] if str(row["id"]) == consumed_id),
        "reference_cycle_fraction": 0.0,
        "dominant_component_cycle_fraction": 0.0,
        "coherent_component_support": False,
        "positive_extent_gate": False,
        "commit": False,
        "correct": False,
    }
    result["decisions"] = {consumed_id: consumed_decision, **result["decisions"]}
    result["prompt_receipts"] = {
        consumed_id: {
            "selection_authority": "CONSUMED_ZERO_REFERENCE_CYCLES_DETERMINISTIC_NON_COMMIT",
            "all_cycle_pixels": 0,
            "all_cycle_fraction": 0.0,
            "selected_component_fraction_of_cycles": 0.0,
            "prompt_box_xyxy": None,
            "target_bbox_iou_evaluation_only": 0.0,
            "execution_receipt": resume["failure_receipt_path"],
        },
        **result["prompt_receipts"],
    }
    result["query_support_receipts"] = {
        consumed_id: {"status": "NOT_RUN_ZERO_REFERENCE_CYCLES", "masker_calls": 0},
        **result["query_support_receipts"],
    }
    positives = [row for row in result["decisions"].values() if row["label"] == "target_present"]
    negatives = [row for row in result["decisions"].values() if row["label"] == "target_absent"]
    positive_commits = sum(bool(row["commit"]) for row in positives)
    false_commits = sum(bool(row["commit"]) for row in negatives)
    gate = protocol["decision_gate"]
    gate_met = (
        len(positives) == int(gate["required_positive_pairs"])
        and len(negatives) == int(gate["required_target_absent_pairs"])
        and positive_commits >= int(gate["minimum_positive_commits"])
        and false_commits <= int(gate["maximum_target_absent_false_commits"])
    )
    result["gate_met"] = gate_met
    result["conclusion"] = (
        "L10_3RSCAN_CYCLE_COMPONENT_OPEN_SET_POSTHOC_DEVELOPMENT_GATE_MET"
        if gate_met else "L10_3RSCAN_CYCLE_COMPONENT_OPEN_SET_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
    )
    result["metrics"] = {
        "positive_pairs": len(positives),
        "positive_commits": positive_commits,
        "target_absent_pairs": len(negatives),
        "target_absent_false_commits": false_commits,
    }
    result["runtime"]["resumed_roma_calls"] = result["runtime"]["roma_calls"]
    result["runtime"]["prior_aborted_roma_calls"] = 1
    result["runtime"]["roma_calls"] += 1
    result["resume_revision"] = resume
    result["execution_revision"] = revision
    result["runtime_loader"] = {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))}
    pixel.atomic_write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import named_poi_facade_fingerprint as facade  # noqa: E402
import named_poi_multifacet_entrance as entrance  # noqa: E402
from named_poi_target_support_field import (  # noqa: E402
    SupportBindingState,
    SupportFieldConfig,
    SupportProposal,
    bind_support_field_to_entrances,
    build_target_support_field,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _best_iou(box: list[float] | tuple[float, ...] | None, truth: list[list[float]]) -> float:
    if box is None or not truth:
        return 0.0
    return max(_iou(box, target) for target in truth)


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    correct = sum(bool(row[f"{prefix}_correct"]) for row in rows)
    committed = sum(row[f"{prefix}_proposal_id"] is not None for row in rows)
    false = committed - correct
    positives = sum(bool(row["truth_boxes_xyxy"]) for row in rows)
    return {
        "correct_unique_commits": correct,
        "false_commits": false,
        "commits": committed,
        "commit_precision": correct / committed if committed else 0.0,
        "positive_image_recall": correct / positives if positives else 0.0,
    }


def _truth(protocol: dict[str, Any], target_id: str, source_index: int) -> dict[str, Any] | None:
    return protocol["exact_entrance_truth"].get(target_id, {}).get(str(source_index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("named_poi_target_support_protocol_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-target-support-v1",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_rows, query_rows, _, source_info = entrance._load_rows(protocol)
    target_ids = list(protocol["targets"])
    references_by_target = {
        target_id: [row for row in reference_rows if row.target_id == target_id]
        for target_id in target_ids
    }

    clip_path = ROOT / protocol["models"]["clip"]["path"]
    dino_path = ROOT / protocol["models"]["dinov2"]["path"]
    grounder_path = ROOT / protocol["models"]["grounding_dino"]["path"]
    if _sha256(clip_path / "pytorch_model.bin") != protocol["models"]["clip"]["weights_sha256"]:
        raise ValueError("CLIP_HASH_MISMATCH")
    if _sha256(dino_path / "model.safetensors") != protocol["models"]["dinov2"]["weights_sha256"]:
        raise ValueError("DINO_HASH_MISMATCH")
    if _sha256(grounder_path / "model.safetensors") != protocol["models"]["grounding_dino"]["weights_sha256"]:
        raise ValueError("GROUNDER_HASH_MISMATCH")

    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = facade.AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_image = Image.open(reference_rows[0].path).convert("RGB")
    representative = (
        clip_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
        dino_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
    )
    encoder_backend, models = facade._select_backend(
        clip_path,
        dino_path,
        representative,
        output_root / "encoder_backend_receipt.json",
    )
    encoder_device = str(encoder_backend["selected_device_type"])
    patch_grid = int(protocol["models"]["dinov2"]["patch_grid"])
    encoded = facade._encode_images(
        reference_rows + query_rows,
        models,
        clip_processor,
        dino_processor,
        encoder_device,
        patch_grid,
        args.batch_size,
    )
    reference_patches = {
        target_id: [encoded[row.key]["patches"] for row in references_by_target[target_id]]
        for target_id in target_ids
    }

    grounder_processor = AutoProcessor.from_pretrained(grounder_path, local_files_only=True)
    grounder_backend, grounder = entrance._select_grounder(
        grounder_path,
        grounder_processor,
        representative_image,
        protocol["models"]["grounding_dino"]["prompt"],
        output_root / "grounder_backend_receipt.json",
    )
    grounder_device = str(grounder_backend["selected_device_type"])
    config = SupportFieldConfig()
    iou_threshold = float(protocol["evaluation"]["iou_threshold"])
    evaluated = []
    for row in query_rows:
        image = Image.open(row.path).convert("RGB")
        source_index = int(row.key.rsplit(":", 1)[1])
        truth = _truth(protocol, row.target_id, source_index)
        if truth is not None and list(image.size) != truth["image_size"]:
            raise ValueError(f"TRUTH_IMAGE_SIZE_MISMATCH:{row.key}")
        truth_boxes = [] if truth is None else truth["boxes_xyxy"]
        _, proposal_rows = entrance._proposal_evidence(
            grounder,
            grounder_processor,
            image,
            protocol["models"]["grounding_dino"]["prompt"],
            grounder_device,
            float(protocol["models"]["grounding_dino"]["box_threshold"]),
            float(protocol["models"]["grounding_dino"]["text_threshold"]),
        )
        proposal_rows = proposal_rows[: int(protocol["models"]["grounding_dino"]["maximum_proposals"])]
        proposals = []
        by_id = {}
        for index, proposal in enumerate(proposal_rows):
            proposal_id = f"{row.key}:proposal:{index + 1:02d}"
            candidate = SupportProposal(
                proposal_id,
                float(proposal["bound_score"]),
                tuple(float(value) for value in proposal["box_xyxy"]),
            )
            proposals.append(candidate)
            by_id[proposal_id] = candidate
            proposal["proposal_id"] = proposal_id
        field, field_diagnostics = build_target_support_field(
            row.target_id,
            encoded[row.key]["patches"],
            reference_patches,
            patch_grid,
            config,
        )
        decision = bind_support_field_to_entrances(
            field,
            proposals,
            image.width,
            image.height,
            config,
        )
        baseline = proposals[0] if proposals and proposals[0].entrance_score >= config.minimum_entrance_score else None
        successor = by_id.get(decision.selected_proposal_id) if decision.state == SupportBindingState.COMMIT else None
        baseline_box = None if baseline is None else baseline.box_xyxy
        successor_box = None if successor is None else successor.box_xyxy
        evaluated.append(
            {
                "query": row.key,
                "target": row.target_id,
                "source_index": source_index,
                "image_size": list(image.size),
                "truth_boxes_xyxy": truth_boxes,
                "truth_kind": None if truth is None else truth["kind"],
                "proposal_oracle_iou": max(
                    [_best_iou(proposal.box_xyxy, truth_boxes) for proposal in proposals],
                    default=0.0,
                ),
                "baseline_proposal_id": None if baseline is None else baseline.proposal_id,
                "baseline_box_xyxy": baseline_box,
                "baseline_best_iou": _best_iou(baseline_box, truth_boxes),
                "baseline_correct": bool(truth_boxes) and _best_iou(baseline_box, truth_boxes) >= iou_threshold,
                "successor_state": decision.state.value,
                "successor_reason": decision.reason,
                "successor_proposal_id": None if successor is None else successor.proposal_id,
                "successor_box_xyxy": successor_box,
                "successor_best_iou": _best_iou(successor_box, truth_boxes),
                "successor_correct": bool(truth_boxes) and _best_iou(successor_box, truth_boxes) >= iou_threshold,
                "candidate_set": list(decision.candidate_set),
                "candidate_set_truth_coverage": any(
                    _best_iou(by_id[proposal_id].box_xyxy, truth_boxes) >= iou_threshold
                    for proposal_id in decision.candidate_set
                ),
                "field": field.tolist(),
                "field_diagnostics": field_diagnostics,
                "support_edges": [asdict(edge) for edge in decision.edges],
                "proposals": proposal_rows,
            }
        )

    baseline_metrics = _metrics(evaluated, "baseline")
    successor_metrics = _metrics(evaluated, "successor")
    positive_count = sum(bool(row["truth_boxes_xyxy"]) for row in evaluated)
    proposal_oracle = sum(
        row["proposal_oracle_iou"] >= iou_threshold for row in evaluated if row["truth_boxes_xyxy"]
    )
    state_coverage = sum(
        row["successor_correct"] or row["candidate_set_truth_coverage"]
        for row in evaluated
        if row["truth_boxes_xyxy"]
    )
    retained = (
        successor_metrics["correct_unique_commits"] >= 1
        and successor_metrics["correct_unique_commits"] > baseline_metrics["correct_unique_commits"]
        and successor_metrics["false_commits"] < baseline_metrics["false_commits"]
    )
    result = {
        "schema": "l10-named-poi-target-support-result-v1",
        "protocol_sha256": _sha256(protocol_path),
        "source": source_info["source"],
        "execution_backends": {"encoder": encoder_backend, "grounder": grounder_backend},
        "ocr_calls": 0,
        "roles": {
            "targets_in_roster": len(target_ids),
            "targets_with_queries": len({row.target_id for row in query_rows}),
            "reference_images": len(reference_rows),
            "evaluation_queries": len(query_rows),
            "positive_queries": positive_count,
            "negative_queries": len(query_rows) - positive_count,
        },
        "evaluation": {"iou_threshold": iou_threshold},
        "baseline_metrics": baseline_metrics,
        "successor_metrics": successor_metrics,
        "secondary_metrics": {
            "proposal_oracle_positive_coverage": proposal_oracle,
            "proposal_oracle_positive_recall": proposal_oracle / positive_count if positive_count else 0.0,
            "commit_or_set_valued_truth_coverage": state_coverage,
            "commit_or_set_valued_truth_recall": state_coverage / positive_count if positive_count else 0.0,
            "state_counts": {
                state.value: sum(row["successor_state"] == state.value for row in evaluated)
                for state in SupportBindingState
            },
        },
        "stop_condition": {
            "retain_target_support_locator": retained,
            "decision": "RETAIN_TARGET_SUPPORT_LOCATOR" if retained else "DO_NOT_TUNE_CHANGE_SPATIAL_IDENTITY_INFORMATION_SOURCE",
        },
        "claim_scope": "Fourth-batch public-image current-view entrance localization only; no OCR, public-access, accessibility, traversability, tracking, metric approach, navigation, arrival, user-benefit, or safety evidence.",
        "evaluation_rows": evaluated,
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(output_path),
                "baseline_metrics": baseline_metrics,
                "successor_metrics": successor_metrics,
                "secondary_metrics": result["secondary_metrics"],
                "stop_condition": result["stop_condition"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

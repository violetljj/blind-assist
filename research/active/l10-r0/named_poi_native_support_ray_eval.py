from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import named_poi_target_support_eval as common  # noqa: E402
from named_poi_native_support_ray import (  # noqa: E402
    RayBindingState,
    SupportRayConfig,
    bind_support_rays_to_entrances,
)
from named_poi_target_support_field import (  # noqa: E402
    SupportFieldConfig,
    SupportProposal,
    build_target_support_field,
)


def _inventory(paths: list[Path], root: Path) -> dict[str, int | str]:
    rows = []
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        rows.append(
            f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\t{common._sha256(path)}\n"
        )
    return {
        "image_count": len(paths),
        "image_bytes": sum(path.stat().st_size for path in paths),
        "inventory_sha256": hashlib.sha256("".join(rows).encode("utf-8")).hexdigest(),
    }


def _load_multifacet_roles(
    protocol: dict[str, Any],
) -> tuple[list[facade.ImageRow], list[facade.ImageRow], dict[str, Any]]:
    reference_spec = protocol["sources"]["reference_multifacet_library"]
    fresh_spec = protocol["sources"]["fresh_multifacet_library"]
    reference_path = ROOT / reference_spec["path"]
    fresh_path = ROOT / fresh_spec["path"]
    if common._sha256(reference_path) != reference_spec["sha256"]:
        raise ValueError("REFERENCE_MULTIFACET_LIBRARY_HASH_MISMATCH")
    if common._sha256(fresh_path) != fresh_spec["sha256"]:
        raise ValueError("FRESH_MULTIFACET_LIBRARY_HASH_MISMATCH")
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    reference_root, fresh_root = reference_path.parent, fresh_path.parent
    for payload, root, expected, label in (
        (references, reference_root, reference_spec, "REFERENCE"),
        (fresh, fresh_root, fresh_spec, "FRESH"),
    ):
        paths = [
            root / "images" / target["id"] / Path(row["local_path"]).name
            for target in payload["targets"]
            for row in target["facets"]
        ]
        observed = _inventory(paths, root)
        for key, value in observed.items():
            if value != expected[key]:
                raise ValueError(f"{label}_{key.upper()}_MISMATCH:expected={expected[key]}:observed={value}")
    references_by_id = {str(row["id"]): row for row in references["targets"]}
    fresh_by_id = {str(row["id"]): row for row in fresh["targets"]}
    reference_rows = []
    query_rows = []
    for target_id, roles in protocol["targets"].items():
        reference_target = references_by_id[target_id]
        for index in roles["reference_indices"]:
            source = reference_target["facets"][int(index) - 1]
            path = reference_root / "images" / target_id / Path(source["local_path"]).name
            if common._sha256(path) != source["sha256"]:
                raise ValueError(f"REFERENCE_IMAGE_HASH_MISMATCH:{target_id}:{index}")
            reference_rows.append(
                facade.ImageRow(
                    f"ref:{target_id}:{int(index):02d}",
                    target_id,
                    "reference",
                    path,
                    source["sha256"],
                    source["commons_file"],
                )
            )
        if roles["evaluation_indices"] and target_id not in fresh_by_id:
            raise ValueError(f"FRESH_TARGET_MISSING:{target_id}")
        fresh_target = fresh_by_id.get(target_id, {"facets": []})
        for index in roles["evaluation_indices"]:
            source = fresh_target["facets"][int(index) - 1]
            path = fresh_root / "images" / target_id / Path(source["local_path"]).name
            if common._sha256(path) != source["sha256"]:
                raise ValueError(f"FRESH_IMAGE_HASH_MISMATCH:{target_id}:{index}")
            query_rows.append(
                facade.ImageRow(
                    f"fresh:{target_id}:{int(index):02d}",
                    target_id,
                    "evaluation",
                    path,
                    source["sha256"],
                    source["commons_file"],
                )
            )
    keys = [row.key for row in reference_rows + query_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("ROLE_KEY_OVERLAP")
    return reference_rows, query_rows, {"source": _inventory(
        [row.path for row in query_rows], fresh_root
    )}


def _encode_native_images(
    rows: list[facade.ImageRow],
    models: dict[str, Any],
    clip_processor: Any,
    dino_processor: Any,
    device: str,
    expected_grid: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    encoded = {}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [Image.open(row.path).convert("RGB") for row in batch_rows]
        clip_pixels = clip_processor(images=images, return_tensors="pt")["pixel_values"]
        dino_pixels = dino_processor(images=images, return_tensors="pt")["pixel_values"]
        _, hidden = facade._forward(models, clip_pixels, dino_pixels, device)
        patch_count = int(hidden.shape[1] - 1)
        grid = int(round(math.sqrt(patch_count)))
        if grid * grid != patch_count or grid != expected_grid:
            raise ValueError(f"NATIVE_PATCH_GRID_MISMATCH:expected={expected_grid}:observed={grid}")
        patches = facade._normalized(hidden[:, 1:])
        for row, value in zip(batch_rows, patches, strict=True):
            encoded[row.key] = value.detach().cpu().numpy().astype(np.float32)
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("named_poi_native_support_ray_protocol_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-native-support-ray-v1",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if "reference_multifacet_library" in protocol["sources"]:
        reference_rows, query_rows, source_info = _load_multifacet_roles(protocol)
    else:
        reference_rows, query_rows, _, source_info = entrance._load_rows(protocol)
    target_ids = list(protocol["targets"])
    references_by_target = {
        target_id: [row for row in reference_rows if row.target_id == target_id]
        for target_id in target_ids
    }

    clip_path = ROOT / protocol["models"]["clip"]["path"]
    dino_path = ROOT / protocol["models"]["dinov2"]["path"]
    grounder_path = ROOT / protocol["models"]["grounding_dino"]["path"]
    if common._sha256(clip_path / "pytorch_model.bin") != protocol["models"]["clip"]["weights_sha256"]:
        raise ValueError("CLIP_HASH_MISMATCH")
    if common._sha256(dino_path / "model.safetensors") != protocol["models"]["dinov2"]["weights_sha256"]:
        raise ValueError("DINO_HASH_MISMATCH")
    if common._sha256(grounder_path / "model.safetensors") != protocol["models"]["grounding_dino"]["weights_sha256"]:
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
    native_grid = int(protocol["models"]["dinov2"]["native_patch_grid"])
    encoded = _encode_native_images(
        reference_rows + query_rows,
        models,
        clip_processor,
        dino_processor,
        encoder_device,
        native_grid,
        args.batch_size,
    )
    reference_patches = {
        target_id: [encoded[row.key] for row in references_by_target[target_id]]
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
    field_config = SupportFieldConfig()
    ray_config = SupportRayConfig()
    iou_threshold = float(protocol["evaluation"]["iou_threshold"])
    evaluated = []
    for row in query_rows:
        image = Image.open(row.path).convert("RGB")
        source_index = int(row.key.rsplit(":", 1)[1])
        truth = common._truth(protocol, row.target_id, source_index)
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
            encoded[row.key],
            reference_patches,
            native_grid,
            field_config,
        )
        decision = bind_support_rays_to_entrances(
            field,
            proposals,
            image.width,
            image.height,
            ray_config,
        )
        baseline = proposals[0] if proposals and proposals[0].entrance_score >= ray_config.minimum_entrance_score else None
        successor = by_id.get(decision.selected_proposal_id) if decision.state == RayBindingState.COMMIT else None
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
                    [common._best_iou(proposal.box_xyxy, truth_boxes) for proposal in proposals],
                    default=0.0,
                ),
                "baseline_proposal_id": None if baseline is None else baseline.proposal_id,
                "baseline_box_xyxy": baseline_box,
                "baseline_best_iou": common._best_iou(baseline_box, truth_boxes),
                "baseline_correct": bool(truth_boxes) and common._best_iou(baseline_box, truth_boxes) >= iou_threshold,
                "successor_state": decision.state.value,
                "successor_reason": decision.reason,
                "successor_proposal_id": None if successor is None else successor.proposal_id,
                "successor_box_xyxy": successor_box,
                "successor_best_iou": common._best_iou(successor_box, truth_boxes),
                "successor_correct": bool(truth_boxes) and common._best_iou(successor_box, truth_boxes) >= iou_threshold,
                "candidate_set": list(decision.candidate_set),
                "candidate_set_truth_coverage": any(
                    common._best_iou(by_id[proposal_id].box_xyxy, truth_boxes) >= iou_threshold
                    for proposal_id in decision.candidate_set
                ),
                "field": field.tolist(),
                "field_diagnostics": field_diagnostics,
                "ray_edges": [asdict(edge) for edge in decision.edges],
                "proposals": proposal_rows,
            }
        )

    baseline_metrics = common._metrics(evaluated, "baseline")
    successor_metrics = common._metrics(evaluated, "successor")
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
        successor_metrics["correct_unique_commits"] > baseline_metrics["correct_unique_commits"]
        and successor_metrics["correct_unique_commits"] >= 2
        and successor_metrics["false_commits"] < baseline_metrics["false_commits"]
    )
    result = {
        "schema": "l10-named-poi-native-support-ray-result-v1",
        "protocol_sha256": common._sha256(protocol_path),
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
        "evaluation": {"iou_threshold": iou_threshold, "native_patch_grid": native_grid},
        "baseline_metrics": baseline_metrics,
        "successor_metrics": successor_metrics,
        "secondary_metrics": {
            "proposal_oracle_positive_coverage": proposal_oracle,
            "proposal_oracle_positive_recall": proposal_oracle / positive_count if positive_count else 0.0,
            "commit_or_set_valued_truth_coverage": state_coverage,
            "commit_or_set_valued_truth_recall": state_coverage / positive_count if positive_count else 0.0,
            "state_counts": {
                state.value: sum(row["successor_state"] == state.value for row in evaluated)
                for state in RayBindingState
            },
        },
        "stop_condition": {
            "retain_native_support_ray_locator": retained,
            "decision": "RETAIN_NATIVE_SUPPORT_RAY_LOCATOR" if retained else "DO_NOT_TUNE_CHANGE_TARGET_ENTRANCE_RELATION_SOURCE",
        },
        "claim_scope": "Fifth-batch public-image current-view entrance localization only; no OCR, public-access, accessibility, traversability, tracking, metric approach, navigation, arrival, user-benefit, or safety evidence.",
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

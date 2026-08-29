"""Run the frozen PB6 SCIL + semantic presence + portal-set confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import ultralytics
from rapidocr import RapidOCR
from ultralytics import YOLO

import named_poi_biscript_eval as scil
from named_poi_portal_lattice import propose_portal_sets


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(spec: dict[str, str]) -> Path:
    path = _path(spec["path"])
    actual = _sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(f"HASH_MISMATCH:{path}:{actual}:{spec['sha256']}")
    return path


def _member(box: Sequence[float], region: Sequence[float]) -> bool:
    center_x = 0.5 * (box[0] + box[2])
    center_y = 0.5 * (box[1] + box[3])
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    coverage = intersection / area if area else 0.0
    return (
        region[0] <= center_x <= region[2]
        and region[1] <= center_y <= region[3]
        and coverage >= 0.5
    )


def _components(mask: np.ndarray, door_class: int) -> list[dict[str, Any]]:
    binary = (mask == door_class).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    image_area = float(width * height)
    rows = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        rows.append(
            {
                "box_xyxy": [x, y, x + component_width, y + component_height],
                "component_pixels": area,
                "component_area_fraction": area / image_area,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["component_area_fraction"], row["box_xyxy"]),
    )


def _authority_token(frame: str, target: str, image_sha256: str) -> str:
    payload = json.dumps(
        {
            "authority": "PB6_SCIL_SPPF_SAME_FRAME",
            "frame": frame,
            "target": target,
            "image_sha256": image_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise ValueError(f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = _json(protocol_path)
    if _sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("EVALUATOR_HASH_MISMATCH")
    sources = {name: _verify(spec) for name, spec in protocol["sources"].items()}
    frozen = {
        name: _verify(spec) for name, spec in protocol["frozen_files"].items()
    }
    if ultralytics.__version__ != protocol["runtime"]["ultralytics_version"]:
        raise ValueError("ULTRALYTICS_VERSION_MISMATCH")
    if not torch.cuda.is_available():
        raise ValueError("FROZEN_CUDA_PROVIDER_UNAVAILABLE")

    source = _json(sources["source_spec"])
    manifest = _json(sources["source_manifest"])
    audit = _json(sources["source_audit"])
    if any(
        audit["checks"][key] != 0
        for key in (
            "ocr_calls_before_freeze",
            "semantic_calls_before_freeze",
            "geometry_calls_before_freeze",
        )
    ):
        raise ValueError("PRE_FREEZE_MODEL_ACCESS_DETECTED")
    if len(source["cohort"]) != 8 or manifest["image_count"] != 8:
        raise ValueError("FRESH_COHORT_COUNT_MISMATCH")
    by_index = {int(row["index"]): row for row in source["cohort"]}
    audit_by_index = {int(row["index"]): row for row in audit["frames"]}
    manifest_by_index = {int(row["index"]): row for row in manifest["frames"]}
    if set(by_index) != set(audit_by_index) or set(by_index) != set(manifest_by_index):
        raise ValueError("FRESH_COHORT_INDEX_MISMATCH")

    entities = list(source["cohort"])
    profiles = scil._profiles(entities)
    thresholds = protocol["scil"]["thresholds"]
    model_root = _path(protocol["runtime"]["ocr_model_root"])
    for filename, expected in protocol["runtime"]["ocr_models"].items():
        path = model_root / filename
        if _sha256(path) != expected:
            raise ValueError(f"OCR_MODEL_HASH_MISMATCH:{filename}")
    ocr = RapidOCR(
        params={
            "Global.model_root_dir": str(model_root),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    semantic_model = YOLO(str(frozen["semantic_model"]))
    if semantic_model.task != "semantic" or semantic_model.names != {
        0: "background",
        1: "door",
    }:
        raise ValueError("SEMANTIC_MODEL_CONTRACT_MISMATCH")
    semantic_threshold = float(protocol["semantic_presence"]["component_area_fraction"])
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for index in sorted(by_index):
        entity = by_index[index]
        frame = manifest_by_index[index]
        truth = audit_by_index[index]
        if entity["commons_file"] != frame["commons_file"]:
            raise ValueError(f"SOURCE_FILE_ORDER_MISMATCH:{index}")
        if entity["id"] != truth["id"] or frame["sha256"] != truth["image_sha256"]:
            raise ValueError(f"SOURCE_AUDIT_IDENTITY_MISMATCH:{index}")
        image_path = Path(frame["local_path"])
        if _sha256(image_path) != frame["sha256"]:
            raise ValueError(f"IMAGE_HASH_MISMATCH:{index}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"IMAGE_DECODE_FAILED:{index}")
        image_size = [int(image.shape[1]), int(image.shape[0])]
        if image_size != truth["local_image_size"]:
            raise ValueError(f"IMAGE_SIZE_MISMATCH:{index}:{image_size}")

        ocr_started = time.perf_counter()
        ocr_output = ocr(image)
        ocr_seconds = time.perf_counter() - ocr_started
        ocr_texts = [
            str(value) for value in (ocr_output.txts if ocr_output.txts is not None else [])
        ]
        observations = scil._windows(ocr_texts)
        scores, carrier_scores, conflict = scil._lattice_scores(
            entities, profiles, observations
        )
        ranked_entities = sorted(scores, key=lambda key: (-scores[key], key))
        top, second = ranked_entities[:2]
        top_score = float(scores[top])
        margin = float(scores[top] - scores[second])
        proof = (
            top
            if not conflict
            and top_score >= thresholds["score"]
            and margin >= thresholds["margin"]
            else None
        )
        correct_proof = proof == entity["id"]

        semantic_started = time.perf_counter()
        semantic_result = semantic_model.predict(
            source=str(image_path),
            device=0,
            imgsz=int(protocol["runtime"]["semantic_image_size"]),
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        semantic_seconds = time.perf_counter() - semantic_started
        if semantic_result.semantic_mask is None:
            raise ValueError(f"SEMANTIC_MASK_MISSING:{index}")
        mask = (
            semantic_result.semantic_mask.data.detach()
            .to("cpu")
            .numpy()
            .astype(np.uint8)
        )
        components = _components(mask, int(protocol["semantic_presence"]["door_class"]))[:20]
        semantic_present = any(
            component["component_area_fraction"] >= semantic_threshold
            for component in components
        )

        geometry_started = time.perf_counter()
        proposals, diagnostics = propose_portal_sets(image)
        geometry_seconds = time.perf_counter() - geometry_started
        truth_box = truth["portal_set_box_xyxy"]
        annotated = [
            {
                **asdict(proposal),
                "truth_member": _member(proposal.box_xyxy, truth_box)
                if truth_box is not None
                else None,
            }
            for proposal in proposals[:10]
        ]
        raw_truth_top3 = bool(truth_box) and any(
            proposal["truth_member"] for proposal in annotated[:3]
        )
        source_request_authorized = bool(correct_proof and semantic_present and annotated[:3])
        token = (
            _authority_token(str(index), entity["id"], frame["sha256"])
            if source_request_authorized
            else None
        )
        authorized = [
            {
                "proposal_id": proposal["proposal_id"],
                "requested_entity": entity["id"],
                "authority_token": token,
            }
            for proposal in annotated[:3]
        ] if source_request_authorized else []
        joined_truth_top3 = bool(source_request_authorized and raw_truth_top3)
        wrong_proof_authorization = bool(
            proof is not None
            and proof != entity["id"]
            and semantic_present
            and annotated[:3]
        )

        status = truth["status"]
        if proof is None:
            state = "IDENTITY_UNKNOWN"
            deficit_action = "APPROACH_TEXT_OR_SWEEP_SIGN"
        elif not correct_proof:
            state = "IDENTITY_WRONG_PROOF"
            deficit_action = "HOLD_IDENTITY_CONFLICT"
        elif not semantic_present:
            state = "IDENTITY_PROVED_DOOR_SEMANTIC_NOT_OBSERVABLE"
            deficit_action = "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif not annotated:
            state = "IDENTITY_PROVED_PORTAL_GEOMETRY_NOT_OBSERVABLE"
            deficit_action = "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif status == "IN_SCOPE_DOOR_LEVEL_POSITIVE":
            state = (
                "ENTITY_BOUND_SEMANTIC_PORTAL_SET"
                if joined_truth_top3
                else "SEMANTIC_PORTAL_AUTHORIZED_TRUTH_MISS"
            )
            deficit_action = None if joined_truth_top3 else "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif status == "NO_PORTAL_NEGATIVE":
            state = "FALSE_PORTAL_AUTHORIZATION"
            deficit_action = "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif status == "LARGE_OPEN_ENTRANCE_OOD":
            state = "OOD_PORTAL_AUTHORIZATION_LEAKAGE"
            deficit_action = "SIDESTEP_OR_APPROACH_DOOR_LEVEL_PORTAL"
        else:
            raise ValueError(f"UNKNOWN_SOURCE_STATUS:{status}")

        rows.append(
            {
                "index": index,
                "id": entity["id"],
                "role": status,
                "image_sha256": frame["sha256"],
                "image_size": image_size,
                "ocr": {
                    "texts": ocr_texts,
                    "seconds": ocr_seconds,
                    "top_entity": top,
                    "top_score": top_score,
                    "second_score": float(scores[second]),
                    "margin": margin,
                    "proof": proof,
                    "correct_proof": correct_proof,
                    "carrier_conflict": conflict,
                    "carrier_scores": carrier_scores[top],
                },
                "semantic": {
                    "seconds": semantic_seconds,
                    "mask_size": [int(mask.shape[1]), int(mask.shape[0])],
                    "door_pixel_fraction": float(np.mean(mask == 1)),
                    "present": semantic_present,
                    "threshold": semantic_threshold,
                    "components": components,
                },
                "geometry": {
                    "seconds": geometry_seconds,
                    "diagnostics": diagnostics,
                    "truth_box_xyxy": truth_box,
                    "raw_truth_retained_top3": raw_truth_top3,
                    "proposals": annotated,
                },
                "state": state,
                "deficit_action": deficit_action,
                "source_request_authorized": source_request_authorized,
                "joined_truth_retained_top3": joined_truth_top3,
                "authorized_proposals": authorized,
                "wrong_proof_authorization": wrong_proof_authorization,
                "source_label_wrong_request_count": len(entities) - 1,
                "joined_wrong_request_authorizations": 0 if not wrong_proof_authorization else len(entities) - 1,
            }
        )

    actual_semantic_device = str(next(semantic_model.model.parameters()).device)
    if not actual_semantic_device.startswith("cuda"):
        raise ValueError(f"SILENT_SEMANTIC_DEVICE_FALLBACK:{actual_semantic_device}")
    positives = [row for row in rows if row["role"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE"]
    negatives = [row for row in rows if row["role"] == "NO_PORTAL_NEGATIVE"]
    ood = [row for row in rows if row["role"] == "LARGE_OPEN_ENTRANCE_OOD"]
    controls = negatives + ood
    correct_proofs = sum(row["ocr"]["correct_proof"] for row in rows)
    wrong_proofs = sum(
        row["ocr"]["proof"] is not None and not row["ocr"]["correct_proof"]
        for row in rows
    )
    unknown = sum(row["ocr"]["proof"] is None for row in rows)
    positive_identity = sum(row["ocr"]["correct_proof"] for row in positives)
    control_identity = sum(row["ocr"]["correct_proof"] for row in controls)
    semantic_positive = sum(row["semantic"]["present"] for row in positives)
    semantic_negative = sum(row["semantic"]["present"] for row in negatives)
    semantic_ood = sum(row["semantic"]["present"] for row in ood)
    raw_truth = sum(row["geometry"]["raw_truth_retained_top3"] for row in positives)
    joined_truth = sum(row["joined_truth_retained_top3"] for row in positives)
    negative_false = sum(row["source_request_authorized"] for row in negatives)
    ood_leakage = sum(row["source_request_authorized"] for row in ood)
    wrong_request_authorizations = sum(
        row["joined_wrong_request_authorizations"] for row in rows
    )
    unbound_wrong_requests = sum(
        row["source_label_wrong_request_count"]
        for row in rows
        if row["semantic"]["present"] and row["geometry"]["proposals"]
    )
    safe_controls = len(controls) - negative_false - ood_leakage
    balanced_accuracy = 0.5 * (
        joined_truth / len(positives) + safe_controls / len(controls)
    )
    forbidden_states = {
        "COMMIT",
        "HANDOFF",
        "ARRIVED",
        "GUIDE_LEFT",
        "GUIDE_RIGHT",
        "GUIDE_FORWARD",
    }
    gate = protocol["promotion_gate"]
    gates = {
        "minimum_total_correct_scil_proofs": correct_proofs
        >= gate["minimum_total_correct_scil_proofs"],
        "zero_wrong_scil_proofs": wrong_proofs == 0,
        "minimum_positive_identity_proofs": positive_identity
        >= gate["minimum_positive_identity_proofs"],
        "minimum_identity_evaluable_controls": control_identity
        >= gate["minimum_identity_evaluable_controls"],
        "minimum_joined_top3_truth_retention": joined_truth
        >= gate["minimum_joined_top3_truth_retention"],
        "zero_no_portal_semantic_presence": semantic_negative == 0,
        "zero_no_portal_authorization": negative_false == 0,
        "zero_ood_authorization": ood_leakage == 0,
        "zero_wrong_request_authorization": wrong_request_authorizations == 0,
        "zero_identity_unknown_authorization": all(
            not row["authorized_proposals"]
            for row in rows
            if row["ocr"]["proof"] is None
        ),
        "no_commit_or_navigation_state": forbidden_states.isdisjoint(
            {row["state"] for row in rows}
        ),
    }
    gate_met = all(gates.values())
    decision = (
        "L10_PB6_SCRIPT_PROVED_SEMANTIC_GATED_PORTAL_LATTICE_GATE_MET"
        if gate_met
        else "L10_PB6_SCRIPT_PROVED_SEMANTIC_GATED_PORTAL_LATTICE_GATE_NOT_MET"
    )
    metrics = {
        "frames": len(rows),
        "in_scope_positive_frames": len(positives),
        "no_portal_negative_frames": len(negatives),
        "large_open_entrance_ood_frames": len(ood),
        "scil_correct_proofs": correct_proofs,
        "scil_wrong_proofs": wrong_proofs,
        "scil_unknown": unknown,
        "positive_identity_proofs": positive_identity,
        "identity_evaluable_controls": control_identity,
        "semantic_positive_presence": semantic_positive,
        "semantic_negative_presence": semantic_negative,
        "semantic_ood_presence": semantic_ood,
        "raw_geometry_truth_retained_top3": raw_truth,
        "joined_truth_retained_top3": joined_truth,
        "joined_truth_retained_top3_rate": joined_truth / len(positives),
        "no_portal_false_authorization": negative_false,
        "ood_authorization_leakage": ood_leakage,
        "wrong_request_authorizations": wrong_request_authorizations,
        "unbound_wrong_request_authorizations": unbound_wrong_requests,
        "wrong_request_relative_reduction": (
            (unbound_wrong_requests - wrong_request_authorizations)
            / unbound_wrong_requests
            if unbound_wrong_requests
            else 0.0
        ),
        "balanced_accuracy": balanced_accuracy,
    }
    result = {
        "schema": "l10-named-poi-semantic-portal-fresh-result-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "decision": decision,
        "metrics": metrics,
        "gates": gates,
        "backend": {
            "ocr": "CPUExecutionProvider",
            "semantic": "semantic-door-torch-cuda",
            "actual_semantic_device": actual_semantic_device,
            "actual_semantic_device_name": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__,
            "peak_semantic_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_semantic_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "rows": rows,
        "next_step": protocol["next_step"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "decision": decision,
                "metrics": metrics,
                "failed_gates": [name for name, passed in gates.items() if not passed],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

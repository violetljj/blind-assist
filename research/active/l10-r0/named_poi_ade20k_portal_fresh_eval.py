"""Confirm the frozen SCIL + broad-domain door-component join on fresh entities."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from rapidocr import RapidOCR
from ultralytics import YOLO

import named_poi_ade20k_portal_development_eval as door_field
import named_poi_biscript_eval as scil


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


def _token(index: int, target: str, image_sha256: str) -> str:
    payload = json.dumps(
        {
            "authority": "PB7_SCIL_BD_DCF_SAME_FRAME",
            "frame": index,
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
    frozen = {name: _verify(spec) for name, spec in protocol["frozen_files"].items()}
    if ultralytics.__version__ != protocol["runtime"]["ultralytics_version"]:
        raise ValueError("ULTRALYTICS_VERSION_MISMATCH")
    if not torch.cuda.is_available():
        raise ValueError("CUDA_UNAVAILABLE")

    source = _json(sources["source_spec"])
    manifest = _json(sources["source_manifest"])
    audit = _json(sources["source_audit"])
    if any(
        audit["checks"][key] != 0
        for key in (
            "ocr_calls_before_freeze",
            "semantic_calls_before_freeze",
        )
    ):
        raise ValueError("PRE_FREEZE_MODEL_ACCESS_DETECTED")
    if len(source["cohort"]) != 8 or manifest["image_count"] != 8:
        raise ValueError("COHORT_COUNT_MISMATCH")
    source_by_index = {int(row["index"]): row for row in source["cohort"]}
    manifest_by_index = {int(row["index"]): row for row in manifest["frames"]}
    audit_by_index = {int(row["index"]): row for row in audit["frames"]}
    if set(source_by_index) != set(audit_by_index) or set(source_by_index) != set(manifest_by_index):
        raise ValueError("COHORT_INDEX_MISMATCH")

    entities = list(source["cohort"])
    profiles = scil._profiles(entities)
    score_threshold = float(protocol["scil"]["score_threshold"])
    margin_threshold = float(protocol["scil"]["margin_threshold"])
    model_root = _path(protocol["runtime"]["ocr_model_root"])
    for filename, expected in protocol["runtime"]["ocr_models"].items():
        if _sha256(model_root / filename) != expected:
            raise ValueError(f"OCR_MODEL_HASH_MISMATCH:{filename}")
    ocr = RapidOCR(
        params={
            "Global.model_root_dir": str(model_root),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    model = YOLO(str(frozen["semantic_model"]))
    classes = {14: "door", 58: "screen door"}
    if model.task != "semantic" or any(model.names[key] != value for key, value in classes.items()):
        raise ValueError("ADE20K_DOOR_CLASS_CONTRACT_MISMATCH")
    component_threshold = float(protocol["door_field"]["component_area_fraction"])
    bottom_floor = float(protocol["door_field"]["minimum_box_bottom_fraction"])
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for index in sorted(source_by_index):
        entity = source_by_index[index]
        frame = manifest_by_index[index]
        truth = audit_by_index[index]
        if entity["commons_file"] != frame["commons_file"] or entity["id"] != truth["id"]:
            raise ValueError(f"SOURCE_ORDER_MISMATCH:{index}")
        image_path = Path(frame["local_path"])
        if _sha256(image_path) != frame["sha256"] or frame["sha256"] != truth["image_sha256"]:
            raise ValueError(f"IMAGE_HASH_MISMATCH:{index}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or [image.shape[1], image.shape[0]] != truth["local_image_size"]:
            raise ValueError(f"IMAGE_DECODE_OR_SIZE_MISMATCH:{index}")

        ocr_started = time.perf_counter()
        ocr_output = ocr(image)
        ocr_seconds = time.perf_counter() - ocr_started
        texts = [str(value) for value in (ocr_output.txts if ocr_output.txts is not None else [])]
        observations = scil._windows(texts)
        scores, carrier_scores, conflict = scil._lattice_scores(entities, profiles, observations)
        ranked = sorted(scores, key=lambda key: (-scores[key], key))
        top, second = ranked[:2]
        top_score = float(scores[top])
        margin = float(scores[top] - scores[second])
        proof = (
            top
            if not conflict and top_score >= score_threshold and margin >= margin_threshold
            else None
        )
        correct_proof = proof == entity["id"]

        semantic_started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            device=0,
            imgsz=int(protocol["runtime"]["image_size"]),
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        semantic_seconds = time.perf_counter() - semantic_started
        if result.semantic_mask is None:
            raise ValueError(f"SEMANTIC_MASK_MISSING:{index}")
        mask = result.semantic_mask.data.detach().to("cpu").numpy().astype(np.uint8)
        truth_box = truth["portal_set_box_xyxy"]
        components = [
            {
                **component,
                "truth_member": door_field._member(component["box_xyxy"], truth_box)
                if truth_box is not None
                else None,
            }
            for component in door_field._components(mask, set(classes))[:30]
        ]
        eligible = [
            component
            for component in components
            if component["component_area_fraction"] >= component_threshold
            and component["box_bottom_fraction"] >= bottom_floor
        ][:3]
        semantic_present = bool(eligible)
        raw_truth_top3 = bool(truth_box) and any(component["truth_member"] for component in eligible)
        source_authorized = bool(correct_proof and eligible)
        authority_token = _token(index, entity["id"], frame["sha256"]) if source_authorized else None
        authorized = [
            {
                "rank": rank,
                "box_xyxy": component["box_xyxy"],
                "requested_entity": entity["id"],
                "authority_token": authority_token,
            }
            for rank, component in enumerate(eligible, start=1)
        ] if source_authorized else []
        joined_truth = bool(source_authorized and raw_truth_top3)
        wrong_proof_authorization = bool(
            proof is not None and not correct_proof and eligible
        )
        status = truth["status"]
        if proof is None:
            state = "IDENTITY_UNKNOWN"
            action = "APPROACH_TEXT_OR_SWEEP_SIGN"
        elif not correct_proof:
            state = "IDENTITY_WRONG_PROOF"
            action = "HOLD_IDENTITY_CONFLICT"
        elif not eligible:
            state = "IDENTITY_PROVED_DOOR_COMPONENT_NOT_OBSERVABLE"
            action = "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif status == "IN_SCOPE_DOOR_LEVEL_POSITIVE":
            state = "ENTITY_BOUND_DOOR_COMPONENT_SET" if joined_truth else "DOOR_COMPONENT_TRUTH_MISS"
            action = None if joined_truth else "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif status == "NO_PORTAL_NEGATIVE":
            state = "FALSE_PORTAL_AUTHORIZATION"
            action = "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif status == "LARGE_OPEN_ENTRANCE_OOD":
            state = "OOD_PORTAL_AUTHORIZATION_LEAKAGE"
            action = "SIDESTEP_OR_APPROACH_DOOR_LEVEL_PORTAL"
        else:
            raise ValueError(f"UNKNOWN_STATUS:{status}")

        rows.append(
            {
                "index": index,
                "id": entity["id"],
                "role": status,
                "image_sha256": frame["sha256"],
                "ocr": {
                    "texts": texts,
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
                "door_field": {
                    "seconds": semantic_seconds,
                    "mask_size": [int(mask.shape[1]), int(mask.shape[0])],
                    "door_or_screen_door_pixel_fraction": float(np.mean(np.isin(mask, list(classes)))),
                    "component_area_threshold": component_threshold,
                    "eligible": eligible,
                    "all_components": components,
                    "truth_box_xyxy": truth_box,
                    "raw_truth_retained_top3": raw_truth_top3,
                },
                "state": state,
                "deficit_action": action,
                "source_request_authorized": source_authorized,
                "joined_truth_retained_top3": joined_truth,
                "authorized_proposals": authorized,
                "wrong_proof_authorization": wrong_proof_authorization,
                "wrong_request_proxy_count": len(entities) - 1,
                "joined_wrong_request_authorizations": 0 if not wrong_proof_authorization else len(entities) - 1,
            }
        )

    actual_device = str(next(model.model.parameters()).device)
    if not actual_device.startswith("cuda"):
        raise ValueError(f"SILENT_DEVICE_FALLBACK:{actual_device}")
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
    positive_presence = sum(bool(row["door_field"]["eligible"]) for row in positives)
    joined_truth = sum(row["joined_truth_retained_top3"] for row in positives)
    negative_presence = sum(bool(row["door_field"]["eligible"]) for row in negatives)
    ood_presence = sum(bool(row["door_field"]["eligible"]) for row in ood)
    negative_false = sum(row["source_request_authorized"] for row in negatives)
    ood_leakage = sum(row["source_request_authorized"] for row in ood)
    wrong_request = sum(row["joined_wrong_request_authorizations"] for row in rows)
    unbound_wrong = sum(
        row["wrong_request_proxy_count"] for row in rows if row["door_field"]["eligible"]
    )
    safe_controls = len(controls) - negative_false - ood_leakage
    balanced_accuracy = 0.5 * (
        joined_truth / len(positives) + safe_controls / len(controls)
    )
    gate = protocol["promotion_gate"]
    forbidden = {"COMMIT", "HANDOFF", "ARRIVED", "GUIDE_LEFT", "GUIDE_RIGHT", "GUIDE_FORWARD"}
    gates = {
        "minimum_total_correct_scil_proofs": correct_proofs >= gate["minimum_total_correct_scil_proofs"],
        "zero_wrong_scil_proofs": wrong_proofs == 0,
        "minimum_positive_identity_proofs": positive_identity >= gate["minimum_positive_identity_proofs"],
        "minimum_identity_evaluable_controls": control_identity >= gate["minimum_identity_evaluable_controls"],
        "minimum_joined_top3_truth_retention": joined_truth >= gate["minimum_joined_top3_truth_retention"],
        "zero_no_portal_presence": negative_presence == 0,
        "zero_no_portal_authorization": negative_false == 0,
        "zero_ood_authorization": ood_leakage == 0,
        "zero_wrong_request_authorization": wrong_request == 0,
        "zero_identity_unknown_authorization": all(
            not row["authorized_proposals"] for row in rows if row["ocr"]["proof"] is None
        ),
        "no_commit_or_navigation_state": forbidden.isdisjoint({row["state"] for row in rows}),
    }
    gate_met = all(gates.values())
    decision = (
        "L10_PB7_SCRIPT_PROVED_BROAD_DOMAIN_DOOR_COMPONENT_GATE_MET"
        if gate_met
        else "L10_PB7_SCRIPT_PROVED_BROAD_DOMAIN_DOOR_COMPONENT_GATE_NOT_MET"
    )
    metrics = {
        "frames": len(rows),
        "positive_frames": len(positives),
        "negative_frames": len(negatives),
        "ood_frames": len(ood),
        "scil_correct_proofs": correct_proofs,
        "scil_wrong_proofs": wrong_proofs,
        "scil_unknown": unknown,
        "positive_identity_proofs": positive_identity,
        "identity_evaluable_controls": control_identity,
        "positive_door_component_presence": positive_presence,
        "joined_truth_retained_top3": joined_truth,
        "joined_truth_retained_top3_rate": joined_truth / len(positives),
        "no_portal_presence": negative_presence,
        "ood_presence": ood_presence,
        "no_portal_false_authorization": negative_false,
        "ood_authorization_leakage": ood_leakage,
        "wrong_request_authorizations": wrong_request,
        "unbound_wrong_request_authorizations": unbound_wrong,
        "wrong_request_relative_reduction": (
            (unbound_wrong - wrong_request) / unbound_wrong if unbound_wrong else 0.0
        ),
        "balanced_accuracy": balanced_accuracy,
    }
    result_payload = {
        "schema": "l10-named-poi-ade20k-portal-fresh-result-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "decision": decision,
        "metrics": metrics,
        "gates": gates,
        "backend": {
            "ocr": "CPUExecutionProvider",
            "semantic": "ade20k-semantic-torch-cuda",
            "actual_semantic_device": actual_device,
            "actual_semantic_device_name": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "rows": rows,
        "next_step": protocol["next_step"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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

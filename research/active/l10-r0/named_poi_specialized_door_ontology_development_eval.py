"""Evaluate one fixed specialized doorway/door ontology on consumed PB7."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO


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


def _postprocessor(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pb9_author_mirror_suppress", path)
    if spec is None or spec.loader is None:
        raise ValueError("POSTPROCESSOR_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _member(candidate: list[float], truth: list[int] | None) -> bool:
    if truth is None:
        return False
    x1, y1, x2, y2 = candidate
    tx1, ty1, tx2, ty2 = truth
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    if not (tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2):
        return False
    intersection = max(0.0, min(x2, tx2) - max(x1, tx1)) * max(
        0.0, min(y2, ty2) - max(y1, ty1)
    )
    area = max(1.0, (x2 - x1) * (y2 - y1))
    return intersection / area >= 0.5


def _sorted_portals(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    portals = [row for row in detections if int(row["cls"]) in {0, 1}]
    return sorted(
        portals,
        key=lambda row: (
            -float(row["conf"]),
            int(row["cls"]),
            tuple(round(float(value), 6) for value in row["box"]),
        ),
    )


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
    model_path = _verify(protocol["model"]["weights"])
    postprocessor_path = _verify(protocol["model"]["postprocessor"])
    _verify(protocol["model"]["model_card"])
    if ultralytics.__version__ != protocol["runtime"]["ultralytics_version"]:
        raise ValueError("ULTRALYTICS_VERSION_MISMATCH")
    if not torch.cuda.is_available():
        raise ValueError("CUDA_UNAVAILABLE")

    source = _json(sources["source_spec"])
    manifest = _json(sources["source_manifest"])
    audit = _json(sources["source_audit"])
    pb7 = _json(frozen["pb7_formal_result"])
    if pb7["metrics"]["scil_correct_proofs"] != 8 or pb7["metrics"]["scil_wrong_proofs"] != 0:
        raise ValueError("PB7_IDENTITY_AUTHORITY_MISMATCH")
    if len(source["cohort"]) != 8 or manifest["image_count"] != 8 or len(audit["frames"]) != 8:
        raise ValueError("COHORT_COUNT_MISMATCH")
    source_by_index = {int(row["index"]): row for row in source["cohort"]}
    manifest_by_index = {int(row["index"]): row for row in manifest["frames"]}
    audit_by_index = {int(row["index"]): row for row in audit["frames"]}
    pb7_by_index = {int(row["index"]): row for row in pb7["rows"]}
    expected_indices = set(range(1, 9))
    if not all(
        set(values) == expected_indices
        for values in (source_by_index, manifest_by_index, audit_by_index, pb7_by_index)
    ):
        raise ValueError("COHORT_INDEX_MISMATCH")

    author = _postprocessor(postprocessor_path)
    model = YOLO(str(model_path))
    expected_names = {0: "doorway", 1: "door", 2: "people", 3: "window", 4: "mirror"}
    if model.task != "segment" or model.names != expected_names:
        raise ValueError(f"MODEL_ONTOLOGY_MISMATCH:{model.task}:{model.names}")
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index in sorted(source_by_index):
        entity = source_by_index[index]
        frame = manifest_by_index[index]
        truth = audit_by_index[index]
        identity = pb7_by_index[index]
        if (
            entity["id"] != truth["id"]
            or entity["id"] != identity["id"]
            or entity["commons_file"] != frame["commons_file"]
            or entity["role"] != truth["status"]
        ):
            raise ValueError(f"SOURCE_ORDER_MISMATCH:{index}")
        image_path = Path(frame["local_path"])
        if _sha256(image_path) != frame["sha256"] or frame["sha256"] != truth["image_sha256"]:
            raise ValueError(f"IMAGE_HASH_MISMATCH:{index}")
        if not identity["ocr"]["correct_proof"]:
            raise ValueError(f"IDENTITY_PROOF_NOT_CORRECT:{index}")

        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            device=0,
            conf=float(protocol["author_contract"]["predict_confidence"]),
            imgsz=int(protocol["author_contract"]["image_size"]),
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        raw = author.dets_from_result(result)
        final = author.finalize(
            raw,
            out_conf=float(protocol["author_contract"]["output_confidence"]),
        )
        portals = _sorted_portals(final)
        top3 = [
            {
                "rank": rank,
                "class_id": int(detection["cls"]),
                "class_name": expected_names[int(detection["cls"])],
                "confidence": float(detection["conf"]),
                "box_xyxy": [float(value) for value in detection["box"]],
                "truth_member": _member(detection["box"], truth["portal_set_box_xyxy"]),
            }
            for rank, detection in enumerate(portals[:3], start=1)
        ]
        presence = bool(portals)
        joined_truth = bool(
            truth["status"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE"
            and any(candidate["truth_member"] for candidate in top3)
        )
        if not presence:
            state = "IDENTITY_PROVED_SPECIALIZED_PORTAL_NONE"
            action = "APPROACH_OR_SWEEP_FOR_DOOR_LEVEL_BOUNDARY"
        elif truth["status"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE":
            state = "ENTITY_BOUND_SPECIALIZED_PORTAL_SET" if joined_truth else "SPECIALIZED_PORTAL_TRUTH_MISS"
            action = None if joined_truth else "APPROACH_OR_SWEEP_FOR_DOOR_LEVEL_BOUNDARY"
        elif truth["status"] == "NO_PORTAL_NEGATIVE":
            state = "FALSE_SPECIALIZED_PORTAL_AUTHORIZATION"
            action = "HOLD_FALSE_PORTAL"
        else:
            state = "OOD_SPECIALIZED_PORTAL_AUTHORIZATION_LEAKAGE"
            action = "SIDESTEP_OR_APPROACH_FOR_DOOR_LEVEL_PORTAL"
        rows.append(
            {
                "index": index,
                "id": entity["id"],
                "role": truth["status"],
                "image_sha256": frame["sha256"],
                "identity_proof": identity["ocr"]["proof"],
                "seconds": seconds,
                "raw_detections": raw,
                "author_final_detections": final,
                "portal_top3": top3,
                "portal_presence": presence,
                "source_request_authorized": presence,
                "joined_truth_retained_top3": joined_truth,
                "state": state,
                "deficit_action": action,
                "wrong_request_proxy_count": len(source_by_index) - 1,
                "joined_wrong_request_authorizations": 0,
            }
        )

    actual_device = str(next(model.model.parameters()).device)
    if not actual_device.startswith("cuda"):
        raise ValueError(f"SILENT_DEVICE_FALLBACK:{actual_device}")
    positives = [row for row in rows if row["role"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE"]
    negatives = [row for row in rows if row["role"] == "NO_PORTAL_NEGATIVE"]
    ood = [row for row in rows if row["role"] == "LARGE_OPEN_ENTRANCE_OOD"]
    positive_presence = sum(row["portal_presence"] for row in positives)
    joined_truth = sum(row["joined_truth_retained_top3"] for row in positives)
    negative_false = sum(row["source_request_authorized"] for row in negatives)
    ood_leakage = sum(row["source_request_authorized"] for row in ood)
    wrong_request = sum(row["joined_wrong_request_authorizations"] for row in rows)
    unbound_wrong = sum(
        row["wrong_request_proxy_count"] for row in rows if row["portal_presence"]
    )
    safe_controls = len(negatives) + len(ood) - negative_false - ood_leakage
    balanced_accuracy = 0.5 * (
        joined_truth / len(positives) + safe_controls / (len(negatives) + len(ood))
    )
    gate = protocol["development_gate"]
    forbidden = {"COMMIT", "HANDOFF", "ARRIVED", "GUIDE_LEFT", "GUIDE_RIGHT", "GUIDE_FORWARD"}
    gates = {
        "minimum_positive_presence": positive_presence >= gate["minimum_positive_presence"],
        "minimum_joined_top3_truth_retention": joined_truth >= gate["minimum_joined_top3_truth_retention"],
        "zero_no_portal_authorization": negative_false == 0,
        "zero_large_open_mouth_authorization": ood_leakage == 0,
        "zero_wrong_request_authorization": wrong_request == 0,
        "no_commit_or_navigation_state": forbidden.isdisjoint({row["state"] for row in rows}),
    }
    gate_met = all(gates.values())
    decision = (
        "L10_PB9_SPECIALIZED_DOOR_ONTOLOGY_DEVELOPMENT_GATE_MET"
        if gate_met
        else "L10_PB9_SPECIALIZED_DOOR_ONTOLOGY_DEVELOPMENT_GATE_NOT_MET"
    )
    metrics = {
        "frames": len(rows),
        "positive_portal_presence": positive_presence,
        "joined_truth_retained_top3": joined_truth,
        "joined_truth_retained_top3_rate": joined_truth / len(positives),
        "no_portal_false_authorization": negative_false,
        "large_open_mouth_authorization_leakage": ood_leakage,
        "wrong_request_authorizations": wrong_request,
        "unbound_wrong_request_authorizations": unbound_wrong,
        "wrong_request_relative_reduction": (
            (unbound_wrong - wrong_request) / unbound_wrong if unbound_wrong else 0.0
        ),
        "balanced_accuracy": balanced_accuracy,
    }
    payload = {
        "schema": "l10-named-poi-specialized-door-ontology-development-result-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "decision": decision,
        "metrics": metrics,
        "gates": gates,
        "backend": {
            "provider": "perseusdg-yolo26n-seg-torch-cuda",
            "actual_device": actual_device,
            "actual_device_name": torch.cuda.get_device_name(0),
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
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

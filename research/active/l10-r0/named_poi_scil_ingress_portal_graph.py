"""Evaluate a proof-seeded, host-connected functional ingress portal graph.

SCIL supplies a localized requested-entity proof.  The unchanged structural
portal proposer supplies candidate regions.  A candidate can carry the proof
only when it lies below the proof anchor, contains a frozen minimum amount of
walkable/ingress semantics, and intersects the same semantic host component as
the proof anchor.  Human portal boxes are evaluator-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import ultralytics
from rapidocr import RapidOCR
from ultralytics import YOLO

import named_poi_biscript_eval as scil
from named_poi_portal_lattice import PortalSetProposal, propose_portal_sets


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


def _union_box(boxes: list[list[list[float]]]) -> list[float]:
    xs = [float(point[0]) for box in boxes for point in box]
    ys = [float(point[1]) for box in boxes for point in box]
    return [min(xs), min(ys), max(xs), max(ys)]


def _best_anchor(
    texts: list[str],
    boxes: list[list[list[float]]],
    target: str,
    entities: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    for start in range(len(texts)):
        for span in range(1, min(4, len(texts) - start) + 1):
            window = texts[start : start + span]
            scores, carriers, conflict = scil._lattice_scores(
                entities, profiles, scil._windows(window)
            )
            ranked = sorted(scores, key=lambda entity: (-scores[entity], entity))
            margin = scores[target] - max(
                value for entity, value in scores.items() if entity != target
            )
            candidates.append(
                {
                    "start": start,
                    "span": span,
                    "texts": window,
                    "box_xyxy": _union_box(boxes[start : start + span]),
                    "target_score": float(scores[target]),
                    "target_margin": float(margin),
                    "target_is_top1": ranked[0] == target,
                    "target_top_carrier": float(max(carriers[target].values())),
                    "carrier_conflict": bool(conflict),
                }
            )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["target_is_top1"] and not row["carrier_conflict"],
            row["target_score"],
            row["target_margin"],
            -row["span"],
            -row["start"],
        ),
    )


def _map_box(
    box: Sequence[float], source_size: tuple[int, int], mask_size: tuple[int, int]
) -> list[int]:
    source_width, source_height = source_size
    mask_width, mask_height = mask_size
    x1, y1, x2, y2 = box
    return [
        max(0, min(mask_width, round(x1 * mask_width / source_width))),
        max(0, min(mask_height, round(y1 * mask_height / source_height))),
        max(0, min(mask_width, round(x2 * mask_width / source_width))),
        max(0, min(mask_height, round(y2 * mask_height / source_height))),
    ]


def _anchor_host_label(labels: np.ndarray, anchor_box: list[int]) -> tuple[int, str]:
    x1, y1, x2, y2 = anchor_box
    local = labels[y1:y2, x1:x2]
    values, counts = np.unique(local[local > 0], return_counts=True)
    if values.size:
        return int(values[np.argmax(counts)]), "MAXIMUM_ANCHOR_OVERLAP"
    center = np.asarray([(y1 + y2) / 2.0, (x1 + x2) / 2.0])
    points = np.argwhere(labels > 0)
    if not points.size:
        return 0, "NO_HOST_COMPONENT"
    nearest = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
    return int(labels[tuple(nearest)]), "NEAREST_HOST_COMPONENT"


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


def _token(frame_key: str, target: str, image_sha256: str, protocol_sha256: str) -> str:
    payload = json.dumps(
        {
            "authority": "PB18_SCIL_HOST_INGRESS_GRAPH",
            "frame_key": frame_key,
            "target": target,
            "image_sha256": image_sha256,
            "protocol_sha256": protocol_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_row(
    proposal: PortalSetProposal,
    rank: int,
    mask: np.ndarray,
    labels: np.ndarray,
    host_label: int,
    anchor_center_y: float,
    image_size: tuple[int, int],
    ingress_classes: set[int],
    minimum_ingress_fraction: float,
    truth_box: Sequence[float] | None,
    decoy_box: Sequence[float] | None,
) -> dict[str, Any]:
    mask_size = (mask.shape[1], mask.shape[0])
    mapped = _map_box(proposal.box_xyxy, image_size, mask_size)
    x1, y1, x2, y2 = mapped
    semantic_crop = mask[y1:y2, x1:x2]
    label_crop = labels[y1:y2, x1:x2]
    ingress_fraction = (
        float(np.mean(np.isin(semantic_crop, list(ingress_classes))))
        if semantic_crop.size
        else 0.0
    )
    same_host_pixels = int(np.sum(label_crop == host_label)) if host_label else 0
    candidate_center_y = 0.5 * (proposal.box_xyxy[1] + proposal.box_xyxy[3])
    below_anchor = candidate_center_y >= anchor_center_y
    eligible = bool(
        host_label
        and same_host_pixels > 0
        and below_anchor
        and ingress_fraction >= minimum_ingress_fraction
    )
    return {
        "raw_rank": rank,
        "proposal_id": proposal.proposal_id,
        "family": proposal.family,
        "box_xyxy": [float(value) for value in proposal.box_xyxy],
        "proposal_score": float(proposal.proposal_score),
        "candidate_center_y": float(candidate_center_y),
        "below_anchor": bool(below_anchor),
        "ingress_fraction": ingress_fraction,
        "same_host_pixels": same_host_pixels,
        "eligible": eligible,
        "truth_member": _member(proposal.box_xyxy, truth_box)
        if truth_box is not None
        else None,
        "decoy_member": _member(proposal.box_xyxy, decoy_box)
        if decoy_box is not None
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise ValueError(f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol_sha256 = _sha256(protocol_path)
    protocol = _json(protocol_path)
    if _sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("EVALUATOR_HASH_MISMATCH")
    cohort_path = _verify(protocol["source"])
    scil_path = _verify(protocol["frozen_components"]["scil_evaluator"])
    if scil_path.resolve() != Path(scil.__file__).resolve():
        raise ValueError("SCIL_EVALUATOR_PATH_MISMATCH")
    proposer_path = _verify(protocol["frozen_components"]["portal_proposer"])
    if proposer_path.resolve() != (Path(__file__).resolve().parent / "named_poi_portal_lattice.py"):
        raise ValueError("PORTAL_PROPOSER_PATH_MISMATCH")
    semantic_model_path = _verify(protocol["frozen_components"]["semantic_model"])
    actual_versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "rapidocr": importlib.metadata.version("rapidocr"),
    }
    if actual_versions != protocol["runtime"]["versions"]:
        raise ValueError(f"RUNTIME_VERSION_MISMATCH:{actual_versions}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA_UNAVAILABLE")

    cohort = _json(cohort_path)
    if cohort["model_calls_before_source_freeze"] != {
        "ocr": 0,
        "semantic": 0,
        "portal_proposer": 0,
    }:
        raise ValueError("SOURCE_NOT_MODEL_BLIND_AT_FREEZE")
    frames = cohort["frames"]
    roles = Counter(frame["role"] for frame in frames)
    if roles != Counter(
        {
            "TARGET_PORTAL_PRESENT": 4,
            "TARGET_PORTAL_NOT_VISIBLE": 2,
            "WRONG_PORTAL_DECOY_VISIBLE": 2,
        }
    ):
        raise ValueError(f"ROLE_COUNT_MISMATCH:{dict(roles)}")
    for frame in frames:
        role = frame["role"]
        has_target = "target_portal_box_xyxy" in frame
        has_decoy = "decoy_portal_box_xyxy" in frame
        if role == "TARGET_PORTAL_PRESENT" and (not has_target or has_decoy):
            raise ValueError(f"POSITIVE_BOX_CONTRACT_MISMATCH:{frame['key']}")
        if role == "TARGET_PORTAL_NOT_VISIBLE" and (has_target or has_decoy):
            raise ValueError(f"NO_PORTAL_BOX_CONTRACT_MISMATCH:{frame['key']}")
        if role == "WRONG_PORTAL_DECOY_VISIBLE" and not (has_target and has_decoy):
            raise ValueError(f"DECOY_BOX_CONTRACT_MISMATCH:{frame['key']}")
    entities = [
        {
            "id": frame["id"],
            "name": frame["name"],
            "aliases": frame["aliases"],
        }
        for frame in frames
    ]
    if len({entity["id"] for entity in entities}) != 8:
        raise ValueError("ENTITY_IDS_NOT_UNIQUE")
    profiles = scil._profiles(entities)

    ocr_model_root = _path(protocol["runtime"]["ocr_model_root"])
    for filename, expected in protocol["runtime"]["ocr_models"].items():
        actual = _sha256(ocr_model_root / filename)
        if actual != expected:
            raise ValueError(f"OCR_MODEL_HASH_MISMATCH:{filename}:{actual}")
    ocr = RapidOCR(
        params={
            "Global.model_root_dir": str(ocr_model_root),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    semantic_model = YOLO(str(semantic_model_path))
    host_classes = {int(key): value for key, value in protocol["graph"]["host_classes"].items()}
    ingress_classes = {
        int(key): value for key, value in protocol["graph"]["ingress_classes"].items()
    }
    expected_classes = {**host_classes, **ingress_classes}
    if semantic_model.task != "semantic" or any(
        semantic_model.names[class_id] != name for class_id, name in expected_classes.items()
    ):
        raise ValueError("SEMANTIC_CLASS_CONTRACT_MISMATCH")

    score_threshold = float(protocol["scil"]["score_threshold"])
    margin_threshold = float(protocol["scil"]["margin_threshold"])
    minimum_ingress_fraction = float(protocol["graph"]["minimum_ingress_fraction"])
    maximum_authorized_candidates = int(protocol["graph"]["maximum_authorized_candidates"])
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for frame in frames:
        image_path = _path(frame["local_path"])
        if _sha256(image_path) != frame["image_sha256"]:
            raise ValueError(f"IMAGE_HASH_MISMATCH:{frame['key']}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or [image.shape[1], image.shape[0]] != frame["image_size"]:
            raise ValueError(f"IMAGE_DECODE_OR_SIZE_MISMATCH:{frame['key']}")
        height, width = image.shape[:2]

        ocr_started = time.perf_counter()
        ocr_output = ocr(image)
        ocr_seconds = time.perf_counter() - ocr_started
        texts = [str(value) for value in (ocr_output.txts if ocr_output.txts is not None else [])]
        boxes = ocr_output.boxes.tolist() if ocr_output.boxes is not None else []
        if len(texts) != len(boxes):
            raise ValueError(f"OCR_BOX_TEXT_ALIGNMENT_MISMATCH:{frame['key']}")
        observations = scil._windows(texts)
        scores, carrier_scores, conflict = scil._lattice_scores(entities, profiles, observations)
        ranked = sorted(scores, key=lambda entity: (-scores[entity], entity))
        top, second = ranked[:2]
        top_score = float(scores[top])
        margin = float(scores[top] - scores[second])
        proof = (
            top
            if not conflict and top_score >= score_threshold and margin >= margin_threshold
            else None
        )
        correct_proof = proof == frame["id"]
        anchor = _best_anchor(texts, boxes, frame["id"], entities, profiles)

        semantic_started = time.perf_counter()
        prediction = semantic_model.predict(
            source=str(image_path),
            device=0,
            imgsz=int(protocol["runtime"]["semantic_image_size"]),
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        semantic_seconds = time.perf_counter() - semantic_started
        if prediction.semantic_mask is None:
            raise ValueError(f"SEMANTIC_MASK_MISSING:{frame['key']}")
        mask = prediction.semantic_mask.data.detach().to("cpu").numpy().astype(np.uint8)
        host_binary = np.isin(mask, list(host_classes)).astype(np.uint8)
        _, labels = cv2.connectedComponents(host_binary, connectivity=8)
        host_label = 0
        host_selection = "NO_SCIL_ANCHOR"
        anchor_center_y = float("inf")
        if anchor is not None:
            anchor_box = _map_box(
                anchor["box_xyxy"], (width, height), (mask.shape[1], mask.shape[0])
            )
            host_label, host_selection = _anchor_host_label(labels, anchor_box)
            anchor_center_y = 0.5 * (anchor["box_xyxy"][1] + anchor["box_xyxy"][3])

        proposer_started = time.perf_counter()
        proposals, proposer_diagnostics = propose_portal_sets(image)
        proposer_seconds = time.perf_counter() - proposer_started
        truth_box = frame.get("target_portal_box_xyxy")
        decoy_box = frame.get("decoy_portal_box_xyxy")
        candidates = [
            _candidate_row(
                proposal,
                rank,
                mask,
                labels,
                host_label,
                anchor_center_y,
                (width, height),
                set(ingress_classes),
                minimum_ingress_fraction,
                truth_box,
                decoy_box,
            )
            for rank, proposal in enumerate(proposals, start=1)
        ]
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        authorized = eligible[:maximum_authorized_candidates] if correct_proof else []
        authority_token = (
            _token(frame["key"], frame["id"], frame["image_sha256"], protocol_sha256)
            if authorized
            else None
        )
        for candidate in authorized:
            candidate["authority_token"] = authority_token
            candidate["requested_entity"] = frame["id"]

        top1_truth = bool(authorized and authorized[0]["truth_member"])
        top3_truth = bool(any(candidate["truth_member"] for candidate in authorized))
        no_portal_false_authorization = bool(
            frame["role"] == "TARGET_PORTAL_NOT_VISIBLE" and authorized
        )
        decoy_authorization = bool(
            frame["role"] == "WRONG_PORTAL_DECOY_VISIBLE"
            and any(candidate["decoy_member"] for candidate in authorized)
        )
        if proof is None:
            state, action = "IDENTITY_UNKNOWN", "APPROACH_TEXT_OR_SWEEP_SIGN"
        elif not correct_proof:
            state, action = "IDENTITY_WRONG_PROOF", "HOLD_IDENTITY_CONFLICT"
        elif decoy_authorization:
            state, action = "TENANT_PORTAL_OWNERSHIP_CONFLICT", "HOLD_PORTAL_OWNERSHIP"
        elif truth_box is not None and top3_truth:
            state, action = "ENTITY_BOUND_INGRESS_PORTAL_SET", None
        elif truth_box is not None:
            state, action = "TARGET_PORTAL_TRUTH_MISS", "APPROACH_PORTAL_OR_SWEEP_ENTRANCE"
        elif no_portal_false_authorization:
            state, action = "FALSE_TARGET_PORTAL_AUTHORIZATION", "HOLD_PORTAL_OWNERSHIP"
        else:
            state, action = "IDENTITY_PROVED_TARGET_PORTAL_NOT_OBSERVABLE", "SWEEP_ENTRANCE"

        rows.append(
            {
                "key": frame["key"],
                "id": frame["id"],
                "role": frame["role"],
                "image_sha256": frame["image_sha256"],
                "ocr": {
                    "seconds": ocr_seconds,
                    "texts": texts,
                    "top_entity": top,
                    "top_score": top_score,
                    "second_score": float(scores[second]),
                    "margin": margin,
                    "proof": proof,
                    "correct_proof": correct_proof,
                    "carrier_conflict": bool(conflict),
                    "carrier_scores": carrier_scores[top],
                    "localized_anchor": anchor,
                },
                "semantic": {
                    "seconds": semantic_seconds,
                    "mask_size": [int(mask.shape[1]), int(mask.shape[0])],
                    "anchor_host_label": host_label,
                    "anchor_host_selection": host_selection,
                },
                "portal_graph": {
                    "seconds": proposer_seconds,
                    "proposer_diagnostics": proposer_diagnostics,
                    "raw_candidate_count": len(candidates),
                    "eligible_candidate_count": len(eligible),
                    "authorized_candidates": authorized,
                    "all_candidates": candidates,
                    "top1_truth": top1_truth,
                    "top3_truth": top3_truth,
                    "no_portal_false_authorization": no_portal_false_authorization,
                    "decoy_authorization": decoy_authorization,
                },
                "state": state,
                "deficit_action": action,
            }
        )

    actual_device = str(next(semantic_model.model.parameters()).device)
    if not actual_device.startswith("cuda"):
        raise ValueError(f"SILENT_DEVICE_FALLBACK:{actual_device}")
    target_portal_frames = [
        row
        for row in rows
        if row["role"] in {"TARGET_PORTAL_PRESENT", "WRONG_PORTAL_DECOY_VISIBLE"}
    ]
    core_positives = [row for row in rows if row["role"] == "TARGET_PORTAL_PRESENT"]
    no_portal = [row for row in rows if row["role"] == "TARGET_PORTAL_NOT_VISIBLE"]
    decoys = [row for row in rows if row["role"] == "WRONG_PORTAL_DECOY_VISIBLE"]
    metrics = {
        "identity_correct_proofs": sum(row["ocr"]["correct_proof"] for row in rows),
        "identity_wrong_proofs": sum(
            row["ocr"]["proof"] is not None and not row["ocr"]["correct_proof"]
            for row in rows
        ),
        "identity_unknown": sum(row["ocr"]["proof"] is None for row in rows),
        "target_portal_top1": sum(
            row["portal_graph"]["top1_truth"] for row in target_portal_frames
        ),
        "target_portal_top3": sum(
            row["portal_graph"]["top3_truth"] for row in target_portal_frames
        ),
        "core_target_portal_top1": sum(
            row["portal_graph"]["top1_truth"] for row in core_positives
        ),
        "core_target_portal_top3": sum(
            row["portal_graph"]["top3_truth"] for row in core_positives
        ),
        "no_portal_false_authorization": sum(
            row["portal_graph"]["no_portal_false_authorization"] for row in no_portal
        ),
        "tenant_decoy_overlap_authorization": sum(
            row["portal_graph"]["decoy_authorization"] for row in decoys
        ),
        "tenant_ownership_pair_exact": sum(
            row["portal_graph"]["top3_truth"]
            and not row["portal_graph"]["decoy_authorization"]
            for row in decoys
        ),
    }
    negative_opportunities = len(no_portal) + len(decoys)
    negative_false = (
        metrics["no_portal_false_authorization"]
        + metrics["tenant_decoy_overlap_authorization"]
    )
    metrics["opportunity_balanced_accuracy_top3"] = 0.5 * (
        metrics["target_portal_top3"] / len(target_portal_frames)
        + (negative_opportunities - negative_false) / negative_opportunities
    )
    gate = {
        "identity_8_of_8_correct_zero_wrong_zero_unknown": (
            metrics["identity_correct_proofs"] == 8
            and metrics["identity_wrong_proofs"] == 0
            and metrics["identity_unknown"] == 0
        ),
        "target_portal_top1_6_of_6": metrics["target_portal_top1"] == 6,
        "target_portal_top3_6_of_6": metrics["target_portal_top3"] == 6,
        "zero_no_portal_false_authorization": metrics["no_portal_false_authorization"] == 0,
        "zero_tenant_decoy_overlap_authorization": (
            metrics["tenant_decoy_overlap_authorization"] == 0
        ),
        "tenant_ownership_pair_exact_2_of_2": metrics["tenant_ownership_pair_exact"] == 2,
    }
    passed = all(gate.values())
    decision = (
        "L10_PB18_SCRIPT_PROVED_INGRESS_CONNECTED_PORTAL_GRAPH_GATE_MET"
        if passed
        else "L10_PB18_SCRIPT_PROVED_INGRESS_CONNECTED_PORTAL_GRAPH_GATE_NOT_MET"
    )
    result = {
        "schema": "l10-pb18-scil-ingress-portal-graph-result-v1",
        "decision": decision,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "cohort_sha256": protocol["source"]["sha256"],
        "evaluator_sha256": protocol["evaluator"]["sha256"],
        "metrics": metrics,
        "gate": {**gate, "passed": passed},
        "runtime": {
            "versions": actual_versions,
            "semantic_device": actual_device,
            "semantic_device_name": torch.cuda.get_device_name(0),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()

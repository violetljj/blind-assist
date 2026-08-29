"""Test one typed VLM portal representation on the consumed PB7 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[3]
ROLE_LABEL = {
    "IN_SCOPE_DOOR_LEVEL_POSITIVE": "1",
    "NO_PORTAL_NEGATIVE": "2",
    "LARGE_OPEN_ENTRANCE_OOD": "3",
}


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


def _model_contract(model_root: Path, spec: dict[str, Any]) -> None:
    if model_root.name != spec["revision"]:
        raise ValueError("MODEL_REVISION_DIRECTORY_MISMATCH")
    for name, expected in spec["hashed_files"].items():
        actual = _sha256(model_root / name)
        if actual != expected:
            raise ValueError(f"MODEL_FILE_HASH_MISMATCH:{name}:{actual}")
    for name, expected in spec["shard_sizes"].items():
        actual = (model_root / name).stat().st_size
        if actual != int(expected):
            raise ValueError(f"MODEL_SHARD_SIZE_MISMATCH:{name}:{actual}")


def _label_ids(processor: Any, labels: list[str]) -> torch.Tensor:
    values = []
    for label in labels:
        encoded = processor.tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"LABEL_NOT_SINGLE_TOKEN:{label}:{encoded}")
        values.append(int(encoded[0]))
    if len(set(values)) != len(values):
        raise ValueError("LABEL_TOKEN_COLLISION")
    return torch.tensor(values, dtype=torch.long)


def _inputs(processor: Any, image: Image.Image, prompt: str) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return dict(
        processor(
            text=[rendered],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
    )


def _scores(
    model: Qwen2VLForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    labels: list[str],
    label_ids: torch.Tensor,
) -> tuple[dict[str, float], float]:
    values = {key: value.to("cuda") for key, value in inputs.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**values, use_cache=False)
        logits = output.logits[0, -1]
        selected = logits.index_select(0, label_ids.to(logits.device)).float()
        normalized = torch.log_softmax(selected, dim=0)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return (
        {label: float(normalized[index].item()) for index, label in enumerate(labels)},
        seconds,
    )


def _grid_box(label: str, width: int, height: int) -> list[int]:
    index = int(label) - 1
    row, column = divmod(index, 3)
    return [
        column * width // 3,
        row * height // 3,
        (column + 1) * width // 3,
        (row + 1) * height // 3,
    ]


def _member(candidate: list[int], truth: list[int] | None) -> bool:
    if truth is None:
        return False
    x1, y1, x2, y2 = candidate
    tx1, ty1, tx2, ty2 = truth
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    if not (tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2):
        return False
    intersection = max(0, min(x2, tx2) - max(x1, tx1)) * max(
        0, min(y2, ty2) - max(y1, ty1)
    )
    area = max(1, (x2 - x1) * (y2 - y1))
    return intersection / area >= 0.5


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
    if transformers.__version__ != protocol["runtime"]["transformers_version"]:
        raise ValueError("TRANSFORMERS_VERSION_MISMATCH")
    if not torch.cuda.is_available():
        raise ValueError("CUDA_UNAVAILABLE")

    source = _json(sources["source_spec"])
    manifest = _json(sources["source_manifest"])
    audit = _json(sources["source_audit"])
    pb7 = _json(frozen["pb7_formal_result"])
    if pb7["metrics"]["scil_correct_proofs"] != 8 or pb7["metrics"]["scil_wrong_proofs"] != 0:
        raise ValueError("PB7_IDENTITY_AUTHORITY_MISMATCH")
    if any(
        audit["checks"][key] != 0
        for key in ("ocr_calls_before_freeze", "semantic_calls_before_freeze")
    ):
        raise ValueError("PRE_FREEZE_MODEL_ACCESS_DETECTED")
    if len(source["cohort"]) != 8 or manifest["image_count"] != 8 or len(audit["frames"]) != 8:
        raise ValueError("COHORT_COUNT_MISMATCH")
    source_by_index = {int(row["index"]): row for row in source["cohort"]}
    manifest_by_index = {int(row["index"]): row for row in manifest["frames"]}
    audit_by_index = {int(row["index"]): row for row in audit["frames"]}
    pb7_by_index = {int(row["index"]): row for row in pb7["rows"]}
    expected_indices = set(range(1, 9))
    if not all(set(values) == expected_indices for values in (
        source_by_index,
        manifest_by_index,
        audit_by_index,
        pb7_by_index,
    )):
        raise ValueError("COHORT_INDEX_MISMATCH")

    model_spec = protocol["model"]
    model_root = _path(model_spec["path"])
    _model_contract(model_root, model_spec)
    processor = AutoProcessor.from_pretrained(
        model_root,
        local_files_only=True,
        min_pixels=int(model_spec["minimum_pixels"]),
        max_pixels=int(model_spec["maximum_pixels"]),
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_root,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).eval().to("cuda")
    actual_device = str(next(model.parameters()).device)
    if not actual_device.startswith("cuda"):
        raise ValueError(f"SILENT_DEVICE_FALLBACK:{actual_device}")
    class_labels = ["1", "2", "3"]
    grid_labels = [str(index) for index in range(1, 10)]
    class_ids = _label_ids(processor, class_labels)
    grid_ids = _label_ids(processor, grid_labels)
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
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if list(image.size) != truth["local_image_size"]:
            raise ValueError(f"IMAGE_SIZE_MISMATCH:{index}")
        if not identity["ocr"]["correct_proof"]:
            raise ValueError(f"IDENTITY_PROOF_NOT_CORRECT:{index}")

        class_scores, class_seconds = _scores(
            model,
            _inputs(processor, image, protocol["prompts"]["typed_scene"]),
            class_labels,
            class_ids,
        )
        typed_label = max(class_labels, key=lambda label: (class_scores[label], -int(label)))
        grid_scores, grid_seconds = _scores(
            model,
            _inputs(processor, image, protocol["prompts"]["grid_localization"]),
            grid_labels,
            grid_ids,
        )
        ranked_grid = sorted(grid_labels, key=lambda label: (-grid_scores[label], int(label)))
        top3 = [
            {
                "rank": rank,
                "cell": int(label),
                "box_xyxy": _grid_box(label, image.width, image.height),
                "log_probability_within_grid_labels": grid_scores[label],
                "truth_member": _member(
                    _grid_box(label, image.width, image.height),
                    truth["portal_set_box_xyxy"],
                ),
            }
            for rank, label in enumerate(ranked_grid[:3], start=1)
        ]
        correct_type = typed_label == ROLE_LABEL[truth["status"]]
        source_authorized = typed_label == "1"
        joined_truth = bool(
            truth["status"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE"
            and source_authorized
            and any(candidate["truth_member"] for candidate in top3)
        )
        if typed_label == "2":
            state = "TYPED_NONE"
            action = "APPROACH_OR_SWEEP_FOR_DOOR_LEVEL_BOUNDARY"
        elif typed_label == "3":
            state = "TYPED_LARGE_OPEN_MOUTH_OOD"
            action = "SIDESTEP_OR_APPROACH_FOR_DOOR_LEVEL_PORTAL"
        elif truth["status"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE":
            state = "ENTITY_BOUND_DOOR_LEVEL_GRID" if joined_truth else "DOOR_LEVEL_GRID_TRUTH_MISS"
            action = None if joined_truth else "APPROACH_OR_SWEEP_FOR_DOOR_LEVEL_BOUNDARY"
        elif truth["status"] == "NO_PORTAL_NEGATIVE":
            state = "FALSE_DOOR_LEVEL_AUTHORIZATION"
            action = "HOLD_FALSE_PORTAL"
        else:
            state = "OOD_DOOR_LEVEL_AUTHORIZATION_LEAKAGE"
            action = "SIDESTEP_OR_APPROACH_FOR_DOOR_LEVEL_PORTAL"
        rows.append(
            {
                "index": index,
                "id": entity["id"],
                "role": truth["status"],
                "image_sha256": frame["sha256"],
                "identity_proof": identity["ocr"]["proof"],
                "typed_scene": {
                    "prediction": typed_label,
                    "truth": ROLE_LABEL[truth["status"]],
                    "correct": correct_type,
                    "log_probabilities_within_labels": class_scores,
                    "seconds": class_seconds,
                },
                "grid_localization": {
                    "top3": top3,
                    "all_log_probabilities_within_labels": grid_scores,
                    "seconds": grid_seconds,
                    "truth_box_xyxy": truth["portal_set_box_xyxy"],
                },
                "source_request_authorized": source_authorized,
                "joined_truth_retained_top3": joined_truth,
                "state": state,
                "deficit_action": action,
                "wrong_request_proxy_count": len(source_by_index) - 1,
                "joined_wrong_request_authorizations": 0,
            }
        )

    positives = [row for row in rows if row["role"] == "IN_SCOPE_DOOR_LEVEL_POSITIVE"]
    negatives = [row for row in rows if row["role"] == "NO_PORTAL_NEGATIVE"]
    ood = [row for row in rows if row["role"] == "LARGE_OPEN_ENTRANCE_OOD"]
    typed_correct = sum(row["typed_scene"]["correct"] for row in rows)
    positive_door = sum(row["typed_scene"]["prediction"] == "1" for row in positives)
    negative_none = sum(row["typed_scene"]["prediction"] == "2" for row in negatives)
    ood_typed = sum(row["typed_scene"]["prediction"] == "3" for row in ood)
    joined_truth = sum(row["joined_truth_retained_top3"] for row in positives)
    negative_false = sum(row["source_request_authorized"] for row in negatives)
    ood_leakage = sum(row["source_request_authorized"] for row in ood)
    wrong_request = sum(row["joined_wrong_request_authorizations"] for row in rows)
    unbound_wrong = sum(
        row["wrong_request_proxy_count"] for row in rows if row["source_request_authorized"]
    )
    safe_controls = len(negatives) + len(ood) - negative_false - ood_leakage
    balanced_accuracy = 0.5 * (
        joined_truth / len(positives) + safe_controls / (len(negatives) + len(ood))
    )
    gate = protocol["development_gate"]
    forbidden = {"COMMIT", "HANDOFF", "ARRIVED", "GUIDE_LEFT", "GUIDE_RIGHT", "GUIDE_FORWARD"}
    gates = {
        "minimum_typed_scene_correct": typed_correct >= gate["minimum_typed_scene_correct"],
        "minimum_positive_door_level_classifications": positive_door >= gate["minimum_positive_door_level_classifications"],
        "minimum_joined_top3_truth_retention": joined_truth >= gate["minimum_joined_top3_truth_retention"],
        "all_no_portal_controls_typed_none": negative_none == len(negatives),
        "all_open_mouth_controls_typed_ood": ood_typed == len(ood),
        "zero_no_portal_authorization": negative_false == 0,
        "zero_ood_authorization": ood_leakage == 0,
        "zero_wrong_request_authorization": wrong_request == 0,
        "no_commit_or_navigation_state": forbidden.isdisjoint({row["state"] for row in rows}),
    }
    gate_met = all(gates.values())
    decision = (
        "L10_PB8_TYPED_VLM_GRID_DEVELOPMENT_GATE_MET"
        if gate_met
        else "L10_PB8_TYPED_VLM_GRID_DEVELOPMENT_GATE_NOT_MET"
    )
    metrics = {
        "frames": len(rows),
        "typed_scene_correct": typed_correct,
        "typed_scene_accuracy": typed_correct / len(rows),
        "positive_door_level_classifications": positive_door,
        "no_portal_typed_none": negative_none,
        "open_mouth_typed_ood": ood_typed,
        "joined_truth_retained_top3": joined_truth,
        "joined_truth_retained_top3_rate": joined_truth / len(positives),
        "no_portal_false_authorization": negative_false,
        "ood_authorization_leakage": ood_leakage,
        "wrong_request_authorizations": wrong_request,
        "unbound_wrong_request_authorizations": unbound_wrong,
        "wrong_request_relative_reduction": (
            (unbound_wrong - wrong_request) / unbound_wrong if unbound_wrong else 0.0
        ),
        "balanced_accuracy": balanced_accuracy,
        "pb7_balanced_accuracy": float(pb7["metrics"]["balanced_accuracy"]),
        "absolute_balanced_accuracy_gain_over_pb7": balanced_accuracy - float(pb7["metrics"]["balanced_accuracy"]),
    }
    payload = {
        "schema": "l10-named-poi-typed-vlm-portal-development-result-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "decision": decision,
        "metrics": metrics,
        "gates": gates,
        "backend": {
            "provider": "qwen2-vl-2b-torch-cuda",
            "actual_device": actual_device,
            "actual_device_name": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
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

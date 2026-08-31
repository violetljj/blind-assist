#!/usr/bin/env python3
"""Goal-conditioned multi-view street-sign verification on official FSNS."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from rapidocr import RapidOCR


PROTOCOL_SCHEMA = "blindassist-l10-fsns-multiview-goal-verifier-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-fsns-multiview-goal-verifier-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        character.lower()
        for character in decomposed
        if not unicodedata.combining(character) and character.isascii() and character.isalnum()
    )


def tokens(text: str) -> list[str]:
    return [token for item in re.findall(r"[^\W_]+", text, flags=re.UNICODE) if (token := normalize(item))]


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, 1):
        current = [index]
        for other_index, other_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + (character != other_character),
                )
            )
        previous = current
    return previous[-1]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_tokens(text: str, contract: dict[str, Any]) -> list[str]:
    ignored = set(contract["ignored_generic_tokens"])
    selected = [
        token
        for token in tokens(text)
        if contract["minimum_target_token_length"] <= len(token) <= contract["maximum_target_token_length"]
        and token not in ignored
        and token.isalpha()
    ]
    return sorted(set(selected), key=lambda token: (-len(token), token))


def candidate_forms(text: str) -> list[str]:
    forms = set(tokens(text))
    whole = normalize(text)
    if whole:
        forms.add(whole)
    return sorted(forms)


def build_cache(
    rows: list[dict[str, Any]],
    images_root: Path,
    models: Path,
    cache_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    payload: dict[str, Any] = {
        "schema": "blindassist-l10-fsns-rapidocr-view-cache-v1",
        "backend": "RapidOCR 3.9.2 / PP-OCRv6 small / ONNX Runtime CPU",
        "view_scale": contract["view_scale"],
        "examples": {},
    }
    started = time.perf_counter()
    for position, row in enumerate(rows, 1):
        path = images_root / row["image"]
        composite = cv2.imread(str(path), cv2.IMREAD_COLOR)
        require(composite is not None and composite.shape[:2] == (150, 600), f"IMAGE_DECODE_FAILED:{path}")
        view_rows = []
        for view_index in range(int(row["view_count"])):
            view = composite[:, view_index * 150 : (view_index + 1) * 150]
            scale = float(contract["view_scale"])
            enlarged = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            output = engine(enlarged)
            boxes = output.boxes if output.boxes is not None else ()
            texts = output.txts if output.txts is not None else ()
            scores = output.scores if output.scores is not None else ()
            detections = []
            for box, text, score in zip(boxes, texts, scores):
                detections.append(
                    {
                        "box": (box / scale).astype(float).tolist(),
                        "text": str(text),
                        "forms": candidate_forms(str(text)),
                        "confidence": float(score),
                        "source": f"view_{view_index}",
                    }
                )
            view_rows.append({"view_index": view_index, "detections": detections})
        payload["examples"][str(row["index"])] = {
            "image": str(path.resolve()),
            "text": row["text"],
            "view_count": row["view_count"],
            "views": view_rows,
        }
        print(json.dumps({"ocr_example": position, "total": len(rows), "index": row["index"]}), flush=True)
    payload["ocr_wall_s"] = round(time.perf_counter() - started, 3)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def exact_match(target: str, form: str) -> bool:
    return target == form or target in form


def match_goal(
    candidates: list[dict[str, Any]],
    expected_tokens: list[str],
    contract: dict[str, Any],
    allow_edit_consensus: bool,
) -> dict[str, Any]:
    exact = []
    edit = []
    for candidate in candidates:
        if candidate["confidence"] < contract["minimum_ocr_confidence"]:
            continue
        for target in expected_tokens:
            for form in candidate["forms"]:
                if exact_match(target, form):
                    exact.append({"target": target, "form": form, "kind": "EXACT", **candidate})
                elif (
                    allow_edit_consensus
                    and len(form) >= contract["minimum_edit_token_length"]
                    and abs(len(form) - len(target)) <= 1
                    and edit_distance(target, form) == 1
                ):
                    edit.append({"target": target, "form": form, "kind": "EDIT_ONE", **candidate})
    if exact:
        selected = max(exact, key=lambda row: (row["confidence"], len(row["target"])))
        return {"accepted": True, "match": selected, "evidence_sources": sorted({row["source"] for row in exact if row["target"] == selected["target"]})}
    if allow_edit_consensus and edit:
        by_target: dict[str, list[dict[str, Any]]] = {}
        for row in edit:
            by_target.setdefault(row["target"], []).append(row)
        admitted = [
            (target, rows)
            for target, rows in by_target.items()
            if len({row["source"] for row in rows}) >= contract["minimum_edit_evidence_views"]
        ]
        if admitted:
            target, rows = max(admitted, key=lambda item: (len({row["source"] for row in item[1]}), max(row["confidence"] for row in item[1]), len(item[0])))
            selected = max(rows, key=lambda row: row["confidence"])
            return {"accepted": True, "match": selected, "evidence_sources": sorted({row["source"] for row in rows})}
    return {"accepted": False, "match": None, "evidence_sources": []}


def flatten_views(example: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    views = example["views"] if limit is None else example["views"][:limit]
    return [detection for view in views for detection in view["detections"]]


def synthetic_token(index: int, salt: str) -> str:
    digest = hashlib.sha256(f"{index}|{salt}".encode("utf-8")).hexdigest()[:12]
    return "zz" + "".join(chr(ord("a") + int(character, 16)) for character in digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path, output_path = args.protocol.resolve(), args.output.resolve()
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    inputs = {key: Path(value).resolve() for key, value in protocol["inputs"].items()}
    for name in protocol["input_sha256"]:
        require(sha256(inputs[name]) == protocol["input_sha256"][name], f"INPUT_HASH_MISMATCH:{name}")
    for name, expected in protocol["model_sha256"].items():
        require(sha256(inputs["models"] / name) == expected, f"MODEL_HASH_MISMATCH:{name}")
    manifest = load_manifest(inputs["manifest"])
    by_index = {int(row["index"]): row for row in manifest}
    selected_indices = [int(index) for index in protocol["selected_indices"]]
    require(len(selected_indices) == len(set(selected_indices)) == protocol["selection_contract"]["cohort_size"], "BAD_SELECTION_SIZE")
    selected = [by_index[index] for index in selected_indices]
    label_counts = Counter(normalize(row["text"]) for row in manifest)
    if "disjoint_reference_manifest" in inputs:
        reference = load_manifest(inputs["disjoint_reference_manifest"])
        reference_hashes = {row["image_sha256"] for row in reference}
        reference_labels = {normalize(row["text"]) for row in reference}
        reference_tokens = {
            token
            for row in reference
            for token in target_tokens(row["text"], protocol["selection_contract"])
        }
        require(not ({row["image_sha256"] for row in selected} & reference_hashes), "REFERENCE_IMAGE_OVERLAP")
        require(not ({normalize(row["text"]) for row in selected} & reference_labels), "REFERENCE_LABEL_OVERLAP")
        require(
            not ({token for row in selected for token in target_tokens(row["text"], protocol["selection_contract"])} & reference_tokens),
            "REFERENCE_DISTINCTIVE_TOKEN_OVERLAP",
        )
    for row in selected:
        require(row["view_count"] >= protocol["selection_contract"]["minimum_view_count"], f"INSUFFICIENT_VIEWS:{row['index']}")
        require(label_counts[normalize(row["text"])] == 1, f"NONUNIQUE_LABEL:{row['index']}")
        require(target_tokens(row["text"], protocol["selection_contract"]), f"NO_DISTINCTIVE_TARGET:{row['index']}")
    cache_path = inputs["ocr_cache"]
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else build_cache(selected, inputs["images"], inputs["models"], cache_path, protocol["algorithm_contract"])
    require(set(cache["examples"]) == {str(index) for index in selected_indices}, "CACHE_SELECTION_MISMATCH")
    counts = {"single_view_exact": Counter(), "multiview_goal": Counter()}
    challenge_accepts = []
    synthetic_accepts = []
    result_rows = []
    for position, row in enumerate(selected):
        example = cache["examples"][str(row["index"])]
        expected = target_tokens(row["text"], protocol["selection_contract"])
        baseline = match_goal(flatten_views(example, 1), expected, protocol["algorithm_contract"], False)
        successor = match_goal(flatten_views(example), expected, protocol["algorithm_contract"], True)
        counts["single_view_exact"]["CORRECT" if baseline["accepted"] else "UNKNOWN"] += 1
        counts["multiview_goal"]["CORRECT" if successor["accepted"] else "UNKNOWN"] += 1
        first_acceptance = None
        for view_count in range(1, int(row["view_count"]) + 1):
            prefix = match_goal(flatten_views(example, view_count), expected, protocol["algorithm_contract"], True)
            if prefix["accepted"]:
                first_acceptance = view_count
                break
        challenge_row = selected[(position + 1) % len(selected)]
        challenge_tokens = target_tokens(challenge_row["text"], protocol["selection_contract"])
        require(not any(token in normalize(row["text"]) for token in challenge_tokens), f"CHALLENGE_IN_CANONICAL_LABEL:{row['index']}")
        challenge = match_goal(flatten_views(example), challenge_tokens, protocol["algorithm_contract"], True)
        if challenge["accepted"]:
            challenge_accepts.append({"index": row["index"], "challenge_index": challenge_row["index"], "challenge_text": challenge_row["text"], "match": challenge})
        canary = synthetic_token(int(row["index"]), protocol["synthetic_negative_salt"])
        synthetic = match_goal(flatten_views(example), [canary], protocol["algorithm_contract"], True)
        if synthetic["accepted"]:
            synthetic_accepts.append({"index": row["index"], "token": canary, "match": synthetic})
        result_rows.append(
            {
                "index": row["index"],
                "text": row["text"],
                "view_count": row["view_count"],
                "target_tokens": expected,
                "single_view_exact": baseline,
                "multiview_goal": {**successor, "first_acceptance_view_count": first_acceptance},
                "canonical_label_disjoint_challenge": {
                    "challenge_index": challenge_row["index"],
                    "challenge_text": challenge_row["text"],
                    "accepted": challenge["accepted"],
                    "match": challenge,
                },
            }
        )
    baseline_correct = counts["single_view_exact"]["CORRECT"]
    successor_correct = counts["multiview_goal"]["CORRECT"]
    accepted_view_counts = [row["multiview_goal"]["first_acceptance_view_count"] for row in result_rows if row["multiview_goal"]["first_acceptance_view_count"] is not None]
    gate = {
        "cohort_size_met": len(result_rows) == protocol["selection_contract"]["cohort_size"],
        "minimum_correct_gain_met": successor_correct - baseline_correct >= protocol["gate"]["minimum_correct_gain"],
        "minimum_successor_correct_rate_met": successor_correct / len(result_rows) >= protocol["gate"].get("minimum_successor_correct_rate", 0.0),
        "zero_synthetic_negative_accepts": len(synthetic_accepts) == 0,
        "canonical_label_disjoint_challenge_limit_met": len(challenge_accepts) <= protocol["gate"]["maximum_canonical_label_disjoint_challenge_accepts"],
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
    metrics = {
        "examples": len(result_rows),
        "single_view_exact_correct_unknown": [baseline_correct, len(result_rows) - baseline_correct],
        "multiview_goal_correct_unknown": [successor_correct, len(result_rows) - successor_correct],
        "correct_gain": successor_correct - baseline_correct,
        "canonical_label_disjoint_challenges": len(result_rows),
        "canonical_label_disjoint_challenge_accepts": len(challenge_accepts),
        "synthetic_negative_queries": len(result_rows),
        "synthetic_negative_accepts": len(synthetic_accepts),
        "accepted_mean_views_to_first_candidate": round(sum(accepted_view_counts) / len(accepted_view_counts), 3) if accepted_view_counts else None,
        "identity_bindings_emitted": 0,
        "portal_bindings_emitted": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": protocol["authority"],
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "ocr_cache": str(cache_path),
        "ocr_cache_sha256": sha256(cache_path),
        "metrics": metrics,
        "gate": gate,
        "canonical_label_disjoint_challenge_accepts": challenge_accepts,
        "synthetic_negative_accepts": synthetic_accepts,
        "rows": result_rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"decision": result["decision"], "metrics": metrics, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

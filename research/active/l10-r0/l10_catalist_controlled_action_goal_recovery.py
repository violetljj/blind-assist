#!/usr/bin/env python3
"""Evaluate goal-text recovery during controlled CATALIST camera actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
from rapidocr import RapidOCR


PROTOCOL_SCHEMA = "blindassist-l10-catalist-controlled-action-goal-recovery-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-catalist-controlled-action-goal-recovery-result-v1"
ACTIONS = ("TR", "P", "T", "Z", "R")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        character.lower()
        for character in decomposed
        if not unicodedata.combining(character) and character.isascii() and character.isalnum()
    )


def tokens(text: str, minimum: int = 1, maximum: int = 1000) -> list[str]:
    return [
        token
        for item in re.findall(r"[A-Za-z]+", text)
        if minimum <= len(token := normalize(item)) <= maximum
    ]


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


def read_annotations(path: Path) -> list[dict[str, Any]]:
    rows = []
    for source_position, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        fields = line.split("\t")
        if len(fields) < 3 or not fields[0].endswith(".mp4"):
            continue
        rows.append(
            {
                "source_position": source_position,
                "video": fields[0],
                "label": fields[1],
                "action": fields[2].upper(),
                "action_start_s": float(fields[3]) if len(fields) >= 4 and fields[3] else 0.0,
                "action_end_s": float(fields[4]) if len(fields) >= 5 and fields[4] else None,
            }
        )
    return rows


def label_documents(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, set[str]]:
    return {
        row["label"]: set(
            tokens(
                row["label"],
                contract["minimum_target_token_length"],
                contract["maximum_target_token_length"],
            )
        )
        for row in rows
    }


def select_target(
    label: str,
    document_frequency: Counter[str],
    reference_tokens: set[str],
    contract: dict[str, Any],
) -> str | None:
    candidates = {
        token
        for token in tokens(
            label,
            contract["minimum_target_token_length"],
            contract["maximum_target_token_length"],
        )
        if token not in reference_tokens
    }
    return min(candidates, key=lambda token: (document_frequency[token], -len(token), token)) if candidates else None


def deterministic_panel(
    reference_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    reference_documents = label_documents(reference_rows, contract)
    reference_tokens = {token for document in reference_documents.values() for token in document}
    all_documents = label_documents(reference_rows + evaluation_rows, contract)
    document_frequency = Counter(token for document in all_documents.values() for token in document)
    first_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in evaluation_rows:
        target = select_target(row["label"], document_frequency, reference_tokens, contract)
        if row["action"] not in ACTIONS or target is None:
            continue
        first_pair.setdefault((row["label"], row["action"]), {**row, "target_token": target})
    by_action = {
        action: [row for (label, candidate_action), row in first_pair.items() if candidate_action == action]
        for action in ACTIONS
    }
    selected: list[dict[str, Any]] = []
    used_labels: set[str] = set()
    quota = int(contract["episodes_per_action"])
    for round_index in range(quota):
        action_order = ACTIONS if round_index % 2 == 0 else tuple(reversed(ACTIONS))
        for action in action_order:
            options = [row for row in by_action[action] if row not in selected]
            require(bool(options), f"INSUFFICIENT_ACTION_CANDIDATES:{action}")
            chosen = next((row for row in options if row["label"] not in used_labels), options[0])
            selected.append(chosen)
            used_labels.add(chosen["label"])
    return selected


def candidate_forms(text: str) -> list[str]:
    forms = set(tokens(text))
    whole = normalize(text)
    if whole:
        forms.add(whole)
    return sorted(forms)


def frame_indices(
    frame_count: int,
    fps: float,
    action_start_s: float,
    action_end_s: float | None,
    fractions: list[float],
) -> tuple[int, list[int], float]:
    require(frame_count > 1 and fps > 0, "INVALID_VIDEO_TIMING")
    duration_s = (frame_count - 1) / fps
    start_s = min(max(action_start_s, 0.0), duration_s)
    end_s = duration_s if action_end_s is None else min(max(action_end_s, start_s), duration_s)
    baseline = max(0, int(math.floor(start_s * fps)) - 1) if start_s > 0 else 0
    action_frames = []
    for fraction in fractions:
        timestamp = start_s + (end_s - start_s) * float(fraction)
        action_frames.append(min(frame_count - 1, max(0, int(round(timestamp * fps)))))
    action_frames = [index for index in dict.fromkeys(action_frames) if index != baseline]
    require(bool(action_frames), "NO_DISTINCT_ACTION_FRAMES")
    return baseline, action_frames, duration_s


def ocr_frame(engine: RapidOCR, frame: Any, frame_index: int) -> dict[str, Any]:
    output = engine(frame)
    boxes = output.boxes if output.boxes is not None else ()
    texts = output.txts if output.txts is not None else ()
    scores = output.scores if output.scores is not None else ()
    detections = []
    for box, text, score in zip(boxes, texts, scores):
        detections.append(
            {
                "frame_index": frame_index,
                "box": box.astype(float).tolist(),
                "text": str(text),
                "forms": candidate_forms(str(text)),
                "confidence": float(score),
            }
        )
    return {"frame_index": frame_index, "detections": detections}


def build_cache(
    selected: list[dict[str, Any]],
    videos_root: Path,
    models: Path,
    cache_path: Path,
    algorithm: dict[str, Any],
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
        "schema": "blindassist-l10-catalist-controlled-action-ocr-cache-v1",
        "backend": "RapidOCR / PP-OCRv6 small / ONNX Runtime CPU",
        "videos": {},
    }
    started = time.perf_counter()
    for position, row in enumerate(selected, 1):
        video_path = videos_root / row["video"]
        capture = cv2.VideoCapture(str(video_path))
        require(capture.isOpened(), f"VIDEO_OPEN_FAILED:{video_path}")
        declared_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        decoded_frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                decoded_frame_count += 1
        finally:
            capture.release()
        require(decoded_frame_count > 1, f"INSUFFICIENT_DECODABLE_FRAMES:{row['video']}")
        frame_count = decoded_frame_count
        baseline, action_frames, duration_s = frame_indices(
            frame_count,
            fps,
            float(row["action_start_s"]),
            row["action_end_s"],
            algorithm["action_checkpoint_fractions"],
        )
        wanted = [baseline, *action_frames]
        wanted_set = set(wanted)
        frames_by_index: dict[int, dict[str, Any]] = {}
        capture = cv2.VideoCapture(str(video_path))
        require(capture.isOpened(), f"VIDEO_REOPEN_FAILED:{video_path}")
        try:
            frame_index = 0
            while frame_index <= max(wanted):
                ok, frame = capture.read()
                require(ok and frame is not None, f"FRAME_DECODE_FAILED:{row['video']}:{frame_index}")
                if frame_index in wanted_set:
                    frames_by_index[frame_index] = ocr_frame(engine, frame, frame_index)
                frame_index += 1
        finally:
            capture.release()
        require(set(frames_by_index) == wanted_set, f"SAMPLED_FRAME_SET_MISMATCH:{row['video']}")
        frames = [frames_by_index[frame_index] for frame_index in wanted]
        payload["videos"][row["video"]] = {
            "path": str(video_path.resolve()),
            "declared_frame_count": declared_frame_count,
            "frame_count": frame_count,
            "fps": fps,
            "duration_s": duration_s,
            "baseline_frame": baseline,
            "action_frames": action_frames,
            "frames": frames,
        }
        print(json.dumps({"ocr_video": position, "total": len(selected), "video": row["video"]}), flush=True)
    payload["ocr_wall_s"] = round(time.perf_counter() - started, 3)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return payload


def exact_match(target: str, form: str) -> bool:
    return target == form or target in form


def match_goal(
    frames: list[dict[str, Any]],
    target: str,
    algorithm: dict[str, Any],
    allow_edit_consensus: bool,
) -> dict[str, Any]:
    exact = []
    edits = []
    for frame in frames:
        for detection in frame["detections"]:
            if detection["confidence"] < algorithm["minimum_ocr_confidence"]:
                continue
            for form in detection["forms"]:
                if exact_match(target, form):
                    exact.append({"kind": "EXACT", "target": target, "form": form, **detection})
                elif (
                    allow_edit_consensus
                    and len(form) >= algorithm["minimum_edit_token_length"]
                    and abs(len(form) - len(target)) <= 1
                    and edit_distance(target, form) == 1
                ):
                    edits.append({"kind": "EDIT_ONE", "target": target, "form": form, **detection})
    if exact:
        selected = max(exact, key=lambda row: (row["confidence"], len(row["form"])))
        return {
            "accepted": True,
            "match": selected,
            "evidence_frames": sorted({row["frame_index"] for row in exact}),
        }
    edit_frames = {row["frame_index"] for row in edits}
    if allow_edit_consensus and len(edit_frames) >= algorithm["minimum_edit_evidence_frames"]:
        selected = max(edits, key=lambda row: row["confidence"])
        return {"accepted": True, "match": selected, "evidence_frames": sorted(edit_frames)}
    return {"accepted": False, "match": None, "evidence_frames": []}


def synthetic_token(video: str, salt: str) -> str:
    digest = hashlib.sha256(f"{video}|{salt}".encode("utf-8")).hexdigest()[:12]
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
    for name, expected in protocol["input_sha256"].items():
        require(sha256(inputs[name]) == expected, f"INPUT_HASH_MISMATCH:{name}")
    for name, expected in protocol["model_sha256"].items():
        require(sha256(inputs["models"] / name) == expected, f"MODEL_HASH_MISMATCH:{name}")
    reference_rows = read_annotations(inputs["reference_annotations"])
    evaluation_rows = read_annotations(inputs["evaluation_annotations"])
    selected = deterministic_panel(reference_rows, evaluation_rows, protocol["selection_contract"])
    require(
        [row["video"] for row in selected] == protocol["selected_videos"],
        "DETERMINISTIC_SELECTION_MISMATCH",
    )
    require(len(selected) == protocol["selection_contract"]["cohort_size"], "BAD_SELECTION_SIZE")
    require(
        Counter(row["action"] for row in selected)
        == Counter({action: protocol["selection_contract"]["episodes_per_action"] for action in ACTIONS}),
        "ACTION_BALANCE_MISMATCH",
    )
    for row in selected:
        video_path = inputs["videos_root"] / row["video"]
        require(video_path.is_file(), f"MISSING_VIDEO:{row['video']}")
        require(sha256(video_path) == protocol["selected_video_sha256"][row["video"]], f"VIDEO_HASH_MISMATCH:{row['video']}")
    cache_path = inputs["ocr_cache"]
    cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else build_cache(selected, inputs["videos_root"], inputs["models"], cache_path, protocol["algorithm_contract"])
    )
    require(set(cache["videos"]) == set(protocol["selected_videos"]), "CACHE_SELECTION_MISMATCH")

    unique_targets = list(dict.fromkeys(row["target_token"] for row in selected))
    counts = {"baseline_first_frame_exact": Counter(), "controlled_action_accumulation": Counter()}
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    challenge_accepts = []
    synthetic_accepts = []
    result_rows = []
    first_recovery_steps = []
    for row in selected:
        example = cache["videos"][row["video"]]
        baseline_frames = [example["frames"][0]]
        all_frames = example["frames"]
        baseline = match_goal(baseline_frames, row["target_token"], protocol["algorithm_contract"], False)
        successor = match_goal(all_frames, row["target_token"], protocol["algorithm_contract"], True)
        baseline_state = "CORRECT" if baseline["accepted"] else "UNKNOWN"
        successor_state = "CORRECT" if successor["accepted"] else "UNKNOWN"
        counts["baseline_first_frame_exact"][baseline_state] += 1
        counts["controlled_action_accumulation"][successor_state] += 1
        by_action[row["action"]][f"baseline_{baseline_state.lower()}"] += 1
        by_action[row["action"]][f"successor_{successor_state.lower()}"] += 1

        first_acceptance_step = None
        for step in range(1, len(all_frames) + 1):
            prefix = match_goal(all_frames[:step], row["target_token"], protocol["algorithm_contract"], True)
            if prefix["accepted"]:
                first_acceptance_step = step - 1
                break
        recovered = not baseline["accepted"] and successor["accepted"]
        if recovered and first_acceptance_step is not None:
            first_recovery_steps.append(first_acceptance_step)
            by_action[row["action"]]["recovered"] += 1

        challenge = next(target for target in unique_targets if target != row["target_token"] and target not in normalize(row["label"]))
        challenge_result = match_goal(all_frames, challenge, protocol["algorithm_contract"], True)
        if challenge_result["accepted"]:
            challenge_accepts.append({"video": row["video"], "challenge": challenge, "match": challenge_result})
        canary = synthetic_token(row["video"], protocol["synthetic_negative_salt"])
        synthetic_result = match_goal(all_frames, canary, protocol["algorithm_contract"], True)
        if synthetic_result["accepted"]:
            synthetic_accepts.append({"video": row["video"], "token": canary, "match": synthetic_result})
        result_rows.append(
            {
                **row,
                "video_sha256": protocol["selected_video_sha256"][row["video"]],
                "baseline_first_frame_exact": baseline,
                "controlled_action_accumulation": {
                    **successor,
                    "first_acceptance_action_step": first_acceptance_step,
                    "decision": (
                        "ALREADY_EVIDENCED_HOLD"
                        if baseline["accepted"]
                        else f"RECOVERED_AFTER_{row['action']}"
                        if successor["accepted"]
                        else f"UNKNOWN_AFTER_{row['action']}"
                    ),
                },
                "canonical_label_disjoint_challenge": {"target": challenge, "accepted": challenge_result["accepted"], "match": challenge_result},
                "synthetic_negative": {"target": canary, "accepted": synthetic_result["accepted"]},
                "sampled_frames": {
                    "baseline": example["baseline_frame"],
                    "action": example["action_frames"],
                    "fps": example["fps"],
                    "duration_s": example["duration_s"],
                },
            }
        )

    cohort_size = len(result_rows)
    baseline_correct = counts["baseline_first_frame_exact"]["CORRECT"]
    successor_correct = counts["controlled_action_accumulation"]["CORRECT"]
    metrics = {
        "episodes": cohort_size,
        "unique_verified_labels": len({row["label"] for row in selected}),
        "unique_train_label_disjoint_target_tokens": len(set(unique_targets)),
        "baseline_first_frame_exact_correct_unknown": [baseline_correct, cohort_size - baseline_correct],
        "controlled_action_accumulation_correct_unknown": [successor_correct, cohort_size - successor_correct],
        "correct_gain": successor_correct - baseline_correct,
        "correct_rate_gain_pp": round((successor_correct - baseline_correct) / cohort_size * 100.0, 3),
        "mean_action_steps_to_recovery": round(sum(first_recovery_steps) / len(first_recovery_steps), 3) if first_recovery_steps else None,
        "by_action": {action: dict(by_action[action]) for action in ACTIONS},
        "canonical_label_disjoint_challenges": cohort_size,
        "canonical_label_disjoint_challenge_accepts": len(challenge_accepts),
        "synthetic_negative_queries": cohort_size,
        "synthetic_negative_accepts": len(synthetic_accepts),
        "identity_bindings_emitted": 0,
        "portal_bindings_emitted": 0,
    }
    gate = {
        "cohort_size_met": cohort_size == protocol["selection_contract"]["cohort_size"],
        "minimum_unique_verified_labels_met": metrics["unique_verified_labels"] >= protocol["gate"]["minimum_unique_verified_labels"],
        "balanced_action_classes_met": all(sum(by_action[action].values()) > 0 for action in ACTIONS),
        "minimum_correct_gain_met": metrics["correct_gain"] >= protocol["gate"]["minimum_correct_gain"],
        "minimum_successor_correct_rate_met": successor_correct / cohort_size >= protocol["gate"]["minimum_successor_correct_rate"],
        "canonical_label_disjoint_challenge_limit_met": len(challenge_accepts) <= protocol["gate"]["maximum_canonical_label_disjoint_challenge_accepts"],
        "zero_synthetic_negative_accepts": len(synthetic_accepts) == 0,
        "zero_identity_or_portal_bindings": True,
    }
    gate["passed"] = all(gate.values())
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

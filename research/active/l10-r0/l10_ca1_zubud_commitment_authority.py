#!/usr/bin/env python3
"""Run the frozen L10-CA1 unary commitment gate on ZuBuD exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROTOCOL_SCHEMA = "blindassist-l10-ca1-zubud-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-ca1-zubud-result-v1"
SCALES = (1.0, 0.5, 0.25)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def center_crop(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * scale))
    crop_height = max(1, round(height * scale))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def parse_truth(path: Path) -> dict[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines and lines[0].split() == ["TEST", "TRAIN"], "GROUNDTRUTH_HEADER")
    truth: dict[int, int] = {}
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split()
        require(len(fields) == 2, f"GROUNDTRUTH_FIELDS:{line_number}")
        query_id, building_id = map(int, fields)
        require(query_id not in truth, f"DUPLICATE_QUERY:{query_id}")
        truth[query_id] = building_id
    require(sorted(truth) == list(range(1, 116)), "QUERY_IDS")
    require(all(1 <= value <= 201 for value in truth.values()), "BUILDING_IDS")
    return truth


def validate_source(protocol: dict[str, Any], dataset_root: Path) -> None:
    for key in ("database_archive", "query_archive", "groundtruth"):
        frozen = protocol["source_snapshot"][key]
        path = dataset_root / frozen["filename"]
        require(path.is_file(), f"MISSING_SOURCE:{path}")
        require(path.stat().st_size == int(frozen["bytes"]), f"SOURCE_SIZE:{path.name}")
        require(sha256(path) == frozen["sha256"], f"SOURCE_SHA256:{path.name}")


def source_paths(dataset_root: Path) -> tuple[list[Path], list[int], list[Path]]:
    extracted = dataset_root / "extracted"
    query_root = extracted / "1000city" / "qimage"
    database_root = extracted / "png-ZuBuD"
    queries = [query_root / f"qimg{query_id:04d}.JPG" for query_id in range(1, 116)]
    building_ids = list(range(1, 202))
    references = [
        database_root / f"object{building_id:04d}.view{view:02d}.png"
        for building_id in building_ids
        for view in range(1, 6)
    ]
    missing = [str(path) for path in queries + references if not path.is_file()]
    require(not missing, f"MISSING_EXTRACTED:{missing[:3]}")
    require(len(queries) == 115 and len(references) == 1005, "IMAGE_COUNTS")
    return queries, building_ids, references


def encode(
    paths: list[Path], model_root: Path, batch_size: int, device: str
) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True).eval().to(device)
    batches: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        crops: list[Image.Image] = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as source:
                image = source.convert("RGB")
            crops.extend(center_crop(image, scale) for scale in SCALES)
        inputs = processor(images=crops, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            values = model(**inputs).last_hidden_state[:, 0]
            values = torch.nn.functional.normalize(values.float(), dim=1)
        batches.append(
            values.cpu().numpy().astype(np.float32).reshape(len(crops) // 3, 3, -1)
        )
    return np.concatenate(batches, axis=0)


def save_cache(path: Path, identity: str, queries: np.ndarray, references: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez(
                handle,
                identity=np.asarray(identity),
                queries=queries,
                references=references,
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_or_encode(
    cache_path: Path,
    identity: str,
    query_paths: list[Path],
    reference_paths: list[Path],
    model_root: Path,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file():
        with np.load(cache_path) as cached:
            require(str(cached["identity"]) == identity, "CACHE_IDENTITY")
            return cached["queries"], cached["references"]
    queries = encode(query_paths, model_root, batch_size, device)
    references = encode(reference_paths, model_root, batch_size, device)
    save_cache(cache_path, identity, queries, references)
    return queries, references


def unary_score(left: np.ndarray, right: np.ndarray) -> float:
    matrix = left @ right.T
    baseline = float(matrix[0, 0])
    local = np.concatenate((matrix[1:, :].ravel(), matrix[:, 1:].ravel()))
    return float(0.25 * baseline + 0.75 * local.max())


def arm_metrics(correct: np.ndarray, committed: np.ndarray) -> dict[str, Any]:
    correct_commits = int(np.logical_and(correct, committed).sum())
    wrong_commits = int(np.logical_and(~correct, committed).sum())
    unknown = int((~committed).sum())
    total_commits = correct_commits + wrong_commits
    return {
        "correct": correct_commits,
        "wrong": wrong_commits,
        "unknown": unknown,
        "commit_precision": round(correct_commits / total_commits, 6) if total_commits else None,
        "coverage": round(total_commits / len(correct), 6),
    }


def run(
    protocol_path: Path,
    dataset_root: Path,
    model_root: Path,
    result_path: Path,
    cache_path: Path,
    batch_size: int,
) -> None:
    import torch

    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    validate_source(protocol, dataset_root)
    truth = parse_truth(dataset_root / "zubud_groundtruth.txt")
    query_paths, building_ids, reference_paths = source_paths(dataset_root)
    model_hash = sha256(model_root / "model.safetensors")
    require(
        model_hash == protocol["frozen_commitment_witness"]["model_safetensors_sha256"],
        "MODEL_SHA256",
    )
    require(torch.cuda.is_available(), "CUDA_REQUIRED_BY_FROZEN_RUNTIME")
    device = "cuda"
    cache_identity = hashlib.sha256(
        (
            sha256(protocol_path)
            + model_hash
            + protocol["source_snapshot"]["database_archive"]["sha256"]
            + protocol["source_snapshot"]["query_archive"]["sha256"]
            + "|cuda-fp32|1.0,0.5,0.25"
        ).encode("utf-8")
    ).hexdigest()
    query_embeddings, reference_embeddings = load_or_encode(
        cache_path,
        cache_identity,
        query_paths,
        reference_paths,
        model_root,
        batch_size,
        device,
    )
    require(query_embeddings.shape[:2] == (115, 3), "QUERY_EMBEDDINGS")
    require(reference_embeddings.shape[:2] == (1005, 3), "REFERENCE_EMBEDDINGS")

    reference_embeddings = reference_embeddings.reshape(201, 5, 3, -1)
    global_scores = query_embeddings[:, 0] @ reference_embeddings[:, :, 0].reshape(1005, -1).T
    identity_scores = global_scores.reshape(115, 201, 5).max(axis=2)
    proposed_indices = identity_scores.argmax(axis=1)
    proposed_ids = np.asarray([building_ids[index] for index in proposed_indices])
    truth_ids = np.asarray([truth[query_id] for query_id in range(1, 116)])
    correct = proposed_ids == truth_ids

    canonical = reference_embeddings[:, 0]
    verifier_scores = np.asarray(
        [
            unary_score(query_embeddings[index], canonical[proposed_indices[index]])
            for index in range(115)
        ],
        dtype=np.float64,
    )
    threshold = float(protocol["frozen_commitment_witness"]["commit_threshold"])
    committed = verifier_scores >= threshold
    baseline = arm_metrics(correct, np.ones(115, dtype=bool))
    gated = arm_metrics(correct, committed)
    correct_retention = gated["correct"] / baseline["correct"] if baseline["correct"] else 0.0
    wrong_reduction = (baseline["wrong"] - gated["wrong"]) / baseline["wrong"] if baseline["wrong"] else 0.0
    precision_delta = gated["commit_precision"] - baseline["commit_precision"]
    gate_met = correct_retention >= 0.80 and wrong_reduction >= 0.50 and precision_delta >= 0.10

    rows = []
    for index in range(115):
        rows.append(
            {
                "query_id": index + 1,
                "truth_building_id": int(truth_ids[index]),
                "proposed_building_id": int(proposed_ids[index]),
                "proposal_correct": bool(correct[index]),
                "unary_score": round(float(verifier_scores[index]), 6),
                "ca1_decision": "COMMIT" if committed[index] else "UNKNOWN",
            }
        )

    result = {
        "schema": RESULT_SCHEMA,
        "authority": "SOURCE_DISJOINT_REAL_IMAGE_COMMITMENT_CANARY_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "source": {
            "provider": protocol["provider"]["name"],
            "queries": 115,
            "candidate_identities": 201,
            "database_views": 1005,
            "groundtruth_sha256": protocol["source_snapshot"]["groundtruth"]["sha256"],
        },
        "model": {
            "path": str(model_root.resolve()),
            "model_safetensors_sha256": model_hash,
            "device": torch.cuda.get_device_name(0),
            "dtype": "float32",
        },
        "frozen_threshold": threshold,
        "arms": {"commit_all": baseline, "ca1_unary_gate": gated},
        "deltas": {
            "correct_commit_retention": round(correct_retention, 6),
            "wrong_commit_reduction": round(wrong_reduction, 6),
            "commit_precision": round(precision_delta, 6),
            "coverage": round(gated["coverage"] - baseline["coverage"], 6),
        },
        "gate_met": gate_met,
        "conclusion": (
            "L10_CA1_C1_KEEP_FOR_NEW_EPISODE_INTEGRATION"
            if gate_met
            else "L10_CA1_C1_REJECT_CROSS_PROVIDER_COMMITMENT_WITNESS"
        ),
        "decision_gate": protocol["evaluation"]["decision_gate"],
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_path, result)


def self_check() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]], dtype=np.float32)
    right = np.asarray([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    matrix = left @ right.T
    expected = 0.25 * matrix[0, 0] + 0.75 * max(matrix[1:, :].max(), matrix[:, 1:].max())
    require(abs(unary_score(left, right) - expected) < 1e-7, "UNARY_FORMULA")
    correct = np.asarray([True, True, False, False])
    committed = np.asarray([True, False, False, True])
    require(arm_metrics(correct, committed)["wrong"] == 1, "ARM_WRONG")
    print("SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("self-check")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--dataset-root", type=Path, required=True)
    run_parser.add_argument("--model-root", type=Path, required=True)
    run_parser.add_argument("--result", type=Path, required=True)
    run_parser.add_argument("--cache", type=Path, required=True)
    run_parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.action == "self-check":
        self_check()
    else:
        run(
            args.protocol,
            args.dataset_root,
            args.model_root,
            args.result,
            args.cache,
            args.batch_size,
        )


if __name__ == "__main__":
    main()

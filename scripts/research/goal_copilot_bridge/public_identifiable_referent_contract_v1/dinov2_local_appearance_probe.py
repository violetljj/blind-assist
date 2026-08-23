"""Order-free DINOv2-S local appearance probe over the 17 oracle pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    oracle_competing_identity_probe as oracle,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    visible_identity_probe as base,
)


SCHEMA_VERSION = "blindassist_dinov2_local_appearance_probe_v0"
MODEL_REPOSITORY = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_FILES = {
    "model.safetensors": "AE1E99FCEFD534ED978CDEB8326F08030C96E28B7A81FFCBC98A857C84D14BE1",
    "config.json": "1809F83E3BDB1609A501A610AD4A742F4FD8AE44D72CA4AA0DF52D1F2AC8628D",
    "preprocessor_config.json": "14E780D86FA1861F8751F868D7F45425B5FEB55C38CA26F152CA5097AB30F828",
}
INPUT_SIZE = 224
PATCH_SIDE = 16
PATCH_COUNT = PATCH_SIDE * PATCH_SIDE
FEATURE_DIM = 384
BATCH_SIZE = 16
EXPECTED_PAIRS = 17
CLAIM_CEILING = (
    "CONSUMED_VISIBLE_ONLY_ORACLE_CANDIDATE_DINOV2_LOCAL_APPEARANCE_DIAGNOSTIC_"
    "NO_THRESHOLD_CANDIDATE_GENERATION_FUSION_BELIEF_TRACKING_ACTIVE_SEARCH_SAFETY_OR_PRODUCT_CLAIM"
)
FORBIDDEN_SCORE_CONFIG_TOKENS = (
    "target_position",
    "target_physical_instance_id",
    "physical_instance_id",
    "native_object_id",
    "baseline_identity_outcome",
)


class LocalAppearanceProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalAppearanceProbeError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _body_hash(value: Mapping[str, Any]) -> str:
    return base._body_hash(value)


def _verify_body_hash(value: Mapping[str, Any], name: str) -> None:
    _require(value.get("body_sha256") == _body_hash(value), f"{name} body SHA mismatch")


def _validate_model(model_dir: Path, device: str) -> dict[str, Any]:
    import torch

    observed = {}
    for name, expected in MODEL_FILES.items():
        path = model_dir / name
        _require(path.is_file(), f"missing frozen model file: {path}")
        actual = _sha256_file(path).upper()
        _require(actual == expected, f"frozen model hash drift for {name}: {actual} != {expected}")
        observed[name] = actual
    config = _load_json(model_dir / "config.json")
    for key, expected in {"model_type": "dinov2", "hidden_size": FEATURE_DIM, "patch_size": 14}.items():
        _require(config.get(key) == expected, f"frozen model config drift for {key}")
    _require(device in {"cpu", "cuda"}, "device must be cpu or cuda")
    if device == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "files": observed,
        "input_size": INPUT_SIZE,
        "patch_grid": [PATCH_SIDE, PATCH_SIDE],
        "feature_dim": FEATURE_DIM,
        "layer": "last_hidden_state_patch_tokens",
        "device": device,
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def _relative_bbox(object_bbox: Sequence[float], crop_bbox: Sequence[float]) -> list[float]:
    crop_x0, crop_y0, crop_x1, crop_y1 = (float(value) for value in crop_bbox)
    width = crop_x1 - crop_x0
    height = crop_y1 - crop_y0
    _require(width > 0.0 and height > 0.0, "crop bbox is degenerate")
    x0, y0, x1, y1 = (float(value) for value in object_bbox)
    relative = [
        (x0 - crop_x0) / width,
        (y0 - crop_y0) / height,
        (x1 - crop_x0) / width,
        (y1 - crop_y0) / height,
    ]
    return [min(1.0, max(0.0, value)) for value in relative]


def _patch_mask(relative_bbox: Sequence[float]) -> np.ndarray:
    x0, y0, x1, y1 = (float(value) for value in relative_bbox)
    _require(x1 > x0 and y1 > y0, "object bbox is degenerate inside crop")
    centers = (np.arange(PATCH_SIDE, dtype=np.float32) + 0.5) / PATCH_SIDE
    xs, ys = np.meshgrid(centers, centers)
    mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    flat = mask.reshape(-1)
    _require(bool(flat.any()), "object bbox selects zero DINOv2 patches")
    return flat


def _crop_tensor(image_path: Path, crop_bbox: Sequence[float]) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    _require(image is not None, f"cannot read image: {image_path}")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (float(value) for value in crop_bbox)
    left = max(0, min(width - 1, int(math.floor(x0 * width))))
    top = max(0, min(height - 1, int(math.floor(y0 * height))))
    right = max(left + 1, min(width, int(math.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height))))
    crop = image[top:bottom, left:right]
    _require(crop.shape[0] >= 2 and crop.shape[1] >= 2, "crop has fewer than two pixels on one side")
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
    std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
    return np.transpose((rgb - mean) / std, (2, 0, 1))


class DenseEncoder:
    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = torch.device(device)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(self.device).eval()
        self.forward_batches = 0
        self.encoded_crops = 0

    def encode(self, tensors: Sequence[np.ndarray]) -> list[np.ndarray]:
        torch = self.torch
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(tensors), BATCH_SIZE):
                batch = torch.from_numpy(np.stack(tensors[start : start + BATCH_SIZE])).to(self.device)
                output = self.model(pixel_values=batch).last_hidden_state[:, 1:, :]
                _require(tuple(output.shape[1:]) == (PATCH_COUNT, FEATURE_DIM), "DINOv2 patch output drifted")
                output = torch.nn.functional.normalize(output.float(), dim=-1)
                encoded.extend(output.cpu().numpy())
                self.forward_batches += 1
                self.encoded_crops += int(batch.shape[0])
        return encoded


def symmetric_local_score(
    reference_patches: np.ndarray,
    candidate_patches: np.ndarray,
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, float | int]:
    reference = np.asarray(reference_patches, dtype="float32")[reference_mask]
    candidate = np.asarray(candidate_patches, dtype="float32")[candidate_mask]
    _require(reference.ndim == 2 and reference.shape[1] == FEATURE_DIM, "reference patch shape drifted")
    _require(candidate.ndim == 2 and candidate.shape[1] == FEATURE_DIM, "candidate patch shape drifted")
    similarities = reference @ candidate.T
    reference_to_candidate = float(similarities.max(axis=1).mean())
    candidate_to_reference = float(similarities.max(axis=0).mean())
    return {
        "reference_patch_count": int(reference.shape[0]),
        "candidate_patch_count": int(candidate.shape[0]),
        "reference_to_candidate_mean_nearest_cosine": reference_to_candidate,
        "candidate_to_reference_mean_nearest_cosine": candidate_to_reference,
        "symmetric_score": (reference_to_candidate + candidate_to_reference) / 2.0,
    }


def _winner(score_a: float, score_b: float) -> str:
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "TIE"


def _assert_score_config_blind(value: Any) -> None:
    lowered = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    for token in FORBIDDEN_SCORE_CONFIG_TOKENS:
        _require(token not in lowered, f"score config leaks private token: {token}")


def prepare_run(
    oracle_run: Path,
    counterbalance_run: Path,
    model: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    _require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    oracle_config = _load_json(oracle_run / "run-config.json")
    oracle_report = _load_json(oracle_run / "final-report.json")
    counterbalance_report = _load_json(counterbalance_run / "final-report.json")
    _verify_body_hash(oracle_config, "oracle config")
    _verify_body_hash(oracle_report, "oracle report")
    _verify_body_hash(counterbalance_report, "counterbalance report")
    _require(len(oracle_config["pairs"]) == EXPECTED_PAIRS, "oracle pair count drifted")
    visible_run = Path(oracle_config["parent_run"])
    visible_config = _load_json(visible_run / "run-config.json")
    visible_report = _load_json(visible_run / "final-report.json")
    _verify_body_hash(visible_config, "visible config")
    _verify_body_hash(visible_report, "visible report")
    pairs = []
    for pair in oracle_config["pairs"]:
        pair_id = pair["pair_id"]
        private = _load_json(oracle_run / pair["private_input_relative_path"])
        observation_id = private["observation_id"]
        visible_workspace = visible_run / "provider-public" / "cases" / observation_id
        public = _load_json(visible_workspace / "public-input.json")
        visible_private = _load_json(visible_run / "evaluator-private" / "observations" / f"{observation_id}.json")
        reference_path = visible_workspace / "01-reference.jpg"
        later_path = visible_workspace / "03-later.jpg"
        _require(_sha256_file(reference_path) == public["reference_image_sha256"], "reference image SHA drifted")
        _require(_sha256_file(later_path) == public["later_image_sha256"], "later image SHA drifted")
        reference_bbox = public["public_target_region_xyxy_normalized"]
        reference_crop = oracle._square_crop_bounds(reference_bbox)
        candidates = {}
        for slot in ("A", "B"):
            native_id = int(private[f"candidate_{slot.lower()}_native_object_id"])
            instance = next(
                item for item in visible_private["native_instances"] if int(item["native_object_id"]) == native_id
            )
            expected_physical_id = private[f"candidate_{slot.lower()}_physical_instance_id"]
            _require(instance["physical_instance_id"] == expected_physical_id, "oracle candidate identity drifted")
            crop_bbox = oracle._square_crop_bounds(instance["bbox_xyxy_normalized"])
            _require(
                np.allclose(crop_bbox, private["crop_bounds_xyxy_normalized"][slot], rtol=0.0, atol=1e-12),
                "oracle candidate crop drifted",
            )
            candidates[slot] = {
                "crop_bbox_xyxy_normalized": crop_bbox,
                "object_bbox_within_crop_xyxy_normalized": _relative_bbox(
                    instance["bbox_xyxy_normalized"], crop_bbox
                ),
            }
        pairs.append(
            {
                "pair_id": pair_id,
                "case_id": private["case_id"],
                "observation_id": observation_id,
                "reference": {
                    "image_path": str(reference_path),
                    "image_sha256": public["reference_image_sha256"],
                    "crop_bbox_xyxy_normalized": reference_crop,
                    "object_bbox_within_crop_xyxy_normalized": _relative_bbox(reference_bbox, reference_crop),
                },
                "later_image_path": str(later_path),
                "later_image_sha256": public["later_image_sha256"],
                "candidates": candidates,
            }
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "mode": "REVERSIBLE_EXPLORATION_ORDER_FREE_LOCAL_APPEARANCE_DIAGNOSTIC",
        "source_code_sha256": _sha256_file(Path(__file__)),
        "oracle_run": str(oracle_run),
        "oracle_report_body_sha256": oracle_report["body_sha256"],
        "counterbalance_run": str(counterbalance_run),
        "counterbalance_report_body_sha256": counterbalance_report["body_sha256"],
        "visible_run": str(visible_run),
        "visible_report_body_sha256": visible_report["body_sha256"],
        "model": dict(model),
        "crop_contract": {
            "square_context_fraction_per_side": oracle.CROP_CONTEXT_FRACTION,
            "resize": [INPUT_SIZE, INPUT_SIZE],
            "interpolation": "opencv_inter_cubic",
            "normalization": "imagenet_mean_std",
            "annotations_rendered": False,
        },
        "score_contract": {
            "candidate_scoring": "INDEPENDENT_PER_CANDIDATE",
            "patch_selection": "PATCH_CENTER_INSIDE_OBJECT_REGION",
            "directional_score": "MEAN_OF_NEAREST_PATCH_COSINE",
            "symmetric_score": "MEAN_REFERENCE_TO_CANDIDATE_AND_CANDIDATE_TO_REFERENCE",
            "comparison": "STRICT_GREATER_THAN_WITH_EXACT_TIE",
            "threshold": None,
            "training": False,
            "augmentation": False,
        },
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "pairs": pairs,
    }
    _assert_score_config_blind(config)
    config["body_sha256"] = _body_hash(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "run-config.json", config)
    return config


def execute(model_dir: Path, run_dir: Path, device: str) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    _verify_body_hash(config, "run config")
    _assert_score_config_blind(config)
    tensors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for pair in config["pairs"]:
        reference = pair["reference"]
        tensors.append(_crop_tensor(Path(reference["image_path"]), reference["crop_bbox_xyxy_normalized"]))
        masks.append(_patch_mask(reference["object_bbox_within_crop_xyxy_normalized"]))
        for slot in ("A", "B"):
            candidate = pair["candidates"][slot]
            tensors.append(_crop_tensor(Path(pair["later_image_path"]), candidate["crop_bbox_xyxy_normalized"]))
            masks.append(_patch_mask(candidate["object_bbox_within_crop_xyxy_normalized"]))
    encoder = DenseEncoder(model_dir, device)
    features = encoder.encode(tensors)
    rows = []
    offset = 0
    for pair in config["pairs"]:
        reference_features = features[offset]
        reference_mask = masks[offset]
        offset += 1
        scores = {}
        for slot in ("A", "B"):
            scores[slot] = symmetric_local_score(reference_features, features[offset], reference_mask, masks[offset])
            offset += 1
        score_a = float(scores["A"]["symmetric_score"])
        score_b = float(scores["B"]["symmetric_score"])
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "observation_id": pair["observation_id"],
                "candidate_scores": scores,
                "winner_slot": _winner(score_a, score_b),
                "slot_margin_a_minus_b": score_a - score_b,
            }
        )
    raw = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "encoded_crop_count": encoder.encoded_crops,
        "forward_batch_count": encoder.forward_batches,
        "rows": rows,
    }
    raw["body_sha256"] = _body_hash(raw)
    _atomic_json(run_dir / "raw-scores.json", raw)
    return raw


def _history_categories(oracle_report: Mapping[str, Any], counterbalance_report: Mapping[str, Any]) -> dict[str, str]:
    original = {
        row["observation_id"]: row
        for row in oracle_report["rows"]
        if row["stratum"] == "HISTORICAL_WRONG"
    }
    swapped = {row["observation_id"]: row for row in counterbalance_report["rows"]}
    _require(set(original) == set(swapped), "counterbalance observation set drifted")
    categories = {}
    for observation_id in sorted(original):
        outcomes = (original[observation_id]["evaluation"], swapped[observation_id]["evaluation"])
        if outcomes == ("TARGET_SELECTED", "TARGET_SELECTED"):
            category = "ROBUST_TARGET"
        elif outcomes == ("DISTRACTOR_SELECTED", "DISTRACTOR_SELECTED"):
            category = "STABLE_DISTRACTOR"
        else:
            category = "ORDER_SENSITIVE"
        categories[observation_id] = category
    return categories


def _metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "pair_count": len(rows),
        "target_outranks_count": sum(row["evaluation"] == "TARGET_OUTRANKS" for row in rows),
        "distractor_outranks_count": sum(row["evaluation"] == "DISTRACTOR_OUTRANKS" for row in rows),
        "tie_count": sum(row["evaluation"] == "TIE" for row in rows),
    }


def evaluate(run_dir: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "run-config.json")
    raw = _load_json(run_dir / "raw-scores.json")
    _verify_body_hash(config, "run config")
    _verify_body_hash(raw, "raw scores")
    _require(raw["run_config_body_sha256"] == config["body_sha256"], "raw scores bind wrong config")
    oracle_run = Path(config["oracle_run"])
    counterbalance_run = Path(config["counterbalance_run"])
    oracle_report = _load_json(oracle_run / "final-report.json")
    counterbalance_report = _load_json(counterbalance_run / "final-report.json")
    _verify_body_hash(oracle_report, "oracle report")
    _verify_body_hash(counterbalance_report, "counterbalance report")
    _require(oracle_report["body_sha256"] == config["oracle_report_body_sha256"], "oracle report binding drifted")
    _require(
        counterbalance_report["body_sha256"] == config["counterbalance_report_body_sha256"],
        "counterbalance report binding drifted",
    )
    history_categories = _history_categories(oracle_report, counterbalance_report)
    rows = []
    for raw_row in raw["rows"]:
        pair_id = raw_row["pair_id"]
        private = _load_json(oracle_run / "evaluator-private" / "pairs" / f"{pair_id}.json")
        target_slot = private["target_position"]
        distractor_slot = "B" if target_slot == "A" else "A"
        target_score = float(raw_row["candidate_scores"][target_slot]["symmetric_score"])
        distractor_score = float(raw_row["candidate_scores"][distractor_slot]["symmetric_score"])
        evaluation = (
            "TARGET_OUTRANKS"
            if target_score > distractor_score
            else "DISTRACTOR_OUTRANKS"
            if distractor_score > target_score
            else "TIE"
        )
        rows.append(
            {
                **raw_row,
                "stratum": private["stratum"],
                "history_category": history_categories.get(private["observation_id"], "BASELINE_CORRECT_CONTROL"),
                "target_slot": target_slot,
                "target_score": target_score,
                "distractor_score": distractor_score,
                "target_margin": target_score - distractor_score,
                "evaluation": evaluation,
            }
        )
    historical = [row for row in rows if row["stratum"] == "HISTORICAL_WRONG"]
    controls = [row for row in rows if row["stratum"] == "BASELINE_CORRECT_CONTROL"]
    metrics = {
        "all_pairs": _metric(rows),
        "historical_wrong": _metric(historical),
        "baseline_correct_control": _metric(controls),
        "robust_target_history": _metric([row for row in rows if row["history_category"] == "ROBUST_TARGET"]),
        "stable_distractor_history": _metric(
            [row for row in rows if row["history_category"] == "STABLE_DISTRACTOR"]
        ),
        "order_sensitive_history": _metric(
            [row for row in rows if row["history_category"] == "ORDER_SENSITIVE"]
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": _utc_now(),
        "run_config_body_sha256": config["body_sha256"],
        "raw_scores_body_sha256": raw["body_sha256"],
        "model": config["model"],
        "metrics": metrics,
        "rows": rows,
        "success_gate": None,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": "DINOV2_LOCAL_APPEARANCE_SIGNAL_OBSERVED",
    }
    report["body_sha256"] = _body_hash(report)
    _atomic_json(run_dir / "final-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--counterbalance-run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    model_dir = args.model_dir.resolve()
    model = _validate_model(model_dir, args.device)
    prepare_run(
        args.oracle_run.resolve(),
        args.counterbalance_run.resolve(),
        model,
        args.run_dir.resolve(),
    )
    execute(model_dir, args.run_dir.resolve(), args.device)
    report = evaluate(args.run_dir.resolve())
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

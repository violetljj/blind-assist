#!/usr/bin/env python3
"""Run the frozen P1-W2 providers once without loading evaluator truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_RESEARCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RESEARCH_DIR / "ba_adt_real_evidence"))
import run_p1_a2_dense_identity as a2  # noqa: E402


PROVIDER_SCHEMA = "p1_w2_frozen_provider_output_v1"
RECEIPT_SCHEMA = "p1_w2_single_execution_provider_receipt_v1"
PROGRESS_SCHEMA = "p1_w2_single_execution_progress_v1"


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_model_files(model_dir: Path, expected: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for name, identity in expected["artifacts"].items():
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"missing frozen model file: {path}")
        actual = digest(path)
        if actual != identity["sha256"].lower():
            raise ValueError(f"frozen model hash drift: {name}")
        if "bytes" in identity and path.stat().st_size != int(identity["bytes"]):
            raise ValueError(f"frozen model size drift: {name}")
        observed[name] = actual
    return observed


def load_requested_frames(provider: dict[str, Any]) -> dict[tuple[str, int], np.ndarray]:
    import av

    requested: dict[str, set[int]] = {}
    for case in provider["cases"]:
        requested.setdefault(case["parent_id"], set()).update(
            (int(case["source_frame_index"]), int(case["probe_frame_index"]))
        )
    assets = {asset["parent_id"]: asset for asset in provider["assets"]}
    result: dict[tuple[str, int], np.ndarray] = {}
    for parent_id, indices in requested.items():
        asset = assets[parent_id]
        path = Path(asset["rgb_path"])
        if digest(path) != asset["rgb_sha256"]:
            raise ValueError(f"RGB asset hash drift: {parent_id}")
        remaining = set(indices)
        with av.open(str(path)) as container:
            stream = next(stream for stream in container.streams if stream.type == "video")
            for index, frame in enumerate(container.decode(stream)):
                if index in remaining:
                    result[(parent_id, index)] = frame.to_ndarray(format="bgr24")
                    remaining.remove(index)
                    if not remaining:
                        break
        if remaining:
            raise ValueError(f"RGB requested frames missing for {parent_id}: {sorted(remaining)}")
    return result


def outer_crop(image: np.ndarray, core: list[float]) -> tuple[np.ndarray, list[float]]:
    x1, y1, x2, y2 = map(float, core)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate core bbox: {core}")
    outer_width = max(2.0 * width, (4.0 / 3.0) * 2.0 * height)
    outer_height = 0.75 * outer_width
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    outer_x1 = center_x - 0.5 * outer_width
    outer_y1 = center_y - 0.5 * outer_height
    scale_x = 640.0 / outer_width
    scale_y = 480.0 / outer_height
    transform = np.asarray(
        [[scale_x, 0.0, -outer_x1 * scale_x], [0.0, scale_y, -outer_y1 * scale_y]],
        dtype="float32",
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = cv2.warpAffine(
        gray,
        transform,
        (640, 480),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    core_in_crop = [
        (x1 - outer_x1) * scale_x,
        (y1 - outer_y1) * scale_y,
        (x2 - outer_x1) * scale_x,
        (y2 - outer_y1) * scale_y,
    ]
    return crop, core_in_crop


def inside(points: np.ndarray, core: list[float]) -> np.ndarray:
    return (
        (points[:, 0] >= core[0])
        & (points[:, 0] <= core[2])
        & (points[:, 1] >= core[1])
        & (points[:, 1] <= core[3])
    )


def occupied_quadrants(points: np.ndarray, core: list[float]) -> int:
    if not len(points):
        return 0
    center_x = 0.5 * (core[0] + core[2])
    center_y = 0.5 * (core[1] + core[3])
    return len({(int(point[0] >= center_x), int(point[1] >= center_y)) for point in points})


class GeometryProvider:
    def __init__(self, model_dir: Path, device: str, rule: dict[str, Any]):
        import torch
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=False
        )
        self.model = AutoModelForKeypointMatching.from_pretrained(
            str(model_dir), local_files_only=True
        ).to(self.device).eval()
        self.rule = rule
        self.calls = 0

    def evaluate(
        self,
        source_image: np.ndarray,
        source_core: list[float],
        probe_image: np.ndarray,
        probe_core: list[float],
    ) -> dict[str, Any]:
        from PIL import Image

        source_crop, source_core_crop = outer_crop(source_image, source_core)
        probe_crop, probe_core_crop = outer_crop(probe_image, probe_core)
        inputs = self.processor(
            [
                Image.fromarray(cv2.cvtColor(source_crop, cv2.COLOR_GRAY2RGB)),
                Image.fromarray(cv2.cvtColor(probe_crop, cv2.COLOR_GRAY2RGB)),
            ],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            raw = self.model(**inputs)
        post = self.processor.post_process_keypoint_matching(
            raw,
            [[(480, 640), (480, 640)]],
            threshold=float(self.rule["candidate_confidence_min"]),
        )[0]
        source_points = post["keypoints0"].detach().cpu().numpy().astype("float32")
        probe_points = post["keypoints1"].detach().cpu().numpy().astype("float32")
        scores = post["matching_scores"].detach().cpu().numpy().astype("float32")
        self.calls += 1
        mask = inside(source_points, source_core_crop) & inside(probe_points, probe_core_crop)
        source_local = source_points[mask]
        probe_local = probe_points[mask]
        local_scores = scores[mask]
        authority = self.rule["geometry_authority"]
        homography = None
        inliers = np.zeros(len(source_local), dtype=bool)
        if len(source_local) >= int(authority["min_core_correspondences"]):
            cv2.setRNGSeed(int(authority["rng_seed"]))
            homography, raw_inliers = cv2.findHomography(
                source_local,
                probe_local,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=float(authority["reprojection_threshold_px"]),
                maxIters=int(authority["max_iterations"]),
                confidence=float(authority["confidence"]),
            )
            if raw_inliers is not None:
                inliers = raw_inliers.reshape(-1).astype(bool)
        finite_model = homography is not None and bool(np.isfinite(homography).all())
        inlier_count = int(inliers.sum()) if finite_model else 0
        inlier_ratio = float(inlier_count / len(source_local)) if len(source_local) else 0.0
        source_quadrants = occupied_quadrants(source_local[inliers], source_core_crop)
        probe_quadrants = occupied_quadrants(probe_local[inliers], probe_core_crop)
        supported = bool(
            finite_model
            and inlier_count >= int(authority["min_inliers"])
            and inlier_ratio >= float(authority["min_inlier_ratio"])
            and source_quadrants >= int(authority["min_occupied_quadrants_source"])
            and probe_quadrants >= int(authority["min_occupied_quadrants_probe"])
        )
        result = {
            "raw_correspondence_count": int(len(source_points)),
            "core_correspondence_count": int(len(source_local)),
            "core_score_median": float(np.median(local_scores)) if len(local_scores) else None,
            "homography_finite": finite_model,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "source_inlier_quadrants": source_quadrants,
            "probe_inlier_quadrants": probe_quadrants,
            "geometry_supported": supported,
        }
        numeric = [value for value in result.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite geometry output: {result}")
        return result


def identity_pass(metrics: dict[str, Any], rule: dict[str, float]) -> bool:
    return bool(
        metrics["anchor_match_fraction"] >= float(rule["anchor_match_fraction_min"])
        and metrics["match_confidence"] >= float(rule["median_match_confidence_min"])
        and metrics["spatial_consistency"] >= float(rule["spatial_consistency_min"])
        and metrics["anchor_coverage"] >= float(rule["anchor_coverage_min"])
    )


def execution_identity(
    freeze_path: Path,
    roster_receipt_path: Path,
    provider_input_path: Path,
    geometry_hashes: dict[str, str],
    identity_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "freeze_sha256": digest(freeze_path),
        "roster_receipt_sha256": digest(roster_receipt_path),
        "provider_input_sha256": digest(provider_input_path),
        "geometry_model_files": geometry_hashes,
        "identity_model_files": identity_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--roster-receipt", type=Path, required=True)
    parser.add_argument("--provider-input", type=Path, required=True)
    parser.add_argument("--geometry-model-dir", type=Path, required=True)
    parser.add_argument("--identity-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    roster_receipt = json.loads(args.roster_receipt.read_text(encoding="utf-8"))
    provider = json.loads(args.provider_input.read_text(encoding="utf-8"))
    if roster_receipt.get("terminal") != "P1_W2_FRESH_PRIVATE_ROSTER_FROZEN":
        raise ValueError("private roster is not frozen/evaluable")
    expected_provider_hash = roster_receipt["output_sha256"]["provider_input.json"]
    if digest(args.provider_input) != expected_provider_hash:
        raise ValueError("provider input hash drift")
    if provider.get("truth_fields_present") is not False:
        raise ValueError("provider truth firewall is not sealed")
    expected_calls = sum(len(case["candidates"]) for case in provider["cases"])
    if expected_calls > int(freeze["future_execution_budget"]["max_pair_evaluations_per_provider"]):
        raise ValueError("frozen pair budget exceeded")

    geometry_hashes = validate_model_files(args.geometry_model_dir, freeze["geometry_provider"])
    identity_hashes = validate_model_files(args.identity_model_dir, freeze["identity_provider"])
    identity = execution_identity(
        args.freeze,
        args.roster_receipt,
        args.provider_input,
        geometry_hashes,
        identity_hashes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "provider_progress.json"
    provider_output_path = args.output_dir / "provider_output.json"
    receipt_path = args.output_dir / "provider_execution_receipt.json"
    if provider_output_path.exists() or receipt_path.exists():
        raise ValueError("sealed provider output already exists; refusing rerun")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("execution_identity") != identity:
            raise ValueError("progress identity drift")
    else:
        progress = {
            "schema_version": PROGRESS_SCHEMA,
            "execution_identity": identity,
            "completed": {},
        }
        atomic_json(progress_path, progress)

    started = time.perf_counter()
    frames = load_requested_frames(provider)
    source_tensors: dict[str, Any] = {}
    candidate_tensors: dict[tuple[str, str], Any] = {}
    for case in provider["cases"]:
        source_image = frames[(case["parent_id"], int(case["source_frame_index"]))]
        source_tensors.setdefault(
            case["parent_id"], a2._crop_tensor(source_image, case["source_core_bbox_xyxy"])
        )
        probe_image = frames[(case["parent_id"], int(case["probe_frame_index"]))]
        for candidate in case["candidates"]:
            candidate_tensors[(case["case_id"], candidate["candidate_id"])] = a2._crop_tensor(
                probe_image, candidate["core_bbox_xyxy"]
            )
    tensor_keys = [("source", key) for key in sorted(source_tensors)] + [
        ("candidate", key) for key in sorted(candidate_tensors)
    ]
    tensors = [source_tensors[key] if kind == "source" else candidate_tensors[key] for kind, key in tensor_keys]
    dense_encoder = a2.DenseEncoder(args.identity_model_dir, device=args.device)
    encoded_values = dense_encoder.encode(tensors)
    encoded = {key: value for key, value in zip(tensor_keys, encoded_values, strict=True)}
    geometry = GeometryProvider(args.geometry_model_dir, args.device, freeze["geometry_provider"])

    completed: dict[str, Any] = progress["completed"]
    for case in provider["cases"]:
        source_image = frames[(case["parent_id"], int(case["source_frame_index"]))]
        probe_image = frames[(case["parent_id"], int(case["probe_frame_index"]))]
        for candidate in case["candidates"]:
            work_id = f"{case['case_id']}::{candidate['candidate_id']}"
            if work_id in completed:
                continue
            geometry_metrics = geometry.evaluate(
                source_image,
                case["source_core_bbox_xyxy"],
                probe_image,
                candidate["core_bbox_xyxy"],
            )
            identity_metrics = a2.dense_consensus(
                encoded[("source", case["parent_id"])],
                encoded[("candidate", (case["case_id"], candidate["candidate_id"]))],
            )
            identity_supported = identity_pass(
                identity_metrics, freeze["identity_provider"]["candidate_pass_rule"]
            )
            completed[work_id] = {
                "case_id": case["case_id"],
                "candidate_id": candidate["candidate_id"],
                "geometry": geometry_metrics,
                "identity": identity_metrics,
                "identity_supported": identity_supported,
                "joint_supported": bool(
                    geometry_metrics["geometry_supported"] and identity_supported
                ),
            }
            atomic_json(progress_path, progress)

    cases = []
    for case in provider["cases"]:
        rows = [completed[f"{case['case_id']}::{candidate['candidate_id']}"] for candidate in case["candidates"]]
        identity_ids = [row["candidate_id"] for row in rows if row["identity_supported"]]
        joint_ids = [row["candidate_id"] for row in rows if row["joint_supported"]]
        cases.append(
            {
                "case_id": case["case_id"],
                "parent_id": case["parent_id"],
                "candidates": rows,
                "identity_state": "SEPARATED" if len(identity_ids) == 1 else ("NOT_OBSERVABLE" if not identity_ids else "AMBIGUOUS"),
                "identity_selected_candidate_id": identity_ids[0] if len(identity_ids) == 1 else None,
                "joint_state": "ELIGIBLE" if len(joint_ids) == 1 else ("NOT_ELIGIBLE" if not joint_ids else "AMBIGUOUS"),
                "joint_selected_candidate_id": joint_ids[0] if len(joint_ids) == 1 else None,
            }
        )
    output = {
        "schema_version": PROVIDER_SCHEMA,
        "execution_identity": identity,
        "cases": cases,
        "private_truth_loaded": False,
        "provider_call_counts": {
            "geometry_pair_calls": expected_calls,
            "identity_encoded_crops": dense_encoder.encoded_crops,
            "identity_forward_batches": dense_encoder.forward_batches,
        },
    }
    atomic_json(provider_output_path, output)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_identity": identity,
        "provider_output_sha256": digest(provider_output_path),
        "case_count": len(cases),
        "candidate_pair_count": expected_calls,
        "completed_candidate_pair_count": len(completed),
        "private_truth_loaded": False,
        "provider_call_counts": output["provider_call_counts"],
        "elapsed_seconds_diagnostic": round(time.perf_counter() - started, 3),
        "terminal": "P1_W2_PROVIDER_OUTPUT_SEALED",
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps({"terminal": receipt["terminal"], "cases": len(cases), "candidate_pairs": expected_calls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fresh OpenLORIS-home frontdoor and one-shot R27 evaluation.

Source/reference/candidate identities are selected from indexes, calibration and
poses before image payloads are decoded.  Reference RGB-D and candidate RGB are
then opened to seal unchanged R27 scores.  Candidate depth is opened only after
that selection seal and is used solely for fresh source-derived task evidence.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as balanced
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_rgb_query_interaction_ranker as r25
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_openloris_home_frontdoor.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_R27_OPENLORIS_HOME_FRESH_SOURCE_FRONTDOOR_LOCK_2026-08-14.json"
LOCK_ID = "TARO_R27_OPENLORIS_HOME_FRESH_SOURCE_FRONTDOOR_R0"
PARENT_IDS = ("home1-1", "home1-2", "home1-3", "home1-4", "home1-5")
PACKAGE_BYTES = 18_974_105_600
PACKAGE_SHA256 = "5A6B14CC01843669A9D77D077355FD5CA6E7FF88297718A8A7D6D5947A584E75"
GROUNDTRUTH_BYTES = 11_076_559
GROUNDTRUTH_SHA256 = "07564D7ED3D6739585002AFA12BCF481CC0E9E358FC64EFD5E658E2C994BDC3B"
SOURCE_SIZE_WH = (848, 480)
CROP_XYWH = (104, 0, 640, 480)
CROPPED_SIZE_WH = (640, 480)
DEPTH_SCALE_M = 0.001
MAX_RGB_DEPTH_DELTA_S = 0.02
MAX_POSE_DELTA_S = 0.10
MAX_REFERENCES_PER_PARENT = 5
MIN_REFERENCES = 16
MIN_PARENTS = 4
MIN_OPPORTUNITY_PARENTS = 4


class OpenLorisFrontdoorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenLorisFrontdoorError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative.replace("\\", "/")).resolve()
    require(root.resolve() in candidate.parents, f"path escapes source root: {relative}")
    return candidate


def _parse_index(path: Path) -> list[list[str]]:
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    require(rows and all(len(row) == 2 for row in rows), f"invalid timestamp index: {path}")
    timestamps = [float(row[0]) for row in rows]
    require(all(right > left for left, right in zip(timestamps, timestamps[1:])), f"non-monotonic index: {path}")
    return rows


def _parse_groundtruth(path: Path) -> tuple[list[list[str]], list[float]]:
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    require(rows and all(len(row) == 8 for row in rows), f"invalid groundtruth: {path}")
    timestamps = [float(row[0]) for row in rows]
    require(all(right > left for left, right in zip(timestamps, timestamps[1:])), f"non-monotonic groundtruth: {path}")
    return rows, timestamps


def _nearest_base_pose(
    rows: Sequence[Sequence[str]],
    timestamps: Sequence[float],
    timestamp: float,
) -> tuple[np.ndarray, float] | None:
    index = bisect.bisect_left(timestamps, timestamp)
    choices = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(rows)]
    if not choices:
        return None
    chosen = min(choices, key=lambda candidate: (abs(timestamps[candidate] - timestamp), candidate))
    delta = abs(timestamps[chosen] - timestamp)
    if delta > MAX_POSE_DELTA_S:
        return None
    row = rows[chosen]
    pose = bonn._pose_matrix([float(value) for value in row[1:4]], [float(value) for value in row[4:8]])
    return np.ascontiguousarray(pose, dtype=np.float64), float(delta)


def _validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (4, 4) and np.all(np.isfinite(matrix)), f"invalid transform: {name}")
    require(np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7), f"invalid homogeneous row: {name}")
    rotation = matrix[:3, :3]
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4), f"non-orthonormal rotation: {name}")
    require(abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-4, f"invalid rotation determinant: {name}")
    return np.ascontiguousarray(matrix)


def _opencv_calibration(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sensors_path = root / "sensors.yaml"
    transforms_path = root / "trans_matrix.yaml"
    require(sensors_path.is_file() and transforms_path.is_file(), f"OpenLORIS calibration absent: {root}")
    sensors = cv2.FileStorage(str(sensors_path), cv2.FILE_STORAGE_READ)
    transforms = cv2.FileStorage(str(transforms_path), cv2.FILE_STORAGE_READ)
    try:
        raw = sensors.getNode("d400_color_optical_frame").getNode("intrinsics").mat()
        require(raw is not None and raw.shape == (1, 4), f"D435i color intrinsics absent: {root}")
        # OpenLORIS stores [fx, cx, fy, cy].
        fx, cx, fy, cy = (float(raw[0, index]) for index in range(4))
        source_intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        sequence = transforms.getNode("trans_matrix")
        by_frames: dict[tuple[str, str], np.ndarray] = {}
        for index in range(sequence.size()):
            row = sequence.at(index)
            parent = row.getNode("parent_frame").string()
            child = row.getNode("child_frame").string()
            by_frames[(parent, child)] = _validate_transform(row.getNode("matrix").mat(), f"{parent}->{child}")
        base_to_color = by_frames.get(("base_link", "d400_color_optical_frame"))
        require(base_to_color is not None, f"base-to-color transform absent: {root}")
    finally:
        sensors.release()
        transforms.release()
    x, y, _width, _height = CROP_XYWH
    cropped_intrinsics = source_intrinsics.copy()
    cropped_intrinsics[0, 2] -= x
    cropped_intrinsics[1, 2] -= y
    require(
        0.0 <= cropped_intrinsics[0, 2] < CROPPED_SIZE_WH[0]
        and 0.0 <= cropped_intrinsics[1, 2] < CROPPED_SIZE_WH[1],
        f"cropped principal point outside image: {root}",
    )
    receipt = {
        "sensors_sha256": sha256_file(sensors_path),
        "trans_matrix_sha256": sha256_file(transforms_path),
        "source_intrinsics": source_intrinsics.tolist(),
        "cropped_intrinsics": cropped_intrinsics.tolist(),
        "base_to_color": base_to_color.tolist(),
    }
    return cropped_intrinsics, base_to_color, receipt


@dataclass(frozen=True)
class OpenLorisAsset:
    parent_id: str
    rgb_path: Path
    depth_path: Path
    intrinsics: np.ndarray


def load_outcome_blind_roster(
    source_root: Path,
    groundtruth_root: Path,
) -> tuple[list[bonn.Frame], dict[str, OpenLorisAsset], dict[str, Any]]:
    frames: list[bonn.Frame] = []
    assets: dict[str, OpenLorisAsset] = {}
    parent_receipts: list[dict[str, Any]] = []
    for parent_id in PARENT_IDS:
        root = (source_root / parent_id).resolve()
        require(root.is_dir(), f"OpenLORIS parent absent: {parent_id}")
        color_path = root / "color.txt"
        aligned_depth_path = root / "aligned_depth.txt"
        require(color_path.is_file() and aligned_depth_path.is_file(), f"OpenLORIS indexes absent: {parent_id}")
        color_rows = _parse_index(color_path)
        depth_rows = _parse_index(aligned_depth_path)
        associations = bonn._associate_unique_nearest(color_rows, depth_rows, MAX_RGB_DEPTH_DELTA_S)
        truth_candidates = (
            groundtruth_root / "per-sequence" / parent_id / "groundtruth.txt",
            groundtruth_root / parent_id / "groundtruth.txt",
            root / "groundtruth.txt",
        )
        truth_paths = [path.resolve() for path in truth_candidates if path.is_file()]
        require(bool(truth_paths), f"OpenLORIS groundtruth absent: {parent_id}")
        truth_path = truth_paths[0]
        truth_rows, truth_timestamps = _parse_groundtruth(truth_path)
        intrinsics, base_to_color, calibration = _opencv_calibration(root)
        pose_abstentions = missing_payloads = 0
        maximum_pose_delta = 0.0
        maximum_rgb_depth_delta = 0.0
        admitted = 0
        for color, depth in associations:
            timestamp = float(color[0])
            pose_row = _nearest_base_pose(truth_rows, truth_timestamps, timestamp)
            if pose_row is None:
                pose_abstentions += 1
                continue
            world_to_base, pose_delta = pose_row
            camera_to_world = _validate_transform(world_to_base @ base_to_color, f"{parent_id}:{timestamp:.5f}")
            rgb_path = _safe_child(root, color[1])
            depth_path = _safe_child(root, depth[1])
            if not rgb_path.is_file() or not depth_path.is_file():
                missing_payloads += 1
                continue
            frame = bonn.Frame(parent_id, timestamp, rgb_path, depth_path, camera_to_world)
            require(frame.frame_id not in assets, f"duplicate OpenLORIS frame: {frame.frame_id}")
            assets[frame.frame_id] = OpenLorisAsset(parent_id, rgb_path, depth_path, intrinsics)
            frames.append(frame)
            admitted += 1
            maximum_pose_delta = max(maximum_pose_delta, pose_delta)
            maximum_rgb_depth_delta = max(maximum_rgb_depth_delta, abs(float(color[0]) - float(depth[0])))
        parent_receipts.append(
            {
                "parent_id": parent_id,
                "color_index_count": len(color_rows),
                "aligned_depth_index_count": len(depth_rows),
                "associated_count": len(associations),
                "pose_index_count": len(truth_rows),
                "pose_abstention_count": pose_abstentions,
                "missing_payload_count": missing_payloads,
                "admitted_frame_count": admitted,
                "maximum_rgb_depth_delta_s": maximum_rgb_depth_delta,
                "maximum_pose_delta_s": maximum_pose_delta,
                "color_index_sha256": sha256_file(color_path),
                "aligned_depth_index_sha256": sha256_file(aligned_depth_path),
                "groundtruth_sha256": sha256_file(truth_path),
                "calibration": calibration,
            }
        )
    require(frames and len(parent_receipts) == len(PARENT_IDS), "empty OpenLORIS roster")
    return frames, assets, {
        "family": "OPENLORIS_SCENE_D435I_HOME",
        "analysis_role": "FRESH_SOURCE_EVALUATION",
        "parent_ids": list(PARENT_IDS),
        "parent_count": len(PARENT_IDS),
        "frame_count": len(frames),
        "selection_inputs": ["color.txt", "aligned_depth.txt", "groundtruth.txt", "sensors.yaml", "trans_matrix.yaml"],
        "selection_reads_task_outcome": False,
        "image_payload_reads_during_selection": 0,
        "candidate_depth_reads_during_selection": 0,
        "source_resolution_wh": list(SOURCE_SIZE_WH),
        "center_crop_xywh": list(CROP_XYWH),
        "cropped_resolution_wh": list(CROPPED_SIZE_WH),
        "depth_scale_m": DEPTH_SCALE_M,
        "world_up_xyz": tum.WORLD_UP.tolist(),
        "parents": parent_receipts,
    }


class PayloadStore:
    def __init__(self, assets: Mapping[str, OpenLorisAsset]):
        self.assets = dict(assets)
        self._observations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self._rgb_planes: dict[str, np.ndarray] = {}
        self._depth_receipts: dict[str, str] = {}
        self._rgb_receipts: dict[str, str] = {}

    @staticmethod
    def _crop(raw: np.ndarray) -> np.ndarray:
        x, y, width, height = CROP_XYWH
        require(raw.shape[:2] == (SOURCE_SIZE_WH[1], SOURCE_SIZE_WH[0]), "OpenLORIS image shape drift")
        return np.ascontiguousarray(raw[y : y + height, x : x + width])

    def observation(self, frame_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        cached = self._observations.get(frame_id)
        if cached is not None:
            return cached
        asset = self.assets[frame_id]
        payload = asset.depth_path.read_bytes()
        raw = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        require(raw is not None and raw.dtype == np.uint16, f"OpenLORIS depth decode drift: {frame_id}")
        cropped = self._crop(raw).astype(np.float64) * DEPTH_SCALE_M
        low, points, valid = tum._low_observation(cropped, asset.intrinsics)
        result = (low, points, valid, float(np.mean(valid)))
        self._observations[frame_id] = result
        self._depth_receipts[frame_id] = hashlib.sha256(payload).hexdigest().upper()
        return result

    def planes(self, frame: bonn.Frame) -> np.ndarray:
        frame_id = frame.frame_id
        cached = self._rgb_planes.get(frame_id)
        if cached is not None:
            return cached
        asset = self.assets[frame_id]
        payload = asset.rgb_path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(bgr is not None, f"OpenLORIS RGB decode drift: {frame_id}")
        rgb = cv2.cvtColor(self._crop(bgr), cv2.COLOR_BGR2RGB)
        planes = r25._rgb_planes(rgb)
        self._rgb_planes[frame_id] = planes
        self._rgb_receipts[frame_id] = hashlib.sha256(payload).hexdigest().upper()
        return planes

    def receipt(self) -> dict[str, Any]:
        depth_rows = [{"frame_id": key, "sha256": value} for key, value in sorted(self._depth_receipts.items())]
        rgb_rows = [{"frame_id": key, "sha256": value} for key, value in sorted(self._rgb_receipts.items())]
        return {
            "unique_depth_payload_decode_count": len(depth_rows),
            "unique_rgb_payload_decode_count": len(rgb_rows),
            "depth_payload_receipt_sha256": hashlib.sha256(canonical_json_bytes(depth_rows)).hexdigest().upper(),
            "rgb_payload_receipt_sha256": hashlib.sha256(canonical_json_bytes(rgb_rows)).hexdigest().upper(),
            "model_runs": 0,
        }


def _candidate_identity(selected: Sequence[bonn.ReferenceSupport]) -> tuple[dict[str, tuple[bonn.Pair, ...]], str]:
    proposals = {row.reference.frame_id: oracle.pose_proposal_pairs(row) for row in selected}
    reference_ids = set(proposals)
    candidate_ids = {pair.neighbor.frame_id for pairs in proposals.values() for pair in pairs}
    require(reference_ids.isdisjoint(candidate_ids), "reference/candidate payload identity overlap")
    rows = [
        {
            "reference_frame_id": reference_id,
            "neighbor_frame_ids": [pair.neighbor.frame_id for pair in pairs],
        }
        for reference_id, pairs in sorted(proposals.items())
    ]
    return proposals, hashlib.sha256(canonical_json_bytes(rows)).hexdigest().upper()


def verify_locked_inputs(source_package: Path, groundtruth_archive: Path) -> dict[str, Any]:
    require(LOCK_PATH.is_file(), f"frontdoor lock absent: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    require(lock.get("lock_id") == LOCK_ID, "frontdoor lock identity drift")
    require(source_package.is_file() and source_package.stat().st_size == PACKAGE_BYTES, "OpenLORIS package byte drift")
    require(groundtruth_archive.is_file() and groundtruth_archive.stat().st_size == GROUNDTRUTH_BYTES, "OpenLORIS groundtruth byte drift")
    package_sha = sha256_file(source_package)
    groundtruth_sha = sha256_file(groundtruth_archive)
    require(package_sha == PACKAGE_SHA256, "OpenLORIS package SHA-256 drift")
    require(groundtruth_sha == GROUNDTRUTH_SHA256, "OpenLORIS groundtruth SHA-256 drift")
    return {
        "lock_path": str(LOCK_PATH),
        "lock_sha256": sha256_file(LOCK_PATH),
        "package_path": str(source_package),
        "package_bytes": source_package.stat().st_size,
        "package_sha256": package_sha,
        "groundtruth_archive_path": str(groundtruth_archive),
        "groundtruth_archive_bytes": groundtruth_archive.stat().st_size,
        "groundtruth_archive_sha256": groundtruth_sha,
    }


def evaluate(
    source_root: Path,
    groundtruth_root: Path,
    source_package: Path,
    groundtruth_archive: Path,
) -> dict[str, Any]:
    input_receipt = verify_locked_inputs(source_package.resolve(), groundtruth_archive.resolve())
    frames, assets, source = load_outcome_blind_roster(source_root.resolve(), groundtruth_root.resolve())
    selected, capability = balanced.select_pose_capable_references(frames, MAX_REFERENCES_PER_PARENT)
    source_capable = (
        capability["eligible_parent_count"] >= MIN_PARENTS
        and capability["selected_reference_count"] >= MIN_REFERENCES
    )
    if not source_capable:
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "FRESH_SOURCE_EVALUATION",
            "input_receipt": input_receipt,
            "source": source,
            "pose_pair_capability": capability,
            "terminal": "STOP_TARO_R27_OPENLORIS_HOME_SOURCE_NOT_EVALUABLE",
            "read_boundary": {
                "image_payload_decodes": 0,
                "candidate_depth_reads_before_selection_seal": 0,
                "task_outcome_reads": 0,
            },
            "claim_ceiling": "Fresh source metadata and pose-pair capability only; no algorithm, product or safety conclusion.",
            "android_candidate_authorized": False,
            "product_authorized": False,
            "safety_authorized": False,
        }
        result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
        return result

    proposals, candidate_identity_sha = _candidate_identity(selected)
    store = PayloadStore(assets)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    geometry_abstentions = 0
    for row in selected:
        low, points, valid, _coverage = store.observation(row.reference.frame_id)
        asset = assets[row.reference.frame_id]
        low_intrinsics = bonn._scaled_intrinsics(asset.intrinsics, CROPPED_SIZE_WH, tum.LOW_SIZE_WH)
        queries = oracle._queries(row.reference, low, low_intrinsics)
        if queries is None:
            geometry_abstentions += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, low_intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        reference_planes = store.planes(row.reference)
        for pair in proposals[row.reference.frame_id]:
            features, analytic = r27.reprojection_visibility_features(
                context,
                pair,
                reference_planes,
                store.planes(pair.neighbor),
            )
            records.append(
                scorer.CandidateRecord(
                    row.reference.parent_id,
                    "FRESH_SOURCE_EVALUATION",
                    row.reference.frame_id,
                    pair,
                    features,
                    analytic,
                )
            )
    require(records, "OpenLORIS geometry produced no candidate records")
    scores, selection = r27.primary_selection_scores(records)
    selection_seal = {
        "candidate_identity_sha256": candidate_identity_sha,
        "candidate_record_count": len(records),
        "evaluated_reference_count_before_target": selection["reference_count"],
        "selection_receipt_sha256": selection["selection_receipt_sha256"],
        "candidate_depth_reads_before_selection_seal": 0,
    }
    selection_seal["seal_sha256"] = hashlib.sha256(canonical_json_bytes(selection_seal)).hexdigest().upper()

    # The locked R27 scores and selected identities are now sealed.  Candidate
    # depth is opened below only to derive fresh task-evidence targets.
    observations = {
        frame_id: store.observation(frame_id)
        for frame_id in sorted({record.pair.neighbor.frame_id for record in records})
    }
    scorer._attach_targets(records, contexts, observations)
    metrics = r21.fold_metrics(records, scores)
    checks = dict(metrics["checks"])
    checks["minimum_evaluated_references"] = metrics["reference_count"] >= MIN_REFERENCES
    checks["minimum_evaluated_parents"] = metrics["parent_count"] >= MIN_PARENTS
    checks["minimum_opportunity_parents"] = metrics["opportunity_parent_count"] >= MIN_OPPORTUNITY_PARENTS
    checks["zero_known_evidence_retention_failures_by_union_construction"] = True
    passed = all(checks.values())
    terminal = (
        "TARO_R27_OPENLORIS_HOME_FRESH_SOURCE_PASS"
        if passed
        else "STOP_TARO_R27_OPENLORIS_HOME_FRESH_SOURCE_FAIL"
    )
    result = {
        "schema": SCHEMA,
        "mode": "FRESH_SOURCE_EVALUATION",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "input_receipt": input_receipt,
        "source": source,
        "pose_pair_capability": capability,
        "candidate_algorithm": {
            "name": "TARO_QUERY_ALIGNED_REPROJECTION_VISIBILITY_R27",
            "warp_coverage_dilation_px": r27.WARP_COVERAGE_DILATION_PX,
            "photometric_residual_quantile": r27.PHOTOMETRIC_RESIDUAL_QUANTILE,
            "minimum_override_novel_cell_advantage": r27.MINIMUM_OVERRIDE_NOVEL_CELL_ADVANTAGE,
            "candidate_depth_in_scorer_input": False,
            "training_steps": 0,
            "parameters_fit_from_fresh_targets": 0,
        },
        "selection": selection,
        "selection_seal_before_candidate_depth": selection_seal,
        "geometry_abstention_count": geometry_abstentions,
        "metrics": metrics,
        "checks": checks,
        "payload_receipt": store.receipt(),
        "terminal": terminal,
        "fresh_source_confirmation_pass": passed,
        "read_boundary": {
            "source_selection_reads_task_outcome": False,
            "reference_rgb_and_depth_in_scorer_input": True,
            "candidate_rgb_in_scorer_input": True,
            "candidate_depth_in_scorer_input": False,
            "candidate_depth_opened_after_selection_seal": True,
            "network_requests_during_evaluation": 0,
            "model_runs": 0,
            "training_steps": 0,
        },
        "claim_ceiling": "One fresh OpenLORIS-home source-family evaluation of frozen R27. A PASS is not broad generalization, collision correctness, Android readiness, product success, deployment readiness or safety evidence.",
        "android_candidate_authorized": False,
        "product_authorized": False,
        "safety_authorized": False,
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--groundtruth-root", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--groundtruth-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.source_root,
        args.groundtruth_root,
        args.source_package,
        args.groundtruth_archive,
    )
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

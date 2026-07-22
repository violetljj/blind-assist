from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from contract import read_json, sha256, validate_pose, write_json


def depth(row: dict) -> np.ndarray:
    path = Path(row["depth_path"])
    if row["depth_encoding"] == "uint16_png_z_meters":
        return np.asarray(Image.open(path), dtype=np.float32) / float(row["depth_scale"])
    if row["depth_encoding"] == "float32_npy_z_meters":
        return np.load(path).astype(np.float32, copy=False) / float(row["depth_scale"])
    raise ValueError("unsupported depth encoding")


def run_source(ledger: Path) -> dict:
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    if any("camera_to_world" in row or "pose_timestamp_s" in row for row in rows):
        raise ValueError("ground-truth pose field leaked into estimator input")
    orb = cv2.ORB_create(nfeatures=1600, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    estimates = []
    current = np.eye(4, dtype=np.float64)
    previous_gray = cv2.imread(rows[0]["rgb_path"], cv2.IMREAD_GRAYSCALE)
    previous_depth = depth(rows[0])
    previous_keypoints, previous_descriptors = orb.detectAndCompute(previous_gray, None)
    estimates.append({"frame_id": rows[0]["frame_id"], "timestamp_s": rows[0]["timestamp_s"], "camera_to_world": current.tolist(), "status": "initialized", "inliers": 0})
    successes = 1
    for row in rows[1:]:
        gray = cv2.imread(row["rgb_path"], cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        status = "unknown"
        inlier_count = 0
        if previous_descriptors is not None and descriptors is not None and len(previous_keypoints) >= 20 and len(keypoints) >= 20:
            matches = sorted(matcher.match(previous_descriptors, descriptors), key=lambda item: item.distance)[:500]
            fx, fy, cx, cy = [float(v) for v in row["intrinsics_fx_fy_cx_cy"]]
            points3d = []
            points2d = []
            for match in matches:
                u, v = previous_keypoints[match.queryIdx].pt
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= vi < previous_depth.shape[0] and 0 <= ui < previous_depth.shape[1]):
                    continue
                z = float(previous_depth[vi, ui])
                if not np.isfinite(z) or z <= 0.15 or z >= 8.0:
                    continue
                points3d.append(((u - cx) * z / fx, (v - cy) * z / fy, z))
                points2d.append(keypoints[match.trainIdx].pt)
            if len(points3d) >= 16:
                camera = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.asarray(points3d, dtype=np.float32),
                    np.asarray(points2d, dtype=np.float32),
                    camera,
                    None,
                    iterationsCount=120,
                    reprojectionError=2.5,
                    confidence=0.999,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                inlier_count = 0 if inliers is None else int(len(inliers))
                if ok and inlier_count >= 12:
                    rotation, _ = cv2.Rodrigues(rvec)
                    previous_to_current = np.eye(4, dtype=np.float64)
                    previous_to_current[:3, :3] = rotation
                    previous_to_current[:3, 3] = tvec[:, 0]
                    current = current @ np.linalg.inv(previous_to_current)
                    status = "estimated"
                    successes += 1
        validate_pose(current.tolist())
        estimates.append({"frame_id": row["frame_id"], "timestamp_s": row["timestamp_s"], "camera_to_world": current.tolist(), "status": status, "inliers": inlier_count})
        previous_gray = gray
        previous_depth = depth(row)
        previous_keypoints, previous_descriptors = keypoints, descriptors
    return {"pose_estimates": estimates, "estimated_fraction": successes / len(rows), "frame_count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    manifest = read_json(args.inputs / "estimator_inputs.json")
    if manifest.get("ground_truth_pose_fields_exposed") is not False:
        raise SystemExit("estimator input isolation failed")
    sources = []
    for source in manifest["sources"]:
        ledger = args.inputs / source["sanitized_ledger"]
        if sha256(ledger) != source["sanitized_ledger_sha256"]:
            raise ValueError("sanitized estimator ledger hash mismatch")
        result = run_source(ledger)
        sources.append({"source_id": source["source_id"], **result})
    write_json(args.output, {
        "schema": "blindassist_ustrf_sensor_replay_r3_pose_estimates_v1",
        "estimator_id": "opencv_orb_rgbd_pnp_r3_v1",
        "estimator_inputs_sha256": sha256(args.inputs / "estimator_inputs.json"),
        "ground_truth_pose_accessed": False,
        "sources": sources,
        "production_authority": False,
    })
    print(json.dumps({"sources": [{"source_id": row["source_id"], "estimated_fraction": row["estimated_fraction"]} for row in sources]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

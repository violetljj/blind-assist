from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from contract import BUNDLE_SCHEMA, nearest_pose, parse_rows, quaternion_matrix, read_json, sha256, validate_pose, write_json


def _eth3d(root: Path, source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    associated = parse_rows(root / "associated.txt", 4)[:limit]
    poses = [(float(row[0]), [float(v) for v in row[1:]]) for row in parse_rows(root / "groundtruth.txt", 8)]
    intrinsics = [float(v) for v in (root / "calibration.txt").read_text(encoding="utf-8").split()]
    result = []
    for index, row in enumerate(associated):
        rgb_ts, rgb_path, depth_ts, depth_path = float(row[0]), row[1], float(row[2]), row[3]
        pose_ts, pose_values = nearest_pose(poses, rgb_ts)
        result.append(_frame(root, index, rgb_ts, depth_ts, pose_ts, rgb_path, depth_path, intrinsics, quaternion_matrix(pose_values), source, "uint16_png_z_meters"))
    return result


def _icl(root: Path, source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    associated = parse_rows(root / "associations.txt", 4)[:limit]
    pose_rows = {int(row[0]): [float(v) for v in row[1:]] for row in parse_rows(root / "livingRoom0.gt.freiburg", 8)}
    offset = int(source["pose_index_offset"])
    result = []
    for index, row in enumerate(associated):
        depth_id, depth_path, rgb_id, rgb_path = int(row[0]), row[1], int(row[2]), row[3]
        if depth_id != rgb_id or depth_id + offset not in pose_rows:
            raise ValueError(f"ICL frame binding failed at {depth_id}")
        timestamp = depth_id / 30.0
        result.append(_frame(root, index, timestamp, timestamp, timestamp, rgb_path, depth_path, source["intrinsics"], quaternion_matrix(pose_rows[depth_id + offset]), source, "uint16_png_z_meters"))
    return result


def _tartanair(root: Path, source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    ids = sorted(path.name.removesuffix("_rgb.png") for path in root.glob("*_rgb.png"))[:limit]
    result = []
    for index, frame_id in enumerate(ids):
        camera_path = root / f"{frame_id}_cam.npz"
        depth_path = root / f"{frame_id}_depth.npy"
        if not camera_path.is_file() or not depth_path.is_file():
            raise ValueError(f"TartanAir modality missing: {frame_id}")
        camera = np.load(camera_path)
        pose = np.asarray(camera["camera_pose"], dtype=np.float64)
        intrinsics = np.asarray(camera["camera_intrinsics"], dtype=np.float64)
        validate_pose(pose.tolist())
        timestamp = int(frame_id) / float(source["frame_rate_hz"])
        result.append(_frame(root, index, timestamp, timestamp, timestamp, f"{frame_id}_rgb.png", f"{frame_id}_depth.npy", [intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]], pose.tolist(), source, "float32_npy_z_meters", camera_path=f"{frame_id}_cam.npz"))
    return result


def _timestamp_rows(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        timestamp, relative = line.split()
        rows.append((float(timestamp), relative))
    if not rows:
        raise ValueError(f"empty timestamp table: {path}")
    return rows


def _associate_nearest(
    first: list[tuple[float, str]],
    second: list[tuple[float, str]],
    maximum_delta_s: float,
) -> list[tuple[float, str, float, str]]:
    """Greedy one-to-one association matching the public TUM benchmark tool."""
    candidates = sorted(
        (abs(a[0] - b[0]), ai, bi)
        for ai, a in enumerate(first)
        for bi, b in enumerate(second)
        if abs(a[0] - b[0]) <= maximum_delta_s
    )
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches = []
    for _, ai, bi in candidates:
        if ai in used_first or bi in used_second:
            continue
        used_first.add(ai)
        used_second.add(bi)
        matches.append((*first[ai], *second[bi]))
    return sorted(matches, key=lambda row: row[0])


def _tum_rgbd_dynamic(root: Path, source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rgb = _timestamp_rows(root / "rgb.txt")
    depth = _timestamp_rows(root / "depth.txt")
    maximum_delta_s = float(source.get("maximum_rgb_depth_delta_s", 0.02))
    associated = _associate_nearest(rgb, depth, maximum_delta_s)
    start = int(source.get("start_association_index", 0))
    if limit > 0:
        associated = associated[start:start + limit]
    else:
        associated = associated[start:]
    poses = [(float(row[0]), [float(v) for v in row[1:]]) for row in parse_rows(root / "groundtruth.txt", 8)]
    intrinsics = [float(v) for v in source["intrinsics"]]
    result = []
    for index, (rgb_ts, rgb_path, depth_ts, depth_path) in enumerate(associated):
        pose_ts, pose_values = nearest_pose(poses, rgb_ts)
        result.append(
            _frame(
                root,
                index,
                rgb_ts,
                depth_ts,
                pose_ts,
                rgb_path,
                depth_path,
                intrinsics,
                quaternion_matrix(pose_values),
                source,
                "uint16_png_z_meters",
            )
        )
    return result


def _opencv_calibration(root: Path) -> tuple[list[float], dict[tuple[str, str], np.ndarray]]:
    sensors = cv2.FileStorage(str(root / "sensors.yaml"), cv2.FILE_STORAGE_READ)
    transforms = cv2.FileStorage(str(root / "trans_matrix.yaml"), cv2.FILE_STORAGE_READ)
    try:
        raw = sensors.getNode("d400_color_optical_frame").getNode("intrinsics").mat()
        if raw is None or raw.shape != (1, 4):
            raise ValueError("OpenLORIS D435i color intrinsics missing")
        # OpenLORIS stores [fx, cx, fy, cy]; the replay contract uses [fx, fy, cx, cy].
        intrinsics = [float(raw[0, 0]), float(raw[0, 2]), float(raw[0, 1]), float(raw[0, 3])]
        sequence = transforms.getNode("trans_matrix")
        by_frames: dict[tuple[str, str], np.ndarray] = {}
        for index in range(sequence.size()):
            row = sequence.at(index)
            key = (row.getNode("parent_frame").string(), row.getNode("child_frame").string())
            matrix = np.asarray(row.getNode("matrix").mat(), dtype=np.float64)
            validate_pose(matrix.tolist())
            by_frames[key] = matrix
        return intrinsics, by_frames
    finally:
        sensors.release()
        transforms.release()


def _openloris_camera_pose(
    groundtruth_pose: np.ndarray,
    base_to_color: np.ndarray,
    base_to_groundtruth: np.ndarray | None,
    groundtruth_frame: str,
) -> np.ndarray:
    if groundtruth_frame == "base_link":
        world_to_base = groundtruth_pose
    elif groundtruth_frame == "marker":
        if base_to_groundtruth is None:
            raise ValueError("OpenLORIS marker ground truth requires base_link-to-marker calibration")
        world_to_base = groundtruth_pose @ np.linalg.inv(base_to_groundtruth)
    else:
        raise ValueError(f"unsupported OpenLORIS ground-truth frame: {groundtruth_frame}")
    camera_pose = world_to_base @ base_to_color
    validate_pose(camera_pose.tolist())
    return camera_pose


def _openloris_package(root: Path, source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    color = _timestamp_rows(root / "color.txt")
    aligned_depth = _timestamp_rows(root / "aligned_depth.txt")
    maximum_delta_s = float(source.get("maximum_rgb_depth_delta_s", 0.02))
    associated = _associate_nearest(color, aligned_depth, maximum_delta_s)
    start = int(source.get("start_association_index", 0))
    associated = associated[start:start + limit] if limit > 0 else associated[start:]
    poses = [(float(row[0]), [float(v) for v in row[1:]]) for row in parse_rows(root / "groundtruth.txt", 8)]
    intrinsics, transforms = _opencv_calibration(root)
    base_to_color = transforms.get(("base_link", "d400_color_optical_frame"))
    if base_to_color is None:
        raise ValueError("OpenLORIS base_link-to-D435i color calibration missing")
    groundtruth_frame = source["groundtruth_frame"]
    base_to_groundtruth = transforms.get(("base_link", groundtruth_frame))
    result = []
    for index, (rgb_ts, rgb_path, depth_ts, depth_path) in enumerate(associated):
        pose_ts, pose_values = nearest_pose(poses, rgb_ts)
        groundtruth_pose = np.asarray(quaternion_matrix(pose_values), dtype=np.float64)
        camera_pose = _openloris_camera_pose(
            groundtruth_pose,
            base_to_color,
            base_to_groundtruth,
            groundtruth_frame,
        )
        result.append(
            _frame(
                root,
                index,
                rgb_ts,
                depth_ts,
                pose_ts,
                rgb_path,
                depth_path,
                intrinsics,
                camera_pose.tolist(),
                source,
                "uint16_png_z_meters",
            )
        )
    return result


def _frame(root: Path, index: int, rgb_ts: float, depth_ts: float, pose_ts: float, rgb_path: str, depth_path: str, intrinsics: list[float], pose: list[list[float]], source: dict[str, Any], depth_encoding: str, camera_path: str | None = None) -> dict[str, Any]:
    rgb = (root / rgb_path).resolve(); depth = (root / depth_path).resolve()
    if root.resolve() not in rgb.parents or root.resolve() not in depth.parents or not rgb.is_file() or not depth.is_file():
        raise ValueError(f"unsafe or missing modality at frame {index}")
    validate_pose(pose)
    row = {
        "frame_id": f"{index:06d}", "rgb_timestamp_s": rgb_ts, "depth_timestamp_s": depth_ts, "pose_timestamp_s": pose_ts,
        "rgb_path": rgb.relative_to(root).as_posix(), "depth_path": depth.relative_to(root).as_posix(),
        "rgb_sha256": sha256(rgb), "depth_sha256": sha256(depth), "intrinsics_fx_fy_cx_cy": intrinsics,
        "camera_to_world": pose, "depth_encoding": depth_encoding, "depth_scale": source["depth_scale"],
        "rgb_depth_registered": True, "pose_stability": source["pose_stability"],
    }
    if camera_path:
        row["camera_path"] = camera_path
    return row


ADAPTERS = {
    "eth3d_tum": _eth3d,
    "icl_nuim_tum": _icl,
    "tartanair_preprocessed": _tartanair,
    "tum_rgbd_dynamic": _tum_rgbd_dynamic,
    "openloris_package": _openloris_package,
}


def normalize(repo: Path, sources_path: Path, prereg_path: Path, output: Path) -> dict[str, Any]:
    sources = read_json(sources_path); prereg = read_json(prereg_path)
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    summaries = []
    for source in sources["sources"]:
        root = (repo / source["root"]).resolve()
        requested = int(source.get("frame_count", prereg["frames_per_source"]))
        frames = ADAPTERS[source["adapter"]](root, source, requested)
        minimum = int(source.get("minimum_frame_count", requested))
        if len(frames) < minimum:
            raise ValueError(f"source {source['source_id']} has only {len(frames)} frames; needs {minimum}")
        target = output / source["source_id"]
        target.mkdir()
        ledger = target / "frames.jsonl"
        ledger.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in frames), encoding="utf-8")
        receipt = {"schema": BUNDLE_SCHEMA, "source": source, "source_root": str(root), "frame_count": len(frames), "frames_sha256": sha256(ledger), "prereg_sha256": sha256(prereg_path), "production_authority": False, "u0_authority": False}
        write_json(target / "bundle.json", receipt)
        summaries.append({"source_id": source["source_id"], "frame_count": len(frames), "bundle_sha256": sha256(target / "bundle.json")})
    report = {"schema": "blindassist_ustrf_sensor_replay_normalization_v1", "ok": len(summaries) >= int(prereg["minimum_admitted_sources"]), "sources": summaries, "source_count": len(summaries), "production_authority": False}
    write_json(output / "normalization_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path.cwd()); parser.add_argument("--sources", type=Path, required=True); parser.add_argument("--prereg", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = normalize(args.repo.resolve(), args.sources.resolve(), args.prereg.resolve(), args.output.resolve())
        print(json.dumps(report)); return 0 if report["ok"] else 1
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)})); return 2


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Generate and audit a small, analytic RGB-D-equivalent USTRF temporal geometry benchmark.

The samples are numerical depth rasters, not photorealistic images.  Ground, camera mount,
intrinsics, static-object world positions and exact adjacent-frame pose deltas are all generated
from one analytic scene definition, so they can exercise geometry/temporal contracts without
claiming real-device or real-user validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "blindassist_ustrf_synthetic_temporal_geometry_v1"
WIDTH, HEIGHT, FOCAL = 64, 48, 52.0
PRINCIPAL_X, PRINCIPAL_Y, CAMERA_HEIGHT = 31.5, 23.5, 1.5
FRAME_INTERVAL_NS = 100_000_000
POSE_FORWARD_METERS = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_definition(kind: str, lateral: float) -> dict[str, Any] | None:
    if kind == "clear":
        return None
    if kind == "visibility_gap":
        # Keep the gap in view in both frames; it is a missing lower-body-band observation,
        # not a synthetic below-horizon drop label.
        return {"kind": kind, "lateral_m": lateral, "forward_m": 3.5, "height_m": 0.60}
    if kind == "drop":
        # Lower in the image than a body target so the reference ground intersection remains
        # within the five-metre metric range in both frames.
        return {"kind": kind, "lateral_m": lateral, "forward_m": 3.5, "height_m": 0.40}
    return {
        "kind": kind,
        "lateral_m": lateral,
        "forward_m": 3.5,
        "height_m": 0.60 if kind == "lower_body" else 1.70,
    }


def render_depth(camera_forward: float, target: dict[str, Any] | None) -> np.ndarray:
    """Render a ground plane plus a small, nearest-depth object patch in metres."""
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    denominator = rows - PRINCIPAL_Y
    ground = np.where(denominator > 0, CAMERA_HEIGHT * FOCAL / denominator, 0.0).astype(np.float32)
    depth = np.broadcast_to(ground, (HEIGHT, WIDTH)).copy()
    depth[(depth < 0.20) | (depth > 5.0)] = 0.0
    if target is None:
        return depth
    forward = target["forward_m"] - camera_forward
    if forward <= 0.20:
        return depth
    column = int(round(PRINCIPAL_X + target["lateral_m"] * FOCAL / forward))
    row = int(round(PRINCIPAL_Y - (target["height_m"] - CAMERA_HEIGHT) * FOCAL / forward))
    for vertical in range(row - 2, row + 3):
        for horizontal in range(column - 2, column + 3):
            if 0 <= vertical < HEIGHT and 0 <= horizontal < WIDTH:
                if target["kind"] == "visibility_gap":
                    depth[vertical, horizontal] = 0.0
                elif target["kind"] == "drop" and depth[vertical, horizontal] > 0.0:
                    depth[vertical, horizontal] += 0.80
                elif depth[vertical, horizontal] == 0.0 or forward < depth[vertical, horizontal]:
                    depth[vertical, horizontal] = np.float32(forward)
    return depth


def generate(root: Path) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark root: {root}")
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    scenarios = [
        ("clear", 0.0),
        ("lower_body", -0.6),
        ("lower_body", 0.0),
        ("lower_body", 0.6),
        ("head", -0.5),
        ("head", 0.5),
        ("visibility_gap", 0.0),
        ("clear", 0.0),
        ("lower_body", -0.2),
        ("head", 0.2),
        ("visibility_gap", -0.4),
        ("lower_body", 0.4),
        ("drop", -0.3),
        ("drop", 0.3),
    ]
    rows: list[dict[str, Any]] = []
    for sequence_index, (kind, lateral) in enumerate(scenarios):
        target = object_definition(kind, lateral)
        sequence_id = f"synthetic-sequence-{sequence_index:02d}-{kind}"
        frame_rows: list[dict[str, Any]] = []
        for frame_index, camera_forward in enumerate((0.0, POSE_FORWARD_METERS)):
            frame_id = sequence_index * 2 + frame_index
            timestamp_ns = 1_000_000_000 + frame_id * FRAME_INTERVAL_NS
            filename = f"{sequence_id}-frame-{frame_index:02d}.npz"
            path = frames_dir / filename
            depth = render_depth(camera_forward, target)
            np.savez_compressed(path, depth_meters=depth)
            csv_filename = filename.removesuffix(".npz") + ".csv"
            csv_path = frames_dir / csv_filename
            np.savetxt(csv_path, depth, delimiter=",", fmt="%.6f")
            frame_rows.append({
                "frame_id": frame_id,
                "captured_at_ns": timestamp_ns,
                "camera_forward_m": camera_forward,
                "depth_path": f"frames/{filename}",
                "depth_sha256": sha256(path),
                "depth_csv_path": f"frames/{csv_filename}",
                "depth_csv_sha256": sha256(csv_path),
            })
        rows.append({
            "sequence_id": sequence_id,
            "frames": frame_rows,
            "target": target,
            "expected": {
                "temporal_static_match": kind in {"lower_body", "head"},
                "must_not_emit_drop": kind == "visibility_gap",
                "must_emit_drop": kind == "drop",
            },
            "pose_delta": {
                "forward_meters": POSE_FORWARD_METERS,
                "lateral_meters": 0.0,
                "yaw_radians": 0.0,
                "verified_for_offline_replay": True,
            },
        })
    specification = {
        "format": SCHEMA,
        "purpose": "offline_theory_geometry_and_temporal_contract_benchmark",
        "production_authority": False,
        "image_width_px": WIDTH,
        "image_height_px": HEIGHT,
        "intrinsics": {"fx_px": FOCAL, "fy_px": FOCAL, "cx_px": PRINCIPAL_X, "cy_px": PRINCIPAL_Y},
        "camera_to_body": {"translation_m": [0.0, CAMERA_HEIGHT, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "ground_plane_body": {"normal": [0.0, 1.0, 0.0], "offset_m": 0.0},
        "sequence_count": len(rows),
        "frame_count": sum(len(row["frames"]) for row in rows),
    }
    (root / "dataset_spec.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
    with (root / "kotlin_replay.tsv").open("w", encoding="utf-8") as destination:
        destination.write("\t".join((
            "sequence_id", "target_kind", "target_lateral_m", "target_forward_m", "target_height_m",
            "expected_temporal_static_match", "expected_must_not_emit_drop", "expected_must_emit_drop",
            "frame0_id", "frame0_captured_at_ns", "frame0_camera_forward_m", "frame0_depth_csv_path",
            "frame1_id", "frame1_captured_at_ns", "frame1_camera_forward_m", "frame1_depth_csv_path",
            "pose_forward_meters", "pose_lateral_meters", "pose_yaw_radians", "pose_verified",
        )) + "\n")
        for row in rows:
            target = row["target"] or {}
            first, second = row["frames"]
            values = (
                row["sequence_id"], target.get("kind", ""), str(target.get("lateral_m", "")),
                str(target.get("forward_m", "")), str(target.get("height_m", "")),
                str(row["expected"]["temporal_static_match"]).lower(),
                str(row["expected"]["must_not_emit_drop"]).lower(),
                str(row["expected"]["must_emit_drop"]).lower(),
                str(first["frame_id"]), str(first["captured_at_ns"]), str(first["camera_forward_m"]), first["depth_csv_path"],
                str(second["frame_id"]), str(second["captured_at_ns"]), str(second["camera_forward_m"]), second["depth_csv_path"],
                str(row["pose_delta"]["forward_meters"]), str(row["pose_delta"]["lateral_meters"]),
                str(row["pose_delta"]["yaw_radians"]), str(row["pose_delta"]["verified_for_offline_replay"]).lower(),
            )
            destination.write("\t".join(values) + "\n")
    return specification


def audit(root: Path, *, require_cuda: bool) -> dict[str, Any]:
    specification = json.loads((root / "dataset_spec.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    errors: list[str] = []
    static_pairs: list[tuple[float, float]] = []
    target_samples: list[tuple[float, float, int]] = []
    gap_samples: list[float] = []
    gap_pairs = 0
    drop_samples: list[tuple[float, float]] = []
    finite_depth_count = 0
    valid_depth_count = 0
    for sequence in rows:
        frames = sequence["frames"]
        if len(frames) != 2 or frames[1]["frame_id"] != frames[0]["frame_id"] + 1:
            errors.append(f"frame_binding:{sequence['sequence_id']}")
            continue
        if frames[1]["captured_at_ns"] - frames[0]["captured_at_ns"] != FRAME_INTERVAL_NS:
            errors.append(f"timestamp_binding:{sequence['sequence_id']}")
        arrays = []
        for frame in frames:
            path = root / frame["depth_path"]
            if not path.is_file() or sha256(path) != frame["depth_sha256"]:
                errors.append(f"depth_hash:{sequence['sequence_id']}:{frame['frame_id']}")
                continue
            depth = np.load(path)["depth_meters"]
            csv_path = root / frame["depth_csv_path"]
            if not csv_path.is_file() or sha256(csv_path) != frame["depth_csv_sha256"]:
                errors.append(f"depth_csv_hash:{sequence['sequence_id']}:{frame['frame_id']}")
            else:
                csv_depth = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
                if csv_depth.shape != depth.shape or not np.allclose(csv_depth, depth, rtol=0.0, atol=1e-6):
                    errors.append(f"depth_csv_values:{sequence['sequence_id']}:{frame['frame_id']}")
            if depth.shape != (HEIGHT, WIDTH):
                errors.append(f"depth_shape:{sequence['sequence_id']}:{frame['frame_id']}")
            if not np.isfinite(depth).all() or (depth < 0).any():
                errors.append(f"depth_values:{sequence['sequence_id']}:{frame['frame_id']}")
            finite_depth_count += int(np.isfinite(depth).sum())
            valid_depth_count += int((depth > 0).sum())
            arrays.append(depth)
        target = sequence["target"]
        if target and sequence["expected"]["temporal_static_match"]:
            expected_current_forward = target["forward_m"] - sequence["pose_delta"]["forward_meters"]
            static_pairs.append((target["forward_m"] - POSE_FORWARD_METERS, expected_current_forward))
            expected_code = 1 if target["kind"] == "lower_body" else 2
            for frame, depth in zip(frames, arrays):
                forward = target["forward_m"] - frame["camera_forward_m"]
                horizontal = int(round(PRINCIPAL_X + target["lateral_m"] * FOCAL / forward))
                vertical = int(round(PRINCIPAL_Y - (target["height_m"] - CAMERA_HEIGHT) * FOCAL / forward))
                target_samples.append((float(depth[vertical, horizontal]), float(vertical), expected_code))
        if sequence["expected"]["must_not_emit_drop"]:
            gap_pairs += 1
            for frame, depth in zip(frames, arrays):
                forward = target["forward_m"] - frame["camera_forward_m"]
                horizontal = int(round(PRINCIPAL_X + target["lateral_m"] * FOCAL / forward))
                vertical = int(round(PRINCIPAL_Y - (target["height_m"] - CAMERA_HEIGHT) * FOCAL / forward))
                gap_samples.append(float(depth[vertical, horizontal]))
        if sequence["expected"].get("must_emit_drop", False):
            for frame, depth in zip(frames, arrays):
                forward = target["forward_m"] - frame["camera_forward_m"]
                horizontal = int(round(PRINCIPAL_X + target["lateral_m"] * FOCAL / forward))
                vertical = int(round(PRINCIPAL_Y - (target["height_m"] - CAMERA_HEIGHT) * FOCAL / forward))
                expected_ground = CAMERA_HEIGHT * FOCAL / (vertical - PRINCIPAL_Y)
                drop_samples.append((float(depth[vertical, horizontal]), expected_ground))
    backend: dict[str, Any] = {"name": "numpy", "cuda": False}
    reprojection_rmse = 0.0
    target_classification_accuracy = 0.0
    gap_false_drop_count = 0
    drop_detected_count = 0
    try:
        import torch
        cuda = torch.cuda.is_available()
        if require_cuda and not cuda:
            raise RuntimeError("CUDA required for synthetic temporal benchmark audit")
        if cuda:
            device = torch.device("cuda")
            prior = torch.tensor([pair[0] for pair in static_pairs], dtype=torch.float32, device=device)
            current = torch.tensor([pair[1] for pair in static_pairs], dtype=torch.float32, device=device)
            reprojection_rmse = float(torch.sqrt(torch.mean((prior - current) ** 2)).cpu()) if static_pairs else 0.0
            sampled_depth = torch.tensor([sample[0] for sample in target_samples], dtype=torch.float32, device=device)
            sampled_row = torch.tensor([sample[1] for sample in target_samples], dtype=torch.float32, device=device)
            expected = torch.tensor([sample[2] for sample in target_samples], dtype=torch.int64, device=device)
            height = -(sampled_row - PRINCIPAL_Y) * sampled_depth / FOCAL + CAMERA_HEIGHT
            predicted = torch.where(
                sampled_depth <= 0.0,
                torch.zeros_like(expected),
                torch.where(height < 1.35, torch.ones_like(expected), 2 * torch.ones_like(expected)),
            )
            target_classification_accuracy = float((predicted == expected).float().mean().cpu()) if target_samples else 0.0
            gap_false_drop_count = int((torch.tensor(gap_samples, dtype=torch.float32, device=device) > 0.0).sum().cpu())
            if drop_samples:
                observed = torch.tensor([item[0] for item in drop_samples], dtype=torch.float32, device=device)
                expected_ground = torch.tensor([item[1] for item in drop_samples], dtype=torch.float32, device=device)
                drop_detected_count = int((observed - expected_ground >= .35).sum().cpu())
            backend = {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(device)}
    except ImportError:
        if require_cuda:
            raise
    report = {
        "format": "blindassist_ustrf_synthetic_temporal_geometry_audit_v1",
        "dataset_format": specification["format"],
        "ok": not errors,
        "errors": errors,
        "sequence_count": len(rows),
        "frame_count": sum(len(row["frames"]) for row in rows),
        "static_target_pair_count": len(static_pairs),
        "visibility_gap_pair_count": gap_pairs,
        "static_reprojection_rmse_meters": reprojection_rmse,
        "target_classification_count": len(target_samples),
        "target_classification_accuracy": target_classification_accuracy,
        "visibility_gap_missing_depth_count": len(gap_samples),
        "visibility_gap_false_drop_count": gap_false_drop_count,
        "drop_expected_sample_count": len(drop_samples),
        "drop_detected_sample_count": drop_detected_count,
        "drop_recall": float(drop_detected_count / len(drop_samples)) if drop_samples else 0.0,
        "finite_depth_samples": finite_depth_count,
        "valid_depth_samples": valid_depth_count,
        "compute_backend": backend,
        "kotlin_replay_tsv_sha256": sha256(root / "kotlin_replay.tsv"),
        "production_authority": False,
    }
    qa = root / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        generate(args.output)
    report = audit(args.output, require_cuda=args.require_cuda)
    print(json.dumps({"ok": report["ok"], "sequences": report["sequence_count"], "backend": report["compute_backend"]}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

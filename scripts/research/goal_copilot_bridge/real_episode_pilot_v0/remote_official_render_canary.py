"""Run a zero-provider ABotN canary against the pinned official render server.

This entrypoint is intended for an isolated Linux/CUDA worker. It uses the
official evaluator client from a pinned ABot-Navigation checkout, renders only
predeclared source poses, and writes lossless pixels plus a call journal. It
never imports or invokes a BlindAssist provider, teacher, or baseline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SCHEMA = "blindassist_abotn_official_render_canary_v0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _pixel_stats(image: Any) -> dict[str, Any]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    sample = rgb[::8, ::8].reshape(-1, 3)
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "luma_mean": float(luma.mean()),
        "luma_stddev": float(luma.std()),
        "black_fraction": float((luma <= 2).mean()),
        "white_fraction": float((luma >= 253).mean()),
        "sampled_distinct_rgb": int(np.unique(sample, axis=0).shape[0]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    annotation = args.annotation.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.finalize_existing:
        raise FileExistsError(f"one-shot output directory already exists: {output_dir}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.expected_commit:
        raise ValueError(f"repository revision drift: {actual_commit}")
    if _sha256(annotation) != args.expected_annotation_sha256:
        raise ValueError("annotation SHA-256 mismatch")

    task = json.loads(annotation.read_text(encoding="utf-8"))
    trajectory = task.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("annotation trajectory is missing")
    pose_indices = [int(value) for value in args.pose_indices.split(",")]
    if len(set(pose_indices)) != len(pose_indices) or any(
        index < 0 or index >= len(trajectory) for index in pose_indices
    ):
        raise ValueError("pose indices must be unique and within the frozen trajectory")

    sys.path.insert(0, str(repository))
    from abotn_evaluator.render_client import CameraConfig, GaussianRenderer  # noqa: PLC0415
    from abotn_evaluator.scene import GaussianScene  # noqa: PLC0415
    import diff_plane_rasterization._C as diff_plane_rasterization_c  # noqa: PLC0415
    import simple_knn._C as simple_knn_c  # noqa: PLC0415
    import torch  # noqa: PLC0415

    camera = CameraConfig()
    renderer = GaussianRenderer(
        render_url=args.render_url,
        camera_config=camera,
        num_views=1,
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    journal_path = output_dir / "render-journal.json"
    if args.finalize_existing:
        from PIL import Image  # noqa: PLC0415

        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("schema_version") != "blindassist_abotn_official_render_journal_v0":
            raise ValueError("existing render journal schema mismatch")
        if journal.get("render_calls_dispatched") != len(pose_indices) or journal.get(
            "render_calls_completed"
        ) != len(pose_indices):
            raise ValueError("existing render journal is not exactly complete")
        calls = journal.get("calls")
        if not isinstance(calls, list) or [row.get("pose_index") for row in calls] != pose_indices:
            raise ValueError("existing render journal pose roster mismatch")
        frames = []
        for row in calls:
            if row.get("status") != "COMPLETED" or row.get("scene_id") != args.scene_id:
                raise ValueError("existing render call is not a completed frozen-scene call")
            frame = output_dir / str(row["output"])
            if _sha256(frame) != row.get("output_sha256"):
                raise ValueError(f"existing frame hash mismatch: {frame.name}")
            with Image.open(frame) as image:
                stats = _pixel_stats(image)
            frames.append({
                "pose_index": int(row["pose_index"]),
                "path": frame.name,
                "bytes": frame.stat().st_size,
                "sha256": row["output_sha256"],
                "pixel_stats": stats,
            })
    else:
        output_dir.mkdir(parents=True)
        journal = {
            "schema_version": "blindassist_abotn_official_render_journal_v0",
            "created_at_utc": _utc_now(),
            "render_calls_dispatched": 0,
            "render_calls_completed": 0,
            "calls": [],
        }
        _atomic_json(journal_path, journal)

        frames = []
        for index in pose_indices:
            source_pose = trajectory[index]
            dispatch = {
                "pose_index": index,
                "scene_id": args.scene_id,
                "dispatched_at_utc": _utc_now(),
                "status": "DISPATCHED",
            }
            journal["render_calls_dispatched"] += 1
            journal["calls"].append(dispatch)
            _atomic_json(journal_path, journal)
            try:
                pose = GaussianScene.get_gaussian_pose(source_pose)
                image = renderer.render_at_pose(pose, args.scene_id)[0]
                frame = output_dir / f"pose-{index:03d}-front.png"
                temporary = frame.with_suffix(".png.tmp")
                image.save(temporary, format="PNG")
                os.replace(temporary, frame)
                stats = _pixel_stats(image)
                dispatch.update({
                    "status": "COMPLETED",
                    "completed_at_utc": _utc_now(),
                    "output": frame.name,
                    "output_sha256": _sha256(frame),
                })
                journal["render_calls_completed"] += 1
                frames.append({
                    "pose_index": index,
                    "path": frame.name,
                    "bytes": frame.stat().st_size,
                    "sha256": dispatch["output_sha256"],
                    "pixel_stats": stats,
                })
                _atomic_json(journal_path, journal)
            except Exception as error:
                dispatch.update({
                    "status": "FAILED",
                    "completed_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                _atomic_json(journal_path, journal)
                raise

    receipt = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "terminal": "ABOTN_OFFICIAL_RENDER_CANARY_PASS",
        "official_repository": str(repository),
        "official_repository_commit": actual_commit,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "diff_plane_rasterization_sha256": _sha256(
                Path(diff_plane_rasterization_c.__file__)
            ),
            "simple_knn_sha256": _sha256(Path(simple_knn_c.__file__)),
            "simple_knn_compile_compatibility": "NVCC_PREPEND_FLAGS=-include cfloat",
        },
        "annotation_path": str(annotation),
        "annotation_sha256": args.expected_annotation_sha256,
        "scene_id": args.scene_id,
        "pose_indices": pose_indices,
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "extrinsic_height": camera.extrinsic_height,
            "views": ["front"],
            "server_render_scale": args.server_render_scale,
        },
        "frames": frames,
        "render_calls_dispatched": journal["render_calls_dispatched"],
        "render_calls_completed": journal["render_calls_completed"],
        "render_calls_in_doubt": (
            journal["render_calls_dispatched"] - journal["render_calls_completed"]
        ),
        "teacher_calls": 0,
        "provider_calls": 0,
        "baseline_calls": 0,
        "sealed_episode_reruns": 0,
        "receipt_finalized_from_existing_outputs": args.finalize_existing,
        "claim_ceiling": "PINNED_OFFICIAL_RENDERER_PIXEL_TRANSPORT_ONLY",
        "forbidden_inferences": [
            "Do not treat this zero-provider canary as a rerun of the sealed episode.",
            "Do not infer functional entrance truth from rendered pixels alone.",
            "Do not claim navigation, arrival, selection, bearing, range, or product success.",
        ],
    }
    _atomic_json(output_dir / "terminal-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--expected-annotation-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--pose-indices", default="0,1,2")
    parser.add_argument("--render-url", default="http://127.0.0.1:17036/render_gs")
    parser.add_argument("--server-render-scale", type=float, default=1.5)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "terminal": result["terminal"],
        "frames": len(result["frames"]),
        "render_calls": result["render_calls_completed"],
        "provider_calls": result["provider_calls"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

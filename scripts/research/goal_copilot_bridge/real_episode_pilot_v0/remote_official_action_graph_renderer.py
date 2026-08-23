"""Materialize a frozen public ABotN action graph with the official renderer.

The remote worker receives only the provider-public graph and scene identity.
Evaluator-private endpoint/distance truth is intentionally absent.
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


SCHEMA = "blindassist_abotn_official_action_graph_pixels_v0"
JOURNAL_SCHEMA = "blindassist_abotn_official_action_graph_render_journal_v0"
PUBLIC_SCHEMA = "blindassist_abotn_v0_action_graph_public_v0"
FORBIDDEN_PUBLIC_LITERALS = ("endpoint_xy", "distance_to_goal_m", "target_position")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _source_pose(node: dict[str, Any]) -> dict[str, float]:
    camera = node["source_camera"]
    position = camera["position"]
    euler = camera["euler_radians"]
    if len(position) != 3 or len(euler) != 3:
        raise ValueError(f"invalid source camera: {node['node_id']}")
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    graph_path = args.public_graph.resolve()
    output_dir = args.output_dir.resolve()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.expected_commit:
        raise ValueError(f"repository revision drift: {actual_commit}")
    if _sha256(graph_path) != args.expected_public_graph_sha256:
        raise ValueError("public graph SHA-256 mismatch")
    graph_text = graph_path.read_text(encoding="utf-8")
    forbidden_hits = [value for value in FORBIDDEN_PUBLIC_LITERALS if value in graph_text]
    if forbidden_hits:
        raise ValueError(f"private truth literal in public graph: {forbidden_hits}")
    graph = json.loads(graph_text)
    if graph.get("schema_version") != PUBLIC_SCHEMA or graph.get("private_truth_access") is not False:
        raise ValueError("public graph schema or firewall mismatch")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("public graph has no nodes")
    if len(nodes) != args.expected_render_calls:
        raise ValueError("frozen render-call budget does not equal graph node count")
    graph_paths = [str(node["rendered_frame_path"]) for node in nodes]
    if len(set(graph_paths)) != len(graph_paths):
        raise ValueError("public graph contains duplicate frame paths")

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
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("schema_version") != JOURNAL_SCHEMA:
            raise ValueError("render journal schema drift")
        if journal.get("public_graph_sha256") != args.expected_public_graph_sha256:
            raise ValueError("render journal graph identity drift")
        if any(row.get("status") != "COMPLETED" for row in journal.get("calls", [])):
            raise ValueError("resume refused because a previous render call is in_doubt or failed")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        journal = {
            "schema_version": JOURNAL_SCHEMA,
            "created_at_utc": _utc_now(),
            "public_graph_sha256": args.expected_public_graph_sha256,
            "render_calls_authorized": args.expected_render_calls,
            "render_calls_dispatched": 0,
            "render_calls_completed": 0,
            "calls": [],
        }
        _atomic_json(journal_path, journal)

    completed_by_id = {row["observation_id"]: row for row in journal["calls"]}
    frames: list[dict[str, Any]] = []
    for node in nodes:
        observation_id = str(node["node_id"])
        frame_path = output_dir / str(node["rendered_frame_path"])
        completed = completed_by_id.get(observation_id)
        if completed is not None:
            if _sha256(frame_path) != completed.get("output_sha256"):
                raise ValueError(f"completed frame hash mismatch: {observation_id}")
            frames.append(completed["frame_receipt"])
            continue
        dispatch = {
            "observation_id": observation_id,
            "pose_index": int(node["pose_index"]),
            "viewport_yaw_index": int(node["viewport_yaw_index"]),
            "scene_id": args.scene_id,
            "output": str(node["rendered_frame_path"]),
            "dispatched_at_utc": _utc_now(),
            "status": "DISPATCHED",
        }
        journal["render_calls_dispatched"] += 1
        journal["calls"].append(dispatch)
        _atomic_json(journal_path, journal)
        pose = GaussianScene.get_gaussian_pose(_source_pose(node))
        image = renderer.render_at_pose(pose, args.scene_id)[0]
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = frame_path.with_suffix(frame_path.suffix + ".tmp")
        image.save(temporary, format="PNG")
        os.replace(temporary, frame_path)
        stats = _pixel_stats(image)
        frame_receipt = {
            "observation_index": len(frames),
            "observation_id": observation_id,
            "pose_index": int(node["pose_index"]),
            "viewport_yaw_index": int(node["viewport_yaw_index"]),
            "path": str(node["rendered_frame_path"]),
            "bytes": frame_path.stat().st_size,
            "sha256": _sha256(frame_path),
            "pixel_stats": stats,
            "nondegenerate": (
                stats["luma_stddev"] >= 8.0 and stats["sampled_distinct_rgb"] >= 256
            ),
        }
        dispatch.update({
            "status": "COMPLETED",
            "completed_at_utc": _utc_now(),
            "output_sha256": frame_receipt["sha256"],
            "frame_receipt": frame_receipt,
        })
        journal["render_calls_completed"] += 1
        frames.append(frame_receipt)
        _atomic_json(journal_path, journal)
        print(
            f"RENDER {journal['render_calls_completed']}/{args.expected_render_calls} COMPLETE",
            flush=True,
        )

    if journal["render_calls_dispatched"] != args.expected_render_calls or journal[
        "render_calls_completed"
    ] != args.expected_render_calls:
        raise ValueError("terminal render call accounting mismatch")
    if not all(frame["nondegenerate"] for frame in frames):
        raise ValueError("official action graph contains a degenerate frame")
    roster = {
        "schema_version": "blindassist_abotn_official_action_graph_roster_v0",
        "public_graph_sha256": args.expected_public_graph_sha256,
        "observations": [
            {
                "observation_index": index,
                "observation_id": node["node_id"],
                "pose_index": node["pose_index"],
                "viewport_yaw_index": node["viewport_yaw_index"],
                "output_path": node["rendered_frame_path"],
            }
            for index, node in enumerate(nodes)
        ],
    }
    roster_path = output_dir / "roster.json"
    _atomic_json(roster_path, roster)
    receipt = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "terminal": "ABOTN_OFFICIAL_ACTION_GRAPH_PIXELS_PASS",
        "official_repository_commit": actual_commit,
        "scene_id": args.scene_id,
        "episode_id": graph["episode_id"],
        "public_graph_sha256": args.expected_public_graph_sha256,
        "roster_sha256": _sha256(roster_path),
        "renderer": {
            "kind": "PINNED_OFFICIAL_ABOTN_RENDERER",
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
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "diff_plane_rasterization_sha256": _sha256(Path(diff_plane_rasterization_c.__file__)),
            "simple_knn_sha256": _sha256(Path(simple_knn_c.__file__)),
            "simple_knn_compile_compatibility": "NVCC_PREPEND_FLAGS=-include cfloat",
        },
        "frames": frames,
        "render_calls_dispatched": journal["render_calls_dispatched"],
        "render_calls_completed": journal["render_calls_completed"],
        "render_calls_in_doubt": 0,
        "provider_calls": 0,
        "teacher_calls": 0,
        "baseline_calls": 0,
        "private_truth_access": False,
        "claim_ceiling": "PINNED_OFFICIAL_RENDERER_ACTION_GRAPH_PIXELS_ONLY",
    }
    receipt_path = output_dir / "terminal-receipt.json"
    _atomic_json(receipt_path, receipt)
    manifest = {
        "schema_version": "blindassist_abotn_official_action_graph_manifest_v0",
        "created_at_utc": _utc_now(),
        "terminal": "ABOTN_OFFICIAL_ACTION_GRAPH_PIXELS_PASS",
        "public_graph_sha256": args.expected_public_graph_sha256,
        "roster_sha256": _sha256(roster_path),
        "terminal_receipt_sha256": _sha256(receipt_path),
        "render_calls": args.expected_render_calls,
        "provider_calls": 0,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--public-graph", type=Path, required=True)
    parser.add_argument("--expected-public-graph-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--expected-render-calls", type=int, required=True)
    parser.add_argument("--render-url", default="http://127.0.0.1:17036/render_gs")
    parser.add_argument("--server-render-scale", type=float, default=1.5)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps({
        "terminal": receipt["terminal"],
        "frames": len(receipt["frames"]),
        "render_calls": receipt["render_calls_completed"],
        "provider_calls": receipt["provider_calls"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

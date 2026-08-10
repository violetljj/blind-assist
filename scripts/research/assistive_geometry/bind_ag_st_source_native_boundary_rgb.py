#!/usr/bin/env python3
"""Bind each source-native boundary sidecar to its exact RGB evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ag_st_tum_rgbd import load_tum_role_payloads
from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-source-native-boundary-corpus-r0/result.json"
)
DEFAULT_ARKIT_STAGE0_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-stage0a-mapanything-apache-train16-block64-r1/result.json"
)
DEFAULT_TUM_DEPTH_RESULT = (
    Path("F:/ba-data/blindassist-artifacts-20260805/experiments/ag-st-tum-third-teacher-r2/result.json")
)
DEFAULT_ICL_RGB_ROOT = REPO_ROOT / "artifacts.local/downloads/ag-st-icl-boundary-r1/selected12/rgb"
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-source-native-boundary-corpus-r0/rgb_binding.json"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _image_shape(source: Path | bytes) -> list[int]:
    if isinstance(source, bytes):
        import io

        handle: Any = io.BytesIO(source)
    else:
        handle = source
    with Image.open(handle) as image:
        return [int(image.height), int(image.width), len(image.getbands())]


def _arkit_bindings(stage0_result_path: Path) -> dict[str, dict[str, Any]]:
    stage0 = json.loads(stage0_result_path.read_text(encoding="utf-8"))
    manifest_path = Path(stage0["source"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = {
        str(video["video_id"]): video
        for video in manifest["videos"]
        if str(video.get("role")) == "TRAIN"
    }
    orientations = {
        str(frame["frame_stem"]): str(frame["orientation"])
        for parent in stage0["parent_runs"]
        for frame in parent["frame_summaries"]
    }
    output: dict[str, dict[str, Any]] = {}
    for parent_id in stage0["source"]["parents"]:
        video = videos[str(parent_id)]
        for index, row in enumerate(video["extracted"]["lowres_wide"]):
            path = Path(row["path"])
            if path.stem not in orientations:
                continue
            output[path.stem] = {
                "rgb_storage_kind": "file",
                "rgb_path": str(path.resolve()),
                "rgb_sha256": sha256_file(path),
                "rgb_shape_hwc": _image_shape(path),
                "source_frame_index": index,
                "adapter": "load_factor_source_frame canonical upright rotation then MapAnything crop_resize to label HW",
                "orientation": orientations[path.stem],
                "source_manifest": str(manifest_path.resolve()),
                "source_manifest_sha256": sha256_file(manifest_path),
            }
    require(len(output) == 48, "ARKit RGB binding count drift")
    return output


def _tum_bindings(depth_result_path: Path) -> dict[str, dict[str, Any]]:
    result = json.loads(depth_result_path.read_text(encoding="utf-8"))
    cohort_path = Path(result["cohort"]["manifest_path"])
    output: dict[str, dict[str, Any]] = {}
    for role in ("fit", "evaluation"):
        payloads, _ = load_tum_role_payloads(cohort_path, role)
        for payload in payloads:
            frame_id = f"{payload.parent_id}__rgb{payload.rgb.row_index:06d}"
            if payload.rgb_bytes is not None:
                rgb_sha256 = _sha256_bytes(payload.rgb_bytes)
                rgb_shape = _image_shape(payload.rgb_bytes)
                locator = {
                    "rgb_storage_kind": "tar_member",
                    "rgb_source_archive": str(payload.source_path.resolve()),
                    "rgb_member": payload.rgb.relative_path,
                }
            else:
                require(payload.rgb_path is not None, "TUM RGB path missing")
                rgb_sha256 = sha256_file(payload.rgb_path)
                rgb_shape = _image_shape(payload.rgb_path)
                locator = {
                    "rgb_storage_kind": "file",
                    "rgb_path": str(payload.rgb_path.resolve()),
                }
            output[frame_id] = {
                **locator,
                "rgb_sha256": rgb_sha256,
                "rgb_shape_hwc": rgb_shape,
                "source_role": role,
                "adapter": "TumSelectedPayload.load_rgb then MapAnything crop_resize to label HW",
                "cohort_manifest": str(cohort_path.resolve()),
                "cohort_manifest_sha256": sha256_file(cohort_path),
            }
    require(len(output) == 21, "TUM RGB binding count drift")
    return output


def run(
    corpus_result_path: Path,
    arkit_stage0_result_path: Path,
    tum_depth_result_path: Path,
    icl_rgb_root: Path,
) -> dict[str, Any]:
    for path in (corpus_result_path, arkit_stage0_result_path, tum_depth_result_path):
        require(path.is_file(), f"RGB binding input missing: {path}")
    require(icl_rgb_root.is_dir(), "ICL RGB root missing")
    corpus = json.loads(corpus_result_path.read_text(encoding="utf-8"))
    require(corpus.get("status") == "SOURCE_NATIVE_BOUNDARY_CORPUS_PASS", "boundary corpus not passed")
    arkit = _arkit_bindings(arkit_stage0_result_path)
    tum = _tum_bindings(tum_depth_result_path)
    rows: list[dict[str, Any]] = []
    for frame in corpus["frames"]:
        source = str(frame["source"])
        frame_id = str(frame["frame_id"])
        if source == "arkitscenes":
            binding = arkit[frame_id]
        elif source == "tum_rgbd":
            parts = frame_id.split("__")
            require(len(parts) == 3, f"TUM corpus identity invalid: {frame_id}")
            binding = tum["__".join(parts[1:])]
        elif source == "icl_exact":
            index = int(frame_id.rsplit("_", 1)[1])
            rgb_path = icl_rgb_root / f"{index}.png"
            require(rgb_path.is_file(), f"ICL RGB missing: {rgb_path}")
            binding = {
                "rgb_storage_kind": "file",
                "rgb_path": str(rgb_path.resolve()),
                "rgb_sha256": sha256_file(rgb_path),
                "rgb_shape_hwc": _image_shape(rgb_path),
                "adapter": "vertical raster flip matching positive-fy ICL convention, then resize/output at 120x160",
            }
        else:
            raise ValueError(f"unsupported corpus source: {source}")
        label_path = Path(frame["output"])
        require(label_path.is_file(), f"bound label missing: {label_path}")
        with np.load(label_path) as label:
            label_shape = list(label["boundary_probability_hw"].shape)
        rows.append(
            {
                "source": source,
                "parent_id": frame["parent_id"],
                "frame_id": frame_id,
                "label_path": str(label_path.resolve()),
                "label_sha256": sha256_file(label_path),
                "label_shape_hw": label_shape,
                **binding,
            }
        )
    gates = {
        "binding_count_eq_81": len(rows) == 81,
        "source_count_eq_3": len({row["source"] for row in rows}) == 3,
        "parent_count_eq_24": len({(row["source"], row["parent_id"]) for row in rows}) == 24,
        "all_rgb_receipts_present": all(bool(row["rgb_sha256"]) for row in rows),
        "all_label_receipts_present": all(bool(row["label_sha256"]) for row in rows),
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist_ag_st_source_native_boundary_rgb_binding_v1",
        "status": "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_PASS" if passed else "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_FAIL",
        "corpus_result": str(corpus_result_path.resolve()),
        "corpus_result_sha256": sha256_file(corpus_result_path),
        "binding_count": len(rows),
        "source_count": len({row["source"] for row in rows}),
        "parent_count": len({(row["source"], row["parent_id"]) for row in rows}),
        "gates": gates,
        "frames": rows,
        "decision": {
            "rgb_label_loader_contract_ready": passed,
            "masked_source_boundary_training_authorized": passed,
            "teacher_filled_boundary_training_authorized": False,
        },
        "claim_boundary": "Exact RGB-to-source-derived-boundary binding for WILD_LAB masked training only; no formal F1, task, product, deployment, or safety claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-result", type=Path, default=DEFAULT_CORPUS_RESULT)
    parser.add_argument("--arkit-stage0-result", type=Path, default=DEFAULT_ARKIT_STAGE0_RESULT)
    parser.add_argument("--tum-depth-result", type=Path, default=DEFAULT_TUM_DEPTH_RESULT)
    parser.add_argument("--icl-rgb-root", type=Path, default=DEFAULT_ICL_RGB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"RGB binding output exists: {args.output}")
    result = run(args.corpus_result, args.arkit_stage0_result, args.tum_depth_result, args.icl_rgb_root)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "binding_count", "source_count", "parent_count", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

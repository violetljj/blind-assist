#!/usr/bin/env python3
"""Fresh SceneNN local-correspondence/NONE challenge with frozen EfficientLoFTR."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_observed_extent_support as parent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-efficientloftr-fresh-none-protocol-v1"
ADMISSION_SCHEMA = "blindassist-l10-scenenn-efficientloftr-fresh-none-source-admission-v1"
COHORT_SCHEMA = "blindassist-l10-scenenn-efficientloftr-fresh-none-cohort-v1"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-efficientloftr-fresh-none-rgb-receipt-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-efficientloftr-fresh-none-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return parent.sha256(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    predecessor = protocol["predecessor"]
    for prefix in ("temporal_geometry_result", "dinov2_ffa_result", "project_matcher_protocol"):
        dependency = HERE / predecessor[f"{prefix}_path"]
        require(dependency.is_file(), f"PREDECESSOR_MISSING:{dependency}")
        require(sha256(dependency) == predecessor[f"{prefix}_sha256"], f"PREDECESSOR_HASH:{prefix}")
    ffa = load_json(HERE / predecessor["dinov2_ffa_result_path"])
    require(ffa.get("conclusion") == predecessor["dinov2_ffa_required_conclusion"], "PREDECESSOR_CONCLUSION")
    matcher_root = ROOT / protocol["matcher"]["path"]
    require(sha256(matcher_root / "model.safetensors") == protocol["matcher"]["model_safetensors_sha256"], "MATCHER_WEIGHTS")
    require(sha256(matcher_root / "config.json") == protocol["matcher"]["config_sha256"], "MATCHER_CONFIG")
    require(sha256(matcher_root / "preprocessor_config.json") == protocol["matcher"]["preprocessor_sha256"], "MATCHER_PROCESSOR")
    return protocol


def source_paths(source_root: Path, scene_id: str) -> dict[str, Path]:
    scene_root = source_root / "payload" / scene_id
    return {
        "ply": scene_root / f"{scene_id}.ply",
        "xml": scene_root / f"{scene_id}.xml",
        "trajectory": scene_root / "trajectory.log",
        "oni": scene_root / f"{scene_id}.oni",
        "intrinsic": source_root / "payload" / "intrinsic" / "asus.ini",
    }


def selected_image(source_root: Path, scene_id: str, frame: int) -> Path:
    return source_root / "payload" / scene_id / "selected" / "image" / f"frame.{frame:04d}.png"


def validate_length(path: Path, expected: int) -> dict[str, Any]:
    require(path.is_file(), f"SOURCE_MISSING:{path}")
    require(path.stat().st_size == expected, f"SOURCE_LENGTH:{path}")
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def freeze(protocol_path: Path, admission_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    admission = load_json(admission_path)
    require(admission.get("schema") == ADMISSION_SCHEMA, "ADMISSION_SCHEMA")
    require(admission["protocol_sha256"] == sha256(protocol_path), "ADMISSION_PROTOCOL_HASH")
    require(not list(source_root.glob("**/*.oni")), "ONI_OPENED_BEFORE_COHORT_FREEZE")
    require(not list(source_root.glob("**/selected/**/*.png")), "RGB_OPENED_BEFORE_COHORT_FREEZE")
    selected = admission["selected"]
    require([row["scene_id"] for row in selected] == protocol["source_selector"]["expected_selected_scenes"], "SELECTED_SCENES")
    require([int(row["target_instance_id"]) for row in selected] == protocol["source_selector"]["expected_target_instance_ids"], "SELECTED_TARGETS")
    intrinsic_path = source_paths(source_root, selected[0]["scene_id"])["intrinsic"]
    intrinsic_receipt = validate_length(intrinsic_path, int(protocol["provider"]["intrinsic"]["content_length"]))
    require(intrinsic_receipt["sha256"] == protocol["provider"]["intrinsic"]["sha256"], "INTRINSIC_HASH")
    intrinsic = parent.base.parse_intrinsic(intrinsic_path)
    episodes: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}
    renderer_runtime: dict[str, Any] | None = None
    selector = protocol["pre_rgb_selector"]
    for source_row in selected:
        scene_id = str(source_row["scene_id"])
        target_id = int(source_row["target_instance_id"])
        expected = protocol["provider"]["scenes"][scene_id]
        paths = source_paths(source_root, scene_id)
        scene_receipts = {
            name: validate_length(paths[name], int(expected[name]["content_length"]))
            for name in ("ply", "xml", "trajectory")
        }
        require(scene_receipts["xml"]["sha256"] == expected["xml"]["inventory_sha256"], f"XML_HASH:{scene_id}")
        for name, receipt in scene_receipts.items():
            source_manifest[f"payload/{scene_id}/{paths[name].name}"] = receipt
        xyz, labels, faces = parent.visible.read_mesh(paths["ply"])
        poses = parent.base.parse_poses(paths["trajectory"])
        target_points = xyz[labels == target_id]
        require(len(target_points) >= 4, f"TARGET_POINTS:{scene_id}:{target_id}")
        renderer = parent.visible.VisibilityRenderer(xyz, labels, faces, target_id, intrinsic, selector["renderer"])
        if renderer_runtime is None:
            renderer_runtime = renderer.runtime_identity()
        rows, reasons = parent.frame_rows(
            renderer,
            poses,
            np.mean(target_points.astype(np.float64), axis=0),
            selector["eligible_frame"],
            int(selector["trajectory_stride"]),
        )
        pair = parent.visible.select_pair(
            rows,
            minimum_baseline=float(selector["minimum_pair_baseline_metres"]),
            minimum_gap=int(selector["minimum_pair_frame_gap"]),
        )
        require(pair is not None, f"TARGET_PAIR_NOT_EVALUABLE:{scene_id}:{target_id}")
        reference, query, baseline = pair
        pose_by_frame = {int(row["frame"]): row for row in poses}
        role_rows: dict[str, Any] = {}
        for role, row in (("reference", reference), ("query", query)):
            _, mask = renderer.statistics(pose_by_frame[int(row["frame"])]["camera_to_world"], return_mask=True)
            require(mask is not None, f"MASK_MISSING:{scene_id}:{role}")
            ys, xs = np.where(mask)
            require(len(xs) > 0, f"MASK_EMPTY:{scene_id}:{role}")
            role_rows[role] = {
                **row,
                "visible_mask_sha256": parent.visible.mask_sha256(mask),
                "visible_mask_tight_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            }
        episodes.append({
            "episode_id": source_row["episode_id"],
            "scene_id": scene_id,
            "target_instance_id": target_id,
            "target_label": source_row["target_label"],
            "target_mesh_vertices": int(len(target_points)),
            "strict_target_faces": int(renderer.target_face_count),
            "trajectory_frames": len(poses),
            "sampled_frames": len(rows),
            "eligible_frames": sum(bool(row["eligible"]) for row in rows),
            "reason_counts": reasons,
            "camera_baseline_m": float(baseline),
            **role_rows,
        })
    episode_ids = [row["episode_id"] for row in episodes]
    scenarios = parent.scenarios(episode_ids)
    truth_matches = sum(len(set(row["reference_targets"]) & set(row["query_targets"])) for row in scenarios)
    truth_unmatched = sum(len(set(row["reference_targets"]) ^ set(row["query_targets"])) for row in scenarios)
    parent.write_json(output_path, {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGBD_FRESH_MULTISCENE_LOCAL_CORRESPONDENCE_NONE_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "source_admission_path": admission_path.name,
        "source_admission_sha256": sha256(admission_path),
        "entrypoint_path": Path(__file__).name,
        "entrypoint_sha256": sha256(Path(__file__)),
        "source_root": str(source_root.resolve()),
        "source_manifest": dict(sorted(source_manifest.items())),
        "intrinsic_receipt": intrinsic_receipt,
        "renderer_runtime": renderer_runtime,
        "episodes": episodes,
        "selected_frames_by_scene": {row["scene_id"]: sorted([int(row["reference"]["frame"]), int(row["query"]["frame"])]) for row in episodes},
        "scenarios": scenarios,
        "counts": {"physical_targets": 4, "selected_rgb_frames": 8, "scenarios": len(scenarios), "truth_matches_across_scenarios": truth_matches, "truth_unmatched_nodes_across_scenarios": truth_unmatched},
        "geometry_members_opened": 12,
        "rgb_members_opened": 0,
        "depth_members_opened": 0,
        "model_calls": 0,
        "claim_boundary": protocol["claim_boundary"],
    })


def seal(cohort_path: Path, source_root: Path, extraction_root: Path, extractor_exe: Path, output_path: Path) -> None:
    cohort = load_json(cohort_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA")
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__)), "COHORT_ENTRYPOINT_HASH")
    sealed: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for scene_id, frames_raw in cohort["selected_frames_by_scene"].items():
        frames = {int(value) for value in frames_raw}
        scene_extract = extraction_root / scene_id
        expected_names = {f"frame.{frame:04d}.png" for frame in frames}
        images = sorted((scene_extract / "image").glob("*.png"))
        depths = sorted((scene_extract / "depth").glob("*.png"))
        require({path.name for path in images} == expected_names, f"IMAGE_SET:{scene_id}")
        require({path.name for path in depths} == expected_names, f"DEPTH_SET:{scene_id}")
        timestamps = parent.visible.parse_selected_timestamps(scene_extract / "selected_timestamp.txt")
        summary = parent.visible.parse_extraction_summary(scene_extract / "summary.txt")
        require(set(timestamps) == frames, f"TIMESTAMP_SET:{scene_id}")
        require(summary["requested"] == len(frames) and summary["saved"] == len(frames), f"EXTRACTION_COUNT:{scene_id}")
        summaries[scene_id] = summary
        for frame in sorted(frames):
            source_image = scene_extract / "image" / f"frame.{frame:04d}.png"
            source_depth = scene_extract / "depth" / f"frame.{frame:04d}.png"
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
            require(image is not None and image.shape == (480, 640, 3), f"IMAGE_FORMAT:{scene_id}:{frame}")
            require(depth is not None and depth.shape == (480, 640) and depth.dtype == np.uint16, f"DEPTH_FORMAT:{scene_id}:{frame}")
            target_image = selected_image(source_root, scene_id, frame)
            parent.base.atomic_write(target_image, source_image.read_bytes())
            sealed[f"{scene_id}:{frame}"] = {
                "scene_id": scene_id,
                "trajectory_frame": frame,
                "timestamp": timestamps[frame],
                "image_path": target_image.resolve().relative_to(source_root.resolve()).as_posix(),
                "image_sha256": sha256(target_image),
                "depth_extraction_sha256": sha256(source_depth),
                "depth_retained": False,
            }
    parent.write_json(output_path, {
        "schema": RECEIPT_SCHEMA,
        "authority": "POST_COHORT_FREEZE_SPARSE_RGB_ONLY_RECEIPT",
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "extractor_executable": str(extractor_exe.resolve()),
        "extractor_sha256": sha256(extractor_exe),
        "extractor_source_sha256": sha256(parent.visible.EXTRACTOR_SOURCE_PATH),
        "extraction_summaries": summaries,
        "sealed_frames": sealed,
        "rgb_frames_retained": len(sealed),
        "depth_frames_retained": 0,
    })


def expanded_crop(image: np.ndarray, tight_box: list[int], expansion: float) -> tuple[Image.Image, list[int]]:
    x0, y0, x1, y1 = [float(value) for value in tight_box]
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half_width = 0.5 * (x1 - x0) * expansion
    half_height = 0.5 * (y1 - y0) * expansion
    box = [
        max(0, int(math.floor(center_x - half_width))),
        max(0, int(math.floor(center_y - half_height))),
        min(image.shape[1], int(math.ceil(center_x + half_width))),
        min(image.shape[0], int(math.ceil(center_y + half_height))),
    ]
    require(box[0] < box[2] and box[1] < box[3], "CROP_EMPTY")
    return Image.fromarray(cv2.cvtColor(image[box[1] : box[3], box[0] : box[2]], cv2.COLOR_BGR2RGB)), box


def local_support(processor: Any, model: Any, reference: Image.Image, query: Image.Image, matcher: dict[str, Any]) -> dict[str, Any]:
    import torch

    inputs = processor([reference, query], return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    row = processor.post_process_keypoint_matching(
        outputs,
        [[(reference.height, reference.width), (query.height, query.width)]],
        threshold=float(matcher["postprocess_score_threshold"]),
    )[0]
    keypoints0 = row["keypoints0"].detach().cpu().numpy().astype(np.float32)
    keypoints1 = row["keypoints1"].detach().cpu().numpy().astype(np.float32)
    scores = row["matching_scores"].detach().cpu().numpy().astype(np.float32)
    count = int(len(scores))
    result = {"match_count": count, "mean_match_score": float(scores.mean()) if count else 0.0, "homography_inliers": 0, "homography_valid": False, "absolute_support": False}
    if count < int(matcher["minimum_matches"]):
        return result
    homography, mask = cv2.findHomography(keypoints0, keypoints1, cv2.RANSAC, float(matcher["homography_ransac_reprojection_pixels"]))
    if homography is None or mask is None or not np.isfinite(homography).all():
        return result
    inliers = int(mask.reshape(-1).sum())
    result["homography_inliers"] = inliers
    result["homography_valid"] = True
    result["absolute_support"] = inliers >= int(matcher["minimum_homography_inliers"])
    return result


def replay(protocol_path: Path, cohort_path: Path, receipt_path: Path, source_root: Path, output_path: Path) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoModelForKeypointMatching

    protocol = load_protocol(protocol_path)
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "COHORT_PROTOCOL_HASH")
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__)), "COHORT_ENTRYPOINT_HASH")
    require(receipt["cohort_sha256"] == sha256(cohort_path), "RECEIPT_COHORT_HASH")
    intrinsic = parent.base.parse_intrinsic(source_paths(source_root, cohort["episodes"][0]["scene_id"])["intrinsic"])
    crops: dict[str, Image.Image] = {}
    crop_receipts: dict[str, Any] = {}
    expansion = 1.25
    for episode in cohort["episodes"]:
        scene_id = episode["scene_id"]
        paths = source_paths(source_root, scene_id)
        xyz, labels, faces = parent.visible.read_mesh(paths["ply"])
        poses = {int(row["frame"]): row["camera_to_world"] for row in parent.base.parse_poses(paths["trajectory"])}
        renderer = parent.visible.VisibilityRenderer(xyz, labels, faces, int(episode["target_instance_id"]), intrinsic, protocol["pre_rgb_selector"]["renderer"])
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            sealed = receipt["sealed_frames"][f"{scene_id}:{frame}"]
            image_path = selected_image(source_root, scene_id, frame)
            require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{scene_id}:{frame}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(image is not None and image.shape == (480, 640, 3), f"IMAGE_FORMAT:{scene_id}:{frame}")
            visible_device, _ = renderer.visible_mask(poses[frame])
            visible_mask = renderer.cp.asnumpy(visible_device)
            require(parent.visible.mask_sha256(visible_mask) == episode[role]["visible_mask_sha256"], f"MASK_HASH:{scene_id}:{role}")
            ys, xs = np.where(visible_mask)
            tight = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            require(tight == episode[role]["visible_mask_tight_bbox_xyxy"], f"TIGHT_BOX:{scene_id}:{role}")
            crop, box = expanded_crop(image, tight, expansion)
            key = f"{episode['episode_id']}:{role}"
            crops[key] = crop
            crop_receipts[key] = {"scene_id": scene_id, "frame": frame, "source_image_sha256": sealed["image_sha256"], "visible_mask_sha256": episode[role]["visible_mask_sha256"], "tight_bbox_xyxy": tight, "expanded_bbox_xyxy": box, "crop_size": [crop.width, crop.height]}
    dino_root = ROOT / "artifacts.local/models/p1_a2_dinov2_small_ed25f3a"
    require(sha256(dino_root / "model.safetensors") == "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1", "DINO_WEIGHTS")
    dino_processor = AutoImageProcessor.from_pretrained(dino_root, local_files_only=True)
    dino_model = AutoModel.from_pretrained(dino_root, local_files_only=True).eval().to("cpu")
    ordered_keys = sorted(crops)
    dino_inputs = dino_processor(images=[crops[key] for key in ordered_keys], return_tensors="pt")
    with torch.inference_mode():
        dino_hidden = dino_model(**dino_inputs).last_hidden_state[:, 1:].mean(dim=1)
        dino_hidden = torch.nn.functional.normalize(dino_hidden, dim=1)
    dino = {key: dino_hidden[index].cpu().numpy().astype(np.float32) for index, key in enumerate(ordered_keys)}
    del dino_model
    matcher_root = ROOT / protocol["matcher"]["path"]
    matcher_processor = AutoImageProcessor.from_pretrained(matcher_root, local_files_only=True)
    matcher_model = AutoModelForKeypointMatching.from_pretrained(matcher_root, local_files_only=True).eval().to("cpu")
    episode_ids = [row["episode_id"] for row in cohort["episodes"]]
    dino_scores = np.zeros((4, 4), dtype=np.float64)
    local_scores = np.zeros((4, 4), dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    for row, reference_id in enumerate(episode_ids):
        for column, query_id in enumerate(episode_ids):
            dino_scores[row, column] = float(dino[f"{reference_id}:reference"] @ dino[f"{query_id}:query"])
            support = local_support(matcher_processor, matcher_model, crops[f"{reference_id}:reference"], crops[f"{query_id}:query"], protocol["matcher"])
            local_scores[row, column] = float(support["homography_inliers"]) if support["absolute_support"] else 0.0
            diagnostics[f"{reference_id}->{query_id}"] = support
    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenarios: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        dino_matrix = dino_scores[np.ix_(rows, columns)]
        local_matrix = local_scores[np.ix_(rows, columns)]
        dino_matches = parent.open_zero.reciprocal_zero_assignment(dino_matrix)
        local_matches = parent.open_zero.reciprocal_zero_assignment(local_matrix)
        scenarios.append({
            **scenario,
            "dinov2_patch_mean_cosine_matrix": dino_matrix.round(6).tolist(),
            "efficientloftr_supported_inlier_matrix": local_matrix.astype(int).tolist(),
            "methods": {
                "dinov2_reciprocal_no_none_support": parent.open_zero.evaluate_matches(references, queries, dino_matches),
                "efficientloftr_absolute_support_reciprocal": parent.open_zero.evaluate_matches(references, queries, local_matches),
            },
        })
    aggregates = {name: parent.open_zero.aggregate(scenarios, name) for name in scenarios[0]["methods"]}
    baseline = aggregates["dinov2_reciprocal_no_none_support"]
    upgraded = aggregates["efficientloftr_absolute_support_reciprocal"]
    expected_true = int(cohort["counts"]["truth_matches_across_scenarios"])
    diagonal_supported = [bool(diagnostics[f"{episode_id}->{episode_id}"]["absolute_support"]) for episode_id in episode_ids]
    gate_met = all(diagonal_supported) and upgraded["true_positive"] == expected_true and upgraded["false_positive"] == 0 and upgraded["false_negative"] == 0 and upgraded["zero_assignment_exact_scenarios"] == len(scenarios) and upgraded["f1"] > baseline["f1"]
    parent.write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_MULTISCENE_TRUTH_CROP_LOCAL_CORRESPONDENCE_NONE_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "receipt_path": receipt_path.name,
        "receipt_sha256": sha256(receipt_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "matcher": {"path": str(matcher_root.resolve()), "model_safetensors_sha256": protocol["matcher"]["model_safetensors_sha256"], "device": "CPU", "model_calls": 16},
        "conclusion": "L10_SCENENN_EFFICIENTLOFTR_FRESH_NONE_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_EFFICIENTLOFTR_FRESH_NONE_DEVELOPMENT_GATE_NOT_MET",
        "gate_met": gate_met,
        "metrics": {"aggregate": aggregates, "scenarios": scenarios, "full_dinov2_patch_mean_cosine_matrix": dino_scores.round(6).tolist(), "full_efficientloftr_supported_inlier_matrix": local_scores.astype(int).tolist(), "pair_diagnostics": diagnostics, "true_pair_absolute_support": diagonal_supported},
        "incremental_gain_over_dinov2": {"true_positive_delta": upgraded["true_positive"] - baseline["true_positive"], "false_positive_reduction": baseline["false_positive"] - upgraded["false_positive"], "false_negative_reduction": baseline["false_negative"] - upgraded["false_negative"], "f1_delta": round(upgraded["f1"] - baseline["f1"], 6), "exact_scenario_gain": upgraded["zero_assignment_exact_scenarios"] - baseline["zero_assignment_exact_scenarios"]},
        "crop_receipts": crop_receipts,
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--admission", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cohort", type=Path, required=True)
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--extractor-exe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.admission, args.source_root, args.output)
    elif args.action == "seal":
        seal(args.cohort, args.source_root, args.extraction_root, args.extractor_exe, args.output)
    else:
        replay(args.protocol, args.cohort, args.receipt, args.source_root, args.output)


if __name__ == "__main__":
    main()

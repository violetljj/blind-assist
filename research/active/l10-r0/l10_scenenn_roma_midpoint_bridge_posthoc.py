#!/usr/bin/env python3
"""Consumed SceneNN midpoint bridge for temporally local RoMa support."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


_NVIDIA_ROOT = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
_DLL_DIRECTORY_HANDLES = []
if os.name == "nt" and _NVIDIA_ROOT.is_dir():
    for _dll_dir in sorted(_NVIDIA_ROOT.glob("*/bin")):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(_dll_dir)))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_roma_active_none as roma_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-roma-midpoint-bridge-posthoc-protocol-v1"
ADMISSION_SCHEMA = "blindassist-l10-scenenn-roma-midpoint-bridge-posthoc-source-admission-v1"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-roma-midpoint-bridge-posthoc-rgb-receipt-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-roma-midpoint-bridge-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return roma_base.sha256(path)


def bridge_image(source_root: Path, scene_id: str, frame: int) -> Path:
    return source_root / "payload" / scene_id / "bridge_selected" / "image" / f"frame.{frame:04d}.png"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(protocol["implementation"]["sha256"] == sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    require(sha256(predecessor_path) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    require(load_json(predecessor_path)["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    endpoint_receipt_path = HERE / protocol["source"]["endpoint_rgb_receipt_path"]
    require(sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    require(sha256(endpoint_receipt_path) == protocol["source"]["endpoint_rgb_receipt_sha256"], "ENDPOINT_RECEIPT_HASH")
    model_root = ROOT / protocol["matcher"]["path"]
    require(sha256(model_root / "roma_indoor.pth") == protocol["matcher"]["roma_weights_sha256"], "ROMA_WEIGHTS")
    require(sha256(model_root / "dinov2_vitl14_pretrain.pth") == protocol["matcher"]["dinov2_weights_sha256"], "ROMA_DINOV2_WEIGHTS")
    return protocol


def seal(
    protocol_path: Path,
    admission_path: Path,
    source_root: Path,
    extraction_root: Path,
    extractor_exe: Path,
    output_path: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    admission = load_json(admission_path)
    require(admission.get("schema") == ADMISSION_SCHEMA, "ADMISSION_SCHEMA")
    require(admission["protocol_sha256"] == sha256(protocol_path), "ADMISSION_PROTOCOL_HASH")
    require(admission["selected"] == protocol["midpoint_selector"]["selected"], "MIDPOINT_SELECTION")
    sealed: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for row in admission["selected"]:
        scene_id = str(row["scene_id"])
        frame = int(row["midpoint_frame"])
        scene_extract = extraction_root / scene_id
        expected_name = f"frame.{frame:04d}.png"
        images = sorted((scene_extract / "image").glob("*.png"))
        depths = sorted((scene_extract / "depth").glob("*.png"))
        require([path.name for path in images] == [expected_name], f"IMAGE_SET:{scene_id}")
        require([path.name for path in depths] == [expected_name], f"DEPTH_SET:{scene_id}")
        timestamps = roma_base.predecessor.parent.visible.parse_selected_timestamps(scene_extract / "selected_timestamp.txt")
        summary = roma_base.predecessor.parent.visible.parse_extraction_summary(scene_extract / "summary.txt")
        require(set(timestamps) == {frame}, f"TIMESTAMP_SET:{scene_id}")
        require(summary["requested"] == 1 and summary["saved"] == 1, f"EXTRACTION_COUNT:{scene_id}")
        source_image = images[0]
        source_depth = depths[0]
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
        require(image is not None and image.shape == (480, 640, 3), f"IMAGE_FORMAT:{scene_id}")
        require(depth is not None and depth.shape == (480, 640) and depth.dtype == np.uint16, f"DEPTH_FORMAT:{scene_id}")
        target = bridge_image(source_root, scene_id, frame)
        roma_base.predecessor.parent.base.atomic_write(target, source_image.read_bytes())
        summaries[scene_id] = summary
        sealed[f"{scene_id}:{frame}"] = {
            "scene_id": scene_id,
            "trajectory_frame": frame,
            "timestamp": timestamps[frame],
            "image_path": target.resolve().relative_to(source_root.resolve()).as_posix(),
            "image_sha256": sha256(target),
            "depth_extraction_sha256": sha256(source_depth),
            "depth_retained": False,
        }
    roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RECEIPT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_MIDPOINT_SPARSE_RGB_ONLY_RECEIPT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "source_admission_path": admission_path.name,
            "source_admission_sha256": sha256(admission_path),
            "extractor_executable": str(extractor_exe.resolve()),
            "extractor_sha256": sha256(extractor_exe),
            "extractor_source_sha256": sha256(roma_base.predecessor.parent.visible.EXTRACTOR_SOURCE_PATH),
            "extraction_summaries": summaries,
            "sealed_frames": sealed,
            "rgb_frames_retained": len(sealed),
            "depth_frames_retained": 0,
        },
    )


def replay(protocol_path: Path, bridge_receipt_path: Path, source_root: Path, output_path: Path) -> None:
    from transformers import AutoImageProcessor, AutoModel
    import romatch

    protocol = load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    endpoint_receipt_path = HERE / protocol["source"]["endpoint_rgb_receipt_path"]
    cohort = load_json(cohort_path)
    endpoint_receipt = load_json(endpoint_receipt_path)
    bridge_receipt = load_json(bridge_receipt_path)
    require(bridge_receipt.get("schema") == RECEIPT_SCHEMA, "BRIDGE_RECEIPT_SCHEMA")
    require(bridge_receipt["protocol_sha256"] == sha256(protocol_path), "BRIDGE_PROTOCOL_HASH")
    intrinsic = roma_base.predecessor.parent.base.parse_intrinsic(
        roma_base.predecessor.source_paths(source_root, cohort["episodes"][0]["scene_id"])["intrinsic"]
    )
    selected_midpoint = {row["episode_id"]: int(row["midpoint_frame"]) for row in protocol["midpoint_selector"]["selected"]}
    crops: dict[str, Image.Image] = {}
    crop_receipts: dict[str, Any] = {}
    expansion = float(protocol["crop"]["expansion"])
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        scene_id = str(episode["scene_id"])
        target_id = int(episode["target_instance_id"])
        paths = roma_base.predecessor.source_paths(source_root, scene_id)
        xyz, labels, faces = roma_base.predecessor.parent.visible.read_mesh(paths["ply"])
        poses = {int(row["frame"]): row["camera_to_world"] for row in roma_base.predecessor.parent.base.parse_poses(paths["trajectory"])}
        renderer = roma_base.predecessor.parent.visible.VisibilityRenderer(
            xyz, labels, faces, target_id, intrinsic, protocol["renderer"]
        )
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            sealed = endpoint_receipt["sealed_frames"][f"{scene_id}:{frame}"]
            image_path = roma_base.predecessor.selected_image(source_root, scene_id, frame)
            require(sha256(image_path) == sealed["image_sha256"], f"ENDPOINT_IMAGE_HASH:{scene_id}:{frame}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            crop, box = roma_base.predecessor.expanded_crop(
                image, episode[role]["visible_mask_tight_bbox_xyxy"], expansion
            )
            key = f"{episode_id}:{role}"
            crops[key] = crop
            crop_receipts[key] = {
                "scene_id": scene_id,
                "frame": frame,
                "tight_bbox_xyxy": episode[role]["visible_mask_tight_bbox_xyxy"],
                "expanded_bbox_xyxy": box,
                "crop_size": [crop.width, crop.height],
                "source_image_sha256": sealed["image_sha256"],
            }
        midpoint = selected_midpoint[episode_id]
        sealed_midpoint = bridge_receipt["sealed_frames"][f"{scene_id}:{midpoint}"]
        midpoint_path = bridge_image(source_root, scene_id, midpoint)
        require(sha256(midpoint_path) == sealed_midpoint["image_sha256"], f"MIDPOINT_IMAGE_HASH:{scene_id}")
        image = cv2.imread(str(midpoint_path), cv2.IMREAD_COLOR)
        visible_device, _ = renderer.visible_mask(poses[midpoint])
        visible_mask = renderer.cp.asnumpy(visible_device)
        ys, xs = np.where(visible_mask)
        require(len(xs) > 0, f"MIDPOINT_MASK_EMPTY:{scene_id}")
        tight = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        crop, box = roma_base.predecessor.expanded_crop(image, tight, expansion)
        key = f"{episode_id}:midpoint"
        crops[key] = crop
        crop_receipts[key] = {
            "scene_id": scene_id,
            "frame": midpoint,
            "visible_mask_sha256": roma_base.predecessor.parent.visible.mask_sha256(visible_mask),
            "tight_bbox_xyxy": tight,
            "expanded_bbox_xyxy": box,
            "crop_size": [crop.width, crop.height],
            "source_image_sha256": sealed_midpoint["image_sha256"],
        }

    dino_root = ROOT / "artifacts.local/models/p1_a2_dinov2_small_ed25f3a"
    require(sha256(dino_root / "model.safetensors") == protocol["baseline"]["model_safetensors_sha256"], "DINO_WEIGHTS")
    dino_processor = AutoImageProcessor.from_pretrained(dino_root, local_files_only=True)
    dino_model = AutoModel.from_pretrained(dino_root, local_files_only=True).eval().to("cpu")
    ordered_keys = sorted(crops)
    dino_inputs = dino_processor(images=[crops[key] for key in ordered_keys], return_tensors="pt")
    with torch.inference_mode():
        dino_hidden = dino_model(**dino_inputs).last_hidden_state[:, 1:].mean(dim=1)
        dino_hidden = F.normalize(dino_hidden, dim=1)
    dino = {key: dino_hidden[index].cpu().numpy().astype(np.float32) for index, key in enumerate(ordered_keys)}
    del dino_model

    model_root = ROOT / protocol["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
    matcher_model = romatch.roma_indoor(
        device="cuda",
        weights=weights,
        dinov2_weights=dinov2_weights,
        coarse_res=int(protocol["matcher"]["coarse_resolution"]),
        upsample_res=int(protocol["matcher"]["upsample_resolution"]),
        symmetric=True,
        use_custom_corr=False,
        upsample_preds=True,
    )
    episode_ids = [row["episode_id"] for row in cohort["episodes"]]
    count = len(episode_ids)
    dino_scores = np.zeros((count, count), dtype=np.float64)
    roma_scores = np.zeros((count, count), dtype=np.float64)
    entry_diagnostics: dict[str, Any] = {}
    continuation_diagnostics: dict[str, Any] = {}
    continuation_dino: dict[str, float] = {}
    for query_id in episode_ids:
        continuation = roma_base.roma_support(
            matcher_model, crops[f"{query_id}:midpoint"], crops[f"{query_id}:query"], protocol["matcher"]
        )
        continuation_diagnostics[f"{query_id}:midpoint->query"] = continuation
        continuation_dino[query_id] = float(dino[f"{query_id}:midpoint"] @ dino[f"{query_id}:query"])
    for row, reference_id in enumerate(episode_ids):
        for column, query_id in enumerate(episode_ids):
            entry = roma_base.roma_support(
                matcher_model, crops[f"{reference_id}:reference"], crops[f"{query_id}:midpoint"], protocol["matcher"]
            )
            entry_diagnostics[f"{reference_id}->{query_id}:midpoint"] = entry
            entry_dino = float(dino[f"{reference_id}:reference"] @ dino[f"{query_id}:midpoint"])
            dino_scores[row, column] = math.sqrt(max(0.0, entry_dino) * max(0.0, continuation_dino[query_id]))
            continuation = continuation_diagnostics[f"{query_id}:midpoint->query"]
            if entry["absolute_support"] and continuation["absolute_support"]:
                roma_scores[row, column] = math.sqrt(
                    float(entry["symmetric_cycle_score"]) * float(continuation["symmetric_cycle_score"])
                )

    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenarios: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        dino_matrix = dino_scores[np.ix_(rows, columns)]
        roma_matrix = roma_scores[np.ix_(rows, columns)]
        dino_matches = roma_base.predecessor.parent.open_zero.reciprocal_zero_assignment(dino_matrix)
        roma_matches = roma_base.predecessor.parent.open_zero.reciprocal_zero_assignment(roma_matrix)
        scenarios.append(
            {
                **scenario,
                "midpoint_bridge_dinov2_matrix": dino_matrix.round(6).tolist(),
                "midpoint_bridge_roma_matrix": roma_matrix.round(6).tolist(),
                "methods": {
                    "midpoint_bridge_dinov2_reciprocal_no_none_support": roma_base.predecessor.parent.open_zero.evaluate_matches(references, queries, dino_matches),
                    "midpoint_bridge_roma_both_edges_supported_reciprocal": roma_base.predecessor.parent.open_zero.evaluate_matches(references, queries, roma_matches),
                },
            }
        )
    aggregates = {name: roma_base.predecessor.parent.open_zero.aggregate(scenarios, name) for name in scenarios[0]["methods"]}
    upgraded = aggregates["midpoint_bridge_roma_both_edges_supported_reciprocal"]
    diagonal_supported = [
        bool(entry_diagnostics[f"{episode_id}->{episode_id}:midpoint"]["absolute_support"])
        and bool(continuation_diagnostics[f"{episode_id}:midpoint->query"]["absolute_support"])
        for episode_id in episode_ids
    ]
    gate = protocol["decision_gate"]
    gate_met = (
        sum(diagonal_supported) >= int(gate["minimum_true_pairs_with_absolute_support"])
        and upgraded["true_positive"] > int(gate["predecessor_true_positive"])
        and upgraded["false_positive"] == int(gate["maximum_false_positive"])
        and upgraded["f1"] > float(gate["predecessor_f1"])
        and upgraded["zero_assignment_exact_scenarios"] >= int(gate["minimum_exact_scenarios"])
    )
    predecessor_metrics = protocol["predecessor"]["observed_metrics"]
    roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_MIDPOINT_BRIDGE_ROMA_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "source": {
                "cohort_path": cohort_path.name,
                "cohort_sha256": sha256(cohort_path),
                "endpoint_rgb_receipt_path": endpoint_receipt_path.name,
                "endpoint_rgb_receipt_sha256": sha256(endpoint_receipt_path),
                "bridge_rgb_receipt_path": bridge_receipt_path.name,
                "bridge_rgb_receipt_sha256": sha256(bridge_receipt_path),
                "source_root": str(source_root.resolve()),
            },
            "conclusion": "L10_SCENENN_ROMA_MIDPOINT_BRIDGE_POSTHOC_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_ROMA_MIDPOINT_BRIDGE_POSTHOC_DEVELOPMENT_GATE_NOT_MET",
            "gate_met": gate_met,
            "metrics": {
                "aggregate": aggregates,
                "scenarios": scenarios,
                "full_midpoint_bridge_dinov2_matrix": dino_scores.round(6).tolist(),
                "full_midpoint_bridge_roma_matrix": roma_scores.round(6).tolist(),
                "entry_edge_diagnostics": entry_diagnostics,
                "continuation_edge_diagnostics": continuation_diagnostics,
                "true_pair_absolute_support": diagonal_supported,
            },
            "gain_over_predecessor_roma": {
                "true_positive_delta": upgraded["true_positive"] - int(predecessor_metrics["true_positive"]),
                "false_positive_delta": upgraded["false_positive"] - int(predecessor_metrics["false_positive"]),
                "false_negative_reduction": int(predecessor_metrics["false_negative"]) - upgraded["false_negative"],
                "f1_delta": round(upgraded["f1"] - float(predecessor_metrics["f1"]), 6),
                "exact_scenario_gain": upgraded["zero_assignment_exact_scenarios"] - int(predecessor_metrics["exact_scenarios"]),
                "true_support_gain": sum(diagonal_supported) - int(predecessor_metrics["true_pairs_supported"]),
            },
            "crop_receipts": crop_receipts,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--protocol", type=Path, required=True)
    seal_parser.add_argument("--admission", type=Path, required=True)
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--extractor-exe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--bridge-receipt", type=Path, required=True)
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "seal":
        seal(args.protocol, args.admission, args.source_root, args.extraction_root, args.extractor_exe, args.output)
    else:
        replay(args.protocol, args.bridge_receipt, args.source_root, args.output)


if __name__ == "__main__":
    main()

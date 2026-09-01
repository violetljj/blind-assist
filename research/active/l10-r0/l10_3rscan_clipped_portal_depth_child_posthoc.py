#!/usr/bin/env python3
"""Reserve one bounded hypothesis slot for a clipped, depth-occluded portal track."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-clipped-portal-depth-child-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-clipped-portal-depth-child-posthoc-result-v1"


def _depth_cloud(archive: zipfile.ZipFile, frame: int, info: dict[str, Any]) -> tuple[np.ndarray, ...]:
    depth = pixel.decode_depth(archive, frame)
    rows, columns = np.indices(depth.shape)
    metres = depth.astype(np.float64) / 1000.0
    intrinsic = info["depth_intrinsic"]
    camera = np.stack(
        (
            (columns - intrinsic[0, 2]) * metres / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * metres / intrinsic[1, 1],
            metres,
        ),
        axis=-1,
    )
    projected = camera @ info["color_intrinsic"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        color = projected[..., :2] / projected[..., 2:3]
    finite = np.isfinite(color).all(axis=-1)
    xs = np.zeros(depth.shape, dtype=np.int64)
    ys = np.zeros(depth.shape, dtype=np.int64)
    xs[finite] = np.rint(color[..., 0][finite]).astype(np.int64)
    ys[finite] = np.rint(color[..., 1][finite]).astype(np.int64)
    valid = (
        (metres > 0.0)
        & finite
        & (xs >= 0)
        & (xs < int(info["color_width"]))
        & (ys >= 0)
        & (ys < int(info["color_height"]))
    )
    return xs[valid], ys[valid], metres[valid]


def _depth_child(
    box: list[float], cloud: tuple[np.ndarray, ...], rules: dict[str, Any]
) -> tuple[list[float], float, dict[str, Any]] | None:
    xs, ys, metres = cloud
    x1, y1, x2, y2 = [float(value) for value in box]
    width = x2 - x1
    height = y2 - y1
    inset = float(rules["horizontal_inset_fraction"])
    inside = (
        (xs >= x1 + inset * width)
        & (xs <= x2 - inset * width)
        & (ys >= y1)
        & (ys <= y2)
    )
    sample_y = ys[inside]
    sample_depth = metres[inside]
    if len(sample_depth) < int(rules["minimum_depth_points"]):
        return None
    bin_count = int(rules["vertical_bin_count"])
    edges = np.linspace(y1, y2, bin_count + 1)
    medians = np.full(bin_count, np.nan, dtype=np.float64)
    for index in range(bin_count):
        selected = (sample_y >= edges[index]) & (sample_y < edges[index + 1])
        if int(np.count_nonzero(selected)) >= int(rules["minimum_points_per_bin"]):
            medians[index] = float(np.median(sample_depth[selected]))
    finite = medians[np.isfinite(medians)]
    if len(finite) < int(rules["minimum_valid_bins"]):
        return None
    dominant = float(np.quantile(finite, float(rules["dominant_depth_quantile"])))
    tolerance = max(
        float(rules["absolute_depth_tolerance_metres"]),
        float(rules["relative_depth_tolerance"]) * dominant,
    )
    supported = medians >= dominant - tolerance
    low_run = int(rules["consecutive_low_bins"])
    first_search = max(2, bin_count // int(rules["search_start_divisor"]))
    best: tuple[float, int] | None = None
    for index in range(first_search, bin_count - low_run):
        following = medians[index + 1 : index + 1 + low_run]
        if supported[index] and np.isfinite(following).all() and np.all(following < dominant - tolerance):
            drop = float(dominant - np.median(following))
            if best is None or drop > best[0]:
                best = (drop, index + 1)
    if best is None:
        return None
    drop, cut_index = best
    cut_y = float(edges[cut_index])
    retained = float((cut_y - y1) / height)
    if not float(rules["minimum_retained_fraction"]) <= retained <= float(rules["maximum_retained_fraction"]):
        return None
    score = float((drop / dominant) * min(1.0, (1.0 - retained) / float(rules["full_occlusion_fraction"])))
    return [x1, y1, x2, cut_y], score, {
        "dominant_depth_metres": dominant,
        "depth_tolerance_metres": tolerance,
        "post_plane_drop_metres": drop,
        "retained_height_fraction": retained,
        "cut_y": cut_y,
        "valid_depth_bins": int(len(finite)),
    }


def _border(box: list[float], width: int, margin: float) -> str | None:
    if float(box[0]) <= margin:
        return "left"
    if float(box[2]) >= float(width) - margin:
        return "right"
    return None


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    predecessor_path = HERE / protocol["predecessor"]["path"]
    cohort_path = HERE / protocol["cohort"]["path"]
    pixel.require(pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"], "PREDECESSOR_HASH")
    pixel.require(pixel.sha256(cohort_path) == protocol["cohort"]["sha256"], "COHORT_HASH")
    predecessor = pixel.load_json(predecessor_path)
    cohort = pixel.load_json(cohort_path)
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    pixel.require(cohort["schema"] == protocol["cohort"]["required_schema"], "COHORT_SCHEMA")
    episodes = list(predecessor["episodes"])
    pixel.require(len(episodes) == 2, "TWO_VIEW_ONLY")
    scan_ids = {cohort["images"][row["query_key"]]["scan_id"] for row in episodes}
    pixel.require(len(scan_ids) == 1, "QUERY_SCAN_MISMATCH")
    scan_id = next(iter(scan_ids))
    source = cohort["source_manifest"][f"{scan_id}/sequence.zip"]
    zip_path = ROOT / protocol["source"]["artifact_root"] / source["path"]
    pixel.require(zip_path.stat().st_size == int(source["bytes"]), "ZIP_BYTES")
    pixel.require(pixel.sha256(zip_path) == source["sha256"], "ZIP_HASH")
    with zipfile.ZipFile(zip_path) as archive:
        info_payload = archive.read("_info.txt")
        info = pixel.parse_info(info_payload.decode("utf-8"))
        clouds = {
            row["query_key"]: _depth_cloud(archive, int(row["frame"]), info) for row in episodes
        }
        depth_hashes = {
            row["query_key"]: hashlib.sha256(
                archive.read(f"frame-{int(row['frame']):06d}.depth.pgm")
            ).hexdigest()
            for row in episodes
        }

    by_key = {
        episode["query_key"]: {
            int(candidate["candidate_index"]): candidate for candidate in episode["ranked_candidates"]
        }
        for episode in episodes
    }
    first, second = episodes
    rules = protocol["depth_child"]
    candidates = []
    for rank, left in enumerate(first["ranked_candidates"], start=1):
        edges = list(left["edges"])
        if int(left["mutual_best_edge_count"]) != 1 or len(edges) != 1 or not bool(edges[0]["mutual_best"]):
            continue
        right = by_key[second["query_key"]].get(int(edges[0]["partner_index"]))
        if right is None:
            continue
        if left["proposal"].get("proposal_prompt") != rules["required_prompt"]:
            continue
        if right["proposal"].get("proposal_prompt") != rules["required_prompt"]:
            continue
        left_border = _border(left["proposal"]["box_xyxy"], int(info["color_width"]), float(rules["border_margin_pixels"]))
        right_border = _border(right["proposal"]["box_xyxy"], int(info["color_width"]), float(rules["border_margin_pixels"]))
        if left_border is None or left_border != right_border:
            continue
        left_child = _depth_child(left["proposal"]["box_xyxy"], clouds[first["query_key"]], rules)
        right_child = _depth_child(right["proposal"]["box_xyxy"], clouds[second["query_key"]], rules)
        if left_child is None or right_child is None:
            continue
        candidates.append(
            {
                "left_rank": rank,
                "left_candidate_index": int(left["candidate_index"]),
                "right_candidate_index": int(right["candidate_index"]),
                "shared_border": left_border,
                "track_score": float(left["track_score"]),
                "pair_depth_score": float(min(left_child[1], right_child[1])),
                "children": {
                    first["query_key"]: {"box_xyxy": left_child[0], "depth": left_child[2]},
                    second["query_key"]: {"box_xyxy": right_child[0], "depth": right_child[2]},
                },
            }
        )
    pixel.require(candidates, "NO_CLIPPED_PORTAL_DEPTH_CHILD")
    candidates.sort(key=lambda row: (-row["track_score"], -row["pair_depth_score"], row["left_rank"]))
    selected = candidates[0]

    evaluation = []
    for episode in episodes:
        key = episode["query_key"]
        truth = cohort["images"][key]["bbox_xyxy"]
        child = selected["children"][key]
        child["target_metrics_evaluation_only"] = nids.base._bbox_metrics(child["box_xyxy"], truth)
        ordinary = list(episode["ranked_candidates"][: int(protocol["bounded_set"]["ordinary_track_slots"])])
        bounded = [
            {
                "source": "ordinary_3d_track",
                "candidate_index": int(row["candidate_index"]),
                "box_xyxy": row["proposal"]["box_xyxy"],
                "target_metrics_evaluation_only": row["target_metrics_evaluation_only"],
            }
            for row in ordinary
        ]
        bounded.append(
            {
                "source": "clipped_portal_depth_child",
                "candidate_index": int(
                    selected["left_candidate_index"] if key == first["query_key"] else selected["right_candidate_index"]
                ),
                "box_xyxy": child["box_xyxy"],
                "target_metrics_evaluation_only": child["target_metrics_evaluation_only"],
            }
        )
        evaluation.append(
            {
                "query_key": key,
                "frame": int(episode["frame"]),
                "bounded_hypotheses": bounded,
                "original_track_top3_best_iou": float(episode["track_top3_best_iou_evaluation_only"]),
                "new_track_top3_best_iou": float(
                    max(row["target_metrics_evaluation_only"]["iou"] for row in bounded)
                ),
            }
        )
    coverage = sum(row["new_track_top3_best_iou"] >= float(protocol["gate"]["minimum_iou"]) for row in evaluation)
    minimum = min(row["new_track_top3_best_iou"] for row in evaluation)
    mean = float(np.mean([row["new_track_top3_best_iou"] for row in evaluation]))
    original_minimum = min(row["original_track_top3_best_iou"] for row in evaluation)
    met = coverage == len(evaluation) and minimum >= float(protocol["gate"]["minimum_iou"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "posthoc Development on a consumed source; not fresh confirmation",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": protocol["implementation"],
        "predecessor": protocol["predecessor"],
        "cohort": protocol["cohort"],
        "source_receipts": {
            "sequence_zip_sha256": source["sha256"],
            "info_sha256": hashlib.sha256(info_payload).hexdigest(),
            "depth_sha256_by_query": depth_hashes,
        },
        "mechanism": {
            "description": "Two ordinary 3D-track slots plus one independently born clipped-portal depth child; no score fusion and no candidate-set expansion.",
            "truth_used_for_selection": False,
            "rules": rules,
            "eligible_pair_count": len(candidates),
            "selected_pair": selected,
        },
        "episodes": evaluation,
        "metrics": {
            "query_count": len(evaluation),
            "bounded_candidate_count_per_query": int(protocol["bounded_set"]["maximum_candidates"]),
            "original_track_top3_coverage_frames": int(predecessor["metrics"]["track_top3_region_coverage_frames"]),
            "new_track_top3_coverage_frames": int(coverage),
            "original_minimum_track_top3_best_iou": float(original_minimum),
            "new_minimum_track_top3_best_iou": float(minimum),
            "new_mean_track_top3_best_iou": mean,
            "minimum_iou_gain": float(minimum - original_minimum),
        },
        "gate": {
            "minimum_iou": float(protocol["gate"]["minimum_iou"]),
            "met": bool(met),
        },
        "conclusion": (
            "L10_3RSCAN_CLIPPED_PORTAL_DEPTH_CHILD_POSTHOC_DEVELOPMENT_GATE_MET"
            if met
            else "L10_3RSCAN_CLIPPED_PORTAL_DEPTH_CHILD_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "next_action": "Require source-disjoint preregistered confirmation before treating the reserved geometry slot as general evidence.",
        "claim_boundary": "Recovers a bounded referent-region hypothesis only; it does not establish portal ownership, affordance, waypoint, arrival, safety, or handoff.",
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)


if __name__ == "__main__":
    main()

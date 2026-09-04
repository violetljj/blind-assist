#!/usr/bin/env python3
"""Freeze a pixel-blind SEVN address-door reference Development cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
from typing import Any

import l10_sevn_fresh_pan_panel as fresh
import l10_sevn_panolab as sevn
from l10_panolab import require, sha256_file, utc_now


ROOT = Path(__file__).resolve().parents[3]
ROUTE = ROOT / "research/active/l10-r0"
REFERENCE_NAMES = [
    "l10_sevn_panolab_source_v1.json",
    "l10_sevn_address_disjoint_source_v1.json",
    *[f"l10_sevn_fresh_pan_source_v{i}.json" for i in range(1, 6)],
]


def address(row: Any) -> tuple[str, str] | None:
    return sevn.address_key(row)


def box(row: Any) -> list[int]:
    return [int(row[name]) for name in ("x_min", "y_min", "x_max", "y_max")]


def disjoint_boxes(first: list[int], second: list[int]) -> bool:
    return min(first[2], second[2]) <= max(first[0], second[0]) or min(
        first[3], second[3]
    ) <= max(first[1], second[1])


def annotation(row: Any) -> dict[str, Any]:
    return {
        "frame_id": int(row.name),
        "street_name": address(row)[0],
        "house_number": address(row)[1],
        **{key: int(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")},
        "annotation_authority": "SEVN_HUMAN_DOOR_POLYGON_WITH_ADDRESS",
    }


def freeze(metadata_dir: Path, output_prefix: str) -> dict[str, Any]:
    import pandas as pd

    paths = {
        "public_source": ROUTE / f"{output_prefix}_source_v1.json",
        "evaluator_truth": ROUTE / f"{output_prefix}_truth_v1.json",
        "receipt": ROUTE / f"{output_prefix}_source_receipt_v1.json",
    }
    for path in paths.values():
        require(not path.exists(), f"refusing to overwrite: {path}")
    reference_paths = [ROUTE / name for name in REFERENCE_NAMES]
    excluded_addresses, excluded_frames = fresh.union_reference_identity(
        [json.loads(path.read_text(encoding="utf-8")) for path in reference_paths]
    )
    metadata_receipts = sevn.verify_upstream_files(metadata_dir)
    labels = pd.read_hdf(metadata_dir / "label.hdf5", key="df", mode="r")
    coords = pd.read_hdf(metadata_dir / "coord.hdf5", key="df", mode="r")
    with (metadata_dir / "graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    labels_by_frame = sevn.dataframe_rows_by_frame(labels)
    doors_by_address: dict[tuple[str, str], list[Any]] = defaultdict(list)
    doors_by_frame: dict[int, list[Any]] = defaultdict(list)
    for _, row in labels[labels.obj_type == "door"].iterrows():
        key = address(row)
        if key is None:
            continue
        doors_by_frame[int(row.name)].append(row)
        if key not in excluded_addresses and int(row.name) not in excluded_frames and row.name in coords.index:
            doors_by_address[key].append(row)

    eligible = []
    for goal in sevn.candidate_goals(labels, coords, graph):
        key = address(goal)
        frame = int(goal.name)
        if key in excluded_addresses or frame in excluded_frames:
            continue
        rows = doors_by_address[key]
        counts = Counter(int(row.name) for row in rows)
        if counts[frame] != 1:
            continue
        references = [row for row in rows if int(row.name) != frame and counts[int(row.name)] == 1]
        if len(references) < 2:
            continue
        references.sort(key=lambda row: (sevn.distance_xy_m(coords, frame, int(row.name)), int(row.name), int(row.x_min)))
        siblings = [
            row for row in doors_by_frame[frame]
            if address(row) != key and address(row) not in excluded_addresses
            and disjoint_boxes(box(goal), box(row))
        ]
        if not siblings:
            continue
        siblings.sort(key=lambda row: (*address(row), *box(row)))
        eligible.append((goal, references, siblings))

    used_frames = set(excluded_frames)
    used_addresses = set(excluded_addresses)
    selected = []
    for scenario in fresh.PAN_SCENARIOS:
        count = 0
        for goal, references, siblings in eligible:
            key, frame = address(goal), int(goal.name)
            if key in used_addresses or frame in used_frames:
                continue
            remaining = [row for row in references if int(row.name) not in used_frames]
            if len(remaining) < 2:
                continue
            heading = sevn.label_heading_degrees(goal.x_min, goal.x_max, float(coords.loc[frame].angle))
            left = scenario.startswith("PAN_LEFT")
            start = sevn.wrap360(heading + (-sevn.PAN_START_OFFSET_DEGREES if left else sevn.PAN_START_OFFSET_DEGREES))
            destination = sevn.wrap360(start + (sevn.PAN_DEGREES if left else -sevn.PAN_DEGREES))
            before = sevn.truth_for_view(labels_by_frame, coords, frame, [start], key)
            after = sevn.truth_for_view(labels_by_frame, coords, frame, [destination], key)
            if before["binding_state"] == "CORRECT_UNIQUE" or after["binding_state"] != "CORRECT_UNIQUE":
                continue
            chosen_refs = remaining[:2]
            selected.append((scenario, goal, start, destination, chosen_refs, siblings))
            used_addresses.add(key)
            used_frames.update([frame, *[int(row.name) for row in chosen_refs]])
            count += 1
            if count == 4:
                break
    require(len(selected) == 8, f"NOT_EVALUABLE: only {len(selected)}/8 episodes; do not relax selection")

    episodes, observations, truth_episodes = [], {}, {}
    for sequence, (scenario, goal, start, destination, references, siblings) in enumerate(selected, 1):
        frame = int(goal.name)
        episode, truth, views = sevn.build_episode(
            sequence, scenario, goal, frame, start, frame, destination, coords, labels_by_frame, True
        )
        for view in views.values():
            view["image_asset"] = {"container": "high-res-panos.zip", "member_pattern": "panos/pano_{frame_id:06d}.png", "frame_id": frame, "payload_available": True, "selection_accessed_payload": False}
        episode["public_reference_views"] = [
            {
                "reference_id": f"{episode['episode_id']}_REF{index}",
                "role": "PRIVILEGED_LABELED_TARGET_CROP_BOOTSTRAP",
                "identity_scope": "SAME_ADDRESS_DOOR_SURROGATE",
                "inference_interface": "TARGET_CROP_PIXELS_ONLY_NO_FRAME_POSE_ANNOTATION_OR_QUERY_TRUTH",
                "extraction": {
                    "annotation": annotation(row),
                    "panorama_angle_degrees": float(coords.loc[int(row.name)].angle),
                    "heading_degrees": sevn.label_heading_degrees(row.x_min, row.x_max, float(coords.loc[int(row.name)].angle)),
                    "horizontal_fov_degrees": sevn.VIEWPORT_FOV_DEGREES,
                    "image_asset": {"container": "high-res-panos.zip", "member_pattern": "panos/pano_{frame_id:06d}.png", "frame_id": int(row.name)},
                },
            }
            for index, row in enumerate(references, 1)
        ]
        truth["different_address_nonoverlapping_sibling_annotations"] = [annotation(row) for row in siblings]
        truth["overlapping_different_address_annotations_not_negative_controls"] = [
            annotation(row) for row in doors_by_frame[frame]
            if address(row) != address(goal) and not disjoint_boxes(box(goal), box(row))
        ]
        episodes.append(episode)
        observations.update(views)
        truth_episodes[episode["episode_id"]] = truth

    absent_controls = []
    for index, episode in enumerate(episodes):
        key = (episode["mission"]["street_name"], episode["mission"]["house_number"])
        for offset in range(1, len(episodes)):
            other = episodes[(index + offset) % len(episodes)]
            frame = int(observations[other["start_observation_id"]]["frame_id"])
            if all((row["street_name"], row["house_number"]) != key for row in labels_by_frame[frame]):
                absent_controls.append({
                    "control_id": f"{episode['episode_id']}_CROSS_QUERY_LABEL_ABSENT",
                    "reference_episode_id": episode["episode_id"],
                    "query_episode_id": other["episode_id"],
                    "query_frame_id": frame,
                    "truth_scope": "NO_REQUESTED_ADDRESS_LABEL_ANYWHERE_IN_QUERY_PANORAMA_METADATA_NOT_PHYSICAL_ABSENCE",
                })
                break
    require(len(absent_controls) == 8, "NOT_EVALUABLE: missing annotation-level absent controls")
    source = {
        "schema": "blindassist-l10-sevn-panolab-public-cohort-v1", "generated_at_utc": utc_now(),
        "provider": "SEVN 1.0 / Zenodo 3526490", "license": "MIT", "source_receipts": metadata_receipts,
        "action_set": list(sevn.ACTIONS), "observations": observations, "episodes": episodes,
        "reference_contract": "PRIVILEGED_TARGET_CROP_BOOTSTRAP_SAME_ADDRESS_DOOR_SURROGATE_NO_PHYSICAL_INSTANCE_ID",
        "image_payload": {"available": True, "expected_file": "high-res-panos.zip", "status": "NOT_READ_DURING_SELECTION"},
    }
    truth = {"schema": "blindassist-l10-sevn-panolab-evaluator-truth-v1", "generated_at_utc": utc_now(), "truth_authority": "SEVN_HUMAN_ADDRESS_DOOR_ANNOTATIONS_NOT_EXACT_PHYSICAL_IDENTITY", "episodes": truth_episodes, "cross_query_label_absent_controls": absent_controls}
    fresh.validate_pan_public_truth(source, truth, 4)
    selected_frames = used_frames - excluded_frames
    require(len(selected_frames) == 24, "reference/query frames are not globally distinct")
    require(len(used_addresses - excluded_addresses) == 8, "target addresses are not distinct")
    receipt = {
        "schema": "blindassist-l10-sevn-reference-commitment-source-receipt-v1", "generated_at_utc": utc_now(),
        "selection": {"metadata_only": True, "pixel_payloads_opened": 0, "model_calls": 0,
            "candidate_count_two_references_and_nonoverlapping_sibling": len(eligible),
            "episode_count": 8, "reference_count": 16, "distinct_frame_count_including_references": 24,
            "scenario_counts": dict(Counter(row[0] for row in selected)),
            "rule": "Existing sorted candidate_goals; four PAN_LEFT then four PAN_RIGHT. Require original metadata PAN recovery, unique address-door per query/reference frame, two nearest-pose fresh references sorted by XY distance then frame then x_min, and a fresh-address nonoverlapping sibling. Greedy global exclusion of all selected target addresses and all query/reference frames. Cyclic next query with no requested address in any label is the label-absent control.",
            "overlapping_address_aliases_are_not_negative_controls": True},
        "reference_exclusions": {"prior_sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)} for path in reference_paths], "excluded_address_count": len(excluded_addresses), "excluded_frame_count": len(excluded_frames), "selected_address_overlap": [], "selected_frame_overlap_including_references": [], "within_panel_reused_frames": []},
        "inputs": {"metadata_directory": "artifacts.local/l10-sevn/metadata", "metadata": metadata_receipts, "materializer": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha256_file(Path(__file__))}},
        "claim_boundary": "Fresh same-source PAN-only Development; privileged reference crops; same-address door surrogate. No exact physical instance, facade ownership, physical target absence, new-sensor action, arrival, handoff, safety or deployment authority.",
    }
    for name, value in (("public_source", source), ("evaluator_truth", truth)):
        paths[name].write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt["outputs"] = {name: {"path": paths[name].relative_to(ROOT).as_posix(), "sha256": sha256_file(paths[name])} for name in ("public_source", "evaluator_truth")}
    paths["receipt"].write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "artifacts.local/l10-sevn/metadata")
    parser.add_argument("--output-prefix", default="l10_sevn_reference_commitment")
    args = parser.parse_args()
    print(json.dumps(freeze(args.metadata_dir, args.output_prefix), indent=2))


if __name__ == "__main__":
    main()

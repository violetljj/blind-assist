#!/usr/bin/env python3
"""Materialize a metadata-only SEVN panel disjoint from the consumed V1 panel.

The selector deliberately opens only SEVN coordinate, label, and graph metadata.
It excludes every target address and panorama frame present in the reference
panel, prevents cross-episode frame reuse inside the new panel, and preserves
the original PAN/APPROACH recovery construction without looking at pixels.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import l10_sevn_panolab as sevn
from l10_panolab import require, sha256_file, utc_now


SCHEMA = "blindassist-l10-sevn-address-panorama-disjoint-source-protocol-v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def verify_sha256(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"missing frozen input: {path}")
    require(sha256_file(path) == spec["sha256"], f"SHA-256 mismatch: {path}")
    return path


def reference_identity(public: dict[str, Any]) -> tuple[set[tuple[str, str]], set[int]]:
    addresses = {
        (str(episode["mission"]["street_name"]), str(episode["mission"]["house_number"]))
        for episode in public["episodes"]
    }
    frames = {int(row["frame_id"]) for row in public["observations"].values()}
    return addresses, frames


def select_panel(
    labels: Any,
    coords: Any,
    graph: Any,
    excluded_addresses: set[tuple[str, str]],
    excluded_frames: set[int],
    per_scenario: int,
) -> list[tuple[str, Any, int, float, int, float]]:
    labels_by_frame = sevn.dataframe_rows_by_frame(labels)
    goals = sevn.candidate_goals(labels, coords, graph)
    selected: list[tuple[str, Any, int, float, int, float]] = []
    used_addresses = set(excluded_addresses)
    used_frames = set(excluded_frames)

    # APPROACH is the constrained class because it consumes two graph frames.
    # Reserve it first, then fill the much denser single-frame PAN classes.
    scenario = sevn.SCENARIOS[2]
    approach_by_address: dict[
        tuple[str, str],
        list[tuple[float, int, int, float, Any]],
    ] = {}
    for goal in goals:
        target_address = sevn.address_key(goal)
        goal_frame = int(goal.name)
        if (
            target_address is None
            or target_address in excluded_addresses
            or goal_frame in excluded_frames
        ):
            continue
        candidates = []
        for neighbor_value in graph.neighbors(goal_frame):
            neighbor = int(neighbor_value)
            if neighbor not in coords.index or neighbor in excluded_frames:
                continue
            heading = sevn.quantize_heading(sevn.heading_between(coords, neighbor, goal_frame))
            before = sevn.truth_for_view(
                labels_by_frame,
                coords,
                neighbor,
                [heading],
                target_address,
            )
            after = sevn.truth_for_view(
                labels_by_frame,
                coords,
                goal_frame,
                [heading],
                target_address,
            )
            if before["binding_state"] == "CORRECT_UNIQUE" or after["binding_state"] != "CORRECT_UNIQUE":
                continue
            candidates.append(
                (sevn.distance_xy_m(coords, neighbor, goal_frame), neighbor, goal_frame, heading, goal)
            )
        if candidates:
            approach_by_address.setdefault(target_address, []).extend(
                sorted(candidates, key=lambda row: row[:4])
            )

    approach_groups = list(approach_by_address.items())

    def choose_approach(
        group_index: int,
        chosen: list[tuple[tuple[str, str], tuple[float, int, int, float, Any]]],
        occupied_frames: set[int],
    ) -> list[tuple[tuple[str, str], tuple[float, int, int, float, Any]]] | None:
        if len(chosen) == per_scenario:
            return chosen
        if group_index >= len(approach_groups):
            return None
        if len(chosen) + len(approach_groups) - group_index < per_scenario:
            return None
        address, options = approach_groups[group_index]
        for option in options:
            _, start_frame, goal_frame, _, _ = option
            if start_frame in occupied_frames or goal_frame in occupied_frames:
                continue
            result = choose_approach(
                group_index + 1,
                [*chosen, (address, option)],
                occupied_frames.union((start_frame, goal_frame)),
            )
            if result is not None:
                return result
        return choose_approach(group_index + 1, chosen, occupied_frames)

    chosen_approach = choose_approach(0, [], set(excluded_frames))
    require(chosen_approach is not None, "no feasible disjoint APPROACH set at frozen cohort size")
    for target_address, option in chosen_approach:
        _, start_frame, goal_frame, heading, goal = option
        selected.append((scenario, goal, start_frame, heading, goal_frame, heading))
        used_addresses.add(target_address)
        used_frames.update((start_frame, goal_frame))

    for scenario in sevn.SCENARIOS[:2]:
        for goal in goals:
            target_address = sevn.address_key(goal)
            goal_frame = int(goal.name)
            if (
                target_address is None
                or target_address in used_addresses
                or goal_frame in used_frames
            ):
                continue
            target_heading = sevn.label_heading_degrees(
                goal.x_min,
                goal.x_max,
                float(coords.loc[goal_frame].angle),
            )
            offset = (
                -sevn.PAN_START_OFFSET_DEGREES
                if scenario.startswith("PAN_LEFT")
                else sevn.PAN_START_OFFSET_DEGREES
            )
            start_heading = sevn.wrap360(target_heading + offset)
            destination_heading = sevn.wrap360(
                start_heading
                + (sevn.PAN_DEGREES if scenario.startswith("PAN_LEFT") else -sevn.PAN_DEGREES)
            )
            before = sevn.truth_for_view(
                labels_by_frame,
                coords,
                goal_frame,
                [start_heading],
                target_address,
            )
            after = sevn.truth_for_view(
                labels_by_frame,
                coords,
                goal_frame,
                [destination_heading],
                target_address,
            )
            if before["binding_state"] == "CORRECT_UNIQUE" or after["binding_state"] != "CORRECT_UNIQUE":
                continue
            selected.append((scenario, goal, goal_frame, start_heading, goal_frame, destination_heading))
            used_addresses.add(target_address)
            used_frames.add(goal_frame)
            if sum(item[0] == scenario for item in selected) == per_scenario:
                break

    counts = Counter(item[0] for item in selected)
    require(
        all(counts[scenario_name] == per_scenario for scenario_name in sevn.SCENARIOS),
        f"insufficient disjoint SEVN candidates: {dict(counts)}",
    )
    selected.sort(key=lambda item: sevn.SCENARIOS.index(item[0]))
    return selected


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    outputs = [args.source_out.resolve(), args.truth_out.resolve(), args.receipt_out.resolve()]
    for output in outputs:
        require(not output.exists(), f"refusing to overwrite frozen output: {output}")

    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    require(protocol.get("schema") == SCHEMA, "source protocol schema mismatch")
    require(
        sha256_file(Path(__file__).resolve()) == protocol["materializer"]["sha256"],
        "materializer hash mismatch",
    )
    reference_path = verify_sha256(protocol["frozen_inputs"]["reference_public_source"])
    reference = read_json(reference_path)
    sevn.validate_public(reference, int(protocol["reference_panel"]["episodes_per_scenario"]))
    excluded_addresses, excluded_frames = reference_identity(reference)

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise SystemExit("materialization requires pandas<2 and PyTables") from exc

    receipts = sevn.verify_upstream_files(args.metadata_dir.resolve())
    labels = pd.read_hdf(args.metadata_dir / "label.hdf5", key="df", mode="r")
    coords = pd.read_hdf(args.metadata_dir / "coord.hdf5", key="df", mode="r")
    with (args.metadata_dir / "graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)

    per_scenario = int(protocol["cohort"]["episodes_per_scenario"])
    selected = select_panel(
        labels,
        coords,
        graph,
        excluded_addresses,
        excluded_frames,
        per_scenario,
    )
    labels_by_frame = sevn.dataframe_rows_by_frame(labels)
    episodes = []
    truth_episodes: dict[str, Any] = {}
    observations: dict[str, Any] = {}
    episode_frames: dict[str, list[int]] = {}
    for sequence, item in enumerate(selected, start=1):
        scenario, goal, start_frame, start_heading, goal_frame, goal_heading = item
        episode, episode_truth, episode_observations = sevn.build_episode(
            sequence,
            scenario,
            goal,
            start_frame,
            start_heading,
            goal_frame,
            goal_heading,
            coords,
            labels_by_frame,
            True,
        )
        for observation in episode_observations.values():
            frame_id = int(observation["frame_id"])
            observation["image_asset"] = {
                "container": "high-res-panos.zip",
                "member_pattern": "panos/pano_{frame_id:06d}.png",
                "frame_id": frame_id,
                "payload_available": True,
                "selection_accessed_payload": False,
            }
        episodes.append(episode)
        truth_episodes[episode["episode_id"]] = episode_truth
        observations.update(episode_observations)
        episode_frames[episode["episode_id"]] = sorted(
            {int(row["frame_id"]) for row in episode_observations.values()}
        )

    generated_at = utc_now()
    public = {
        "schema": "blindassist-l10-sevn-panolab-public-cohort-v1",
        "generated_at_utc": generated_at,
        "provider": "SEVN 1.0 / Zenodo 3526490",
        "license": "MIT",
        "source_receipts": receipts,
        "image_payload": {
            "available": True,
            "expected_file": "high-res-panos.zip",
            "expected_bytes": int(protocol["high_resolution_archive"]["bytes"]),
            "expected_md5": protocol["high_resolution_archive"]["md5"],
            "status": "FROZEN_EXTERNAL_ARCHIVE_NOT_READ_DURING_SELECTION",
        },
        "action_set": list(sevn.ACTIONS),
        "observation_contract": "PUBLIC_MISSION_POSE_VIEWPORT_AND_IMAGE_LOCATOR_ONLY_NO_SEVN_LABELS",
        "observations": observations,
        "episodes": episodes,
    }
    truth = {
        "schema": "blindassist-l10-sevn-panolab-evaluator-truth-v1",
        "generated_at_utc": generated_at,
        "truth_authority": "SEVN_HUMAN_DOOR_AND_TEXT_ANNOTATIONS_PLUS_FROZEN_VIEWPORT_GEOMETRY",
        "episodes": truth_episodes,
    }
    sevn.validate_public(public, per_scenario)
    sevn.validate_truth(truth, public, per_scenario)

    selected_addresses = {
        (str(episode["mission"]["street_name"]), str(episode["mission"]["house_number"]))
        for episode in episodes
    }
    selected_frames = {int(row["frame_id"]) for row in observations.values()}
    address_overlap = sorted(selected_addresses.intersection(excluded_addresses))
    frame_overlap = sorted(selected_frames.intersection(excluded_frames))
    require(not address_overlap, f"reference address overlap: {address_overlap}")
    require(not frame_overlap, f"reference panorama overlap: {frame_overlap}")
    frame_owners: dict[int, list[str]] = {}
    for episode_id, frames in episode_frames.items():
        for frame_id in frames:
            frame_owners.setdefault(frame_id, []).append(episode_id)
    reused_frames = {str(frame): owners for frame, owners in frame_owners.items() if len(owners) > 1}
    require(not reused_frames, f"within-panel cross-episode frame reuse: {reused_frames}")

    source_bytes = (json.dumps(public, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    truth_bytes = (json.dumps(truth, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.source_out.write_bytes(source_bytes)
    args.truth_out.write_bytes(truth_bytes)
    receipt = {
        "schema": "blindassist-l10-sevn-address-panorama-disjoint-selection-receipt-v1",
        "generated_at_utc": generated_at,
        "selection": {
            "metadata_only": True,
            "pixel_payloads_opened": 0,
            "rule": protocol["selection_rule"],
            "episode_count": len(episodes),
            "scenario_counts": dict(Counter(row["scenario_class"] for row in truth_episodes.values())),
            "distinct_address_count": len(selected_addresses),
            "distinct_panorama_frame_count": len(selected_frames),
        },
        "reference_exclusions": {
            "reference_public_source": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            "excluded_address_count": len(excluded_addresses),
            "excluded_panorama_frame_count": len(excluded_frames),
            "selected_address_overlap": address_overlap,
            "selected_panorama_frame_overlap": frame_overlap,
            "within_panel_cross_episode_reused_frames": reused_frames,
        },
        "selected_episode_frames": episode_frames,
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "materializer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "metadata": receipts,
        },
        "outputs": {
            "public_source": {"path": str(args.source_out), "sha256": sha256_file(args.source_out)},
            "evaluator_truth": {"path": str(args.truth_out), "sha256": sha256_file(args.truth_out)},
        },
    }
    args.receipt_out.write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    parser.add_argument("--truth-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

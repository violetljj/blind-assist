#!/usr/bin/env python3
"""Freeze a third SEVN panel without opening panorama pixels.

The panel excludes every address and panorama frame used by both earlier SEVN
panels.  The prior source has no remaining strictly fresh APPROACH episode, so
this successor freezes only the two still-dense PAN recovery classes.  Pixel
payloads and model outputs are never opened during selection.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from l10_panolab import require, sha256_file, utc_now
import l10_sevn_address_disjoint_panel as prior
import l10_sevn_panolab as sevn


SCHEMA = "blindassist-l10-sevn-dual-reference-fresh-pan-source-protocol-v1"
PAN_SCENARIOS = tuple(sevn.SCENARIOS[:2])


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


def union_reference_identity(
    references: list[dict[str, Any]],
) -> tuple[set[tuple[str, str]], set[int]]:
    addresses: set[tuple[str, str]] = set()
    frames: set[int] = set()
    for public in references:
        current_addresses, current_frames = prior.reference_identity(public)
        addresses.update(current_addresses)
        frames.update(current_frames)
    return addresses, frames


def validate_pan_public_truth(
    public: dict[str, Any],
    truth: dict[str, Any],
    per_scenario: int,
) -> None:
    require(
        public.get("schema") == "blindassist-l10-sevn-panolab-public-cohort-v1",
        "unexpected SEVN public schema",
    )
    require(public.get("provider") == "SEVN 1.0 / Zenodo 3526490", "unexpected provider")
    observations = public.get("observations")
    episodes = public.get("episodes")
    require(isinstance(observations, dict) and observations, "observations must be nonempty")
    require(
        isinstance(episodes, list) and len(episodes) == per_scenario * len(PAN_SCENARIOS),
        "unexpected fresh PAN panel size",
    )
    forbidden = {"binding_state", "target_visible", "target_door_annotation", "visible_door_count"}
    for observation_id, row in observations.items():
        require(row.get("observation_id") == observation_id, f"{observation_id}: ID mismatch")
        require(not forbidden.intersection(row), f"{observation_id}: evaluator truth leaked")
    for episode in episodes:
        start = episode["start_observation_id"]
        require(start in observations, f"{episode['episode_id']}: start observation missing")
        transitions = episode["transitions"][start]
        require(tuple(transitions) == sevn.ACTIONS, f"{episode['episode_id']}: action set mismatch")
        for action, edge in transitions.items():
            destination = edge["to_observation_id"]
            require(destination in observations, f"{episode['episode_id']}/{action}: destination missing")
            if not edge["action_executed"]:
                require(
                    destination == start and edge["movement_distance_m"] == 0.0,
                    f"{episode['episode_id']}/{action}: unavailable action changed state",
                )

    require(
        truth.get("schema") == "blindassist-l10-sevn-panolab-evaluator-truth-v1",
        "unexpected SEVN truth schema",
    )
    episode_truth = truth.get("episodes")
    public_ids = {episode["episode_id"] for episode in episodes}
    require(
        isinstance(episode_truth, dict) and set(episode_truth) == public_ids,
        "truth/public episode mismatch",
    )
    counts = Counter(row["scenario_class"] for row in episode_truth.values())
    require(
        set(counts) == set(PAN_SCENARIOS)
        and all(counts[scenario] == per_scenario for scenario in PAN_SCENARIOS),
        f"fresh PAN scenario count mismatch: {dict(counts)}",
    )
    for episode in episodes:
        episode_id = episode["episode_id"]
        rows = episode_truth[episode_id]["observations"]
        reachable = {
            edge["to_observation_id"]
            for edge in episode["transitions"][episode["start_observation_id"]].values()
        }
        require(reachable.issubset(rows), f"{episode_id}: truth missing reachable observation")
        for observation_id in reachable:
            require(
                rows[observation_id]["binding_state"] in sevn.BINDING_STATES,
                f"{episode_id}/{observation_id}: invalid binding state",
            )


def select_pan_panel(
    labels: Any,
    coords: Any,
    graph: Any,
    excluded_addresses: set[tuple[str, str]],
    excluded_frames: set[int],
    per_scenario: int,
) -> list[tuple[str, Any, int, float, int, float]]:
    labels_by_frame = sevn.dataframe_rows_by_frame(labels)
    goals = sevn.candidate_goals(labels, coords, graph)
    candidates: dict[str, list[tuple[str, Any, int, float, int, float]]] = {
        scenario: [] for scenario in PAN_SCENARIOS
    }
    for scenario in PAN_SCENARIOS:
        for goal in goals:
            address = sevn.address_key(goal)
            frame = int(goal.name)
            if address is None or address in excluded_addresses or frame in excluded_frames:
                continue
            target_heading = sevn.label_heading_degrees(
                goal.x_min,
                goal.x_max,
                float(coords.loc[frame].angle),
            )
            left = scenario.startswith("PAN_LEFT")
            offset = -sevn.PAN_START_OFFSET_DEGREES if left else sevn.PAN_START_OFFSET_DEGREES
            start_heading = sevn.wrap360(target_heading + offset)
            destination_heading = sevn.wrap360(
                start_heading + (sevn.PAN_DEGREES if left else -sevn.PAN_DEGREES)
            )
            before = sevn.truth_for_view(
                labels_by_frame,
                coords,
                frame,
                [start_heading],
                address,
            )
            after = sevn.truth_for_view(
                labels_by_frame,
                coords,
                frame,
                [destination_heading],
                address,
            )
            if before["binding_state"] == "CORRECT_UNIQUE":
                continue
            if after["binding_state"] != "CORRECT_UNIQUE":
                continue
            candidates[scenario].append(
                (scenario, goal, frame, start_heading, frame, destination_heading)
            )

    selected: list[tuple[str, Any, int, float, int, float]] = []
    used_addresses = set(excluded_addresses)
    used_frames = set(excluded_frames)
    for scenario in PAN_SCENARIOS:
        for item in candidates[scenario]:
            _, goal, start_frame, _, destination_frame, _ = item
            address = sevn.address_key(goal)
            require(address is not None, "selected goal lacks an address")
            if address in used_addresses or start_frame in used_frames or destination_frame in used_frames:
                continue
            selected.append(item)
            used_addresses.add(address)
            used_frames.update((start_frame, destination_frame))
            if sum(row[0] == scenario for row in selected) == per_scenario:
                break

    counts = Counter(row[0] for row in selected)
    require(
        all(counts[scenario] == per_scenario for scenario in PAN_SCENARIOS),
        f"insufficient dual-reference fresh PAN candidates: {dict(counts)}",
    )
    selected.sort(key=lambda item: PAN_SCENARIOS.index(item[0]))
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
    reference_paths = [verify_sha256(spec) for spec in protocol["frozen_inputs"]["reference_public_sources"]]
    references = [read_json(path) for path in reference_paths]
    excluded_addresses, excluded_frames = union_reference_identity(references)

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
    selected = select_pan_panel(
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
            frame = int(observation["frame_id"])
            observation["image_asset"] = {
                "container": "high-res-panos.zip",
                "member_pattern": "panos/pano_{frame_id:06d}.png",
                "frame_id": frame,
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
    validate_pan_public_truth(public, truth, per_scenario)

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
        for frame in frames:
            frame_owners.setdefault(frame, []).append(episode_id)
    reused_frames = {str(frame): owners for frame, owners in frame_owners.items() if len(owners) > 1}
    require(not reused_frames, f"within-panel cross-episode frame reuse: {reused_frames}")

    args.source_out.write_bytes((json.dumps(public, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    args.truth_out.write_bytes((json.dumps(truth, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    receipt = {
        "schema": "blindassist-l10-sevn-dual-reference-fresh-pan-selection-receipt-v1",
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
            "reference_public_sources": [
                {"path": str(path), "sha256": sha256_file(path)} for path in reference_paths
            ],
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
    args.receipt_out.write_bytes(
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
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

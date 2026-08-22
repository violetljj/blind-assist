"""Mine goal-driven public-real approach episodes without manual capture/labels.

The prospective route requires the goal and public entrance candidate set to be
frozen before Mapillary metadata, pixels, model output, or evaluator truth are
opened.  A separate consumed-Development adapter exists only to exercise the
pipeline against the already sealed Last-10m Mapillary replay.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .annotation import make_annotation


CLAIM_CEILING = "PUBLIC_REAL_DEVELOPMENT_MECHANICS_ONLY_NO_USER_PRODUCT_SAFETY_OR_NAVIGATION_EFFECT_CLAIM"
TRUTH_PRIORITY = [
    "NATIVE_GT",
    "MAP_OR_TRAJECTORY_DERIVED",
    "INDEPENDENT_TEACHER_CONSENSUS",
    "AMBIGUOUS_OR_UNKNOWN",
    "MANUAL_ANNOTATION_LAST_RESORT",
]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distance_bearing(source: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
    lon1, lat1 = map(float, source)
    lon2, lat2 = map(float, target)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * math.cos(mean_lat)
    return math.hypot(east, north), math.degrees(math.atan2(east, north)) % 360.0


def _angle_error(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _range_bucket(distance_m: float) -> str:
    if distance_m <= 2.0:
        return "RANGE_NEAR"
    if distance_m >= 8.0:
        return "RANGE_FAR"
    return "RANGE_APPROACHING"


def _cardinality(candidate_count: int) -> str:
    if candidate_count == 1:
        return "UNIQUE"
    if candidate_count > 1:
        return "SET_VALUED"
    return "AMBIGUOUS"


def mine_prospective(goal_roster: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Construct approach segments from metadata after enforcing precedence."""
    if goal_roster.get("schema_version") != "blindassist_public_goal_roster_v0":
        raise ValueError("goal roster schema mismatch")
    precedence = goal_roster.get("precedence", {})
    required = {
        "mapillary_metadata_accessed": False,
        "mapillary_pixels_accessed": False,
        "model_outputs_created": False,
        "evaluator_truth_created": False,
    }
    if any(precedence.get(key) is not value for key, value in required.items()):
        raise ValueError("goal roster was not frozen before metadata, pixels, model output, and truth")
    if metadata.get("schema_version") != "blindassist_mapillary_sequence_metadata_v0":
        raise ValueError("Mapillary metadata schema mismatch")

    records_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in metadata.get("images", []):
        coordinates = raw.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 2 or not raw.get("sequence_id"):
            continue
        records_by_sequence[str(raw["sequence_id"])].append(dict(raw))
    for records in records_by_sequence.values():
        records.sort(key=lambda row: (int(row.get("captured_at_ms", 0)), str(row.get("image_id", ""))))

    episodes = []
    for goal in goal_roster.get("goals", []):
        entrances = goal.get("public_entrance_candidates", [])
        cardinality = _cardinality(len(entrances))
        if not entrances:
            continue
        target = [
            sum(float(item["coordinates"][0]) for item in entrances) / len(entrances),
            sum(float(item["coordinates"][1]) for item in entrances) / len(entrances),
        ]
        for sequence_id, records in sorted(records_by_sequence.items()):
            eligible = []
            for row in records:
                distance, bearing = _distance_bearing(row["coordinates"], target)
                heading = row.get("computed_compass_angle", row.get("compass_angle"))
                if not isinstance(heading, (int, float)) or not 2.0 <= distance <= 50.0:
                    continue
                error = _angle_error(heading, bearing)
                if error > 75.0:
                    continue
                eligible.append(dict(row) | {
                    "target_distance_m": round(distance, 3),
                    "target_bearing_deg": round(bearing, 3),
                    "target_bearing_error_deg": round(error, 3),
                })
            if len(eligible) < 3:
                continue
            best: list[dict[str, Any]] = []
            current: list[dict[str, Any]] = []
            for row in eligible:
                if current:
                    gap_ms = int(row.get("captured_at_ms", 0)) - int(current[-1].get("captured_at_ms", 0))
                    if gap_ms < 0 or gap_ms > 10_000 or row["target_distance_m"] > current[-1]["target_distance_m"] + 1.0:
                        if len(current) > len(best):
                            best = current
                        current = []
                current.append(row)
            if len(current) > len(best):
                best = current
            if len(best) < 3 or best[0]["target_distance_m"] - best[-1]["target_distance_m"] < 5.0:
                continue
            observations = [{
                "observation_id": f"{goal['goal_id']}--{sequence_id}--o{index:03d}",
                "timestamp_ms": int(row.get("captured_at_ms", 0)),
                "source_frame_id": str(row["image_id"]),
                "sequence_id": sequence_id,
                "coordinates": row["coordinates"],
                "heading_deg": float(row.get("computed_compass_angle", row.get("compass_angle"))),
                "source_url": row.get("source_url"),
                "image_path": row.get("image_path"),
                "image_sha256": row.get("image_sha256"),
                "map_proxy_distance_m": row["target_distance_m"],
                "map_proxy_bearing_error_deg": row["target_bearing_error_deg"],
            } for index, row in enumerate(best, start=1)]
            episodes.append({
                "episode_id": f"{goal['goal_id']}--{sequence_id}",
                "goal_contract": {"goal_contract": {
                    "goal_type": goal["goal_type"],
                    "target_name": goal["target_name"],
                    "reference_mode": cardinality,
                }},
                "public_entrance_candidates": entrances,
                "observations": observations,
            })
    return {
        "schema_version": "blindassist_real_episode_public_manifest_v0",
        "data_role": "PROSPECTIVE_PUBLIC_REAL_DEVELOPMENT",
        "goal_before_truth": True,
        "goal_before_mapillary_metadata_and_pixels": True,
        "private_truth_access": False,
        "provider_model_calls": 0,
        "sample_rate_hz": 0.0,
        "episode_count": len(episodes),
        "episodes": episodes,
        "claim_ceiling": CLAIM_CEILING,
    }


def _events(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value.get("event_type") == "OBSERVATION_PROCESSED":
            yield value


def adapt_consumed_replay(scene_path: Path, truth_path: Path, run_dir: Path) -> dict[str, Any]:
    """Adapt the sealed public-real replay; never claim it is a fresh cohort."""
    scene, truth = _read(scene_path), _read(truth_path)
    if scene.get("execution_mode") != "ACTION_RESPONSIVE_MAPILLARY_POSE_AND_VIEWPORT_REPLAY":
        raise ValueError("source is not the reviewed Mapillary replay")
    if truth.get("authority") != "EVALUATOR_ONLY_NOT_PROVIDER_VISIBLE":
        raise ValueError("truth firewall is absent")
    nodes = {row["frame_id"]: row for row in scene["nodes"]}
    public_episodes, provider_rows = [], []
    for source_episode in scene["episodes"]:
        episode_id = source_episode["episode_id"]
        event_path = run_dir / "episodes" / episode_id / "events.jsonl"
        observations = []
        for event in _events(event_path):
            node = nodes[event["frame_id"]]
            observation_id = event["observation_id"]
            observations.append({
                "observation_id": observation_id,
                "timestamp_ms": int(node["captured_at_ms"]),
                "source_frame_id": node["source_frame_id"],
                "sequence_id": scene.get("sequence_id") or "dbgdqomGU5W7oPnzKZjxLg",
                "source_url": node["source_url"],
                "image_path": node["image_path"],
                "image_sha256": node["image_sha256"],
                "coordinates": node["coordinates"],
                "heading_deg": node["heading_deg"],
                "map_proxy_distance_m": node["target_distance_m"],
                "map_proxy_bearing_error_deg": node["target_bearing_error_deg"],
            })
            observation_path = run_dir / "episodes" / episode_id / "provider_calls" / observation_id / "observation.json"
            output = _read(observation_path).get("p0_output", {})
            decision = output.get("decision", {})
            ranked = decision.get("ranked_candidate_ids", [])
            by_id = {row["candidate_id"]: row for row in output.get("candidates", [])}
            candidates = []
            for rank, candidate_id in enumerate(ranked, start=1):
                candidate = by_id.get(candidate_id)
                if not candidate:
                    continue
                region = candidate["region"]
                candidates.append({
                    "candidate_id": candidate_id,
                    "rank": rank,
                    "x_center_fraction": (float(region["x_min"]) + float(region["x_max"])) / 2.0,
                    "range_m": None,
                })
            supported = decision.get("goal_identity_support") == "SUPPORTED" and bool(decision.get("selected_candidate_id"))
            provider_rows.append({
                "observation_id": observation_id,
                "candidate_cardinality": "UNIQUE" if supported else "AMBIGUOUS",
                "selection_authorized": supported,
                "candidates": candidates,
                "latency_ms": max(0, int(_read(observation_path).get("processed_at_ms", 0)) - int(_read(observation_path).get("captured_at_ms", 0))),
            })
        public_episodes.append({
            "episode_id": episode_id,
            "goal_contract": {"goal_contract": {
                "goal_type": "NAMED_BUILDING_ENTRANCE",
                "target_name": scene["goal_name"],
                "reference_mode": "UNIQUE",
                "uniqueness_authority": "OSM_ENTRANCE_MAIN_MAP_PROXY_ONLY",
            }},
            "observations": observations,
        })
    public = {
        "schema_version": "blindassist_real_episode_public_manifest_v0",
        "data_role": "PROJECT_CONSUMED_DEVELOPMENT_PIPELINE_SMOKE_ONLY",
        "goal_before_truth": True,
        "goal_before_mapillary_metadata_and_pixels": False,
        "private_truth_access": False,
        "provider_model_calls": 0,
        "sample_rate_hz": 0.0,
        "episode_count": len(public_episodes),
        "episodes": public_episodes,
        "claim_ceiling": CLAIM_CEILING,
    }
    annotation = make_annotation(public)
    node_by_observation = {
        row["observation_id"]: row
        for episode in public_episodes for row in episode["observations"]
    }
    for episode in annotation["episodes"]:
        for row in episode["observations"]:
            source = node_by_observation[row["observation_id"]]
            row["truth_authority_tier"] = "MAP_TRAJECTORY_DERIVED"
            row["functional_authority_sources"] = ["MAP_TRAJECTORY_DERIVED"]
            row["range_truth"] = _range_bucket(float(source["map_proxy_distance_m"]))
            row["range_truth_authority"] = "OSM_ENTRANCE_POSE_PROXY_NOT_PHYSICAL_RANGE_GT"
            row["target_visibility"] = "UNKNOWN"
            row["visibility_reason"] = "EXACT_FRAME_REGION_OR_NATIVE_VISIBILITY_GT_UNAVAILABLE"
    annotation["truth_frozen"] = True
    provider = {
        "schema_version": "blindassist_public_real_provider_observations_v0",
        "source": "SEALED_LAST_10M_PROVIDER_OUTPUT_REUSE_NO_NEW_MODEL_CALLS",
        "observations": provider_rows,
    }
    receipt = {
        "schema_version": "blindassist_public_real_episode_mining_receipt_v0",
        "data_role": public["data_role"],
        "source_sha256": {"scene": _hash(scene_path), "truth": _hash(truth_path)},
        "episode_count": len(public_episodes),
        "observation_count": len(provider_rows),
        "manual_capture_count": 0,
        "manual_annotation_count": 0,
        "new_provider_model_call_count": 0,
        "truth_priority": TRUTH_PRIORITY,
        "fresh_cohort": False,
        "freshness_gap": "GOAL_WAS_NOT_FROZEN_BEFORE_ORIGINAL_MAPILLARY_PIXEL_SELECTION",
        "claim_ceiling": CLAIM_CEILING,
    }
    return {"public": public, "provider": provider, "annotation": annotation, "receipt": receipt}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prospective = sub.add_parser("prospective")
    prospective.add_argument("--goal-roster", type=Path, required=True)
    prospective.add_argument("--mapillary-metadata", type=Path, required=True)
    consumed = sub.add_parser("adapt-consumed-replay")
    consumed.add_argument("--scene", type=Path, required=True)
    consumed.add_argument("--truth", type=Path, required=True)
    consumed.add_argument("--run-dir", type=Path, required=True)
    for command in (prospective, consumed):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("output directory already exists")
    if args.mode == "prospective":
        public = mine_prospective(_read(args.goal_roster), _read(args.mapillary_metadata))
        outputs = {
            "public": public,
            "receipt": {
                "schema_version": "blindassist_public_real_episode_mining_receipt_v0",
                "data_role": public["data_role"],
                "source_sha256": {
                    "goal_roster": _hash(args.goal_roster),
                    "mapillary_metadata": _hash(args.mapillary_metadata),
                },
                "episode_count": public["episode_count"],
                "observation_count": sum(len(row["observations"]) for row in public["episodes"]),
                "manual_capture_count": 0,
                "manual_annotation_count": 0,
                "pixel_download_count": 0,
                "provider_model_call_count": 0,
                "truth_priority": TRUTH_PRIORITY,
                "fresh_cohort": True,
                "claim_ceiling": CLAIM_CEILING,
            },
        }
    else:
        outputs = adapt_consumed_replay(args.scene, args.truth, args.run_dir)
    args.output_dir.mkdir(parents=True)
    for name, value in outputs.items():
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"mode": args.mode, "episode_count": outputs["public"]["episode_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

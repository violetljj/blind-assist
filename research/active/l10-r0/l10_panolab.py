#!/usr/bin/env python3
"""Minimal manifest-driven active-observation lab for named-POI entrances.

The runtime receives only public Panoramax observations and a frozen action
graph.  Evaluator truth is loaded into a separate object and is used only for
offline scoring and the one-step oracle ceiling.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIONS = (
    "HOLD",
    "PAN_LEFT",
    "PAN_RIGHT",
    "SWEEP",
    "APPROACH",
    "SIDESTEP_LEFT",
    "SIDESTEP_RIGHT",
)
MOVEMENT_ACTIONS = {"APPROACH", "SIDESTEP_LEFT", "SIDESTEP_RIGHT"}
BINDING_STATES = {
    "CORRECT_UNIQUE",
    "WRONG_UNIQUE",
    "SET_VALUED",
    "NOT_VISIBLE",
}
SCENARIO_CLASSES = {
    "TENANT_WRONG_ENTRANCE",
    "MULTI_ENTRANCE",
    "TARGET_SELF_OCCLUSION",
    "OTHER_BUILDING_OCCLUSION_REACQUISITION",
}
OCCLUSION_CLASSES = {
    "NONE",
    "OUT_OF_VIEW",
    "TARGET_SELF",
    "OTHER_BUILDING",
    "OTHER_OBJECT",
    "UNKNOWN",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_m(a_lon_lat: list[float], b_lon_lat: list[float]) -> float:
    lon1, lat1 = map(math.radians, a_lon_lat)
    lon2, lat2 = map(math.radians, b_lon_lat)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371008.8 * 2.0 * math.asin(min(1.0, math.sqrt(value)))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_public_manifest(public: dict[str, Any]) -> None:
    require(public.get("schema") == "blindassist-l10-panolab-public-cohort-v1", "unexpected public cohort schema")
    provider = public.get("provider")
    require(isinstance(provider, str) and provider.startswith("Panoramax "), "provider must be a named Panoramax instance")
    require(public.get("world_bearing_to_raw_pixel_contract") == "CLOSED", "world-bearing projection must stay closed")

    observations = public.get("observations")
    episodes = public.get("episodes")
    require(isinstance(observations, dict) and observations, "public observations must be nonempty")
    require(isinstance(episodes, list) and episodes, "public episodes must be nonempty")
    require(len(episodes) == 24, "cohort must contain exactly six episodes for each of four scenarios")

    episode_ids: set[str] = set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        poi_id = episode.get("poi_id")
        require(isinstance(episode_id, str) and episode_id not in episode_ids, "episode IDs must be unique strings")
        require(isinstance(poi_id, str) and poi_id, f"{episode_id}: missing poi_id")
        episode_ids.add(episode_id)
        start = episode.get("start_observation_id")
        require(start in observations, f"{episode_id}: unknown start observation")
        transitions = episode.get("transitions")
        require(isinstance(transitions, dict), f"{episode_id}: transitions must be an object")
        start_edges = transitions.get(start)
        require(isinstance(start_edges, dict), f"{episode_id}: missing start transition row")
        require(set(start_edges) == set(ACTIONS), f"{episode_id}: every action must have exactly one frozen outcome")
        for action, edge in start_edges.items():
            require(isinstance(edge, dict), f"{episode_id}/{action}: edge must be an object")
            destination = edge.get("to_observation_id")
            require(destination in observations, f"{episode_id}/{action}: unknown destination")
            require(edge.get("action") == action, f"{episode_id}/{action}: action receipt mismatch")
            require(edge.get("provider_sequence_id") == observations[start]["sequence_id"],
                    f"{episode_id}/{action}: provider sequence receipt differs from the start image")
            require(isinstance(edge.get("action_executed"), bool),
                    f"{episode_id}/{action}: action_executed receipt must be Boolean")
            if action in MOVEMENT_ACTIONS:
                if edge["action_executed"]:
                    require(destination != start, f"{episode_id}/{action}: executed movement cannot be a no-op")
                    require(observations[start]["image_id"] != observations[destination]["image_id"],
                            f"{episode_id}/{action}: executed movement must change provider image")
                    require(observations[start]["sequence_id"] == observations[destination]["sequence_id"],
                            f"{episode_id}/{action}: executed movement left the frozen provider sequence")
                    require(edge.get("provider_link_relation") in {"prev", "next"}
                            and edge.get("reciprocal_provider_links_verified") is True,
                            f"{episode_id}/{action}: executed movement lacks reciprocal provider adjacency")
                else:
                    require(destination == start,
                            f"{episode_id}/{action}: unavailable movement must preserve the observation")
            else:
                require(edge["action_executed"], f"{episode_id}/{action}: view action cannot be unavailable")

    for observation_id, observation in observations.items():
        require(observation.get("observation_id") == observation_id, f"{observation_id}: ID mismatch")
        require(isinstance(observation.get("image_id"), str), f"{observation_id}: missing image ID")
        require(isinstance(observation.get("sequence_id"), str), f"{observation_id}: missing sequence ID")
        pose = observation.get("camera_pose")
        require(isinstance(pose, dict), f"{observation_id}: missing camera pose")
        lon_lat = pose.get("lon_lat")
        require(isinstance(lon_lat, list) and len(lon_lat) == 2, f"{observation_id}: invalid camera lon/lat")
        require(all(isinstance(value, (int, float)) for value in lon_lat), f"{observation_id}: nonnumeric camera pose")
        viewports = observation.get("raw_viewports")
        require(isinstance(viewports, list) and viewports, f"{observation_id}: missing raw panorama viewport")
        for viewport in viewports:
            require(0.0 <= float(viewport["center_degrees"]) < 360.0, f"{observation_id}: raw center out of range")
            require(0.0 < float(viewport["horizontal_fov_degrees"]) <= 360.0, f"{observation_id}: invalid FOV")
        forbidden = {
            "target_portal_id",
            "binding_state",
            "target_visible",
            "ambiguity_count",
            "occlusion_class",
            "scenario_class",
        }
        require(not forbidden.intersection(observation), f"{observation_id}: evaluator truth leaked into public observation")


def validate_truth(truth: dict[str, Any], public: dict[str, Any]) -> None:
    require(truth.get("schema") == "blindassist-l10-panolab-evaluator-truth-v1", "unexpected evaluator schema")
    episode_truth = truth.get("episodes")
    require(isinstance(episode_truth, dict), "evaluator episode truth must be an object")
    public_episodes = {episode["episode_id"]: episode for episode in public["episodes"]}
    require(set(episode_truth) == set(public_episodes), "public and evaluator episode IDs differ")

    scenario_pois: dict[str, list[str]] = {scenario: [] for scenario in SCENARIO_CLASSES}
    for episode_id, episode in public_episodes.items():
        truth_episode = episode_truth[episode_id]
        scenario = truth_episode.get("scenario_class")
        require(scenario in SCENARIO_CLASSES, f"{episode_id}: unsupported evaluator scenario class")
        scenario_pois[scenario].append(episode["poi_id"])
        rows = truth_episode.get("observations")
        require(isinstance(rows, dict), f"{episode_id}: evaluator observations missing")
        referenced = {episode["start_observation_id"]}
        referenced.update(edge["to_observation_id"] for edge in episode["transitions"][episode["start_observation_id"]].values())
        require(referenced.issubset(rows), f"{episode_id}: evaluator truth missing a reachable observation")
        for observation_id in referenced:
            row = rows[observation_id]
            require(row.get("binding_state") in BINDING_STATES, f"{episode_id}/{observation_id}: invalid binding state")
            require(isinstance(row.get("target_visible"), bool), f"{episode_id}/{observation_id}: target_visible must be Boolean")
            require(isinstance(row.get("ambiguity_count"), int) and row["ambiguity_count"] >= 0,
                    f"{episode_id}/{observation_id}: invalid ambiguity count")
            require(row.get("occlusion_class") in OCCLUSION_CLASSES,
                    f"{episode_id}/{observation_id}: invalid or missing occlusion class")
            if row["binding_state"] == "CORRECT_UNIQUE":
                require(row["target_visible"] and row["ambiguity_count"] == 0,
                        f"{episode_id}/{observation_id}: inconsistent correct-unique truth")
            if row["binding_state"] == "WRONG_UNIQUE":
                require(not row["target_visible"], f"{episode_id}/{observation_id}: wrong-unique cannot expose target")

        start = rows[episode["start_observation_id"]]
        if scenario == "TENANT_WRONG_ENTRANCE":
            require(start["binding_state"] == "WRONG_UNIQUE" and start["occlusion_class"] == "OUT_OF_VIEW",
                    f"{episode_id}: tenant-wrong start must expose only a wrong entrance while target is out of view")
        elif scenario == "MULTI_ENTRANCE":
            require(start["binding_state"] == "SET_VALUED" and start["target_visible"] and start["ambiguity_count"] >= 1,
                    f"{episode_id}: multi-entrance start must expose target plus at least one alternative")
        elif scenario == "TARGET_SELF_OCCLUSION":
            require(start["binding_state"] == "NOT_VISIBLE" and start["occlusion_class"] == "TARGET_SELF",
                    f"{episode_id}: self-occlusion start must be physically target-self occluded")
            require(any(rows[observation_id]["target_visible"] for observation_id in referenced
                        if observation_id != episode["start_observation_id"]),
                    f"{episode_id}: self-occlusion scenario has no action-reachable visible target")
        elif scenario == "OTHER_BUILDING_OCCLUSION_REACQUISITION":
            require(start["binding_state"] == "NOT_VISIBLE" and start["occlusion_class"] == "OTHER_BUILDING",
                    f"{episode_id}: reacquisition start must be physically occluded by another building")
            require(any(rows[observation_id]["target_visible"] for observation_id in referenced if observation_id != episode["start_observation_id"]),
                    f"{episode_id}: named reacquisition scenario has no action-reachable visible target")
    require(all(len(poi_ids) == 6 for poi_ids in scenario_pois.values()),
            "each evaluator scenario must contain exactly six episodes")
    tenant_pois = scenario_pois["TENANT_WRONG_ENTRANCE"]
    require(len(set(tenant_pois)) == 2 and sorted(tenant_pois.count(poi_id) for poi_id in set(tenant_pois)) == [3, 3],
            "tenant stratum must contain two strict target ways with three frozen view variants each")
    for scenario in SCENARIO_CLASSES - {"TENANT_WRONG_ENTRANCE"}:
        require(len(set(scenario_pois[scenario])) == 6,
                f"{scenario}: six source-distinct POIs are required")


@dataclass(frozen=True)
class StepResult:
    before: dict[str, Any]
    after: dict[str, Any]
    action_receipt: dict[str, Any]
    done: bool


class L10PanoLab:
    """One-step environment with no evaluator truth in its constructor."""

    def __init__(self, public_manifest: dict[str, Any]):
        validate_public_manifest(public_manifest)
        self._manifest = copy.deepcopy(public_manifest)
        self._observations = self._manifest["observations"]
        self._episodes = {episode["episode_id"]: episode for episode in self._manifest["episodes"]}
        self._episode: dict[str, Any] | None = None
        self._current_observation_id: str | None = None
        self._stepped = False

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._episodes)

    def reset(self, episode_id: str) -> dict[str, Any]:
        require(episode_id in self._episodes, f"unknown episode: {episode_id}")
        self._episode = self._episodes[episode_id]
        self._current_observation_id = self._episode["start_observation_id"]
        self._stepped = False
        return self._public_observation()

    def _public_observation(self) -> dict[str, Any]:
        require(self._episode is not None and self._current_observation_id is not None, "reset must be called first")
        observation = copy.deepcopy(self._observations[self._current_observation_id])
        return {
            "episode_id": self._episode["episode_id"],
            "poi_id": self._episode["poi_id"],
            **observation,
        }

    def step(self, action: str) -> StepResult:
        require(action in ACTIONS, f"unsupported action: {action}")
        require(self._episode is not None and self._current_observation_id is not None, "reset must be called first")
        require(not self._stepped, "this Development lab permits exactly one active-observation step")
        before = self._public_observation()
        edge = copy.deepcopy(self._episode["transitions"][self._current_observation_id][action])
        destination = edge["to_observation_id"]
        after_source = self._observations[destination]
        measured_move = haversine_m(
            before["camera_pose"]["lon_lat"],
            after_source["camera_pose"]["lon_lat"],
        )
        declared_move = float(edge.get("movement_distance_m", measured_move))
        require(abs(measured_move - declared_move) <= 0.35,
                f"{self._episode['episode_id']}/{action}: movement receipt differs from poses")
        edge["movement_distance_m"] = round(measured_move, 3)
        edge["before_image_id"] = before["image_id"]
        edge["after_image_id"] = after_source["image_id"]
        edge["before_camera_pose"] = copy.deepcopy(before["camera_pose"])
        edge["after_camera_pose"] = copy.deepcopy(after_source["camera_pose"])
        edge["world_bearing_to_raw_pixel_used"] = False
        self._current_observation_id = destination
        self._stepped = True
        return StepResult(before=before, after=self._public_observation(), action_receipt=edge, done=True)


class PanoLabEvaluator:
    """Offline evaluator kept outside the policy/runtime object."""

    def __init__(self, truth_manifest: dict[str, Any], public_manifest: dict[str, Any]):
        validate_truth(truth_manifest, public_manifest)
        self._truth = copy.deepcopy(truth_manifest["episodes"])

    def score(self, episode_id: str, observation_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._truth[episode_id]["observations"][observation_id])

    def scenario_class(self, episode_id: str) -> str:
        return str(self._truth[episode_id]["scenario_class"])


def outcome_key(score: dict[str, Any], action: str) -> tuple[int, int, int, int, int]:
    return (
        int(score["binding_state"] == "CORRECT_UNIQUE"),
        int(score["target_visible"]),
        -int(score["binding_state"] == "WRONG_UNIQUE"),
        -int(score["ambiguity_count"]),
        -ACTIONS.index(action),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    opportunity = [row for row in rows if row["before"]["binding_state"] != "CORRECT_UNIQUE"]
    reacquisition_opportunity = [
        row for row in rows
        if row["scenario_class"] == "OTHER_BUILDING_OCCLUSION_REACQUISITION"
        and not row["before"]["target_visible"]
    ]
    correct = sum(row["after"]["binding_state"] == "CORRECT_UNIQUE" for row in rows)
    wrong = sum(row["after"]["binding_state"] == "WRONG_UNIQUE" for row in rows)
    ambiguity_down = sum(row["after"]["ambiguity_count"] < row["before"]["ambiguity_count"] for row in opportunity)
    reacquired = sum(row["after"]["target_visible"] for row in reacquisition_opportunity)
    summary = {
        "episode_count": total,
        "correct_unique": correct,
        "correct_unique_rate": round(correct / total, 6),
        "wrong_unique": wrong,
        "wrong_unique_rate": round(wrong / total, 6),
        "unresolved": total - correct - wrong,
        "opportunity_count": len(opportunity),
        "ambiguity_reduced": ambiguity_down,
        "ambiguity_reduction_rate": round(ambiguity_down / len(opportunity), 6) if opportunity else None,
        "mean_ambiguity_delta": round(
            sum(row["before"]["ambiguity_count"] - row["after"]["ambiguity_count"] for row in rows) / total,
            6,
        ),
        "reacquisition_opportunity_count": len(reacquisition_opportunity),
        "reacquired": reacquired,
        "reacquisition_rate": round(reacquired / len(reacquisition_opportunity), 6) if reacquisition_opportunity else None,
    }
    summary["by_scenario"] = {}
    for scenario in sorted(SCENARIO_CLASSES):
        subset = [row for row in rows if row["scenario_class"] == scenario]
        subset_correct = sum(row["after"]["binding_state"] == "CORRECT_UNIQUE" for row in subset)
        subset_wrong = sum(row["after"]["binding_state"] == "WRONG_UNIQUE" for row in subset)
        subset_ambiguity_down = sum(
            row["after"]["ambiguity_count"] < row["before"]["ambiguity_count"] for row in subset
        )
        summary["by_scenario"][scenario] = {
            "episode_count": len(subset),
            "correct_unique": subset_correct,
            "correct_unique_rate": round(subset_correct / len(subset), 6),
            "wrong_unique": subset_wrong,
            "ambiguity_reduced": subset_ambiguity_down,
        }
    return summary


def replay_action(
    public: dict[str, Any],
    evaluator: PanoLabEvaluator,
    action_by_episode: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = L10PanoLab(public)
    rows = []
    for episode_id in environment.episode_ids:
        before_observation = environment.reset(episode_id)
        action = action_by_episode[episode_id]
        stepped = environment.step(action)
        before = evaluator.score(episode_id, before_observation["observation_id"])
        after = evaluator.score(episode_id, stepped.after["observation_id"])
        rows.append({
            "episode_id": episode_id,
            "poi_id": before_observation["poi_id"],
            "scenario_class": evaluator.scenario_class(episode_id),
            "action": action,
            "before_observation_id": before_observation["observation_id"],
            "after_observation_id": stepped.after["observation_id"],
            "before": before,
            "after": after,
            "action_receipt": stepped.action_receipt,
        })
    return rows, summarize(rows)


def run_benchmark(public: dict[str, Any], truth: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    require(protocol.get("schema") == "blindassist-l10-panolab-protocol-v1", "unexpected protocol schema")
    require(tuple(protocol.get("action_set", [])) == ACTIONS, "protocol action set differs from implementation")
    require(protocol.get("provider", {}).get("name") == public.get("provider"),
            "public cohort provider differs from the frozen protocol")
    evaluator = PanoLabEvaluator(truth, public)
    episode_ids = tuple(episode["episode_id"] for episode in public["episodes"])

    fixed: dict[str, Any] = {}
    fixed_rows: dict[str, list[dict[str, Any]]] = {}
    for action in ACTIONS:
        rows, metrics = replay_action(public, evaluator, {episode_id: action for episode_id in episode_ids})
        fixed[action] = metrics
        fixed_rows[action] = rows

    oracle_actions: dict[str, str] = {}
    for episode in public["episodes"]:
        episode_id = episode["episode_id"]
        start = episode["start_observation_id"]
        candidates = []
        for action in ACTIONS:
            destination = episode["transitions"][start][action]["to_observation_id"]
            candidates.append((outcome_key(evaluator.score(episode_id, destination), action), action))
        oracle_actions[episode_id] = max(candidates)[1]
    oracle_rows, oracle_metrics = replay_action(public, evaluator, oracle_actions)

    non_hold = [action for action in ACTIONS if action != "HOLD"]
    best_fixed_action = max(
        non_hold,
        key=lambda action: (
            fixed[action]["correct_unique_rate"],
            -fixed[action]["wrong_unique_rate"],
            fixed[action]["ambiguity_reduction_rate"] or 0.0,
            -ACTIONS.index(action),
        ),
    )
    baseline = fixed["HOLD"]
    best_fixed = fixed[best_fixed_action]
    oracle_gain_pp = 100.0 * (oracle_metrics["correct_unique_rate"] - baseline["correct_unique_rate"])
    fixed_gap_pp = 100.0 * (oracle_metrics["correct_unique_rate"] - best_fixed["correct_unique_rate"])
    minimum_oracle_gain_pp = float(protocol["decision_rule"]["minimum_oracle_gain_percentage_points"])
    maximum_fixed_gap_pp = float(protocol["decision_rule"]["maximum_fixed_to_oracle_gap_percentage_points"])

    if oracle_gain_pp < minimum_oracle_gain_pp:
        decision = "L10_PANOLAB_ACTIVE_OBSERVATION_CEILING_NOT_MATERIAL_ROUTE_CLOSED"
    elif fixed_gap_pp <= maximum_fixed_gap_pp:
        decision = "L10_PANOLAB_FIXED_ACTION_POLICY_DEVELOPMENT_SIGNAL"
    else:
        decision = "L10_PANOLAB_ACTION_SELECTOR_TRAINING_SIGNAL"

    action_distribution = {action: sum(selected == action for selected in oracle_actions.values()) for action in ACTIONS}
    action_entropy_bits = -sum(
        (count / len(episode_ids)) * math.log2(count / len(episode_ids))
        for count in action_distribution.values()
        if count
    )
    return {
        "schema": "blindassist-l10-panolab-development-result-v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "claim_scope": "CURATED_REAL_PANORAMAX_ONE_STEP_ACTIVE_OBSERVATION_DEVELOPMENT",
        "provider": public["provider"],
        "world_bearing_to_raw_pixel_contract": public["world_bearing_to_raw_pixel_contract"],
        "cohort": {
            "poi_count": len({episode["poi_id"] for episode in public["episodes"]}),
            "episode_count": len(public["episodes"]),
            "scenario_counts": {
                scenario: sum(row.get("scenario_class") == scenario for row in truth["episodes"].values())
                for scenario in sorted(SCENARIO_CLASSES)
            },
        },
        "fixed_policy_metrics": fixed,
        "best_fixed_action": best_fixed_action,
        "oracle_metrics": oracle_metrics,
        "oracle_action_distribution": action_distribution,
        "oracle_distinct_action_count": sum(count > 0 for count in action_distribution.values()),
        "oracle_action_entropy_bits": round(max(0.0, action_entropy_bits), 6),
        "oracle_gain_percentage_points_vs_hold": round(oracle_gain_pp, 3),
        "best_fixed_to_oracle_gap_percentage_points": round(fixed_gap_pp, 3),
        "decision_rule": copy.deepcopy(protocol["decision_rule"]),
        "episode_results": {
            "best_fixed": fixed_rows[best_fixed_action],
            "oracle": oracle_rows,
        },
        "non_claims": copy.deepcopy(protocol["non_claims"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public = json.loads(args.cohort.read_text(encoding="utf-8"))
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = run_benchmark(public, truth, protocol)
    result["inputs"] = {
        "cohort_path": str(args.cohort),
        "cohort_sha256": sha256_file(args.cohort),
        "truth_path": str(args.truth),
        "truth_sha256": sha256_file(args.truth),
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "best_fixed_action": result["best_fixed_action"],
        "fixed": result["fixed_policy_metrics"][result["best_fixed_action"]],
        "oracle": result["oracle_metrics"],
        "oracle_gain_pp": result["oracle_gain_percentage_points_vs_hold"],
        "fixed_gap_pp": result["best_fixed_to_oracle_gap_percentage_points"],
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

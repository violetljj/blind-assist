"""Stdlib-only deterministic evaluator for P1-R0 target persistence.

Evaluator truth is used only after a system output has been produced. The
evaluator never grants semantic referent validity and never upgrades a
NO_REFERENT handoff.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p1_persistence import baseline


PROTOCOL_ID = "BA-P1-TARGET-PERSISTENCE-R0-V1"
HANDOFF_STATUSES = {"REFERENT_ESTABLISHED", "NO_REFERENT"}
STATES = {"UNBOUND", "TRACKING", "UNCERTAIN", "TEMP_UNOBSERVABLE", "LOST"}
EVENTS = {"NONE", "TARGET_TEMP_UNOBSERVABLE", "LOSS_DETECTED", "REACQUIRED", "REGROUND_REQUIRED"}
PHASES = {"VISIBLE", "SHORT_UNOBSERVABLE", "LOSS_ELIGIBLE", "REACQUISITION_WINDOW"}
SCENARIO_CLASSES = {
    "CONTINUOUS_VISIBLE_CAMERA_MOTION",
    "SHORT_OCCLUSION",
    "TURN_AWAY_AND_RETURN",
    "SAME_CLASS_DISTRACTOR_CROSSING",
    "SIMILAR_CANDIDATE_ALTERNATION",
    "LONG_TARGET_ABSENCE",
    "FALSE_SIMILAR_AFTER_LOSS",
    "NO_REFERENT_GUARD",
}

EPISODE_KEYS = {"schema_version", "protocol_id", "episode_id", "scenario_class", "handoff", "frames", "truth"}
HANDOFF_KEYS = {"status", "goal_id", "referent_id", "grounding_provenance"}
PROVENANCE_KEYS = {"p0_decision_id", "source_frame_index", "authority"}
FRAME_KEYS = {"frame_index", "timestamp_ms", "candidates"}
CANDIDATE_KEYS = {"candidate_id", "identity_support", "identity_contradiction", "stability", "oscillation"}
TRUTH_KEYS = {"referent_instance_id", "frames"}
TRUTH_FRAME_KEYS = {
    "frame_index", "referent_observable", "candidate_instance_map", "allowed_states", "allowed_events", "phase"
}
OUTPUT_KEYS = {"schema_version", "protocol_id", "episode_id", "referent_id", "score_semantics", "frames"}
OUTPUT_FRAME_KEYS = {
    "frame_index", "state", "current_candidate_id", "identity_score", "stability_score", "oscillation_score",
    "frames_since_confirmed", "event"
}


class EpisodeContractError(ValueError):
    """Evaluator episode or truth is malformed."""


class OutputContractError(ValueError):
    """System output is malformed or violates the representation contract."""


def _require(condition: bool, message: str, error_type: type[ValueError]) -> None:
    if not condition:
        raise error_type(message)


def _exact_keys(value: Any, expected: set[str], path: str, error_type: type[ValueError]) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object", error_type)
    observed = set(value)
    _require(
        observed == expected,
        f"{path} keys differ: missing={sorted(expected-observed)} extra={sorted(observed-expected)}",
        error_type,
    )
    return value


def _nonempty_string(value: Any, path: str, error_type: type[ValueError]) -> str:
    _require(isinstance(value, str) and bool(value), f"{path} must be a non-empty string", error_type)
    return value


def _score(value: Any, path: str, error_type: type[ValueError]) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0,
        f"{path} must be an algorithmic score in [0,1]",
        error_type,
    )
    return float(value)


def _string_set(value: Any, allowed: set[str], path: str, error_type: type[ValueError]) -> list[str]:
    _require(isinstance(value, list) and bool(value), f"{path} must be a non-empty array", error_type)
    _require(all(isinstance(item, str) and item in allowed for item in value), f"{path} has unknown values", error_type)
    _require(len(value) == len(set(value)), f"{path} must be unique", error_type)
    return value


def validate_episode(value: Mapping[str, Any]) -> Mapping[str, Any]:
    episode = _exact_keys(value, EPISODE_KEYS, "episode", EpisodeContractError)
    _require(episode["schema_version"] == 1, "episode.schema_version must be 1", EpisodeContractError)
    _require(episode["protocol_id"] == PROTOCOL_ID, "episode protocol drift", EpisodeContractError)
    _nonempty_string(episode["episode_id"], "episode.episode_id", EpisodeContractError)
    _require(episode["scenario_class"] in SCENARIO_CLASSES, "unknown scenario_class", EpisodeContractError)

    handoff = _exact_keys(episode["handoff"], HANDOFF_KEYS, "episode.handoff", EpisodeContractError)
    _require(handoff["status"] in HANDOFF_STATUSES, "unknown handoff status", EpisodeContractError)
    _nonempty_string(handoff["goal_id"], "episode.handoff.goal_id", EpisodeContractError)
    provenance = _exact_keys(
        handoff["grounding_provenance"], PROVENANCE_KEYS, "episode.handoff.grounding_provenance", EpisodeContractError
    )
    _require(provenance["authority"] in {"P0_ESTABLISHED_REFERENT", "P0_NO_REFERENT"}, "invalid P0 authority", EpisodeContractError)
    if handoff["status"] == "REFERENT_ESTABLISHED":
        _nonempty_string(handoff["referent_id"], "episode.handoff.referent_id", EpisodeContractError)
        _nonempty_string(provenance["p0_decision_id"], "grounding_provenance.p0_decision_id", EpisodeContractError)
        _require(isinstance(provenance["source_frame_index"], int) and provenance["source_frame_index"] >= 0,
                 "established referent requires a source_frame_index", EpisodeContractError)
        _require(provenance["authority"] == "P0_ESTABLISHED_REFERENT", "established authority mismatch", EpisodeContractError)
    else:
        _require(handoff["referent_id"] is None, "NO_REFERENT requires null referent_id", EpisodeContractError)
        _require(provenance["p0_decision_id"] is None and provenance["source_frame_index"] is None,
                 "NO_REFERENT cannot carry establishment provenance", EpisodeContractError)
        _require(provenance["authority"] == "P0_NO_REFERENT", "NO_REFERENT authority mismatch", EpisodeContractError)

    _require(isinstance(episode["frames"], list) and bool(episode["frames"]), "episode.frames must not be empty", EpisodeContractError)
    frames_by_index: dict[int, Mapping[str, Any]] = {}
    previous_timestamp = -1
    for position, raw_frame in enumerate(episode["frames"]):
        frame = _exact_keys(raw_frame, FRAME_KEYS, f"episode.frames[{position}]", EpisodeContractError)
        _require(frame["frame_index"] == position, "frame_index must be contiguous from zero", EpisodeContractError)
        _require(isinstance(frame["timestamp_ms"], int) and frame["timestamp_ms"] > previous_timestamp,
                 "timestamps must be strictly increasing integers", EpisodeContractError)
        previous_timestamp = frame["timestamp_ms"]
        _require(isinstance(frame["candidates"], list), "frame candidates must be an array", EpisodeContractError)
        candidate_ids: set[str] = set()
        for candidate_position, raw_candidate in enumerate(frame["candidates"]):
            candidate = _exact_keys(
                raw_candidate, CANDIDATE_KEYS, f"episode.frames[{position}].candidates[{candidate_position}]", EpisodeContractError
            )
            candidate_id = _nonempty_string(candidate["candidate_id"], "candidate_id", EpisodeContractError)
            _require(candidate_id not in candidate_ids, "candidate_id must be unique within a frame", EpisodeContractError)
            candidate_ids.add(candidate_id)
            for key in CANDIDATE_KEYS - {"candidate_id"}:
                _score(candidate[key], f"candidate.{key}", EpisodeContractError)
        frames_by_index[position] = frame

    truth = _exact_keys(episode["truth"], TRUTH_KEYS, "episode.truth", EpisodeContractError)
    if handoff["status"] == "REFERENT_ESTABLISHED":
        _nonempty_string(truth["referent_instance_id"], "truth.referent_instance_id", EpisodeContractError)
    else:
        _require(truth["referent_instance_id"] is None, "NO_REFERENT truth must not create an instance", EpisodeContractError)
    _require(isinstance(truth["frames"], list) and len(truth["frames"]) == len(frames_by_index),
             "truth frames must align one-to-one", EpisodeContractError)
    for position, raw_truth_frame in enumerate(truth["frames"]):
        truth_frame = _exact_keys(raw_truth_frame, TRUTH_FRAME_KEYS, f"truth.frames[{position}]", EpisodeContractError)
        _require(truth_frame["frame_index"] == position, "truth frame_index drift", EpisodeContractError)
        _require(type(truth_frame["referent_observable"]) is bool, "referent_observable must be boolean", EpisodeContractError)
        candidate_map = truth_frame["candidate_instance_map"]
        _require(isinstance(candidate_map, Mapping), "candidate_instance_map must be an object", EpisodeContractError)
        _require(set(candidate_map) == {str(item["candidate_id"]) for item in frames_by_index[position]["candidates"]},
                 "candidate_instance_map must cover exactly the public candidates", EpisodeContractError)
        for candidate_id, instance_id in candidate_map.items():
            _nonempty_string(candidate_id, "truth candidate id", EpisodeContractError)
            _nonempty_string(instance_id, "truth physical instance id", EpisodeContractError)
        _string_set(truth_frame["allowed_states"], STATES, "truth.allowed_states", EpisodeContractError)
        _string_set(truth_frame["allowed_events"], EVENTS, "truth.allowed_events", EpisodeContractError)
        _require(truth_frame["phase"] in PHASES, "unknown truth phase", EpisodeContractError)
        if handoff["status"] == "NO_REFERENT":
            _require(not truth_frame["referent_observable"], "NO_REFERENT cannot expose referent truth", EpisodeContractError)
            _require(truth_frame["allowed_states"] == ["UNBOUND"], "NO_REFERENT only allows UNBOUND", EpisodeContractError)
    return episode


def validate_output(value: Mapping[str, Any], episode: Mapping[str, Any]) -> Mapping[str, Any]:
    output = _exact_keys(value, OUTPUT_KEYS, "output", OutputContractError)
    _require(output["schema_version"] == 1, "output.schema_version must be 1", OutputContractError)
    _require(output["protocol_id"] == PROTOCOL_ID, "output protocol drift", OutputContractError)
    _require(output["episode_id"] == episode["episode_id"], "output episode_id mismatch", OutputContractError)
    _require(output["score_semantics"] == "ALGORITHMIC_EVIDENCE_NOT_CALIBRATED_PROBABILITY",
             "score semantics drift", OutputContractError)
    expected_referent = episode["handoff"]["referent_id"]
    _require(output["referent_id"] == expected_referent, "P1 cannot create or replace the handed-off referent", OutputContractError)
    _require(isinstance(output["frames"], list) and len(output["frames"]) == len(episode["frames"]),
             "output frames must align one-to-one", OutputContractError)

    for position, raw_frame in enumerate(output["frames"]):
        frame = _exact_keys(raw_frame, OUTPUT_FRAME_KEYS, f"output.frames[{position}]", OutputContractError)
        _require(frame["frame_index"] == position, "output frame_index drift", OutputContractError)
        _require(frame["state"] in STATES, "unknown output state", OutputContractError)
        _require(frame["event"] in EVENTS, "unknown output event", OutputContractError)
        candidate_ids = {str(item["candidate_id"]) for item in episode["frames"][position]["candidates"]}
        _require(frame["current_candidate_id"] is None or frame["current_candidate_id"] in candidate_ids,
                 "current_candidate_id must name a candidate in the same frame", OutputContractError)
        for key in ("identity_score", "stability_score", "oscillation_score"):
            if frame[key] is not None:
                _score(frame[key], f"output frame {key}", OutputContractError)
        if frame["state"] == "UNBOUND":
            _require(expected_referent is None and frame["current_candidate_id"] is None,
                     "UNBOUND cannot retain or assert a referent", OutputContractError)
            _require(frame["frames_since_confirmed"] is None, "UNBOUND counter must be null", OutputContractError)
        else:
            _require(expected_referent is not None, "bound state requires an established P0 referent", OutputContractError)
            _require(isinstance(frame["frames_since_confirmed"], int) and frame["frames_since_confirmed"] >= 0,
                     "bound state requires a non-negative confirmation counter", OutputContractError)
        if frame["state"] == "TRACKING":
            _require(frame["current_candidate_id"] is not None, "TRACKING must assert one current candidate", OutputContractError)
        else:
            _require(frame["current_candidate_id"] is None, f"{frame['state']} cannot assert current location", OutputContractError)
        if frame["event"] == "REACQUIRED":
            _require(frame["state"] == "TRACKING", "REACQUIRED must transition to TRACKING", OutputContractError)
        if frame["event"] == "LOSS_DETECTED":
            _require(frame["state"] == "LOST", "LOSS_DETECTED must transition to LOST", OutputContractError)
    return output


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator if denominator else None}


def _wrong_lock_runs(wrong_flags: Sequence[bool], timestamps: Sequence[int]) -> dict[str, int]:
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    cadence = int(statistics.median(deltas)) if deltas else 0
    max_frames = 0
    max_ms = 0
    start: int | None = None
    for index, wrong in enumerate([*wrong_flags, False]):
        if wrong and start is None:
            start = index
        if not wrong and start is not None:
            end = index - 1
            max_frames = max(max_frames, end - start + 1)
            max_ms = max(max_ms, timestamps[end] - timestamps[start] + cadence)
            start = None
    return {"max_frames": max_frames, "max_duration_ms": max_ms}


def evaluate_episode(episode_value: Mapping[str, Any], output_value: Mapping[str, Any]) -> dict[str, Any]:
    episode = validate_episode(episode_value)
    try:
        output = validate_output(output_value, episode)
    except OutputContractError as exc:
        return {
            "episode_id": episode["episode_id"],
            "scenario_class": episode["scenario_class"],
            "valid_system_output": False,
            "contract_error": str(exc),
            "metrics": None,
        }

    truth_frames = episode["truth"]["frames"]
    referent_instance_id = episode["truth"]["referent_instance_id"]
    no_referent = episode["handoff"]["status"] == "NO_REFERENT"
    observable_frames = 0
    correct_assertions = 0
    wrong_assertions = 0
    illegal_bind_frames = 0
    false_loss_frames = 0
    state_violations = 0
    event_violations = 0
    reacquired_correct = 0
    false_reacquisitions = 0
    asserted_instances: list[str] = []
    wrong_flags: list[bool] = []
    reacquisition_window_frames: list[int] = []
    loss_eligible_start: int | None = None
    loss_detection_frame: int | None = None

    for public_frame, truth_frame, output_frame in zip(episode["frames"], truth_frames, output["frames"]):
        observable_frames += int(truth_frame["referent_observable"])
        state_violations += int(output_frame["state"] not in truth_frame["allowed_states"])
        event_violations += int(output_frame["event"] not in truth_frame["allowed_events"])
        if truth_frame["phase"] == "REACQUISITION_WINDOW":
            reacquisition_window_frames.append(output_frame["frame_index"])
        if truth_frame["phase"] == "LOSS_ELIGIBLE" and loss_eligible_start is None:
            loss_eligible_start = output_frame["frame_index"]
        if output_frame["event"] == "LOSS_DETECTED" and loss_detection_frame is None:
            loss_detection_frame = output_frame["frame_index"]

        current_candidate_id = output_frame["current_candidate_id"]
        asserted_instance = None
        if current_candidate_id is not None:
            asserted_instance = truth_frame["candidate_instance_map"][current_candidate_id]
            asserted_instances.append(asserted_instance)
        correct = asserted_instance is not None and asserted_instance == referent_instance_id
        wrong = asserted_instance is not None and asserted_instance != referent_instance_id
        correct_assertions += int(correct)
        wrong_assertions += int(wrong)
        wrong_flags.append(wrong)
        illegal_bind_frames += int(no_referent and (output_frame["state"] != "UNBOUND" or current_candidate_id is not None))
        false_loss_frames += int(output_frame["state"] == "LOST" and truth_frame["referent_observable"])
        if output_frame["event"] == "REACQUIRED":
            if correct and truth_frame["referent_observable"]:
                reacquired_correct += 1
            else:
                false_reacquisitions += 1

    identity_switches = sum(left != right for left, right in zip(asserted_instances, asserted_instances[1:]))
    wrong_lock = _wrong_lock_runs(wrong_flags, [int(frame["timestamp_ms"]) for frame in episode["frames"]])
    assertion_count = len(asserted_instances)
    no_referent_frame_count = len(output["frames"]) if no_referent else 0
    reacquisition_opportunity = bool(reacquisition_window_frames)
    reacquisition_success = reacquisition_opportunity and any(
        frame["event"] == "REACQUIRED"
        and frame["current_candidate_id"] is not None
        and truth_frames[index]["candidate_instance_map"][frame["current_candidate_id"]] == referent_instance_id
        for index, frame in enumerate(output["frames"])
        if truth_frames[index]["phase"] == "REACQUISITION_WINDOW"
    )
    first_reacquisition = next((frame["frame_index"] for frame in output["frames"] if frame["event"] == "REACQUIRED"), None)
    time_to_reacquire_frames = (
        first_reacquisition - reacquisition_window_frames[0]
        if first_reacquisition is not None and reacquisition_window_frames
        else None
    )
    loss_detection_latency = (
        loss_detection_frame - loss_eligible_start
        if loss_eligible_start is not None and loss_detection_frame is not None and loss_detection_frame >= loss_eligible_start
        else None
    )
    short_unobservable = [index for index, item in enumerate(truth_frames) if item["phase"] == "SHORT_UNOBSERVABLE"]
    temporary_recovery_frame = None
    if short_unobservable:
        last_short = short_unobservable[-1]
        for index in range(last_short + 1, len(truth_frames)):
            if truth_frames[index]["phase"] == "LOSS_ELIGIBLE":
                break
            if truth_frames[index]["phase"] == "VISIBLE" and truth_frames[index]["referent_observable"]:
                temporary_recovery_frame = index
                break
    temporary_occlusion_opportunity = temporary_recovery_frame is not None
    temporary_occlusion_recovered = None
    if temporary_recovery_frame is not None:
        recovered_frame = output["frames"][temporary_recovery_frame]
        candidate_id = recovered_frame["current_candidate_id"]
        temporary_occlusion_recovered = (
            recovered_frame["state"] == "TRACKING"
            and candidate_id is not None
            and truth_frames[temporary_recovery_frame]["candidate_instance_map"][candidate_id] == referent_instance_id
        )

    metrics = {
        "illegal_bind_frames": illegal_bind_frames,
        "illegal_bind_rate": _rate(illegal_bind_frames, no_referent_frame_count),
        "wrong_instance_asserted_frames": wrong_assertions,
        "wrong_instance_assertion_rate": _rate(wrong_assertions, assertion_count),
        "identity_switches": identity_switches,
        "identity_switch_rate": _rate(identity_switches, max(0, assertion_count - 1)),
        "false_reacquisitions": false_reacquisitions,
        "false_reacquisition_rate": _rate(
            false_reacquisitions, reacquired_correct + false_reacquisitions
        ),
        "correct_identity_coverage": _rate(correct_assertions, observable_frames),
        "reacquisition_events": {"correct": reacquired_correct, "false": false_reacquisitions},
        "reacquisition_opportunity": reacquisition_opportunity,
        "reacquisition_success": reacquisition_success if reacquisition_opportunity else None,
        "time_to_reacquire_frames": time_to_reacquire_frames,
        "temporary_occlusion_opportunity": temporary_occlusion_opportunity,
        "temporary_occlusion_recovered": temporary_occlusion_recovered,
        "false_loss_frames": false_loss_frames,
        "loss_detection_latency_frames": loss_detection_latency,
        "wrong_lock_persistence": wrong_lock,
        "state_expectation_violations": state_violations,
        "event_expectation_violations": event_violations,
        "lexicographic_vector": [
            illegal_bind_frames,
            wrong_assertions,
            identity_switches,
            false_reacquisitions,
            -correct_assertions,
        ],
    }
    return {
        "episode_id": episode["episode_id"],
        "scenario_class": episode["scenario_class"],
        "valid_system_output": True,
        "contract_error": None,
        "metrics": metrics,
    }


def evaluate_batch(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    episodes = [evaluate_episode(episode, output) for episode, output in pairs]
    invalid = [item for item in episodes if not item["valid_system_output"]]
    if invalid:
        return {"protocol_id": PROTOCOL_ID, "valid": False, "episodes": episodes, "aggregate": None}

    metrics = [item["metrics"] for item in episodes]
    assert all(item is not None for item in metrics)
    typed_metrics = [item for item in metrics if item is not None]
    sums = defaultdict(int)
    correct_coverage_numerator = 0
    correct_coverage_denominator = 0
    reacquisition_opportunities = 0
    reacquisition_successes = 0
    reacquisition_correct_events = 0
    reacquisition_false_events = 0
    no_referent_frames = 0
    asserted_frames = 0
    asserted_transitions = 0
    temporary_occlusion_opportunities = 0
    temporary_occlusion_recoveries = 0
    max_wrong_frames = 0
    max_wrong_ms = 0
    for item in typed_metrics:
        for key in (
            "illegal_bind_frames", "wrong_instance_asserted_frames", "identity_switches", "false_reacquisitions",
            "false_loss_frames", "state_expectation_violations", "event_expectation_violations"
        ):
            sums[key] += int(item[key])
        coverage = item["correct_identity_coverage"]
        correct_coverage_numerator += int(coverage["numerator"])
        correct_coverage_denominator += int(coverage["denominator"])
        if item["reacquisition_opportunity"]:
            reacquisition_opportunities += 1
            reacquisition_successes += int(bool(item["reacquisition_success"]))
        reacquisition_correct_events += int(item["reacquisition_events"]["correct"])
        reacquisition_false_events += int(item["reacquisition_events"]["false"])
        no_referent_frames += int(item["illegal_bind_rate"]["denominator"])
        asserted_frames += int(item["wrong_instance_assertion_rate"]["denominator"])
        asserted_transitions += int(item["identity_switch_rate"]["denominator"])
        if item["temporary_occlusion_opportunity"]:
            temporary_occlusion_opportunities += 1
            temporary_occlusion_recoveries += int(bool(item["temporary_occlusion_recovered"]))
        max_wrong_frames = max(max_wrong_frames, int(item["wrong_lock_persistence"]["max_frames"]))
        max_wrong_ms = max(max_wrong_ms, int(item["wrong_lock_persistence"]["max_duration_ms"]))

    aggregate = {
        **dict(sums),
        "illegal_bind_rate_hard_gate_pass": sums["illegal_bind_frames"] == 0,
        "illegal_bind_rate": _rate(sums["illegal_bind_frames"], no_referent_frames),
        "wrong_instance_assertion_rate": _rate(sums["wrong_instance_asserted_frames"], asserted_frames),
        "identity_switch_rate": _rate(sums["identity_switches"], asserted_transitions),
        "false_reacquisition_rate": _rate(
            sums["false_reacquisitions"], reacquisition_correct_events + reacquisition_false_events
        ),
        "correct_identity_coverage": _rate(correct_coverage_numerator, correct_coverage_denominator),
        "reacquisition_precision": _rate(
            reacquisition_correct_events, reacquisition_correct_events + reacquisition_false_events
        ),
        "reacquisition_recall": _rate(reacquisition_successes, reacquisition_opportunities),
        "temporary_occlusion_recovery_rate": _rate(
            temporary_occlusion_recoveries, temporary_occlusion_opportunities
        ),
        "wrong_lock_persistence_max_frames": max_wrong_frames,
        "wrong_lock_persistence_max_duration_ms": max_wrong_ms,
        "lexicographic_vector": [
            sums["illegal_bind_frames"],
            sums["wrong_instance_asserted_frames"],
            sums["identity_switches"],
            sums["false_reacquisitions"],
            -correct_coverage_numerator,
        ],
    }
    return {"protocol_id": PROTOCOL_ID, "valid": True, "episodes": episodes, "aggregate": aggregate}


def run_fixture_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, list), "fixture root must be an array", EpisodeContractError)
    pairs = []
    for episode in payload:
        validate_episode(episode)
        pairs.append((episode, baseline.run_baseline(baseline.extract_public_input(episode))))
    return evaluate_batch(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic P1-R0 simple baseline and evaluator")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).with_name("scenarios.json"),
        help="P1-R0 evaluator episodes; defaults to the eight synthetic mechanics scenarios",
    )
    args = parser.parse_args()
    print(json.dumps(run_fixture_file(args.fixtures), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

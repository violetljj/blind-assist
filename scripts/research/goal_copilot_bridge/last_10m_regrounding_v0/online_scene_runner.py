"""Prepare and execute the 3x5 network-scene mechanical run.

The public playlist contains only the goal, current image, source identity, and
fixed per-episode frame order.  Evaluator regions are written to a separate
truth sidecar and loaded only for terminal adjudication.  The P0 provider never
receives playlist position, source coordinates, truth, or a previous result.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.p0_grounding import p0_evaluator

from .cli import _append_event, _atomic_json, _failed_provider_observation, _read_json, _sha256_file
from .core import (
    Attribution,
    ContractError,
    EpisodeState,
    Policy,
    State,
    adjudicate_episode,
    apply_observation,
    stop_episode,
    summarize_field_run,
)
from .provider_adapter import ProviderAdapterError, ground_current_frame, preflight_provider


PUBLIC_MANIFEST = "online-scenes-public.json"
TRUTH_SIDECAR = "online-scenes-evaluator-truth.json"
SELECTED_GOALS = ("hofbladelin", "LA Look", "Sint-Jan-Baptist-en-Sint-Jan-Evangelistkerk")
MAPILLARY_URL = "https://www.mapillary.com/app/?focus=photo&pKey={frame_id}"


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def prepare_online_scenes(*, brain_cohort: Path, output_dir: Path) -> dict[str, Any]:
    cohort = _read_json(brain_cohort)
    if cohort.get("claim_ceiling") != "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY":
        raise ContractError("input is not the existing reviewed Silver-B online-scene bank")
    episodes = cohort.get("episodes")
    if not isinstance(episodes, list):
        raise ContractError("brain cohort episodes are missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError("online scene output directory must be absent or empty")

    public_locations = []
    truth_frames: dict[str, Any] = {}
    for location_index, goal_name in enumerate(SELECTED_GOALS, start=1):
        matches = [
            item
            for item in episodes
            if item["evaluator_episode"]["goal_spec"]["target_name"] == goal_name
            and item["evaluator_episode"]["goal_reference_resolution"] == "UNIQUE"
        ]
        matches.sort(key=lambda item: item["evaluator_episode"]["observation_window"]["frame_ids"][0])
        if len(matches) < 2:
            raise ContractError(f"network scene location {goal_name!r} has fewer than two distinct UNIQUE frames")
        location_id = f"online-site-{location_index:02d}"
        public_frames = []
        for item in matches:
            evaluator = item["evaluator_episode"]
            frame_id = str(evaluator["observation_window"]["frame_ids"][0])
            image = Path(str(item["image_path"])).resolve()
            if not image.is_file():
                raise ContractError(f"cached public network frame is missing: {frame_id}")
            image_sha256 = _sha256_file(image)
            if image_sha256 != item["image_sha256"]:
                raise ContractError(f"cached public network frame hash drift: {frame_id}")
            public_frames.append(
                {
                    "frame_id": frame_id,
                    "image_path": str(image),
                    "image_sha256": image_sha256,
                    "source": "Mapillary",
                    "source_url": MAPILLARY_URL.format(frame_id=frame_id),
                    "source_episode_id": item["source_episode_id"],
                }
            )
            truth_frames[frame_id] = {
                "goal_name": goal_name,
                "goal_reference_resolution": evaluator["goal_reference_resolution"],
                "acceptable_spatial_regions": evaluator["acceptable_spatial_regions"],
            }
        mechanical_episodes = []
        for episode_index in range(5):
            offset = episode_index % len(public_frames)
            playlist = public_frames[offset:] + public_frames[:offset]
            mechanical_episodes.append(
                {
                    "episode_id": f"{location_id}-e{episode_index + 1:02d}",
                    "frame_ids": [item["frame_id"] for item in playlist],
                }
            )
        public_locations.append(
            {
                "location_id": location_id,
                "goal_name": goal_name,
                "frames": public_frames,
                "episodes": mechanical_episodes,
            }
        )

    public = {
        "schema_version": 1,
        "milestone_id": "BLINDASSIST_LAST_10M_REGROUNDING_V0",
        "execution_mode": "PUBLIC_NETWORK_SCENE_MECHANICAL_REPLAY",
        "source": {
            "name": "Mapillary",
            "attribution": "Mapillary contributors; image IDs and source links retained",
            "license_note": "Mapillary imagery/open-data attribution obligations retained",
            "input_role": "OPERATIONAL_SCENE_PLAYLIST_NOT_NEW_SCIENTIFIC_COHORT",
        },
        "selection": {
            "rule": "EXISTING_REVIEWED_UNIQUE_GOAL_WITH_AT_LEAST_TWO_DISTINCT_FRAMES",
            "provider_outcome_blind": True,
            "goal_names": list(SELECTED_GOALS),
        },
        "locations": public_locations,
        "claim_ceiling": "NETWORK_SCENE_MECHANICAL_REPLAY_ONLY_NO_REAL_USER_OR_SCIENTIFIC_CONFIRMATION",
    }
    truth = {
        "schema_version": 1,
        "authority": "EVALUATOR_ONLY_NOT_PROVIDER_VISIBLE",
        "frames": truth_frames,
        "public_manifest_sha256": _canonical_hash(public),
    }
    _atomic_json(output_dir / PUBLIC_MANIFEST, public)
    _atomic_json(output_dir / TRUTH_SIDECAR, truth)
    return {
        "status": "ONLINE_SCENES_PREPARED",
        "location_count": len(public_locations),
        "episode_count": sum(len(item["episodes"]) for item in public_locations),
        "distinct_frame_count": sum(len(item["frames"]) for item in public_locations),
        "public_manifest_sha256": truth["public_manifest_sha256"],
    }


def _acquire_lock(run_dir: Path) -> Path:
    lock = run_dir / "online-run.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
        except (ValueError, OSError):
            lock.unlink()
        else:
            raise ContractError(f"online scene runner already owns this run (pid={pid})")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    return lock


def _episode_directory(run_dir: Path, episode_id: str) -> Path:
    return run_dir / "episodes" / episode_id


def _load_last_observation_event(path: Path) -> Mapping[str, Any]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [item for item in events if item.get("event_type") == "OBSERVATION_PROCESSED"]
    if not matches:
        raise ContractError("terminal episode has no observation event")
    return matches[-1]


def _normalize_network_summary(
    *, directory: Path, summary: Mapping[str, Any], execution_mode: str, claim_ceiling: str
) -> dict[str, Any]:
    """Attach wall-clock timing from immutable provider observation receipts.

    Early V0 receipts recorded capture time before the provider call. Receipt
    modification time is therefore used only for user-visible discovery
    latency; provider contents and attempts are never rewritten or replayed.
    """

    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = next(item for item in events if item.get("event_type") == "EPISODE_STARTED")
    grounded = [
        item
        for item in events
        if item.get("event_type") == "OBSERVATION_PROCESSED" and item.get("p0_status") == "GROUNDED"
    ]
    normalized = dict(summary)
    if grounded:
        first = grounded[0]
        receipt = directory / "provider_calls" / first["observation_id"] / "observation.json"
        if not receipt.is_file():
            raise ContractError("grounded event is missing its provider observation receipt")
        normalized["first_discovery_time_ms"] = (
            receipt.stat().st_mtime_ns // 1_000_000 - int(started["started_at_ms"])
        )
    normalized["execution_mode"] = execution_mode
    normalized["claim_ceiling"] = claim_ceiling
    normalized["timing_basis"] = "PROVIDER_OBSERVATION_RECEIPT_MTIME_MINUS_EPISODE_START"
    return normalized


def _false_confirmation(event: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    candidate = event.get("candidate")
    if not isinstance(candidate, Mapping):
        return True
    frame_id = str(event["frame_id"])
    frame_truth = truth["frames"].get(frame_id)
    if not isinstance(frame_truth, Mapping) or frame_truth.get("goal_reference_resolution") != "UNIQUE":
        return True
    region = {
        "frame_id": frame_id,
        "coordinate_space": "NORMALIZED_XYXY",
        **{key: candidate[key] for key in ("x_min", "y_min", "x_max", "y_max")},
    }
    return max(
        (p0_evaluator.region_iou(region, item) for item in frame_truth["acceptable_spatial_regions"]),
        default=0.0,
    ) < p0_evaluator.IOU_THRESHOLD


def _write_progress(run_dir: Path, total: int) -> None:
    summaries = list((run_dir / "episodes").glob("*/episode_summary.json")) if (run_dir / "episodes").exists() else []
    states = list((run_dir / "episodes").glob("*/state.json")) if (run_dir / "episodes").exists() else []
    _atomic_json(
        run_dir / "progress.json",
        {
            "schema_version": 1,
            "completed_or_adjudicated": len(summaries),
            "started": len(states),
            "total": total,
            "percent": round(100.0 * len(summaries) / total, 2),
            "last_activity_at_ms": _now_ms(),
        },
    )


def execute_online_scenes(*, scene_dir: Path, run_dir: Path, codex_exe: Path, model_dir: Path) -> dict[str, Any]:
    public = _read_json(scene_dir / PUBLIC_MANIFEST)
    truth = _read_json(scene_dir / TRUTH_SIDECAR)
    if truth.get("public_manifest_sha256") != _canonical_hash(public):
        raise ContractError("online scene public/truth binding drift")
    locations = public.get("locations")
    if not isinstance(locations, list) or len(locations) != 3:
        raise ContractError("online scene run requires exactly three locations")
    flattened = [(location, episode) for location in locations for episode in location["episodes"]]
    if len(flattened) != 15:
        raise ContractError("online scene run requires exactly fifteen episodes")

    run_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(run_dir)
    try:
        provider_lock_path = run_dir / "provider-lock.json"
        if provider_lock_path.exists():
            provider_lock = _read_json(provider_lock_path)
        else:
            provider_lock = preflight_provider(codex_exe=codex_exe, model_dir=model_dir)
            _atomic_json(provider_lock_path, provider_lock)
        run_manifest = {
            "schema_version": 1,
            "scene_public_sha256": _canonical_hash(public),
            "execution_mode": public["execution_mode"],
            "task_recovery": "SKIP_ADJUDICATED_EPISODES_AND_REUSE_COMPLETED_OBSERVATION_RECEIPTS",
            "iteration_recovery": "ATOMIC_STATE_AFTER_EACH_OBSERVATION_NO_PROVIDER_CALL_REPLAY",
            "claim_ceiling": public["claim_ceiling"],
        }
        run_manifest_path = run_dir / "run-manifest.json"
        if run_manifest_path.exists() and _read_json(run_manifest_path) != run_manifest:
            raise ContractError("online scene run manifest drift; refusing resume")
        if not run_manifest_path.exists():
            _atomic_json(run_manifest_path, run_manifest)
        frame_by_id = {
            frame["frame_id"]: frame for location in locations for frame in location["frames"]
        }
        policy = Policy()
        for ordinal, (location, episode) in enumerate(flattened, start=1):
            episode_id = episode["episode_id"]
            directory = _episode_directory(run_dir, episode_id)
            summary_path = directory / "episode_summary.json"
            if summary_path.exists():
                print(f"episode {ordinal}/15 {episode_id} already adjudicated", flush=True)
                continue
            state_path = directory / "state.json"
            events_path = directory / "events.jsonl"
            if state_path.exists():
                state = EpisodeState.from_dict(_read_json(state_path))
            else:
                state = EpisodeState.start(
                    episode_id=episode_id,
                    location_id=location["location_id"],
                    goal_name=location["goal_name"],
                    started_at_ms=_now_ms(),
                )
                _atomic_json(state_path, state.to_dict())
                _append_event(
                    events_path,
                    {
                        "schema_version": 1,
                        "event_type": "EPISODE_STARTED",
                        "episode_id": episode_id,
                        "location_id": state.location_id,
                        "goal_name": state.goal_name,
                        "started_at_ms": state.started_at_ms,
                        "state": state.state,
                    },
                )

            playlist = episode["frame_ids"]
            while state.state not in {State.COMPLETE.value, State.ABSTAIN.value}:
                index = state.observation_count
                if index >= len(playlist):
                    attribution = (
                        Attribution.CURRENT_FRAME_GROUNDING_BOTTLENECK
                        if state.reliable_observation_count == 0
                        else Attribution.INTERACTION_OR_CONTROL_BOTTLENECK
                    )
                    event = stop_episode(
                        state,
                        stopped_at_ms=_now_ms(),
                        attribution=attribution,
                        reason="PUBLIC_NETWORK_SCENE_PLAYLIST_EXHAUSTED",
                    )
                    _append_event(events_path, event)
                    _atomic_json(state_path, state.to_dict())
                    break
                frame = frame_by_id[playlist[index]]
                observation_id = f"{episode_id}-o{index + 1:03d}"
                call_dir = directory / "provider_calls" / observation_id
                cached_observation = call_dir / "observation.json"
                if cached_observation.exists():
                    observation = _read_json(cached_observation)
                else:
                    try:
                        observation = ground_current_frame(
                            provider_lock=provider_lock,
                            call_dir=call_dir,
                            episode_id=episode_id,
                            goal_name=state.goal_name,
                            image_path=Path(frame["image_path"]),
                            frame_id=frame["frame_id"],
                            observation_id=observation_id,
                            captured_at_ms=_now_ms(),
                        )
                    except ProviderAdapterError as error:
                        observation = _failed_provider_observation(
                            episode_id=episode_id,
                            observation_id=observation_id,
                            frame_id=frame["frame_id"],
                            frame_sha256=frame["image_sha256"],
                            captured_at_ms=_now_ms(),
                            reason=str(error),
                        )
                result = apply_observation(state, observation, policy)
                state = result.state
                _append_event(events_path, result.event)
                _atomic_json(state_path, state.to_dict())
                _write_progress(run_dir, 15)
                print(
                    f"episode {ordinal}/15 {episode_id} observation={state.observation_count}/{len(playlist)} "
                    f"p0={result.event['p0_status']} state={state.state}",
                    flush=True,
                )

            terminal_event = _load_last_observation_event(events_path)
            if state.state == State.COMPLETE.value:
                false_confirmation = _false_confirmation(terminal_event, truth)
                attribution = None
            else:
                false_confirmation = False
                attribution = (
                    Attribution.CURRENT_FRAME_GROUNDING_BOTTLENECK
                    if state.reliable_observation_count == 0
                    or state.consecutive_unreliable >= policy.max_consecutive_unreliable
                    else Attribution.INTERACTION_OR_CONTROL_BOTTLENECK
                )
            summary = adjudicate_episode(
                state,
                adjudicated_at_ms=_now_ms(),
                false_entrance_confirmation=false_confirmation,
                failure_attribution=attribution,
            )
            _atomic_json(summary_path, summary)
            _append_event(events_path, {"event_type": "EPISODE_ADJUDICATED", **summary})
            _write_progress(run_dir, 15)
            print(
                f"episode {ordinal}/15 {episode_id} terminal={state.state} "
                f"false_confirmation={false_confirmation} attribution={summary['failure_attribution']}",
                flush=True,
            )

        summaries = []
        for path in sorted((run_dir / "episodes").glob("*/episode_summary.json")):
            summary = _normalize_network_summary(
                directory=path.parent,
                summary=_read_json(path),
                execution_mode=public["execution_mode"],
                claim_ceiling=public["claim_ceiling"],
            )
            _atomic_json(path, summary)
            summaries.append(summary)
        report = summarize_field_run(summaries)
        report["execution_mode"] = public["execution_mode"]
        report["claim_ceiling"] = public["claim_ceiling"]
        report["source"] = public["source"]
        _atomic_json(run_dir / "field_report.json", report)
        return report
    finally:
        if lock.exists():
            lock.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--brain-cohort", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--scene-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--codex-exe", type=Path, default=Path("E:/codex-tools/bin/codex.exe"))
    run.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts.local/models/grounding-dino-tiny-a2bb814"),
    )
    args = parser.parse_args(argv)
    try:
        result = (
            prepare_online_scenes(brain_cohort=args.brain_cohort, output_dir=args.output_dir)
            if args.command == "prepare"
            else execute_online_scenes(
                scene_dir=args.scene_dir,
                run_dir=args.run_dir,
                codex_exe=args.codex_exe,
                model_dir=args.model_dir,
            )
        )
    except (ContractError, ProviderAdapterError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"fail-closed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

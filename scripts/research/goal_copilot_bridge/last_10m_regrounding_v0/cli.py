"""File-backed field runner for BLINDASSIST_LAST_10M_REGROUNDING_V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import (
    Attribution,
    ContractError,
    EpisodeState,
    Policy,
    adjudicate_episode,
    apply_observation,
    stop_episode,
    summarize_field_run,
)
from .provider_adapter import ProviderAdapterError, ground_current_frame, preflight_provider


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse_site(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("site must be LOCATION_ID=GOAL_NAME")
    location_id, goal_name = value.split("=", 1)
    if not location_id.strip() or not goal_name.strip():
        raise argparse.ArgumentTypeError("site id and goal name must be non-empty")
    return location_id.strip(), goal_name.strip()


def _episode_dir(run_dir: Path, episode_id: str) -> Path:
    return run_dir / "episodes" / episode_id


def command_init_roster(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.site) != 3 or len({site[0] for site in args.site}) != 3:
        raise ContractError("init-roster requires exactly three distinct real locations")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ContractError("run directory must be absent or empty")
    episodes = []
    for location_id, goal_name in args.site:
        for index in range(1, 6):
            episodes.append(
                {
                    "episode_id": f"{location_id}-e{index:02d}",
                    "location_id": location_id,
                    "goal_name": goal_name,
                    "status": "PLANNED",
                }
            )
    roster = {
        "schema_version": 1,
        "milestone_id": "BLINDASSIST_LAST_10M_REGROUNDING_V0",
        "created_at_ms": _now_ms(),
        "required_shape": "3_LOCATIONS_X_5_EPISODES",
        "episodes": episodes,
    }
    _atomic_json(args.run_dir / "roster.json", roster)
    return roster


def _roster_entry(run_dir: Path, episode_id: str) -> Mapping[str, Any]:
    roster = _read_json(run_dir / "roster.json")
    matches = [item for item in roster.get("episodes", []) if item.get("episode_id") == episode_id]
    if len(matches) != 1:
        raise ContractError("episode_id is absent or duplicated in roster")
    return matches[0]


def command_start(args: argparse.Namespace) -> dict[str, Any]:
    entry = _roster_entry(args.run_dir, args.episode_id)
    directory = _episode_dir(args.run_dir, args.episode_id)
    state_path = directory / "state.json"
    if state_path.exists():
        raise ContractError("episode has already started")
    state = EpisodeState.start(
        episode_id=args.episode_id,
        location_id=str(entry["location_id"]),
        goal_name=str(entry["goal_name"]),
        started_at_ms=_now_ms(),
    )
    _atomic_json(state_path, state.to_dict())
    event = {
        "schema_version": 1,
        "event_type": "EPISODE_STARTED",
        "episode_id": state.episode_id,
        "location_id": state.location_id,
        "goal_name": state.goal_name,
        "started_at_ms": state.started_at_ms,
        "state": state.state,
    }
    _append_event(directory / "events.jsonl", event)
    return event


def command_provider_preflight(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = args.run_dir / "provider-lock.json"
    if lock_path.exists():
        raise ContractError("provider lock already exists")
    provider = preflight_provider(codex_exe=args.codex_exe, model_dir=args.model_dir)
    _atomic_json(lock_path, provider)
    return provider


def _load_state(directory: Path) -> EpisodeState:
    return EpisodeState.from_dict(_read_json(directory / "state.json"))


def command_observe(args: argparse.Namespace) -> dict[str, Any]:
    directory = _episode_dir(args.run_dir, args.episode_id)
    state = _load_state(directory)
    observation = _read_json(args.observation)
    policy = Policy(
        center_left=args.center_left,
        center_right=args.center_right,
        arrival_min_height=args.arrival_min_height,
        max_consecutive_unreliable=args.max_consecutive_unreliable,
        max_instructions=args.max_instructions,
    )
    result = apply_observation(state, observation, policy)
    _append_event(directory / "events.jsonl", result.event)
    _atomic_json(directory / "state.json", result.state.to_dict())
    return {"state": result.state.state, "message": result.message, "event": result.event}


def _failed_provider_observation(
    *,
    episode_id: str,
    observation_id: str,
    frame_id: str,
    frame_sha256: str,
    captured_at_ms: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "observation_id": observation_id,
        "frame_id": frame_id,
        "frame_sha256": frame_sha256,
        "captured_at_ms": captured_at_ms,
        "processed_at_ms": _now_ms(),
        "p0_output": {
            "schema_version": 1,
            "episode_id": episode_id,
            "provider_runs": [
                {
                    "provider_id": "frozen-current-p0",
                    "status": "RUN_FAILED",
                    "source_frame_ids": [frame_id],
                    "evidence_ids": [],
                    "candidate_ids": [],
                    "failure_reason": reason[:512],
                }
            ],
            "evidence": [],
            "candidates": [],
            "decision": {
                "status": "INVALID_OBSERVATION",
                "selected_candidate_id": None,
                "ranked_candidate_ids": [],
                "source_frame_id": None,
                "decision_timestamp_ms": captured_at_ms,
                "spatial_region": None,
                "goal_identity_support": "NOT_EVALUABLE",
                "spatial_support": "NOT_EVALUABLE",
                "confidence": None,
                "supporting_evidence_ids": [],
                "competing_candidate_ids": [],
                "abstention_reason": "INVALID_INPUT",
                "persistence_handoff_token": None,
            },
        },
    }


def command_ground_observe(args: argparse.Namespace) -> dict[str, Any]:
    directory = _episode_dir(args.run_dir, args.episode_id)
    state = _load_state(directory)
    image = args.image.resolve()
    if not image.is_file():
        raise ContractError("current frame image is missing")
    frame_sha256 = _sha256_file(image)
    if (
        args.frame_id == state.last_frame_id
        or frame_sha256 == state.last_frame_sha256
        or args.observation_id == state.last_observation_id
    ):
        raise ContractError("ground-observe requires a fresh observation and frame")
    captured_at_ms = args.captured_at_ms if args.captured_at_ms is not None else _now_ms()
    provider_lock = _read_json(args.run_dir / "provider-lock.json")
    call_dir = directory / "provider_calls" / args.observation_id
    try:
        observation = ground_current_frame(
            provider_lock=provider_lock,
            call_dir=call_dir,
            episode_id=args.episode_id,
            goal_name=state.goal_name,
            image_path=image,
            frame_id=args.frame_id,
            observation_id=args.observation_id,
            captured_at_ms=captured_at_ms,
        )
    except ProviderAdapterError as error:
        observation = _failed_provider_observation(
            episode_id=args.episode_id,
            observation_id=args.observation_id,
            frame_id=args.frame_id,
            frame_sha256=frame_sha256,
            captured_at_ms=captured_at_ms,
            reason=str(error),
        )
    policy = Policy(
        center_left=args.center_left,
        center_right=args.center_right,
        arrival_min_height=args.arrival_min_height,
        max_consecutive_unreliable=args.max_consecutive_unreliable,
        max_instructions=args.max_instructions,
    )
    result = apply_observation(state, observation, policy)
    _append_event(directory / "events.jsonl", result.event)
    _atomic_json(directory / "state.json", result.state.to_dict())
    return {"state": result.state.state, "message": result.message, "event": result.event}


def command_stop(args: argparse.Namespace) -> dict[str, Any]:
    directory = _episode_dir(args.run_dir, args.episode_id)
    state = _load_state(directory)
    event = stop_episode(
        state,
        stopped_at_ms=_now_ms(),
        attribution=Attribution(args.attribution),
        reason=args.reason,
    )
    _append_event(directory / "events.jsonl", event)
    _atomic_json(directory / "state.json", state.to_dict())
    return event


def command_adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    directory = _episode_dir(args.run_dir, args.episode_id)
    state = _load_state(directory)
    attribution = Attribution(args.attribution) if args.attribution else None
    summary = adjudicate_episode(
        state,
        adjudicated_at_ms=_now_ms(),
        false_entrance_confirmation=args.false_entrance_confirmation,
        failure_attribution=attribution,
    )
    summary_path = directory / "episode_summary.json"
    if summary_path.exists():
        raise ContractError("episode has already been adjudicated")
    _atomic_json(summary_path, summary)
    _append_event(
        directory / "events.jsonl",
        {
            "schema_version": 1,
            "event_type": "EPISODE_ADJUDICATED",
            **summary,
        },
    )
    return summary


def command_summarize(args: argparse.Namespace) -> dict[str, Any]:
    summaries = []
    episodes_dir = args.run_dir / "episodes"
    if episodes_dir.exists():
        for path in sorted(episodes_dir.glob("*/episode_summary.json")):
            summaries.append(_read_json(path))
    report = summarize_field_run(summaries)
    _atomic_json(args.run_dir / "field_report.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-roster")
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--site", action="append", type=_parse_site, required=True)
    init.set_defaults(handler=command_init_roster)

    start = subparsers.add_parser("start")
    start.add_argument("--run-dir", type=Path, required=True)
    start.add_argument("--episode-id", required=True)
    start.set_defaults(handler=command_start)

    provider_preflight = subparsers.add_parser("provider-preflight")
    provider_preflight.add_argument("--run-dir", type=Path, required=True)
    provider_preflight.add_argument("--codex-exe", type=Path, default=Path("E:/codex-tools/bin/codex.exe"))
    provider_preflight.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts.local/models/grounding-dino-tiny-a2bb814"),
    )
    provider_preflight.set_defaults(handler=command_provider_preflight)

    observe = subparsers.add_parser("observe")
    observe.add_argument("--run-dir", type=Path, required=True)
    observe.add_argument("--episode-id", required=True)
    observe.add_argument("--observation", type=Path, required=True)
    observe.add_argument("--center-left", type=float, default=0.42)
    observe.add_argument("--center-right", type=float, default=0.58)
    observe.add_argument("--arrival-min-height", type=float, default=0.55)
    observe.add_argument("--max-consecutive-unreliable", type=int, default=3)
    observe.add_argument("--max-instructions", type=int, default=12)
    observe.set_defaults(handler=command_observe)

    ground_observe = subparsers.add_parser("ground-observe")
    ground_observe.add_argument("--run-dir", type=Path, required=True)
    ground_observe.add_argument("--episode-id", required=True)
    ground_observe.add_argument("--image", type=Path, required=True)
    ground_observe.add_argument("--frame-id", required=True)
    ground_observe.add_argument("--observation-id", required=True)
    ground_observe.add_argument("--captured-at-ms", type=int)
    ground_observe.add_argument("--center-left", type=float, default=0.42)
    ground_observe.add_argument("--center-right", type=float, default=0.58)
    ground_observe.add_argument("--arrival-min-height", type=float, default=0.55)
    ground_observe.add_argument("--max-consecutive-unreliable", type=int, default=3)
    ground_observe.add_argument("--max-instructions", type=int, default=12)
    ground_observe.set_defaults(handler=command_ground_observe)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--run-dir", type=Path, required=True)
    stop.add_argument("--episode-id", required=True)
    stop.add_argument(
        "--attribution",
        required=True,
        choices=[
            Attribution.CURRENT_FRAME_GROUNDING_BOTTLENECK.value,
            Attribution.INTERACTION_OR_CONTROL_BOTTLENECK.value,
        ],
    )
    stop.add_argument("--reason", required=True)
    stop.set_defaults(handler=command_stop)

    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--run-dir", type=Path, required=True)
    adjudicate.add_argument("--episode-id", required=True)
    truth = adjudicate.add_mutually_exclusive_group(required=True)
    truth.add_argument("--false-entrance-confirmation", action="store_true", dest="false_entrance_confirmation")
    truth.add_argument("--correct-entrance-confirmation", action="store_false", dest="false_entrance_confirmation")
    adjudicate.add_argument("--attribution", choices=[value.value for value in Attribution])
    adjudicate.set_defaults(handler=command_adjudicate)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.set_defaults(handler=command_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (ContractError, ProviderAdapterError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"fail-closed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

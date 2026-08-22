"""Run one frozen ABotN RGB through the unchanged V0 provider behind a truth firewall."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "blindassist_abotn_arrival_provider_firewall_canary_v0"
WEBGL_SCHEMA = "blindassist_abotn_webgl_render_canary_v0"
ARRIVE_THRESHOLD_M = 2.0
PRIVATE_FIELD_NAMES = (
    "target_position",
    "distance_to_goal",
    "end_point",
    "euclidean_distance",
    "private-arrival-truth",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _xy_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def build_frozen_inputs(annotation: Mapping[str, Any], webgl: Mapping[str, Any]) -> tuple[dict, dict]:
    if webgl.get("schema_version") != WEBGL_SCHEMA or webgl.get("terminal") != "WEBGL_RENDER_TRANSPORT_CANARY_PASS":
        raise ValueError("WebGL receipt is not an eligible frozen RGB source")
    trajectory = annotation.get("trajectory")
    extension = annotation.get("label", {}).get("extend", {})
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("ABotN task trajectory is missing")
    goal_name = str(extension.get("goal_label") or "").strip()
    endpoint = extension.get("end_point")
    if not goal_name or not isinstance(endpoint, list) or len(endpoint) != 2:
        raise ValueError("ABotN task lacks named-goal metric endpoint")
    first = trajectory[0]
    last = trajectory[-1]
    initial_xy = [float(first["x"]), float(first["y"])]
    trajectory_final_xy = [float(last["x"]), float(last["y"])]
    endpoint_xy = [float(endpoint[0]), float(endpoint[1])]
    if _xy_distance(endpoint_xy, trajectory_final_xy) > 0.25:
        raise ValueError("label endpoint and trajectory endpoint disagree")

    screenshot = Path(str(webgl.get("screenshot", {}).get("path", ""))).resolve()
    expected_screenshot_hash = str(webgl.get("screenshot", {}).get("sha256", ""))
    if not screenshot.is_file() or _sha256(screenshot) != expected_screenshot_hash:
        raise ValueError("frozen WebGL screenshot is missing or changed")
    camera = webgl.get("camera", {})
    public = {
        "schema_version": "blindassist_abotn_provider_public_envelope_v0",
        "episode_id": "abotn-20260227163550-traj-0",
        "observation_id": "abotn-20260227163550-traj-0-o000",
        "frame_id": "abotn-20260227163550-traj-0-o000",
        "captured_at_ms": 0,
        "goal_contract": {
            "goal_type": "NAMED_POI",
            "target_name": goal_name,
            "instruction": str(annotation.get("instruction") or f"前往{goal_name}"),
        },
        "rgb": {
            "path": str(screenshot),
            "sha256": expected_screenshot_hash,
            "width": int(webgl["screenshot"]["pixel_stats"]["width"]),
            "height": int(webgl["screenshot"]["pixel_stats"]["height"]),
            "source": "FROZEN_ABOTN_WEBGL_INITIAL_VIEW",
        },
        "public_camera_state": {
            "source_initial_position": camera.get("source_initial_position"),
            "source_initial_euler_radians": camera.get("source_initial_euler_radians"),
        },
        "private_truth_access": False,
    }
    initial_distance = _xy_distance(initial_xy, endpoint_xy)
    private = {
        "schema_version": "blindassist_abotn_arrival_private_truth_v0",
        "episode_id": public["episode_id"],
        "observation_id": public["observation_id"],
        "goal_label": goal_name,
        "initial_xy": initial_xy,
        "endpoint_xy": endpoint_xy,
        "trajectory_final_xy": trajectory_final_xy,
        "endpoint_final_residual_m": _xy_distance(endpoint_xy, trajectory_final_xy),
        "initial_distance_to_goal_m": initial_distance,
        "arrive_threshold_m": ARRIVE_THRESHOLD_M,
        "arrival_rule": "distance_to_goal_m < arrive_threshold_m",
        "initial_arrival": initial_distance < ARRIVE_THRESHOLD_M,
        "functional_pixel_region_truth": "NOT_AVAILABLE",
    }
    return public, private


def audit_provider_call(call_dir: Path, private: Mapping[str, Any]) -> dict[str, Any]:
    prompt_path = call_dir / "brain-prompt.txt"
    schema_path = call_dir / "brain-output-schema.json"
    rendered_path = call_dir / "brain-input.jpg"
    for required in (prompt_path, schema_path, rendered_path, call_dir / "observation.json", call_dir / "completion.json"):
        if not required.is_file():
            raise ValueError(f"provider artifact missing: {required.name}")
    prompt = prompt_path.read_text(encoding="utf-8")
    lowered = prompt.lower()
    forbidden_hits = [name for name in PRIVATE_FIELD_NAMES if name.lower() in lowered]
    # Endpoint and derived distance are distinctive private literals.  The
    # threshold value (2.0) is intentionally not scanned as a bare substring:
    # it is too generic to distinguish leakage from public normalized pixels.
    for value in (*private["endpoint_xy"], private["initial_distance_to_goal_m"]):
        literal = str(value)
        if literal in prompt:
            forbidden_hits.append(literal)

    completed_item_types: list[str] = []
    external_action_events = 0
    for stdout_path in sorted(call_dir.glob("attempt-*-stdout.jsonl")):
        for line in stdout_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = str(event.get("type") or "")
            item_type = str(event.get("item", {}).get("type") or "")
            if event_type == "item.completed":
                completed_item_types.append(item_type)
                if item_type != "agent_message":
                    external_action_events += 1
            if "tool" in event_type.lower() or "command" in event_type.lower():
                external_action_events += 1
    return {
        "prompt_sha256": _sha256(prompt_path),
        "schema_sha256": _sha256(schema_path),
        "rendered_provider_input_sha256": _sha256(rendered_path),
        "forbidden_literal_hits": sorted(set(forbidden_hits)),
        "completed_item_types": completed_item_types,
        "external_action_event_count": external_action_events,
        "firewall_pass": not forbidden_hits and external_action_events == 0,
    }


def run_canary(
    *,
    annotation_path: Path,
    webgl_receipt_path: Path,
    output_dir: Path,
    codex_exe: Path,
    grounding_dino: Path,
) -> dict[str, Any]:
    from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.provider_adapter import (
        ground_current_frame,
        preflight_provider,
    )

    # Mechanical preflight intentionally precedes creation of any formal run directory.
    provider_lock = preflight_provider(codex_exe=codex_exe, model_dir=grounding_dino)
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing replay")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    webgl = json.loads(webgl_receipt_path.read_text(encoding="utf-8"))
    public, private = build_frozen_inputs(annotation, webgl)

    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "public-envelope.json", public)
    private_path = output_dir / "evaluator-private" / "private-arrival-truth.json"
    _atomic_json(private_path, private)
    _atomic_json(output_dir / "provider-lock.json", provider_lock)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "mode": "REVERSIBLE_EXPLORATION_CANARY_LITE",
        "status": "TRUTH_FROZEN_PROVIDER_NOT_RUN",
        "inputs": {
            "annotation_path": str(annotation_path.resolve()),
            "annotation_sha256": _sha256(annotation_path),
            "webgl_receipt_path": str(webgl_receipt_path.resolve()),
            "webgl_receipt_sha256": _sha256(webgl_receipt_path),
            "public_envelope_sha256": _sha256(output_dir / "public-envelope.json"),
            "private_truth_sha256": _sha256(private_path),
            "provider_lock_sha256": _sha256(output_dir / "provider-lock.json"),
        },
        "frozen_budget": {"provider_observations": 1, "brain_attempts_maximum": 2},
        "provider_private_truth_access": False,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    journal = {
        "schema_version": "blindassist_abotn_provider_dispatch_journal_v0",
        "status": "DISPATCH_STARTED",
        "started_at_utc": _utc_now(),
        "calls_dispatched": 1,
        "calls_completed": 0,
        "calls_in_doubt": 1,
        "private_truth_sha256_before_dispatch": manifest["inputs"]["private_truth_sha256"],
        "private_truth_access": False,
    }
    _atomic_json(output_dir / "provider-journal.json", journal)
    call_dir = output_dir / "provider-public" / "call-000"
    try:
        observation = ground_current_frame(
            provider_lock=provider_lock,
            call_dir=call_dir,
            episode_id=public["episode_id"],
            goal_name=public["goal_contract"]["target_name"],
            image_path=Path(public["rgb"]["path"]),
            frame_id=public["frame_id"],
            observation_id=public["observation_id"],
            captured_at_ms=int(public["captured_at_ms"]),
        )
    except Exception:
        completion_path = call_dir / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
        journal["status"] = "IN_DOUBT" if completion.get("status") == "IN_DOUBT" else "RUN_FAILED"
        journal["calls_in_doubt"] = 1 if journal["status"] == "IN_DOUBT" else 0
        _atomic_json(output_dir / "provider-journal.json", journal)
        raise

    if _sha256(private_path) != manifest["inputs"]["private_truth_sha256"]:
        raise ValueError("private arrival truth changed during provider call")
    audit = audit_provider_call(call_dir, private)
    if not audit["firewall_pass"]:
        raise ValueError(f"provider firewall audit failed: {audit}")
    completion = json.loads((call_dir / "completion.json").read_text(encoding="utf-8"))
    p0 = observation["p0_output"]
    decision = p0["decision"]
    brain_attempts = len(list(call_dir.glob("attempt-*-dispatch.json")))
    evaluation = {
        "schema_version": "blindassist_abotn_arrival_provider_canary_evaluation_v0",
        "observation_id": public["observation_id"],
        "initial_distance_to_goal_m": private["initial_distance_to_goal_m"],
        "arrive_threshold_m": private["arrive_threshold_m"],
        "arrival_state": "ARRIVED" if private["initial_arrival"] else "NOT_ARRIVED",
        "provider_decision_status": decision["status"],
        "provider_selected_candidate_id": decision.get("selected_candidate_id"),
        "provider_candidate_count": len(p0.get("candidates", [])),
        "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
        "bearing_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
        "range_accuracy": "NOT_EVALUABLE_RANGE_NOT_SHARED_WITH_PROVIDER_AND_SINGLE_INITIAL_STATE_ONLY",
        "arrival_success": "NOT_EVALUABLE_NO_CLOSED_LOOP_ACTION_EXECUTED",
    }
    _atomic_json(output_dir / "evaluation.json", evaluation)
    journal.update({
        "status": "COMPLETED",
        "completed_at_utc": _utc_now(),
        "calls_completed": 1,
        "calls_in_doubt": 0,
        "brain_attempts_dispatched": brain_attempts,
        "provider_completion_status": completion["status"],
    })
    _atomic_json(output_dir / "provider-journal.json", journal)
    result = {
        "schema_version": SCHEMA,
        "closed_at_utc": _utc_now(),
        "terminal": "ABOTN_ARRIVAL_PROVIDER_FIREWALL_CANARY_PASS",
        "provider": {
            "identity": "FROZEN_GROUNDING_DINO_TINY_PLUS_CODEX_TERRA_V0",
            "observation_calls": 1,
            "grounding_dino_calls": 1,
            "brain_attempts": brain_attempts,
            "in_doubt": 0,
            "decision_status": decision["status"],
            "candidate_count": len(p0.get("candidates", [])),
            "selected_candidate_id": decision.get("selected_candidate_id"),
        },
        "firewall": audit,
        "evaluation": evaluation,
        "truth_coverage": {
            "metric_arrival": "EVALUABLE_EVALUATOR_ONLY",
            "functional_pixel_region": "NOT_EVALUABLE",
            "selection_accuracy": "NOT_EVALUABLE",
        },
        "baseline_calls": 0,
        "teacher_calls": 0,
        "claim_ceiling": "REAL_RGB_PROVIDER_FIREWALL_AND_INTERFACE_MECHANICS_ONLY_NO_GROUNDING_NAVIGATION_OR_ARRIVAL_ACCURACY",
        "next_action": "FREEZE_TRAJECTORY_OBSERVATION_ROSTER_AND_RENDER_PIXELS_BEFORE_ANY_ADDITIONAL_PROVIDER_CALL",
        "artifact_sha256": {
            name: _sha256(output_dir / name)
            # manifest.json is finalized after this receipt, so including it
            # here would create a stale self-referential hash.
            for name in ("public-envelope.json", "provider-lock.json", "provider-journal.json", "evaluation.json")
        },
        "private_truth_sha256": _sha256(private_path),
        "provider_observation_sha256": _sha256(call_dir / "observation.json"),
    }
    _atomic_json(output_dir / "terminal-receipt.json", result)
    manifest.update({
        "status": "SEALED_PROVIDER_FIREWALL_CANARY_PASS",
        "terminal_receipt_sha256": _sha256(output_dir / "terminal-receipt.json"),
    })
    _atomic_json(output_dir / "manifest.json", manifest)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--webgl-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codex-exe", type=Path, default=Path("E:/codex-tools/bin/codex.exe"))
    parser.add_argument("--grounding-dino", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_canary(
        annotation_path=args.annotation.resolve(),
        webgl_receipt_path=args.webgl_receipt.resolve(),
        output_dir=args.output_dir.resolve(),
        codex_exe=args.codex_exe.resolve(),
        grounding_dino=args.grounding_dino.resolve(),
    )
    print(json.dumps({
        "terminal": result["terminal"],
        "provider": result["provider"],
        "evaluation": result["evaluation"],
        "next_action": result["next_action"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

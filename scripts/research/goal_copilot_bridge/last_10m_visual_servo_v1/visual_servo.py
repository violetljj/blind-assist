#!/usr/bin/env python3
"""Automated current-frame visual servoing over frozen real-facade view transforms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_TEXT_ENCODER_SHA256,
    EXPECTED_ULTRALYTICS_VERSION,
    PRIVATE_SCHEMA,
    PUBLIC_SCHEMA,
    iou,
    sha256,
    validated_box,
)


MANIFEST_SCHEMA = "blindassist_last_10m_visual_servo_manifest_v1"
AUTH_SCHEMA = "blindassist_last_10m_visual_servo_authorization_v1"
RUN_SCHEMA = "blindassist_last_10m_visual_servo_run_v1"
EVAL_SCHEMA = "blindassist_last_10m_visual_servo_evaluation_v1"
PROTOCOL_ID = "BLINDASSIST_LAST_10M_CURRENT_FRAME_VISUAL_SERVO_V1"
RENDER_SIZE = 640


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _manifest_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema mismatch")
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "manifest protocol mismatch")
    _require(manifest.get("created_before_selected_pixel_access") is True, "manifest pixel precedence missing")
    _require(manifest.get("created_before_private_truth_access") is True, "manifest truth precedence missing")
    _require(manifest.get("retry_or_replay_authorized") is False, "manifest permits replay")
    config = manifest.get("frozen_config")
    _require(isinstance(config, dict), "frozen_config missing")
    _require(config.get("goal_type") == "LEFTMOST_BUILDING_ENTRANCE", "goal type drift")
    _require(config.get("canonical_prompt") == "building entrance", "prompt drift")
    _require(config.get("zoom_factors") == [1.5, 1.8, 2.16, 2.592], "zoom grid drift")
    _require(config.get("pan_fractions") == [0.0, 0.25, 0.5, 0.75, 1.0], "pan grid drift")
    _require(config.get("start_zoom_index") == 0 and config.get("start_pan_index") == 2, "start state drift")
    _require(config.get("align_left") == 0.45 and config.get("align_right") == 0.55, "alignment contract drift")
    _require(config.get("arrival_min_height") == 0.55, "arrival height drift")
    _require(config.get("bounded_pool_size") == 10, "candidate pool drift")
    _require(config.get("max_observations") == 12, "observation budget drift")
    _require(config.get("no_candidate_action") == "TURN_LEFT", "active scan rule drift")
    _require(config.get("selection_rule") == "LEFTMOST_CANDIDATE_X_CENTER", "selection rule drift")
    return config


def crop_geometry(width: int, height: int, zoom: float, pan_fraction: float, confirm_phase: int) -> dict[str, float]:
    _require(width > 0 and height > 0 and zoom >= 1.0, "invalid source geometry")
    crop_width = width / zoom
    crop_height = height / zoom
    x_center = crop_width / 2.0 + pan_fraction * (width - crop_width)
    vertical_jitter = (0.01 * height) if confirm_phase else 0.0
    y_center = min(height - crop_height / 2.0, height / 2.0 + vertical_jitter)
    return {
        "x0": x_center - crop_width / 2.0,
        "y0": y_center - crop_height / 2.0,
        "x1": x_center + crop_width / 2.0,
        "y1": y_center + crop_height / 2.0,
    }


def transform_box(box: list[float], geometry: Mapping[str, float]) -> tuple[list[float] | None, float]:
    x0, y0, x1, y1 = box
    ix0 = max(x0, geometry["x0"])
    iy0 = max(y0, geometry["y0"])
    ix1 = min(x1, geometry["x1"])
    iy1 = min(y1, geometry["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return None, 0.0
    original_area = (x1 - x0) * (y1 - y0)
    visible_fraction = ((ix1 - ix0) * (iy1 - iy0)) / original_area
    rendered = [
        (ix0 - geometry["x0"]) / (geometry["x1"] - geometry["x0"]) * RENDER_SIZE,
        (iy0 - geometry["y0"]) / (geometry["y1"] - geometry["y0"]) * RENDER_SIZE,
        (ix1 - geometry["x0"]) / (geometry["x1"] - geometry["x0"]) * RENDER_SIZE,
        (iy1 - geometry["y0"]) / (geometry["y1"] - geometry["y0"]) * RENDER_SIZE,
    ]
    return rendered, visible_fraction


def choose_action(candidates: Sequence[Mapping[str, Any]], *, pending_confirmation: bool, config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
    if not candidates:
        return str(config["no_candidate_action"]), None
    selected = min(candidates, key=lambda candidate: (((candidate["bbox_xyxy"][0] + candidate["bbox_xyxy"][2]) / 2.0), int(candidate["provider_rank"])))
    box = validated_box(selected["bbox_xyxy"], "selected candidate")
    normalized = [value / RENDER_SIZE for value in box]
    if normalized[2] < float(config["align_left"]):
        return "TURN_LEFT", selected
    if normalized[0] > float(config["align_right"]):
        return "TURN_RIGHT", selected
    if normalized[3] - normalized[1] >= float(config["arrival_min_height"]):
        return ("COMPLETE" if pending_confirmation else "ARRIVAL_CONFIRM"), selected
    return "FORWARD", selected


def authorize(manifest_path: Path, public_path: Path, private_path: Path, run_output: Path, journal_output: Path, authorization_output: Path) -> dict[str, Any]:
    _require(not authorization_output.exists(), "authorization is immutable and already exists")
    _require(not run_output.exists() and not journal_output.exists(), "formal output path already exists")
    manifest = _read(manifest_path)
    _manifest_config(manifest)
    public = _read(public_path)
    private = _read(private_path)
    _require(public.get("schema_version") == PUBLIC_SCHEMA, "public schema mismatch")
    _require(private.get("schema_version") == PRIVATE_SCHEMA, "private schema mismatch")
    _require(private.get("public_input_sha256") == sha256(public_path), "private/public binding mismatch")
    public_cases = {case["case_id"]: case for case in public.get("cases", [])}
    private_cases = {case["case_id"]: case for case in private.get("cases", [])}
    _require(set(public_cases) == set(private_cases), "public/private roster mismatch")
    for case in public_cases.values():
        goal = case.get("goal_contract", {})
        _require(goal.get("goal_type") == "LEFTMOST_BUILDING_ENTRANCE" and goal.get("reference_mode") == "UNIQUE", "public goal contract drift")
    visible = sum(case.get("target_visibility") == "VISIBLE" for case in private_cases.values())
    minimum = int(manifest.get("minimum_visible_case_count", 0))
    _require(visible >= minimum >= 1, "visible-case authorization denominator not met")
    receipt = {
        "schema_version": AUTH_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(manifest_path),
        "public_input_sha256": sha256(public_path),
        "private_input_sha256": sha256(private_path),
        "visible_case_count": visible,
        "minimum_visible_case_count": minimum,
        "provider_model_calls_before_authorization": 0,
        "run_output_path": str(run_output.resolve()),
        "journal_output_path": str(journal_output.resolve()),
        "retry_or_replay_authorized": False,
        "execution_authorized": True,
    }
    _atomic_json(authorization_output, receipt)
    return receipt


def _render(image: Image.Image, geometry: Mapping[str, float], output: Path) -> None:
    crop = image.crop((geometry["x0"], geometry["y0"], geometry["x1"], geometry["y1"]))
    rendered = crop.resize((RENDER_SIZE, RENDER_SIZE), Image.Resampling.BILINEAR).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.jpg")
    rendered.save(temporary, format="JPEG", quality=95, subsampling=0)
    os.replace(temporary, output)


def run(manifest_path: Path, public_path: Path, authorization_path: Path, model_path: Path, text_encoder_path: Path, journal_path: Path, output_path: Path, render_root: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists() and not journal_path.exists(), "formal run is immutable and already exists")
    manifest = _read(manifest_path)
    config = _manifest_config(manifest)
    public = _read(public_path)
    authorization = _read(authorization_path)
    _require(authorization.get("schema_version") == AUTH_SCHEMA and authorization.get("execution_authorized") is True, "execution is not authorized")
    _require(authorization.get("manifest_sha256") == sha256(manifest_path), "authorization/manifest binding mismatch")
    _require(authorization.get("public_input_sha256") == sha256(public_path), "authorization/public binding mismatch")
    _require(authorization.get("run_output_path") == str(output_path.resolve()), "authorization output path mismatch")
    _require(authorization.get("journal_output_path") == str(journal_path.resolve()), "authorization journal path mismatch")
    _require(model_path.is_file() and sha256(model_path) == EXPECTED_MODEL_SHA256, "frozen YOLOE model mismatch")
    _require(text_encoder_path.is_file() and sha256(text_encoder_path) == EXPECTED_TEXT_ENCODER_SHA256, "frozen text encoder mismatch")

    import torch
    import ultralytics
    from ultralytics import YOLOE

    _require(ultralytics.__version__ == EXPECTED_ULTRALYTICS_VERSION, "Ultralytics version mismatch")
    _require(not device.startswith("cuda") or torch.cuda.is_available(), "requested CUDA device unavailable")
    model = YOLOE(str(model_path.resolve()))
    journal = {
        "schema_version": "blindassist_last_10m_visual_servo_journal_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "STARTED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_calls_dispatched": 0,
        "provider_calls_completed": 0,
        "retry_or_replay_authorized": False,
    }
    _atomic_json(journal_path, journal)
    episode_outputs = []
    try:
        for public_case in public.get("cases", []):
            case_id = public_case["case_id"]
            image_path = Path(public_case["query"]["image_path"])
            _require(image_path.is_file() and sha256(image_path) == public_case["query"]["image_sha256"], "public image binding mismatch")
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                width, height = source.size
                zoom_index = int(config["start_zoom_index"])
                pan_index = int(config["start_pan_index"])
                confirm_phase = 0
                pending_confirmation = False
                observations = []
                terminal = "MAX_OBSERVATIONS"
                visited: set[tuple[int, int, int, bool]] = set()
                for observation_index in range(1, int(config["max_observations"]) + 1):
                    state_key = (zoom_index, pan_index, confirm_phase, pending_confirmation)
                    if state_key in visited:
                        terminal = "CONTROL_LOOP"
                        break
                    visited.add(state_key)
                    zoom = float(config["zoom_factors"][zoom_index])
                    pan_fraction = float(config["pan_fractions"][pan_index])
                    geometry = crop_geometry(width, height, zoom, pan_fraction, confirm_phase)
                    rendered_path = (render_root / case_id / f"o{observation_index:03d}.jpg").resolve()
                    _require(not rendered_path.exists(), "refusing to overwrite rendered observation")
                    _render(source, geometry, rendered_path)
                    journal.update({"status": "DISPATCHING", "active_case_id": case_id, "active_observation": observation_index, "provider_calls_dispatched": journal["provider_calls_dispatched"] + 1})
                    _atomic_json(journal_path, journal)
                    previous = Path.cwd()
                    os.chdir(text_encoder_path.resolve().parent)
                    try:
                        model.set_classes([str(config["canonical_prompt"])])
                    finally:
                        os.chdir(previous)
                    started = time.perf_counter()
                    result = model.predict(source=str(rendered_path), verbose=False, device=device, imgsz=640, conf=0.001, max_det=100)[0]
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    ranked = sorted(zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True), key=lambda pair: pair[0], reverse=True)
                    candidates = [{"provider_rank": rank, "bbox_xyxy": [float(value) for value in box], "proposal_score": float(score)} for rank, (score, box) in enumerate(ranked[: int(config["bounded_pool_size"])], start=1)]
                    action, selected = choose_action(candidates, pending_confirmation=pending_confirmation, config=config)
                    observations.append({
                        "observation_index": observation_index,
                        "frame_id": f"{case_id}-z{zoom_index}-p{pan_index}-c{confirm_phase}-o{observation_index}",
                        "rendered_image_path": str(rendered_path),
                        "rendered_image_sha256": sha256(rendered_path),
                        "source_geometry": geometry,
                        "zoom_index": zoom_index,
                        "pan_index": pan_index,
                        "confirm_phase": confirm_phase,
                        "pending_confirmation_before": pending_confirmation,
                        "candidates": candidates,
                        "selected_candidate": selected,
                        "action": action,
                        "latency_ms": latency_ms,
                    })
                    journal.update({"status": "ACTIVE", "active_case_id": None, "active_observation": None, "provider_calls_completed": journal["provider_calls_completed"] + 1})
                    _atomic_json(journal_path, journal)
                    if action == "COMPLETE":
                        terminal = "COMPLETE"
                        break
                    if action == "ARRIVAL_CONFIRM":
                        pending_confirmation = True
                        confirm_phase = 1
                        continue
                    pending_confirmation = False
                    confirm_phase = 0
                    if action == "TURN_LEFT":
                        if pan_index == 0:
                            terminal = "ACTION_UNAVAILABLE"
                            break
                        pan_index -= 1
                    elif action == "TURN_RIGHT":
                        if pan_index == len(config["pan_fractions"]) - 1:
                            terminal = "ACTION_UNAVAILABLE"
                            break
                        pan_index += 1
                    elif action == "FORWARD":
                        if zoom_index == len(config["zoom_factors"]) - 1:
                            terminal = "ACTION_UNAVAILABLE"
                            break
                        zoom_index += 1
                    else:
                        raise ValueError(f"unsupported action {action}")
                episode_outputs.append({"case_id": case_id, "terminal": terminal, "observations": observations})
        payload = {
            "schema_version": RUN_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": sha256(manifest_path),
            "public_input_sha256": sha256(public_path),
            "authorization_sha256": sha256(authorization_path),
            "private_truth_access": False,
            "provider": {"name": "YOLOE-26n-seg", "model_sha256": sha256(model_path), "text_encoder_sha256": sha256(text_encoder_path), "ultralytics_version": ultralytics.__version__, "device": device},
            "episodes": episode_outputs,
            "claim_ceiling": manifest.get("claim_ceiling"),
        }
        _atomic_json(output_path, payload)
        journal.update({"status": "COMPLETED", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_output_sha256": sha256(output_path)})
        _atomic_json(journal_path, journal)
        return payload
    except Exception:
        journal.update({"status": "FAILED", "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        _atomic_json(journal_path, journal)
        raise


def evaluate(manifest_path: Path, public_path: Path, private_path: Path, run_path: Path, journal_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "evaluation is immutable and already exists")
    manifest = _read(manifest_path)
    config = _manifest_config(manifest)
    public = _read(public_path)
    private = _read(private_path)
    run_payload = _read(run_path)
    journal = _read(journal_path)
    _require(private.get("public_input_sha256") == sha256(public_path), "private/public binding mismatch")
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "run schema/private boundary mismatch")
    _require(run_payload.get("manifest_sha256") == sha256(manifest_path) and run_payload.get("public_input_sha256") == sha256(public_path), "run binding mismatch")
    _require(journal.get("status") == "COMPLETED" and journal.get("run_output_sha256") == sha256(run_path), "journal is not complete")
    _require(journal.get("provider_calls_dispatched") == journal.get("provider_calls_completed"), "provider call accounting mismatch")
    private_cases = {case["case_id"]: case for case in private.get("cases", [])}
    public_cases = {case["case_id"]: case for case in public.get("cases", [])}
    run_cases = {case["case_id"]: case for case in run_payload.get("episodes", [])}
    _require(set(private_cases) == set(public_cases) == set(run_cases), "evaluation roster mismatch")
    rows = []
    for case_id in sorted(run_cases):
        truth = private_cases[case_id]
        if truth.get("target_visibility") != "VISIBLE":
            continue
        target = validated_box(truth["legal_target_bboxes_xyxy"][0], f"{case_id} target")
        observations = run_cases[case_id]["observations"]
        target_available_any = False
        target_selected_any = False
        truth_arrival_flags = []
        selected_target_flags = []
        for observation in observations:
            transformed, visible_fraction = transform_box(target, observation["source_geometry"])
            candidates = observation.get("candidates", [])
            available = transformed is not None and any(iou(validated_box(candidate["bbox_xyxy"], f"{case_id} candidate"), transformed) >= 0.30 for candidate in candidates)
            selected = observation.get("selected_candidate")
            selected_hit = transformed is not None and selected is not None and iou(validated_box(selected["bbox_xyxy"], f"{case_id} selected"), transformed) >= 0.30
            truth_arrival = transformed is not None and visible_fraction >= 0.80 and transformed[0] <= RENDER_SIZE / 2.0 <= transformed[2] and (transformed[3] - transformed[1]) / RENDER_SIZE >= float(config["arrival_min_height"])
            target_available_any = target_available_any or available
            target_selected_any = target_selected_any or selected_hit
            truth_arrival_flags.append(truth_arrival)
            selected_target_flags.append(selected_hit)
        completed = run_cases[case_id]["terminal"] == "COMPLETE"
        true_completion = completed and len(observations) >= 2 and all(truth_arrival_flags[-2:]) and all(selected_target_flags[-2:]) and observations[-2]["action"] == "ARRIVAL_CONFIRM" and observations[-1]["action"] == "COMPLETE"
        false_completion = completed and not true_completion
        if true_completion:
            failure_layer = None
        elif not target_available_any:
            failure_layer = "NO_TARGET_CANDIDATE"
        elif not target_selected_any:
            failure_layer = "TARGET_NOT_SELECTED"
        else:
            failure_layer = "CURRENT_FRAME_CONTROL_OR_COMPLETION"
        rows.append({
            "case_id": case_id,
            "observation_count": len(observations),
            "terminal": run_cases[case_id]["terminal"],
            "target_available_any": target_available_any,
            "target_selected_any": target_selected_any,
            "completed": completed,
            "true_completion": true_completion,
            "false_completion": false_completion,
            "failure_layer": failure_layer,
            "actions": [observation["action"] for observation in observations],
        })
    denominator = len(rows)
    _require(denominator >= int(manifest["minimum_visible_case_count"]), "evaluation denominator drift")
    true_completions = sum(row["true_completion"] for row in rows)
    false_completions = sum(row["false_completion"] for row in rows)
    payload = {
        "schema_version": EVAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "evaluable_case_count": denominator,
        "true_completion_count": true_completions,
        "true_completion_rate": true_completions / denominator,
        "false_completion_count": false_completions,
        "target_available_case_count": sum(row["target_available_any"] for row in rows),
        "target_selected_case_count": sum(row["target_selected_any"] for row in rows),
        "failure_layer_counts": {layer: sum(row["failure_layer"] == layer for row in rows) for layer in ("NO_TARGET_CANDIDATE", "TARGET_NOT_SELECTED", "CURRENT_FRAME_CONTROL_OR_COMPLETION")},
        "provider_call_count": journal["provider_calls_completed"],
        "rows": rows,
        "inputs": {"manifest_sha256": sha256(manifest_path), "public_sha256": sha256(public_path), "private_sha256": sha256(private_path), "run_sha256": sha256(run_path), "journal_sha256": sha256(journal_path)},
        "claim_ceiling": manifest.get("claim_ceiling"),
        "terminal": "CURRENT_FRAME_VISUAL_SERVO_SIGNAL_ESTABLISHED" if true_completions > 0 and false_completions == 0 else ("CURRENT_FRAME_VISUAL_SERVO_FALSE_COMPLETION" if false_completions else "CURRENT_FRAME_VISUAL_SERVO_COMPLETION_NOT_ESTABLISHED"),
    }
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("authorize")
    for name in ("manifest", "public", "private", "run-output", "journal-output", "authorization-output"):
        auth.add_argument(f"--{name}", required=True, type=Path)
    execute = sub.add_parser("run")
    for name in ("manifest", "public", "authorization", "model", "text-encoder", "journal", "output", "render-root"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    execute.add_argument("--device", default="cuda:0")
    evaluation = sub.add_parser("evaluate")
    for name in ("manifest", "public", "private", "run", "journal", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "authorize":
        receipt = authorize(args.manifest, args.public, args.private, args.run_output, args.journal_output, args.authorization_output)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "run":
        run(args.manifest, args.public, args.authorization, args.model, args.text_encoder, args.journal, args.output, args.render_root, args.device)
    else:
        payload = evaluate(args.manifest, args.public, args.private, args.run, args.journal, args.output)
        print(json.dumps({key: payload[key] for key in ("evaluable_case_count", "true_completion_count", "false_completion_count", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

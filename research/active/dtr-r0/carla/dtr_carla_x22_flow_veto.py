from __future__ import annotations

"""Frozen CARLA V18 X21/X22 paired replay and one-shot evaluator.

The prediction phase consumes only observable, sealed tracker refreshes.  It
compares the unchanged X21 refresh-first transition with one intervention:
when current optical flow supports the transported stable state and the
proposed partial-observation velocity is discontinuous, X22 vetoes that
replacement.  Simulator identity and geometry remain evaluator-only.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
DTR_ROOT = HERE.parent
if str(DTR_ROOT) not in sys.path:
    sys.path.insert(0, str(DTR_ROOT))

import dtr_x21_ancestry_core as ancestry  # noqa: E402
import carla_raw_source as carla_source  # noqa: E402


EXPERIMENT_ID = "CARLA_C2_WEARABLE_PHYSICAL_OCCLUSION_SOURCE_V18_FLOW_ONSET"
ONLINE_MANIFEST_SCHEMA = "blindassist-dtr-carla-v18-online-refresh-manifest-v1"
ONLINE_FRAME_SCHEMA = "blindassist-dtr-carla-v18-online-refresh-frame-v1"
FREEZE_SCHEMA = "blindassist-dtr-carla-v18-x22-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-v18-x21-x22-predictions-v1"
RESULT_SCHEMA = "blindassist-dtr-carla-v18-x22-one-score-result-v1"
ARM_X21 = "X21_SAME_SOURCE_REFRESH_FIRST_ANCESTRY"
ARM_X22 = "X22_FLOW_STABLE_REPLACEMENT_VETO"

MIN_FLOW_COHERENT_RATIO = 0.60
MAX_DEGRADED_SUPPORT_RATIO = 0.65
MAX_DEGRADED_FOOTPRINT_OVERLAP = 0.25
MIN_VELOCITY_JUMP_MPS = 0.50
MIN_STABLE_RESIDUAL_ADVANTAGE_PX = 0.75
CARRY_WINDOW_S = 0.50
ROUTE_HALF_WIDTH_M = 0.65
ROUTE_HORIZON_S = 3.0
MIN_CLOSING_SPEED_MPS = 0.05
EPSILON = 1e-9

ONLINE_FORBIDDEN_KEYS = frozenset(
    {
        "actor_id",
        "actor_state",
        "instance",
        "instance_id",
        "obb",
        "target_obb_polygon_xy",
        "contact",
        "current_contact",
        "future_contact",
        "future_contact_within_horizon",
        "route_truth",
        "truth",
        "physical_loss",
        "evaluator",
    }
)


StateKey = tuple[str, int]


@dataclass(frozen=True)
class MotionState:
    track_token: str
    component: int
    time_s: float
    evidence_time_s: float
    forward_m: float
    left_m: float
    velocity_forward_mps: float
    velocity_left_mps: float
    radius_m: float
    disposition: str


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"x22_json_object:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"x22_jsonl_object:{path}:{line_number}")
        rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        raise


def assert_online_clean(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            require(str(key).lower() not in ONLINE_FORBIDDEN_KEYS, f"x22_online_forbidden_key:{path}.{key}")
            assert_online_clean(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_online_clean(child, f"{path}[{index}]")


def finite_number(value: Any, label: str) -> float:
    output = float(value)
    require(math.isfinite(output), f"x22_nonfinite:{label}")
    return output


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path.resolve(strict=True))
    require(protocol.get("experiment_id") == EXPERIMENT_ID, "x22_protocol_identity")
    constants = protocol["x22_contract"]["fixed_transition_constants"]
    expected = {
        "minimum_flow_coherent_ratio": MIN_FLOW_COHERENT_RATIO,
        "maximum_support_ratio_for_degraded_observation": MAX_DEGRADED_SUPPORT_RATIO,
        "maximum_footprint_overlap_for_degraded_observation": MAX_DEGRADED_FOOTPRINT_OVERLAP,
        "minimum_velocity_jump_mps": MIN_VELOCITY_JUMP_MPS,
        "minimum_stable_residual_advantage_px": MIN_STABLE_RESIDUAL_ADVANTAGE_PX,
        "carry_window_seconds": CARRY_WINDOW_S,
    }
    require(constants == expected, "x22_protocol_transition_constants")
    route = protocol["route_contract"]
    require(
        float(route["future_horizon_seconds"]) == ROUTE_HORIZON_S
        and float(route["route_body_radius_m"]) == ROUTE_HALF_WIDTH_M,
        "x22_protocol_route_constants",
    )
    frozen = protocol["frozen_files"]
    files = {
        "project_adapter": Path(carla_source.__file__).resolve(),
        "project_x22": Path(__file__).resolve(),
        "ancestry_core": Path(ancestry.__file__).resolve(),
    }
    for name, file_path in files.items():
        require(name in frozen, f"x22_protocol_frozen_file_missing:{name}")
        reference = frozen[name]
        require(Path(reference["path"]).resolve() == file_path, f"x22_protocol_frozen_path:{name}")
        require(sha256_file(file_path) == str(reference["sha256"]).upper(), f"x22_protocol_frozen_drift:{name}")
    return protocol


def _online_rows_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    raw = Path(str(manifest["rows"]["path"]))
    path = raw.resolve() if raw.is_absolute() else (manifest_path.parent / raw).resolve()
    require(path.is_file(), f"x22_online_rows_missing:{path}")
    return path


def load_online_refreshes(
    manifest_path: Path,
    protocol: Mapping[str, Any],
    *,
    expected_model_manifest_sha256: str,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema") == ONLINE_MANIFEST_SCHEMA
        and manifest.get("truth_blind") is True,
        "x22_online_manifest_schema",
    )
    require(
        manifest.get("identity_authority") == "ONLINE_TRACKER_EPISODE_LOCAL_NOT_SIMULATOR_ID",
        "x22_online_identity_authority",
    )
    require(
        manifest.get("coordinate_frame") == "WEARER_LOCAL_X_FORWARD_Y_LEFT"
        and manifest.get("velocity_semantics") == "WEARER_RELATIVE_METERS_PER_SECOND",
        "x22_online_motion_semantics",
    )
    require(
        str(manifest.get("source_model_manifest_sha256", "")).upper()
        == expected_model_manifest_sha256.upper(),
        "x22_online_source_manifest_drift",
    )
    assert_online_clean({key: value for key, value in manifest.items() if key != "forbidden_fields"})
    rows_path = _online_rows_path(manifest_path, manifest)
    require(sha256_file(rows_path) == str(manifest["rows"]["sha256"]).upper(), "x22_online_rows_hash")
    rows = read_jsonl(rows_path)
    assert_online_clean(rows)
    scenarios = list(protocol["scripted_motion"]["scenarios"])
    require(len(rows) == 804 == int(manifest["rows"]["frames"]), f"x22_online_frame_count:{len(rows)}")
    cursor = 0
    for scenario in scenarios:
        for sample_index in range(201):
            row = rows[cursor]
            cursor += 1
            require(row.get("schema") == ONLINE_FRAME_SCHEMA, f"x22_online_frame_schema:{cursor}")
            require(row.get("sequence") == scenario, f"x22_online_sequence:{cursor}:{scenario}")
            require(int(row.get("sample_index", -1)) == sample_index, f"x22_online_sample:{scenario}:{sample_index}")
            time_s = finite_number(row.get("time_s"), f"time:{scenario}:{sample_index}")
            require(abs(time_s - sample_index * 0.05) <= EPSILON, f"x22_online_clock:{scenario}:{sample_index}:{time_s}")
            live_tracks = row.get("live_tracks")
            carry_keys = row.get("carry_keys")
            refreshes = row.get("refreshes")
            require(isinstance(live_tracks, list) and len(live_tracks) == len(set(live_tracks)), f"x22_live_tracks:{scenario}:{sample_index}")
            require(all(isinstance(token, str) and token for token in live_tracks), f"x22_live_track_token:{scenario}:{sample_index}")
            require(isinstance(carry_keys, list), f"x22_carry_key_list:{scenario}:{sample_index}")
            parsed_carry: set[StateKey] = set()
            for carry in carry_keys:
                require(isinstance(carry, dict), f"x22_carry_key_object:{scenario}:{sample_index}")
                token = carry.get("track_token")
                require(isinstance(token, str) and token in live_tracks, f"x22_carry_track:{scenario}:{sample_index}")
                key = (token, int(carry.get("component")))
                require(key not in parsed_carry, f"x22_carry_key_duplicate:{scenario}:{sample_index}:{key}")
                parsed_carry.add(key)
            require(isinstance(refreshes, list), f"x22_refresh_list:{scenario}:{sample_index}")
            seen: set[StateKey] = set()
            for refresh in refreshes:
                require(isinstance(refresh, dict), f"x22_refresh_object:{scenario}:{sample_index}")
                token = refresh.get("track_token")
                require(isinstance(token, str) and token in live_tracks, f"x22_refresh_track:{scenario}:{sample_index}")
                key = (token, int(refresh.get("component")))
                require(key not in seen, f"x22_refresh_duplicate:{scenario}:{sample_index}:{key}")
                seen.add(key)
                for name in (
                    "forward_m",
                    "left_m",
                    "velocity_forward_mps",
                    "velocity_left_mps",
                    "radius_m",
                    "support_ratio",
                    "footprint_overlap",
                ):
                    finite_number(refresh.get(name), f"{name}:{scenario}:{sample_index}:{key}")
                require(float(refresh["radius_m"]) >= 0.0, f"x22_radius:{scenario}:{sample_index}:{key}")
                for name in ("support_ratio", "footprint_overlap"):
                    require(0.0 <= float(refresh[name]) <= 1.0, f"x22_ratio:{name}:{scenario}:{sample_index}:{key}")
                flow_names = ("flow_coherent_ratio", "stable_flow_residual_px", "proposed_flow_residual_px")
                present = [name in refresh and refresh[name] is not None for name in flow_names]
                require(all(present) or not any(present), f"x22_flow_evidence_partial:{scenario}:{sample_index}:{key}")
                if all(present):
                    for name in flow_names:
                        finite_number(refresh[name], f"{name}:{scenario}:{sample_index}:{key}")
                    require(0.0 <= float(refresh["flow_coherent_ratio"]) <= 1.0, f"x22_flow_ratio:{scenario}:{sample_index}:{key}")
                    require(
                        float(refresh["stable_flow_residual_px"]) >= 0.0
                        and float(refresh["proposed_flow_residual_px"]) >= 0.0,
                        f"x22_flow_residual:{scenario}:{sample_index}:{key}",
                    )
    return manifest, rows_path, rows


def _paths(run_root: Path) -> dict[str, Path]:
    return {
        "freeze": run_root / "freeze-v18.json",
        "predictions": run_root / "predictions-v18.json",
        "score_attempt": run_root / "score-attempt-v18.json",
        "result": run_root / "result-v18.json",
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    paths = _paths(run_root)
    require(not paths["freeze"].exists(), f"x22_freeze_exists:{paths['freeze']}")
    protocol_path = args.protocol.resolve(strict=True)
    protocol = validate_protocol(protocol_path)
    source_root = args.source_root.resolve(strict=True)
    model_manifest_path, _model_manifest = carla_source._load_model_manifest(source_root)
    model_manifest_sha256 = sha256_file(model_manifest_path)
    online_manifest_path = args.online_manifest.resolve(strict=True)
    online_manifest, rows_path, rows = load_online_refreshes(
        online_manifest_path,
        protocol,
        expected_model_manifest_sha256=model_manifest_sha256,
    )
    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_BEFORE_EVALUATOR_TRUTH_OPEN",
        "truth_blind_prediction": True,
        "experiment_id": EXPERIMENT_ID,
        "source_root": str(source_root),
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "source_model_manifest": {"path": str(model_manifest_path), "sha256": model_manifest_sha256},
        "online_refresh_manifest": {"path": str(online_manifest_path), "sha256": sha256_file(online_manifest_path)},
        "online_refresh_rows": {"path": str(rows_path), "sha256": sha256_file(rows_path), "frames": len(rows)},
        "algorithm_files": {
            "project_adapter": {"path": str(Path(carla_source.__file__).resolve()), "sha256": sha256_file(Path(carla_source.__file__).resolve())},
            "project_x22": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "ancestry_core": {"path": str(Path(ancestry.__file__).resolve()), "sha256": sha256_file(Path(ancestry.__file__).resolve())},
        },
        "online_contract": {
            "identity_authority": online_manifest["identity_authority"],
            "coordinate_frame": online_manifest["coordinate_frame"],
            "velocity_semantics": online_manifest["velocity_semantics"],
            "forbidden_fields": sorted(ONLINE_FORBIDDEN_KEYS),
        },
        "arms": [ARM_X21, ARM_X22],
        "fixed_transition_constants": protocol["x22_contract"]["fixed_transition_constants"],
        "one_score_gates": protocol["x22_contract"]["one_score_gates"],
        "forbidden": protocol["x22_contract"]["forbidden"],
    }
    atomic_json(paths["freeze"], value)
    return {**value, "freeze_sha256": sha256_file(paths["freeze"])}


def require_freeze(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = _paths(run_root)
    value = read_json(paths["freeze"].resolve(strict=True))
    require(value.get("schema") == FREEZE_SCHEMA and value.get("status") == "FROZEN_BEFORE_EVALUATOR_TRUTH_OPEN", "x22_freeze_schema")
    protocol_path = Path(value["protocol"]["path"]).resolve(strict=True)
    require(sha256_file(protocol_path) == value["protocol"]["sha256"], "x22_freeze_protocol_drift")
    protocol = validate_protocol(protocol_path)
    for reference in value["algorithm_files"].values():
        path = Path(reference["path"]).resolve(strict=True)
        require(sha256_file(path) == reference["sha256"], f"x22_freeze_algorithm_drift:{path}")
    source_root = Path(value["source_root"]).resolve(strict=True)
    model_path = Path(value["source_model_manifest"]["path"]).resolve(strict=True)
    require(sha256_file(model_path) == value["source_model_manifest"]["sha256"], "x22_freeze_model_manifest_drift")
    online_path = Path(value["online_refresh_manifest"]["path"]).resolve(strict=True)
    require(sha256_file(online_path) == value["online_refresh_manifest"]["sha256"], "x22_freeze_online_manifest_drift")
    _manifest, rows_path, rows = load_online_refreshes(
        online_path,
        protocol,
        expected_model_manifest_sha256=value["source_model_manifest"]["sha256"],
    )
    require(
        str(rows_path) == str(Path(value["online_refresh_rows"]["path"]).resolve())
        and sha256_file(rows_path) == value["online_refresh_rows"]["sha256"],
        "x22_freeze_online_rows_drift",
    )
    require(source_root == model_path.parents[2], "x22_freeze_source_root_drift")
    return value, protocol, rows


def motion_state(refresh: Mapping[str, Any], time_s: float) -> MotionState:
    return MotionState(
        track_token=str(refresh["track_token"]),
        component=int(refresh["component"]),
        time_s=time_s,
        evidence_time_s=time_s,
        forward_m=float(refresh["forward_m"]),
        left_m=float(refresh["left_m"]),
        velocity_forward_mps=float(refresh["velocity_forward_mps"]),
        velocity_left_mps=float(refresh["velocity_left_mps"]),
        radius_m=float(refresh["radius_m"]),
        disposition="OBSERVABLE_REFRESH",
    )


def transport(state: MotionState, target_time_s: float, *, disposition: str = "X21_TRANSPORT") -> MotionState:
    delta_s = target_time_s - state.time_s
    require(delta_s >= -EPSILON, f"x22_transport_future_state:{delta_s}")
    return replace(
        state,
        time_s=target_time_s,
        forward_m=state.forward_m + state.velocity_forward_mps * max(0.0, delta_s),
        left_m=state.left_m + state.velocity_left_mps * max(0.0, delta_s),
        disposition=disposition,
    )


def should_veto(previous: MotionState | None, refresh: Mapping[str, Any], time_s: float) -> bool:
    del time_s
    if previous is None:
        return False
    flow_values = [refresh.get(name) for name in ("flow_coherent_ratio", "stable_flow_residual_px", "proposed_flow_residual_px")]
    if any(value is None for value in flow_values):
        return False
    degraded = (
        float(refresh["support_ratio"]) <= MAX_DEGRADED_SUPPORT_RATIO
        or float(refresh["footprint_overlap"]) <= MAX_DEGRADED_FOOTPRINT_OVERLAP
    )
    velocity_jump = math.hypot(
        float(refresh["velocity_forward_mps"]) - previous.velocity_forward_mps,
        float(refresh["velocity_left_mps"]) - previous.velocity_left_mps,
    )
    residual_advantage = float(refresh["proposed_flow_residual_px"]) - float(refresh["stable_flow_residual_px"])
    return bool(
        degraded
        and float(refresh["flow_coherent_ratio"]) >= MIN_FLOW_COHERENT_RATIO
        and velocity_jump >= MIN_VELOCITY_JUMP_MPS
        and residual_advantage >= MIN_STABLE_RESIDUAL_ADVANTAGE_PX
    )


def stable_replacement(previous: MotionState, time_s: float) -> MotionState:
    carried = transport(previous, time_s, disposition="FLOW_STABLE_REPLACEMENT_VETO")
    return replace(carried, evidence_time_s=time_s)


def advance_arm(
    states: Mapping[StateKey, MotionState],
    refreshes: Mapping[StateKey, MotionState],
    *,
    live_tracks: set[str],
    carry_keys: set[StateKey],
    time_s: float,
) -> tuple[dict[StateKey, MotionState], list[MotionState], dict[str, int]]:
    def carry_filter(key: StateKey, previous: MotionState) -> MotionState | None:
        if key not in carry_keys:
            return None
        return transport(previous, time_s)

    return ancestry.advance(
        states,
        refreshes,
        live_ids=set(live_tracks),
        identity=lambda key: key[0],
        carry_filter=carry_filter,
    )


def first_tube_entry_s(state: MotionState) -> float | None:
    x, y = state.forward_m, state.left_m
    vx, vy = state.velocity_forward_mps, state.velocity_left_mps
    radius = ROUTE_HALF_WIDTH_M + state.radius_m
    distance = math.hypot(x, y)
    speed_squared = vx * vx + vy * vy
    if distance <= radius + EPSILON:
        closing = math.hypot(vx, vy) if distance <= EPSILON else -(x * vx + y * vy) / distance
        return 0.0 if closing >= MIN_CLOSING_SPEED_MPS else None
    if speed_squared <= EPSILON:
        return None
    b = 2.0 * (x * vx + y * vy)
    c = x * x + y * y - radius * radius
    discriminant = b * b - 4.0 * speed_squared * c
    if discriminant < 0.0:
        return None
    root = (-b - math.sqrt(max(0.0, discriminant))) / (2.0 * speed_squared)
    if root < -EPSILON or root > ROUTE_HORIZON_S + EPSILON:
        return None
    entry_s = max(0.0, root)
    entry_x, entry_y = x + vx * entry_s, y + vy * entry_s
    entry_distance = max(EPSILON, math.hypot(entry_x, entry_y))
    inward = -(entry_x * vx + entry_y * vy) / entry_distance
    return entry_s if inward + EPSILON >= MIN_CLOSING_SPEED_MPS else None


def frame_prediction(states: list[MotionState]) -> tuple[bool, float | None, int]:
    entries = [entry for state in states if (entry := first_tube_entry_s(state)) is not None]
    return bool(entries), (min(entries) if entries else None), len(entries)


def continued_states(
    history: list[tuple[float, list[MotionState]]], target_time_s: float
) -> list[MotionState]:
    pieces = ancestry.continue_window(
        history,
        target_time_s=target_time_s,
        transport=lambda payload, delta_s: [
            transport(state, state.time_s + delta_s, disposition="X21_CONTINUATION")
            for state in payload
        ],
        window_s=CARRY_WINDOW_S,
    )
    return [state for piece in pieces for state in piece]


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    paths = _paths(run_root)
    require(not paths["predictions"].exists(), f"x22_predictions_exist:{paths['predictions']}")
    frozen, protocol, rows = require_freeze(run_root)
    arms: dict[str, dict[str, Any]] = {ARM_X21: {"scenarios": {}}, ARM_X22: {"scenarios": {}}}
    total_vetoes = 0
    cursor = 0
    for scenario in protocol["scripted_motion"]["scenarios"]:
        x21_states: dict[StateKey, MotionState] = {}
        x22_states: dict[StateKey, MotionState] = {}
        x21_history: list[tuple[float, list[MotionState]]] = []
        x22_history: list[tuple[float, list[MotionState]]] = []
        arm_frames: dict[str, list[dict[str, Any]]] = {ARM_X21: [], ARM_X22: []}
        scenario_vetoes = 0
        for sample_index in range(201):
            row = rows[cursor]
            cursor += 1
            time_s = float(row["time_s"])
            live_tracks = set(row["live_tracks"])
            carry_keys = {
                (str(value["track_token"]), int(value["component"]))
                for value in row["carry_keys"]
            }
            direct: dict[StateKey, MotionState] = {}
            x22_refreshes: dict[StateKey, MotionState] = {}
            veto_keys: list[str] = []
            for raw in row["refreshes"]:
                key = (str(raw["track_token"]), int(raw["component"]))
                direct[key] = motion_state(raw, time_s)
                previous = x22_states.get(key)
                if should_veto(previous, raw, time_s):
                    assert previous is not None
                    x22_refreshes[key] = stable_replacement(previous, time_s)
                    veto_keys.append(f"{key[0]}::{key[1]}")
                else:
                    x22_refreshes[key] = direct[key]
            x21_states, x21_emitted, x21_diag = advance_arm(
                x21_states,
                direct,
                live_tracks=live_tracks,
                carry_keys=carry_keys,
                time_s=time_s,
            )
            x22_states, x22_emitted, x22_diag = advance_arm(
                x22_states,
                x22_refreshes,
                live_tracks=live_tracks,
                carry_keys=carry_keys,
                time_s=time_s,
            )
            x21_history.append((time_s, x21_emitted))
            x22_history.append((time_s, x22_emitted))
            x21_continued = continued_states(x21_history, time_s)
            x22_continued = continued_states(x22_history, time_s)
            scenario_vetoes += len(veto_keys)
            for arm, raw_emitted, emitted, diagnostics in (
                (ARM_X21, x21_emitted, x21_continued, x21_diag),
                (ARM_X22, x22_emitted, x22_continued, x22_diag),
            ):
                route_risk, minimum_entry_s, risky_states = frame_prediction(emitted)
                arm_frames[arm].append(
                    {
                        "sample_index": sample_index,
                        "time_s": time_s,
                        "route_risk": route_risk,
                        "minimum_entry_s": minimum_entry_s,
                        "emitted_states": len(emitted),
                        "raw_emitted_states": len(raw_emitted),
                        "risky_states": risky_states,
                        "refreshed_states": int(diagnostics["refreshed_states"]),
                        "transported_states": int(diagnostics["transported_states"]),
                        "dropped_states": int(diagnostics["dropped_states"]),
                        "replacement_vetoes": len(veto_keys) if arm == ARM_X22 else 0,
                        "veto_keys": veto_keys if arm == ARM_X22 else [],
                    }
                )
        total_vetoes += scenario_vetoes
        for arm in (ARM_X21, ARM_X22):
            risk_indices = [row["sample_index"] for row in arm_frames[arm] if row["route_risk"]]
            arms[arm]["scenarios"][scenario] = {
                "frames": arm_frames[arm],
                "route_risk_sample_indices": risk_indices,
                "route_risk_frames": len(risk_indices),
                "replacement_vetoes": scenario_vetoes if arm == ARM_X22 else 0,
            }
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED",
        "truth_blind": True,
        "experiment_id": EXPERIMENT_ID,
        "arms": arms,
        "fixed_transition_constants": protocol["x22_contract"]["fixed_transition_constants"],
        "route_contract": {
            "half_width_m": ROUTE_HALF_WIDTH_M,
            "horizon_s": ROUTE_HORIZON_S,
            "minimum_closing_speed_mps": MIN_CLOSING_SPEED_MPS,
        },
        "diagnostics": {"frames": len(rows), "x22_replacement_vetoes": total_vetoes},
        "source": {
            "freeze_sha256": sha256_file(paths["freeze"]),
            "model_manifest_sha256": frozen["source_model_manifest"]["sha256"],
            "online_manifest_sha256": frozen["online_refresh_manifest"]["sha256"],
            "online_rows_sha256": frozen["online_refresh_rows"]["sha256"],
        },
        "forbidden_fields": sorted(ONLINE_FORBIDDEN_KEYS),
    }
    assert_online_clean({key: child for key, child in value.items() if key != "forbidden_fields"})
    atomic_json(paths["predictions"], value)
    return {**value, "predictions_sha256": sha256_file(paths["predictions"])}


def _arm_scenario(predictions: Mapping[str, Any], arm: str, scenario: str) -> dict[str, Any]:
    value = predictions["arms"][arm]["scenarios"][scenario]
    require(len(value["frames"]) == 201, f"x22_prediction_frame_count:{arm}:{scenario}")
    return value


def _first_risk_time(value: Mapping[str, Any]) -> float | None:
    return next((float(row["time_s"]) for row in value["frames"] if row["route_risk"]), None)


def score(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    paths = _paths(run_root)
    require(not paths["score_attempt"].exists(), f"x22_score_attempt_already_consumed:{paths['score_attempt']}")
    require(not paths["result"].exists(), f"x22_result_exists:{paths['result']}")
    frozen, protocol, _rows = require_freeze(run_root)
    predictions_path = paths["predictions"].resolve(strict=True)
    predictions = read_json(predictions_path)
    require(
        predictions.get("schema") == PREDICTION_SCHEMA
        and predictions.get("status") == "SEALED"
        and predictions.get("truth_blind") is True,
        "x22_prediction_schema",
    )
    require(predictions["source"]["freeze_sha256"] == sha256_file(paths["freeze"]), "x22_prediction_freeze_drift")
    source_root = Path(frozen["source_root"]).resolve(strict=True)

    # This exclusive receipt is created before the first evaluator-manifest or
    # evaluator-ledger open.  Any later failure consumes the one score attempt.
    exclusive_json(
        paths["score_attempt"],
        {
            "schema": "blindassist-dtr-carla-v18-x22-score-attempt-v1",
            "attempt": 1,
            "status": "CONSUMED_BEFORE_EVALUATOR_TRUTH_OPEN",
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
            "freeze_sha256": sha256_file(paths["freeze"]),
            "evaluator_manifest_expected_path": str(source_root / carla_source.EVALUATOR_MANIFEST_RELATIVE),
        },
    )

    evaluator_manifest_path = (source_root / carla_source.EVALUATOR_MANIFEST_RELATIVE).resolve(strict=True)
    evaluator = carla_source.load_evaluator_truth(source_root)
    scenarios = list(protocol["scripted_motion"]["scenarios"])
    require(set(evaluator) == set(scenarios), "x22_evaluator_scenarios")
    clear_name = protocol["source_admission"]["clear_crossing_scenario"]
    occluded_name = protocol["source_admission"]["occluded_crossing_scenario"]
    static_name = "static_clear_mirror"
    parallel_name = "moving_parallel_clear_mirror"
    occluded_rows = evaluator[occluded_name]
    physical_gap = {
        int(row["sample_index"])
        for row in occluded_rows
        if bool(row["evaluator"]["physical_loss"])
    }
    require(len(physical_gap) == 11, f"x22_physical_gap_count:{len(physical_gap)}")
    require(
        all(bool(occluded_rows[index]["truth"]["future_contact_within_horizon"]) for index in physical_gap),
        "x22_physical_gap_not_positive",
    )
    x21_occluded = _arm_scenario(predictions, ARM_X21, occluded_name)
    x22_occluded = _arm_scenario(predictions, ARM_X22, occluded_name)
    x21_gap_hits = physical_gap & set(x21_occluded["route_risk_sample_indices"])
    x22_gap_hits = physical_gap & set(x22_occluded["route_risk_sample_indices"])
    denominator = len(physical_gap)
    x21_coverage = len(x21_gap_hits) / denominator
    x22_coverage = len(x22_gap_hits) / denominator
    improvement = x22_coverage - x21_coverage

    contact_times = [
        float(row["elapsed_seconds"])
        for row in occluded_rows
        if bool(row["truth"]["current_contact"])
    ]
    require(bool(contact_times), "x22_occluded_contact_missing")
    first_contact_time = min(contact_times)
    x22_first_occluded = _first_risk_time(x22_occluded)
    lead_s = None if x22_first_occluded is None else first_contact_time - x22_first_occluded

    x21_clear = _arm_scenario(predictions, ARM_X21, clear_name)
    x22_clear = _arm_scenario(predictions, ARM_X22, clear_name)
    x21_first_clear = _first_risk_time(x21_clear)
    x22_first_clear = _first_risk_time(x22_clear)
    clear_first_not_lost = bool(
        x21_first_clear is not None
        and x22_first_clear is not None
        and x22_first_clear <= x21_first_clear + EPSILON
    )

    new_static = sorted(
        set(_arm_scenario(predictions, ARM_X22, static_name)["route_risk_sample_indices"])
        - set(_arm_scenario(predictions, ARM_X21, static_name)["route_risk_sample_indices"])
    )
    new_parallel = sorted(
        set(_arm_scenario(predictions, ARM_X22, parallel_name)["route_risk_sample_indices"])
        - set(_arm_scenario(predictions, ARM_X21, parallel_name)["route_risk_sample_indices"])
    )
    new_prethreat: dict[str, list[int]] = {}
    for scenario in scenarios:
        x21_risk = set(_arm_scenario(predictions, ARM_X21, scenario)["route_risk_sample_indices"])
        x22_risk = set(_arm_scenario(predictions, ARM_X22, scenario)["route_risk_sample_indices"])
        negative = {
            int(row["sample_index"])
            for row in evaluator[scenario]
            if not bool(row["truth"]["future_contact_within_horizon"])
        }
        added = sorted((x22_risk - x21_risk) & negative)
        if added:
            new_prethreat[scenario] = added

    gates = protocol["x22_contract"]["one_score_gates"]
    checks = {
        "occluded_gap_at_least_9_of_11": len(x22_gap_hits) >= int(gates["minimum_occluded_gap_frames"]),
        "x22_minus_x21_gap_coverage_at_least_50pp": improvement + EPSILON >= float(gates["minimum_x22_minus_x21_gap_coverage"]),
        "first_alert_lead_at_least_2_5s": lead_s is not None and lead_s + EPSILON >= float(gates["minimum_first_alert_lead_seconds"]),
        "clear_first_risk_not_lost": clear_first_not_lost,
        "static_zero_new_risk": len(new_static) <= int(gates["maximum_new_static_route_risk_frames"]),
        "parallel_zero_new_risk": len(new_parallel) <= int(gates["maximum_new_parallel_route_risk_frames"]),
        "prethreat_zero_new_risk": sum(map(len, new_prethreat.values())) <= int(gates["maximum_new_prethreat_route_risk_frames"]),
        "prediction_truth_boundary": predictions.get("forbidden_fields") == sorted(ONLINE_FORBIDDEN_KEYS),
        "one_score_attempt_consumed": paths["score_attempt"].is_file(),
    }
    passed = all(checks.values())
    result = {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "CARLA_DTR_X22_FLOW_VETO_SYNTHETIC_DEVELOPMENT_GATE_MET"
            if passed
            else "CARLA_DTR_X22_FLOW_VETO_SYNTHETIC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate": {"passed": passed, "checks": checks},
        "metrics": {
            "physical_gap_sample_indices": sorted(physical_gap),
            "physical_gap_frames": denominator,
            ARM_X21: {"gap_hits": sorted(x21_gap_hits), "gap_frames": len(x21_gap_hits), "gap_coverage": x21_coverage},
            ARM_X22: {"gap_hits": sorted(x22_gap_hits), "gap_frames": len(x22_gap_hits), "gap_coverage": x22_coverage},
            "x22_minus_x21_gap_coverage": improvement,
            "x22_minus_x21_gap_coverage_percentage_points": improvement * 100.0,
            "first_contact_time_s": first_contact_time,
            "x22_first_occluded_risk_time_s": x22_first_occluded,
            "x22_first_alert_lead_s": lead_s,
            "x21_first_clear_risk_time_s": x21_first_clear,
            "x22_first_clear_risk_time_s": x22_first_clear,
            "new_static_risk_sample_indices": new_static,
            "new_parallel_risk_sample_indices": new_parallel,
            "new_prethreat_risk_sample_indices": new_prethreat,
        },
        "sources": {
            "freeze": {"path": str(paths["freeze"]), "sha256": sha256_file(paths["freeze"])},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
            "score_attempt": {"path": str(paths["score_attempt"]), "sha256": sha256_file(paths["score_attempt"])},
            "evaluator_manifest": {"path": str(evaluator_manifest_path), "sha256": sha256_file(evaluator_manifest_path)},
        },
        "decision": {
            "next": "PROMOTE_X22_TO_NEW_SOURCE_DEVELOPMENT" if passed else "CLOSE_FROZEN_X22_ARM_WITHOUT_SWEEP"
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(paths["result"], result)
    return {**result, "result_sha256": sha256_file(paths["result"])}


def status(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.run_root.resolve())
    return {
        name: None if not path.is_file() else {"path": str(path), "sha256": sha256_file(path), "value": read_json(path)}
        for name, path in paths.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--run-root", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--online-manifest", type=Path, required=True)
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    for command in ("predict", "score", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        value = freeze(args)
    elif args.command == "predict":
        value = predict(args)
    elif args.command == "score":
        value = score(args)
    else:
        value = status(args)
    print(json.dumps(value, indent=2, sort_keys=True))
    if args.command == "score":
        return 0 if value["gate"]["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

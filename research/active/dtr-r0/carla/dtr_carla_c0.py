"""Join, predict, and score the DTR-CARLA-C0 observation ladder.

The three modes intentionally separate authority:

* ``join`` validates four deterministic CARLA replays and physically splits
  observable inputs, privileged current state, teacher diagnostics, and future
  truth;
* ``predict`` reads no truth file, runs O0/O1/O2T/O3 through frozen DTR R2, and
  seals predictions with SHA-256;
* ``score`` opens realized truth only after that prediction seal exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DTR_ROOT = HERE.parent
if str(DTR_ROOT) not in sys.path:
    sys.path.insert(0, str(DTR_ROOT))

import real_observation_adapter as real
from dtr_r0 import (
    CausalFrame,
    EgoPose,
    Observation,
    Prediction,
    Signal,
    compute_event_metrics,
)
from dtr_r1 import FROZEN_R1_CONFIG
from dtr_r2 import FROZEN_R2_CONFIG, run_r2_arm

EXPERIMENT_ID = "DTR_CARLA_C0_CAUSAL_BENCHMARK_CANARY_V1"
OBSERVATION_SCHEMA = "blindassist-dtr-carla-c0-observation-index-v1"
ORACLE_SCHEMA = "blindassist-dtr-carla-c0-oracle-current-index-v1"
TEACHER_SCHEMA = "blindassist-dtr-carla-c0-teacher-index-v1"
TRUTH_SCHEMA = "blindassist-dtr-carla-c0-realized-truth-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-c0-predictions-v1"
RESULT_SCHEMA = "blindassist-dtr-carla-c0-result-v1"
ARMS = (
    "O0_RGB",
    "O1_RGB_DEPTH",
    "O2T_RGB_DEPTH_CARLA_FLOW",
    "O3_PRIVILEGED_CURRENT_STATE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("join", "predict", "score"))
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "dtr_carla_c0_protocol.json"
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts.local/models/yolo11n.pt"),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def close(left: float, right: float, tolerance: float, code: str) -> None:
    require(abs(float(left) - float(right)) <= tolerance, code)


def scenario_map(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {str(item["id"]): item for item in protocol["scenarios"]}
    require(len(values) == 12, "protocol_requires_12_unique_episodes")
    return values


def compare_replay_rows(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], label: str
) -> None:
    require(reference["sample_index"] == candidate["sample_index"], f"{label}:sample")
    close(reference["time_s"], candidate["time_s"], 1e-9, f"{label}:time")
    for actor in ("ego", "target"):
        for axis in ("x", "y", "z"):
            close(
                reference[actor]["transform"][axis],
                candidate[actor]["transform"][axis],
                0.001,
                f"{label}:{actor}_position_{axis}",
            )
        for axis in ("x", "y"):
            close(
                reference[actor]["command_velocity"][axis],
                candidate[actor]["command_velocity"][axis],
                0.001,
                f"{label}:{actor}_velocity_{axis}",
            )
    close(
        reference["ego"]["body_yaw_degrees"],
        candidate["ego"]["body_yaw_degrees"],
        0.01,
        f"{label}:body_yaw",
    )
    close(
        reference["ego"]["sensor_yaw_degrees"],
        candidate["ego"]["sensor_yaw_degrees"],
        0.01,
        f"{label}:sensor_yaw",
    )
    for axis in ("x", "y", "z"):
        close(
            reference["camera_transform"][axis],
            candidate["camera_transform"][axis],
            0.001,
            f"{label}:camera_position_{axis}",
        )
    for axis in ("pitch", "yaw", "roll"):
        close(
            reference["camera_transform"][axis],
            candidate["camera_transform"][axis],
            0.01,
            f"{label}:camera_rotation_{axis}",
        )
    require(
        bool(reference["truth"]["current_contact"])
        == bool(candidate["truth"]["current_contact"]),
        f"{label}:contact",
    )


def event_truth(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contacts = [index for index, row in enumerate(rows) if row["truth"]["current_contact"]]
    if not contacts:
        return {
            "critical_event": False,
            "warning_start_s": None,
            "event_start_s": None,
            "event_end_s": None,
            "exit_time_s": None,
        }
    first = contacts[0]
    last = contacts[-1]
    require(
        contacts == list(range(first, last + 1)),
        "critical_truth_must_have_one_contiguous_contact_event",
    )
    require(
        first > 0 and last < len(rows) - 1,
        "critical_truth_event_must_be_bounded_by_known_non_contact",
    )
    warning = next(
        float(row["time_s"])
        for row in rows[: first + 1]
        if row["truth"]["future_contact_within_horizon"]
    )
    exit_time = next(
        (
            float(row["time_s"])
            for row in rows[last + 1 :]
            if not row["truth"]["current_contact"]
        ),
        None,
    )
    return {
        "critical_event": True,
        "warning_start_s": warning,
        "event_start_s": float(rows[first]["time_s"]),
        "event_end_s": float(rows[last]["time_s"]),
        "exit_time_s": exit_time,
    }


def validate_pair_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = scenario_map(protocol)
    reports: list[dict[str, Any]] = []
    for contract in protocol["twin_contracts"]:
        left = scenarios[str(contract["a"])]
        right = scenarios[str(contract["b"])]
        family = str(contract["family"])
        require(left["family"] == family and right["family"] == family, f"{family}:family")
        require(left["ego"]["start_forward_m"] == right["ego"]["start_forward_m"], f"{family}:ego_start_forward")
        require(left["ego"]["start_right_m"] == right["ego"]["start_right_m"], f"{family}:ego_start_right")
        require(left["target"]["start_forward_m"] == right["target"]["start_forward_m"], f"{family}:target_start_forward")
        require(left["target"]["start_right_m"] == right["target"]["start_right_m"], f"{family}:target_start_right")
        if family != "same_motion_different_route":
            require(left["ego"]["segments"] == right["ego"]["segments"], f"{family}:ego_trajectory")
        if family not in {
            "same_geometry_different_motion",
            "same_motion_different_ttc",
            "same_background_static_then_dynamic",
        }:
            require(left["target"]["segments"] == right["target"]["segments"], f"{family}:target_trajectory")
        if family != "same_scene_different_visibility":
            require(left.get("camera_yaw_offsets") == right.get("camera_yaw_offsets"), f"{family}:camera_schedule")
        if family != "same_target_visible_then_occluded":
            require(left.get("occluder") == right.get("occluder"), f"{family}:occluder")
        reports.append(
            {
                "family": family,
                "a": left["id"],
                "b": right["id"],
                "allowed_difference": contract["allowed_difference"],
                "protocol_isolation_pass": True,
            }
        )
    require(len(reports) == 6, "protocol_requires_6_pairs")
    return {"checks": reports, "all_pass": True}


def join(args: argparse.Namespace) -> dict[str, Any]:
    require(args.raw_root is not None, "join_requires_raw_root")
    protocol_path = args.protocol.resolve(strict=True)
    protocol = read_json(protocol_path)
    require(protocol.get("experiment_id") == EXPERIMENT_ID, "protocol_identity")
    protocol_pairs = validate_pair_protocol(protocol)
    raw_root = args.raw_root.resolve(strict=True)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = {
        "observation": root / "observation-index.json",
        "oracle": root / "oracle-current-index.json",
        "teacher": root / "teacher-index.json",
        "truth": root / "truth.json",
        "join": root / "join-result.json",
    }
    require(not any(path.exists() for path in targets.values()), "join_refuses_overwrite")
    sensors = tuple(protocol["capture"]["sensor_order"])
    require(sensors == ("instance", "rgb", "depth", "flow"), "sensor_order")
    scenario_values = scenario_map(protocol)
    sensor_results: dict[str, Any] = {}
    for sensor in sensors:
        result_path = raw_root / sensor / "result.json"
        result = read_json(result_path.resolve(strict=True))
        require(result["status"] == "DTR_CARLA_C0_MODALITY_CAPTURE_COMPLETE", f"{sensor}:capture")
        require(result["protocol_sha256"] == sha256_file(protocol_path), f"{sensor}:protocol_hash")
        sensor_results[sensor] = result

    observations: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    teachers: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    episode_checks: list[dict[str, Any]] = []
    baseline_rows: dict[str, list[dict[str, Any]]] = {}
    for scenario_id, scenario in scenario_values.items():
        rows_by_sensor: dict[str, list[dict[str, Any]]] = {}
        paths_by_sensor: dict[str, list[str]] = {}
        for sensor in sensors:
            episode_dir = (raw_root / sensor / scenario_id).resolve(strict=True)
            manifest = read_json((episode_dir / "manifest.json").resolve(strict=True))
            frames_path = Path(manifest["frames_path"]).resolve(strict=True)
            require(sha256_file(frames_path) == manifest["frames_sha256"], f"{scenario_id}:{sensor}:frames_hash")
            rows = read_jsonl(frames_path)
            require(len(rows) == 161, f"{scenario_id}:{sensor}:frame_count")
            require(int(manifest["payload_count"]) == len(rows), f"{scenario_id}:{sensor}:payload_count")
            payloads: list[str] = []
            for row in rows:
                payload = (episode_dir / row["sensor_path"]).resolve(strict=True)
                require(payload.is_relative_to(episode_dir), f"{scenario_id}:{sensor}:payload_escape")
                require(payload.stat().st_size > 0, f"{scenario_id}:{sensor}:empty_payload")
                payloads.append(str(payload))
            rows_by_sensor[sensor] = rows
            paths_by_sensor[sensor] = payloads
        reference = rows_by_sensor["instance"]
        baseline_rows[scenario_id] = reference
        for sensor in ("rgb", "depth", "flow"):
            for index, (left, right) in enumerate(zip(reference, rows_by_sensor[sensor])):
                compare_replay_rows(left, right, f"{scenario_id}:{sensor}:{index}")

        observable_frames: list[dict[str, Any]] = []
        oracle_frames: list[dict[str, Any]] = []
        teacher_frames: list[dict[str, Any]] = []
        for index, row in enumerate(reference):
            ego = row["ego"]
            target = row["target"]
            observable_frames.append(
                {
                    "sample_index": index,
                    "time_s": float(row["time_s"]),
                    "rgb_path": paths_by_sensor["rgb"][index],
                    "depth_path": paths_by_sensor["depth"][index],
                    "camera_transform": rows_by_sensor["rgb"][index]["camera_transform"],
                    "ego_pose": {
                        "x_m": float(ego["transform"]["x"]),
                        "y_m": -float(ego["transform"]["y"]),
                        "body_yaw_rad": -math.radians(float(ego["body_yaw_degrees"])),
                        "sensor_yaw_rad": -math.radians(float(ego["sensor_yaw_degrees"])),
                        "coordinate_mapping": "DTR_X_EQ_CARLA_X_DTR_Y_EQ_NEG_CARLA_Y",
                    },
                }
            )
            oracle_frames.append(
                {
                    "sample_index": index,
                    "time_s": float(row["time_s"]),
                    "track_id": "oracle-target",
                    "target_x_m": float(target["transform"]["x"]),
                    "target_y_m": -float(target["transform"]["y"]),
                    "radius_m": float(protocol["route_contract"]["target_observation_radius_m"]),
                }
            )
            teacher_frames.append(
                {
                    "sample_index": index,
                    "time_s": float(row["time_s"]),
                    "target_instance": row["observation"]["target_instance"],
                }
            )
        flow_path = (raw_root / "flow" / scenario_id / "flow_raw_float32.npz").resolve(strict=True)
        flow_manifest = read_json(
            (raw_root / "flow" / scenario_id / "manifest.json").resolve(strict=True)
        )
        require(
            Path(flow_manifest["flow_path"]).resolve(strict=True) == flow_path,
            f"{scenario_id}:flow_path",
        )
        require(
            sha256_file(flow_path) == flow_manifest["flow_sha256"],
            f"{scenario_id}:flow_hash",
        )
        observations.append(
            {
                "episode_id": scenario_id,
                "family": scenario["family"],
                "twin": scenario["twin"],
                "frames": observable_frames,
            }
        )
        oracles.append(
            {
                "episode_id": scenario_id,
                "role": "PRIVILEGED_CURRENT_STATE_ONLY",
                "uses_realized_future": False,
                "frames": oracle_frames,
            }
        )
        teachers.append(
            {
                "episode_id": scenario_id,
                "role": "TEACHER_ONLY_CARLA_FLOW_AND_EVALUATOR_ONLY_INSTANCE_VISIBILITY",
                "flow_teacher_path": str(flow_path),
                "frames": teacher_frames,
            }
        )
        truth_value = event_truth(reference)
        require(
            bool(truth_value["critical_event"]) == bool(scenario["expected_critical"]),
            f"{scenario_id}:expected_truth",
        )
        truths.append(
            {
                "episode_id": scenario_id,
                "family": scenario["family"],
                "twin": scenario["twin"],
                "truth": truth_value,
                "frame_times_s": [float(row["time_s"]) for row in reference],
                "contact_by_frame": [bool(row["truth"]["current_contact"]) for row in reference],
            }
        )
        episode_checks.append(
            {
                "episode_id": scenario_id,
                "cross_modality_replay_equivalent": True,
                "frames": len(reference),
                "critical_event": truth_value["critical_event"],
            }
        )

    truth_by_id = {item["episode_id"]: item for item in truths}
    pair_checks: list[dict[str, Any]] = []
    for contract in protocol["twin_contracts"]:
        left_id, right_id = str(contract["a"]), str(contract["b"])
        left_rows, right_rows = baseline_rows[left_id], baseline_rows[right_id]
        boundary = float(contract["identical_before_s"])
        pre_equal = True
        for left, right in zip(left_rows, right_rows):
            if float(left["time_s"]) + 1e-9 >= boundary:
                break
            try:
                compare_replay_rows(left, right, f"pair:{contract['family']}")
            except ValueError:
                pre_equal = False
                break
        family = str(contract["family"])
        left_contact = truth_by_id[left_id]["contact_by_frame"]
        right_contact = truth_by_id[right_id]["contact_by_frame"]
        invariant_timeline = (
            left_contact == right_contact
            if family in {
                "same_scene_different_visibility",
                "same_target_visible_then_occluded",
            }
            else None
        )
        require(pre_equal, f"{family}:pre_intervention_state")
        require(invariant_timeline is not False, f"{family}:contact_timeline")
        pair_checks.append(
            {
                "family": family,
                "a": left_id,
                "b": right_id,
                "pre_intervention_state_equal": pre_equal,
                "contact_timeline_invariant_when_required": invariant_timeline,
            }
        )

    common = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "raw_root": str(raw_root),
    }
    observation_value = {"schema_version": OBSERVATION_SCHEMA, **common, "episodes": observations}
    oracle_value = {"schema_version": ORACLE_SCHEMA, **common, "episodes": oracles}
    teacher_value = {"schema_version": TEACHER_SCHEMA, **common, "episodes": teachers}
    truth_value = {"schema_version": TRUTH_SCHEMA, **common, "episodes": truths}
    write_json_atomic(targets["observation"], observation_value)
    write_json_atomic(targets["oracle"], oracle_value)
    write_json_atomic(targets["teacher"], teacher_value)
    write_json_atomic(targets["truth"], truth_value)
    report = {
        "schema_version": "blindassist-dtr-carla-c0-join-v1",
        "status": "DTR_CARLA_C0_SOURCE_ADMITTED",
        **common,
        "sensor_results": {
            sensor: {
                "status": value["status"],
                "capture_script_sha256": value["capture_script_sha256"],
            }
            for sensor, value in sensor_results.items()
        },
        "protocol_pair_isolation": protocol_pairs,
        "episode_checks": episode_checks,
        "pair_checks": pair_checks,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in targets.items()
            if name != "join"
        },
    }
    write_json_atomic(targets["join"], report)
    return report


def clamp_bbox(bbox: real.BBox, width: int, height: int) -> real.BBox | None:
    x1 = max(0.0, min(float(width - 2), bbox.x1))
    y1 = max(0.0, min(float(height - 2), bbox.y1))
    x2 = max(x1 + 1.0, min(float(width - 1), bbox.x2))
    y2 = max(y1 + 1.0, min(float(height - 1), bbox.y2))
    try:
        return real.BBox(x1, y1, x2, y2)
    except ValueError:
        return None


def decode_depth_m(image: np.ndarray, u: int, v: int) -> float | None:
    height, width = image.shape[:2]
    u = max(0, min(width - 1, int(u)))
    v = max(0, min(height - 1, int(v)))
    x1, x2 = max(0, u - 2), min(width, u + 3)
    y1, y2 = max(0, v - 2), min(height, v + 3)
    values = image[y1:y2, x1:x2]
    if values.ndim != 3 or values.shape[2] < 3:
        return None
    normalized = (
        values[:, :, 2].astype(np.float64)
        + values[:, :, 1].astype(np.float64) * 256.0
        + values[:, :, 0].astype(np.float64) * 65536.0
    ) / 16777215.0
    depth = float(np.median(normalized) * 1000.0)
    return depth if math.isfinite(depth) and 0.05 < depth < 1000.0 else None


def depth_project(
    bbox: real.BBox,
    depth_image: np.ndarray,
    camera: Mapping[str, Any],
    ego_pose: EgoPose,
    *,
    fov_degrees: float,
    radius_m: float,
) -> Observation | None:
    height, width = depth_image.shape[:2]
    u = round((bbox.x1 + bbox.x2) / 2.0)
    v = round(bbox.y1 + 0.80 * (bbox.y2 - bbox.y1))
    depth = decode_depth_m(depth_image, u, v)
    if depth is None:
        return None
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    local_x = depth
    local_y = (u - (width - 1) / 2.0) * depth / focal
    local_z = -((v - (height - 1) / 2.0) * depth / focal)
    pitch = math.radians(float(camera["pitch"]))
    yaw = math.radians(float(camera["yaw"]))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    world_dx = cp * cy * local_x - sy * local_y - sp * cy * local_z
    world_dy = cp * sy * local_x + cy * local_y - sp * sy * local_z
    # CARLA is left-handed (x forward, y right); DTR is right-handed with
    # positive local-left. Reflect CARLA world y before applying DTR's inverse
    # sensor yaw. This is the same mapping used by the joined ego/oracle state.
    world_x = float(camera["x"]) + world_dx
    world_y = -(float(camera["y"]) + world_dy)
    dx, dy = world_x - ego_pose.x_m, world_y - ego_pose.y_m
    cosine, sine = math.cos(ego_pose.sensor_yaw_rad), math.sin(ego_pose.sensor_yaw_rad)
    forward = dx * cosine + dy * sine
    left = -dx * sine + dy * cosine
    if not all(math.isfinite(value) for value in (forward, left)) or forward <= 0.0:
        return None
    return Observation("", forward, left, radius_m)


def shift_from_flow(
    bbox: real.BBox, flow: np.ndarray, width: int, height: int
) -> real.BBox | None:
    x1 = max(0, min(width - 1, math.floor(bbox.x1)))
    y1 = max(0, min(height - 1, math.floor(bbox.y1)))
    x2 = max(x1 + 1, min(width, math.ceil(bbox.x2)))
    y2 = max(y1 + 1, min(height, math.ceil(bbox.y2)))
    region = flow[y1:y2, x1:x2]
    if region.size == 0:
        return None
    delta = np.median(region.reshape(-1, 2), axis=0)
    du, dv = float(delta[0]) * width, float(delta[1]) * height
    if not math.isfinite(du) or not math.isfinite(dv) or math.hypot(du, dv) > 80.0:
        return None
    return clamp_bbox(
        real.BBox(bbox.x1 + du, bbox.y1 + dv, bbox.x2 + du, bbox.y2 + dv),
        width,
        height,
    )


def prediction_dict(value: Prediction) -> dict[str, Any]:
    return value.to_dict()


def predict(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    observation_path = (root / "observation-index.json").resolve(strict=True)
    oracle_path = (root / "oracle-current-index.json").resolve(strict=True)
    teacher_path = (root / "teacher-index.json").resolve(strict=True)
    observation_index = read_json(observation_path)
    oracle_index = read_json(oracle_path)
    teacher_index = read_json(teacher_path)
    require(observation_index["schema_version"] == OBSERVATION_SCHEMA, "observation_schema")
    require(oracle_index["schema_version"] == ORACLE_SCHEMA, "oracle_schema")
    require(teacher_index["schema_version"] == TEACHER_SCHEMA, "teacher_schema")
    require(
        observation_index["protocol_sha256"]
        == oracle_index["protocol_sha256"]
        == teacher_index["protocol_sha256"],
        "index_protocol",
    )
    prediction_path = root / "predictions.json"
    receipt_path = root / "prediction-receipt.json"
    require(not prediction_path.exists() and not receipt_path.exists(), "predict_refuses_overwrite")
    model_path = args.model.resolve(strict=True)
    detector = real.UltralyticsPersonDetector(
        model_path, confidence_threshold=float(args.confidence_threshold)
    )
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("DTR-CARLA-C0 prediction requires cv2") from error
    protocol = read_json(args.protocol.resolve(strict=True))
    width, height = (int(value) for value in protocol["camera"]["resolution"])
    fov = float(protocol["camera"]["fov_degrees"])
    camera_height = float(protocol["environment"]["walker_origin_z_m"]) + float(
        protocol["camera"]["relative_transform"]["z_m"]
    )
    calibration = real.CameraCalibration(
        image_width_px=width,
        image_height_px=height,
        fx_px=width / (2.0 * math.tan(math.radians(fov) / 2.0)),
        fy_px=width / (2.0 * math.tan(math.radians(fov) / 2.0)),
        cx_px=(width - 1) / 2.0,
        cy_px=(height - 1) / 2.0,
        camera_height_m=camera_height,
        pitch_down_rad=math.radians(
            -float(protocol["camera"]["relative_transform"]["pitch_degrees"])
        ),
        person_radius_m=float(protocol["route_contract"]["target_observation_radius_m"]),
    )
    oracle_by_id = {item["episode_id"]: item for item in oracle_index["episodes"]}
    teacher_by_id = {item["episode_id"]: item for item in teacher_index["episodes"]}
    all_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    health: list[dict[str, Any]] = []
    for episode in observation_index["episodes"]:
        episode_id = str(episode["episode_id"])
        oracle_episode = oracle_by_id[episode_id]
        teacher_episode = teacher_by_id[episode_id]
        require(len(episode["frames"]) == len(oracle_episode["frames"]), f"{episode_id}:oracle_count")
        with np.load(
            Path(teacher_episode["flow_teacher_path"]).resolve(strict=True)
        ) as flow_payload:
            flow_xy = np.asarray(flow_payload["flow_xy"], dtype=np.float32)
        require(flow_xy.shape == (len(episode["frames"]), height, width, 2), f"{episode_id}:flow_shape")
        tracker = real.CausalPersonTracker(
            minimum_iou=0.10,
            maximum_footpoint_distance_px=80.0,
            maximum_track_age_s=0.75,
        )
        flow_states: dict[str, dict[str, Any]] = {}
        flow_alias: dict[str, str] = {}
        frames_by_arm: dict[str, list[CausalFrame]] = {arm: [] for arm in ARMS}
        detection_frames = metric0_frames = metric1_frames = flow_carried_frames = 0
        for index, frame in enumerate(episode["frames"]):
            image = cv2.imread(str(Path(frame["rgb_path"])), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(Path(frame["depth_path"])), cv2.IMREAD_UNCHANGED)
            require(image is not None and depth is not None, f"{episode_id}:{index}:decode")
            detections = detector.detect(image)
            tracked = tracker.update(detections, time_s=float(frame["time_s"]))
            if detections:
                detection_frames += 1
            ego_value = frame["ego_pose"]
            ego_pose = EgoPose(
                x_m=float(ego_value["x_m"]),
                y_m=float(ego_value["y_m"]),
                body_yaw_rad=float(ego_value["body_yaw_rad"]),
                sensor_yaw_rad=float(ego_value["sensor_yaw_rad"]),
            )
            o0: list[Observation] = []
            o1: list[Observation] = []
            o2: list[Observation] = []
            seen_flow_ids: set[str] = set()
            current_raw_ids = {value.track_id for value in tracked}
            for tracked_detection in tracked:
                raw_id = tracked_detection.track_id
                bbox = tracked_detection.detection.bbox
                ground = real.project_bbox_bottom_center(bbox, calibration)
                if ground is not None:
                    o0.append(
                        Observation(
                            raw_id,
                            ground.forward_m,
                            ground.left_m,
                            ground.radius_m,
                        )
                    )
                depth_value = depth_project(
                    bbox,
                    depth,
                    frame["camera_transform"],
                    ego_pose,
                    fov_degrees=fov,
                    radius_m=calibration.person_radius_m,
                )
                if depth_value is not None:
                    o1.append(
                        Observation(
                            raw_id,
                            depth_value.forward_m,
                            depth_value.left_m,
                            depth_value.radius_m,
                        )
                    )
                persistent = flow_alias.get(raw_id, raw_id)
                if persistent != raw_id and persistent in current_raw_ids:
                    flow_alias.pop(raw_id, None)
                    persistent = raw_id
                if persistent in seen_flow_ids:
                    persistent = raw_id
                if tracked_detection.is_new_track:
                    candidates = [
                        (real.footpoint_distance(bbox, state["bbox"]), state_id)
                        for state_id, state in flow_states.items()
                        if float(frame["time_s"]) - float(state["last_detection_time_s"]) <= 0.50 + 1e-9
                        and state_id not in seen_flow_ids
                        and state_id not in current_raw_ids
                    ]
                    if candidates and min(candidates)[0] <= 80.0:
                        persistent = min(candidates)[1]
                        flow_alias[raw_id] = persistent
                if depth_value is not None:
                    o2.append(
                        Observation(
                            persistent,
                            depth_value.forward_m,
                            depth_value.left_m,
                            depth_value.radius_m,
                        )
                    )
                    flow_states[persistent] = {
                        "bbox": bbox,
                        "last_detection_time_s": float(frame["time_s"]),
                    }
                    seen_flow_ids.add(persistent)
            if index > 0:
                for state_id, state in list(flow_states.items()):
                    if state_id in seen_flow_ids:
                        continue
                    age = float(frame["time_s"]) - float(state["last_detection_time_s"])
                    if age > 0.50 + 1e-9:
                        flow_states.pop(state_id, None)
                        continue
                    shifted = shift_from_flow(state["bbox"], flow_xy[index - 1], width, height)
                    if shifted is None:
                        continue
                    projected = depth_project(
                        shifted,
                        depth,
                        frame["camera_transform"],
                        ego_pose,
                        fov_degrees=fov,
                        radius_m=calibration.person_radius_m,
                    )
                    if projected is None:
                        continue
                    o2.append(
                        Observation(
                            state_id,
                            projected.forward_m,
                            projected.left_m,
                            projected.radius_m,
                        )
                    )
                    state["bbox"] = shifted
                    flow_carried_frames += 1
            if o0:
                metric0_frames += 1
            if o1:
                metric1_frames += 1
            common = {
                "time_s": float(frame["time_s"]),
                "ego_pose": ego_pose,
            }
            frames_by_arm["O0_RGB"].append(
                CausalFrame(
                    observations=tuple(o0),
                    person_detection_count=len(detections),
                    **common,
                )
            )
            frames_by_arm["O1_RGB_DEPTH"].append(
                CausalFrame(
                    observations=tuple(o1),
                    person_detection_count=len(detections),
                    **common,
                )
            )
            # A flow-carried metric observation is not a fresh detector hit.
            # Keep detector availability unspecified so R2 validates the
            # teacher observation without misreporting it as an RGB detection.
            frames_by_arm["O2T_RGB_DEPTH_CARLA_FLOW"].append(
                CausalFrame(
                    observations=tuple(o2),
                    person_detection_count=None,
                    **common,
                )
            )
            oracle = oracle_episode["frames"][index]
            dx = float(oracle["target_x_m"]) - ego_pose.x_m
            dy = float(oracle["target_y_m"]) - ego_pose.y_m
            cosine, sine = math.cos(ego_pose.sensor_yaw_rad), math.sin(ego_pose.sensor_yaw_rad)
            oracle_observation = Observation(
                str(oracle["track_id"]),
                dx * cosine + dy * sine,
                -dx * sine + dy * cosine,
                float(oracle["radius_m"]),
            )
            frames_by_arm["O3_PRIVILEGED_CURRENT_STATE"].append(
                CausalFrame(
                    time_s=float(frame["time_s"]),
                    ego_pose=ego_pose,
                    observations=(oracle_observation,),
                    person_detection_count=1,
                )
            )
        for arm in ARMS:
            for frame_index, causal_frame in enumerate(frames_by_arm[arm]):
                track_ids = [value.track_id for value in causal_frame.observations]
                require(
                    all(track_ids),
                    f"{episode_id}:{arm}:{frame_index}:empty_track_id",
                )
                require(
                    len(track_ids) == len(set(track_ids)),
                    f"{episode_id}:{arm}:{frame_index}:duplicate_track_id",
                )
            all_predictions[arm][episode_id] = [
                prediction_dict(value) for value in run_r2_arm(frames_by_arm[arm])
            ]
        health.append(
            {
                "episode_id": episode_id,
                "frames": len(episode["frames"]),
                "rgb_detection_frames": detection_frames,
                "o0_metric_frames": metric0_frames,
                "o1_metric_frames": metric1_frames,
                "o2t_flow_carried_observations": flow_carried_frames,
            }
        )
    value = {
        "schema_version": PREDICTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": observation_index["protocol_sha256"],
        "observation_index_sha256": sha256_file(observation_path),
        "oracle_current_index_sha256": sha256_file(oracle_path),
        "teacher_index_sha256": sha256_file(teacher_path),
        "truth_accessed": False,
        "uses_future_frames": False,
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "frozen_r1_config": FROZEN_R1_CONFIG.to_dict(),
        "frozen_r2_config": FROZEN_R2_CONFIG.to_dict(),
        "arm_roles": {
            "O0_RGB": "OBSERVABLE_SYNTHETIC_RGB",
            "O1_RGB_DEPTH": "OBSERVABLE_SYNTHETIC_RGBD",
            "O2T_RGB_DEPTH_CARLA_FLOW": "CARLA_FLOW_TEACHER_NOT_DEPLOYMENT",
            "O3_PRIVILEGED_CURRENT_STATE": "PRIVILEGED_CURRENT_STATE_ORACLE",
        },
        "input_health": health,
        "predictions": all_predictions,
    }
    write_json_atomic(prediction_path, value)
    receipt = {
        "schema_version": "blindassist-dtr-carla-c0-prediction-receipt-v1",
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "truth_accessed": False,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def prediction_from_dict(value: Mapping[str, Any]) -> Prediction:
    return Prediction(
        time_s=float(value["time_s"]),
        signal=Signal(str(value["signal"])),
        raw_alert=(None if value["raw_alert"] is None else bool(value["raw_alert"])),
        reason=str(value["reason"]),
        track_id=(str(value["track_id"]) if value.get("track_id") is not None else None),
        diagnostic=dict(value.get("diagnostic", {})),
    )


def with_f1(metrics: dict[str, Any]) -> dict[str, Any]:
    true_positive = int(metrics["critical_event_recalled_count"])
    false_positive = int(metrics["irrelevant_alert_segments"])
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = metrics["critical_event_recall"]
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return {**metrics, "event_precision": precision, "event_f1": f1}


def score(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(strict=True)
    prediction_path = (root / "predictions.json").resolve(strict=True)
    receipt = read_json((root / "prediction-receipt.json").resolve(strict=True))
    require(receipt["prediction_sha256"] == sha256_file(prediction_path), "prediction_seal")
    predictions_value = read_json(prediction_path)
    require(predictions_value["schema_version"] == PREDICTION_SCHEMA, "prediction_schema")
    require(predictions_value["truth_accessed"] is False, "prediction_truth_boundary")
    truth_path = (root / "truth.json").resolve(strict=True)
    truth_value = read_json(truth_path)
    require(truth_value["schema_version"] == TRUTH_SCHEMA, "truth_schema")
    result_path = root / "result.json"
    manifest_path = root / "result-manifest.json"
    require(not result_path.exists() and not manifest_path.exists(), "score_refuses_overwrite")
    episode_values = [
        {
            "episode_id": item["episode_id"],
            "frames": [{"time_s": value} for value in item["frame_times_s"]],
            "truth": item["truth"],
            "family": item["family"],
            "twin": item["twin"],
        }
        for item in truth_value["episodes"]
    ]
    predictions: dict[str, dict[str, list[Prediction]]] = {
        arm: {
            episode_id: [prediction_from_dict(value) for value in rows]
            for episode_id, rows in predictions_value["predictions"][arm].items()
        }
        for arm in ARMS
    }
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        aggregate = compute_event_metrics(episode_values, predictions[arm])
        by_family = {}
        for family in sorted({item["family"] for item in episode_values}):
            subset = [item for item in episode_values if item["family"] == family]
            by_family[family] = with_f1(
                compute_event_metrics(
                    subset,
                    {item["episode_id"]: predictions[arm][item["episode_id"]] for item in subset},
                )
            )
        metrics[arm] = {"aggregate": with_f1(aggregate), "by_family": by_family}

    protocol = read_json(args.protocol.resolve(strict=True))
    truth_by_id = {item["episode_id"]: item for item in episode_values}
    pair_metrics: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for contract in protocol["twin_contracts"]:
            relation: list[bool] = []
            for episode_id in (str(contract["a"]), str(contract["b"])):
                truth = truth_by_id[episode_id]["truth"]
                rows = predictions[arm][episode_id]
                if truth["critical_event"]:
                    detected = any(
                        float(truth["warning_start_s"]) <= value.time_s <= float(truth["event_end_s"])
                        and value.signal in (Signal.ONSET, Signal.HOLD, Signal.ESCALATE)
                        for value in rows
                    )
                    relation.append(detected)
                else:
                    relation.append(not any(value.signal is Signal.ONSET for value in rows))
            pair_metrics[arm].append(
                {
                    "family": contract["family"],
                    "a": contract["a"],
                    "b": contract["b"],
                    "relation_correct": all(relation),
                }
            )
        metrics[arm]["pair_relation_correct"] = sum(
            bool(value["relation_correct"]) for value in pair_metrics[arm]
        )
        metrics[arm]["pair_relation_total"] = len(pair_metrics[arm])

    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "terminal_status": "DTR_CARLA_C0_PARTIAL_OBSERVATION_LADDER_REPORTED",
        "scientific_advancement_gate_evaluated": False,
        "prediction_sha256_opened_before_truth": receipt["prediction_sha256"],
        "truth_sha256": sha256_file(truth_path),
        "source": {
            "episodes": len(episode_values),
            "pairs": len(protocol["twin_contracts"]),
            "families": len({item["family"] for item in episode_values}),
            "sample_hz": 1.0 / float(protocol["environment"]["sample_seconds"]),
            "modalities": ["rgb", "depth", "carla_flow_teacher", "instance_evaluator"],
        },
        "arm_roles": predictions_value["arm_roles"],
        "metrics": metrics,
        "pair_metrics": pair_metrics,
        "input_health": predictions_value["input_health"],
        "claim_boundary": protocol["claim_boundary"],
        "interpretation_boundary": [
            "O0 and O1 are synthetic observable-input arms; O2T uses CARLA sensor flow and is teacher-only.",
            "O3 uses current privileged target state but never realized future contact during prediction.",
            "This twelve-episode canary proves the execution and comparison chain, not cross-scene or synthetic-to-real generalization.",
            "No threshold, R2 configuration, lifecycle, or consumed real cohort was tuned."
        ],
    }
    write_json_atomic(result_path, result)
    manifest = {
        "schema_version": "blindassist-dtr-carla-c0-result-manifest-v1",
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "truth_path": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
    }
    write_json_atomic(manifest_path, manifest)
    return result


def main() -> int:
    args = parse_args()
    commands = {"join": join, "predict": predict, "score": score}
    value = commands[args.mode](args)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

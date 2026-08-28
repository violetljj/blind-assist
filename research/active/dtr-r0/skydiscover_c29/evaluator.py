"""SkyDiscover evaluator for C29 truth-blind dense-motion authority search."""

from __future__ import annotations

import ast
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
DTR = HERE.parent
REPO = DTR.parents[2]
if str(DTR) not in sys.path:
    sys.path.insert(0, str(DTR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_c27_persistent_point_support as c27  # noqa: E402
from dtr_c1_global_obb_cohort_admission import (  # noqa: E402
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
)
from dtr_c2_fresh_global_obb_replay import _tracks  # noqa: E402
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
from dtr_r5_dropout_canary import cases_from_tracks  # noqa: E402
from dtr_r7_occupancy_flow_canary import FlowLedger, load_flow_ledger  # noqa: E402
try:
    from skydiscover.evaluation import EvaluationResult  # type: ignore  # noqa: E402
except ModuleNotFoundError:  # Shared BlindAssist runtime does not install SkyDiscover.
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class EvaluationResult:  # type: ignore[no-redef]
        metrics: Mapping[str, float]
        artifacts: Mapping[str, str]


C25_ROOT = REPO / "artifacts.local" / "evidence" / "dtr-c25" / "fresh-point-flow-confirmation"
TRACE_PATH = REPO / "artifacts.local" / "evidence" / "dtr-c30" / "raw-residual-authority-trace.json"
C25_PREDICTIONS = C25_ROOT / "predictions.json"
ROSTER_PATH = DTR / "dtr_c25_fresh_confirmation_roster.json"
LABELS_PATH = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_labels.zip"
TIMESTAMPS_PATH = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_timestamps.zip"
INITIAL_PATH = HERE / "initial_program.py"
START = "# EVOLVE-BLOCK-START"
END = "# EVOLVE-BLOCK-END"
FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}


_CONTEXT: dict[str, Any] | None = None


def _parts(text: str) -> tuple[str, str, str]:
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("exactly one EVOLVE-BLOCK pair is required")
    prefix, remainder = text.split(START, 1)
    block, suffix = remainder.split(END, 1)
    return prefix, block, suffix


def _validate_candidate(path: Path) -> Any:
    initial_prefix, _initial_block, initial_suffix = _parts(
        INITIAL_PATH.read_text(encoding="utf-8")
    )
    candidate_text = path.read_text(encoding="utf-8")
    prefix, block, suffix = _parts(candidate_text)
    if prefix != initial_prefix or suffix != initial_suffix:
        raise ValueError("code outside EVOLVE-BLOCK changed")
    tree = ast.parse(block)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise ValueError(f"forbidden candidate syntax:{type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden candidate name:{node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is forbidden")
    module_name = f"blindassist_c29_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("candidate import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "choose", None)):
        raise ValueError("candidate choose adapter missing")
    return module


def _load_context() -> dict[str, Any]:
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    trace = json.loads(TRACE_PATH.resolve(strict=True).read_text(encoding="utf-8"))
    if trace.get("schema") != "blindassist-dtr-c30-truth-blind-raw-residual-authority-trace-v1" or trace.get("truth_blind") is not True:
        raise ValueError("authority trace contract mismatch")
    c25 = json.loads(C25_PREDICTIONS.resolve(strict=True).read_text(encoding="utf-8"))
    roster = json.loads(ROSTER_PATH.resolve(strict=True).read_text(encoding="utf-8"))
    c25_rows = {str(row["sequence"]): row for row in c25["sequences"]}
    trace_rows = {str(row["sequence"]): row for row in trace["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    if not (set(c25_rows) == set(trace_rows) == set(roster_rows)):
        raise ValueError("C29 sequence coverage drift")
    sequence_context: dict[str, Any] = {}
    with zipfile.ZipFile(LABELS_PATH.resolve(strict=True)) as labels, zipfile.ZipFile(
        TIMESTAMPS_PATH.resolve(strict=True)
    ) as timestamps_zip:
        for sequence in sorted(c25_rows):
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(
                frames=frames, timestamps=timestamps, boxes_by_frame=boxes
            )
            sources = c25_rows[sequence]["sources"]["ledgers"]
            pd = c27._load_arrays(
                Path(sources["M1_PD_GLOBAL"]["ledger"]),
                Path(sources["M1_PD_GLOBAL"]["manifest"]),
                {"frames", "frame_time_s", "frame_ego_x_m", "frame_ego_y_m", "frame_ego_yaw_rad", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps"},
            )
            pdc = c27._load_arrays(
                Path(sources["M1_PDC_GLOBAL"]["ledger"]),
                Path(sources["M1_PDC_GLOBAL"]["manifest"]),
                {"frames", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps"},
            )
            frame_poses = {
                frame: c27._pose(pd, index) for index, frame in enumerate(frames)
            }
            cases = cases_from_tracks(
                _tracks(boxes_by_frame=boxes, timestamps=timestamps, frame_poses=frame_poses)
            )
            sequence_context[sequence] = {
                "frames": frames,
                "timestamps": timestamps,
                "timeline": timeline,
                "cases": {(case.label_id, case.segment_index): case for case in cases},
                "trace": trace_rows[sequence],
                "pdc": pdc,
                "pdc_prediction": c25_rows[sequence]["arms"]["M1_PDC_GLOBAL"],
                "r7": load_flow_ledger(
                    Path(sources["R7_P_GLOBAL"]["ledger"]),
                    Path(sources["R7_P_GLOBAL"]["manifest"]),
                    expected_sequence=sequence,
                    expected_frames=frames,
                ),
                "pd": c27.load_point_ledger(
                    Path(sources["M1_PD_GLOBAL"]["ledger"]),
                    Path(sources["M1_PD_GLOBAL"]["manifest"]),
                    expected_sequence=sequence,
                    expected_frames=frames,
                ),
                "roster": roster_rows[sequence],
            }
    _CONTEXT = {"sequences": sequence_context}
    return _CONTEXT


def _ledger_from_rows(
    frames: list[int],
    selected_by_frame: Mapping[int, list[Mapping[str, Any]]],
    pdc: Mapping[str, np.ndarray] | None,
) -> FlowLedger:
    forward: list[float] = []
    left: list[float] = []
    vf: list[float] = []
    vl: list[float] = []
    component: list[int] = []
    offsets = [0]
    for index, frame in enumerate(frames):
        rows = selected_by_frame.get(frame, [])
        for row in rows:
            forward.append(float(row["forward_m"]))
            left.append(float(row["left_m"]))
            vf.append(float(row["velocity_forward_mps"]))
            vl.append(float(row["velocity_left_mps"]))
            component.append(int(row["lineage_id"]))
        if pdc is not None:
            start, stop = int(pdc["offsets"][index]), int(pdc["offsets"][index + 1])
            forward.extend(float(value) for value in pdc["forward_m"][start:stop])
            left.extend(float(value) for value in pdc["left_m"][start:stop])
            vf.extend(float(value) for value in pdc["velocity_forward_mps"][start:stop])
            vl.extend(float(value) for value in pdc["velocity_left_mps"][start:stop])
            component.extend(-(value + 1) for value in range(stop - start))
        offsets.append(len(forward))
    return FlowLedger(
        manifest={"schema": "blindassist-dtr-c29-candidate-ledger-v1"},
        frames=np.asarray(frames, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        forward_m=np.asarray(forward, dtype=np.float32),
        left_m=np.asarray(left, dtype=np.float32),
        velocity_forward_mps=np.asarray(vf, dtype=np.float32),
        velocity_left_mps=np.asarray(vl, dtype=np.float32),
        component_id=np.asarray(component, dtype=np.int32),
    )


def _candidate_prediction(
    frames: list[int],
    timestamps: Mapping[int, float],
    extension: FlowLedger,
    pdc_prediction: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar = {frame: {"predicted_unknown": 0} for frame in frames}
    prediction = c27._predict_memory(
        frames=frames, timestamps=timestamps, ledger=extension, sidecar=sidecar
    )
    for field in ("raw_alert_frames", "active_alert_frames", "urgent_frames"):
        prediction[field] = sorted(
            set(int(value) for value in prediction[field])
            | set(int(value) for value in pdc_prediction[field])
        )
    return prediction


def _compact_feedback(per_sequence: list[dict[str, Any]], aggregate: Mapping[str, Any], recovered: int) -> str:
    lines = [
        f"Aggregate: recall={aggregate['bounded_contact_events_recalled']}/12, false={aggregate['false_alert_segments']}, F1={aggregate['bounded_contact_event_f1']:.4f}, lead={aggregate['median_first_alert_lead_s']:.3f}s, dropout={recovered}/36.",
        "Gate target: recall=12, false<=21, dropout>=30, every event no later than PDC.",
        "Exact statuses available: RAW_PD_RESIDUAL=current reciprocal raw-point velocity (q/flow_support/source_point_count valid); OBSERVED_PD_HIT=existing lineage matched current motion; VISIBILITY_OCCLUDED=hidden old lineage (q=0); VISIBILITY_HIT=current occupancy without motion identity. Use these exact strings.",
    ]
    for index, row in enumerate(per_sequence):
        score = row["score"]
        counts = ",".join(f"{key}:{value}" for key, value in sorted(row["selected_statuses"].items())) or "none"
        lines.append(
            f"S{index}: false={score['false_alert_segments']}, recalled={score['bounded_contact_events_recalled']}/{score['bounded_contact_events']}, dropout={row['dropout_recovered']}, selected={counts}."
        )
    return "\n".join(lines)[:2000]


def evaluate(program_path: str) -> EvaluationResult:
    try:
        candidate = _validate_candidate(Path(program_path).resolve(strict=True))
        context = _load_context()
        scores = []
        per_sequence = []
        event_nonregression: list[bool] = []
        total_recovered = 0
        for sequence, data in context["sequences"].items():
            frames = data["frames"]
            trace_frames = data["trace"]["frames"]
            if [int(row["frame"]) for row in trace_frames] != frames:
                raise ValueError(f"trace frame drift:{sequence}")
            selected_by_frame: dict[int, list[Mapping[str, Any]]] = {}
            selected_statuses: Counter[str] = Counter()
            for trace_frame in trace_frames:
                rows = [dict(row) for row in trace_frame["rows"]]
                public_rows = []
                for row in rows:
                    public = {
                        key: value
                        for key, value in row.items()
                        if key not in {"lineage_id", "emitted"}
                    }
                    public["position"] = (
                        float(row["forward_m"]), float(row["left_m"])
                    )
                    public["velocity"] = (
                        float(row["velocity_forward_mps"]),
                        float(row["velocity_left_mps"]),
                    )
                    public["speed_mps"] = math.hypot(*public["velocity"])
                    public["quality"] = float(row.get("q") or 0.0)
                    public["motion_support"] = float(row.get("flow_support") or 0.0)
                    public["support_count"] = int(row.get("source_point_count") or 0)
                    public["memory_decay"] = float(row.get("h") or 0.0)
                    public["fusion_weight"] = float(row.get("w") or 0.0)
                    public_rows.append(public)
                indices = candidate.choose(public_rows)
                selected = [rows[index] for index in indices if rows[index]["status"] != "OBSERVED_PDC"]
                selected_by_frame[int(trace_frame["frame"])] = selected
                selected_statuses.update(str(row["status"]) for row in selected)
            extension = _ledger_from_rows(frames, selected_by_frame, None)
            combined = _ledger_from_rows(frames, selected_by_frame, data["pdc"])
            prediction = _candidate_prediction(
                frames, data["timestamps"], extension, data["pdc_prediction"]
            )
            score = c27.score_sequence(
                sequence=sequence,
                timeline=data["timeline"],
                prediction_frames=_prediction_frames(frames, prediction),
            )
            scores.append(score)
            pdc_score = c27.score_sequence(
                sequence=sequence,
                timeline=data["timeline"],
                prediction_frames=_prediction_frames(frames, data["pdc_prediction"]),
            )
            pdc_events = {row["event_id"]: row for row in pdc_score["event_rows"]}
            for event in score["event_rows"]:
                baseline = pdc_events[event["event_id"]]
                event_nonregression.append(
                    bool(event["recalled"] and baseline["recalled"])
                    and float(event["first_alert_lead_s"]) + 1e-9 >= float(baseline["first_alert_lead_s"])
                )
            stress = c27.dropout_stress(
                roster_sequence=data["roster"],
                cases=data["cases"],
                r7=data["r7"],
                m1=data["pd"],
                m1_ct=combined,
            )
            recovered = int(stress["m1_ct_recovered_track_only_window_misses"])
            total_recovered += recovered
            per_sequence.append(
                {
                    "score": score,
                    "dropout_recovered": recovered,
                    "selected_statuses": dict(selected_statuses),
                }
            )
        aggregate = c27.aggregate_scores(scores)
        recall = float(aggregate["bounded_contact_events_recalled"])
        false_segments = float(aggregate["false_alert_segments"])
        lead = float(aggregate["median_first_alert_lead_s"])
        f1 = float(aggregate["bounded_contact_event_f1"])
        nonregression = all(event_nonregression)
        combined_score = (
            0.36 * (recall / 12.0)
            + 0.28 * (total_recovered / 36.0)
            + 0.20 * max(0.0, 1.0 - false_segments / 52.0)
            + 0.10 * f1
            + 0.06 * min(1.0, lead / 4.2)
        )
        if not nonregression:
            combined_score *= 0.8
        diagnostics = {
            "event_lead_nonregression": nonregression,
            "per_sequence": [
                {
                    "false_segments": row["score"]["false_alert_segments"],
                    "recalled": row["score"]["bounded_contact_events_recalled"],
                    "dropout_recovered": row["dropout_recovered"],
                    "selected_statuses": row["selected_statuses"],
                }
                for row in per_sequence
            ],
        }
        return EvaluationResult(
            metrics={
                "combined_score": combined_score,
                "validity": 1.0,
                "contact_recall": recall,
                "dropout_recovery": float(total_recovered),
                "false_alert_segments": false_segments,
                "median_first_alert_lead_s": lead,
                "event_f1": f1,
                "event_lead_nonregression": 1.0 if nonregression else 0.0,
            },
            artifacts={
                "feedback": _compact_feedback(per_sequence, aggregate, total_recovered),
                "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
            },
        )
    except Exception as error:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "validity": 0.0,
                "contact_recall": 0.0,
                "dropout_recovery": 0.0,
                "false_alert_segments": 999.0,
                "median_first_alert_lead_s": 0.0,
                "event_f1": 0.0,
                "event_lead_nonregression": 0.0,
            },
            artifacts={"feedback": f"Invalid candidate: {type(error).__name__}: {error}"},
        )

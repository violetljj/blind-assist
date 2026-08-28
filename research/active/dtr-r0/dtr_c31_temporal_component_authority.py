"""C31 causal short-window velocity-component authority.

C30 established a strong point-wise confidence and local-consensus baseline,
but its raw residual rows carried zero-filled temporal deltas.  C31 adds the
missing online state: compatible point decisions form local velocity components
which must persist before they can authorize isolated reacquisition or a brief
occluded lineage.  No future frame, sequence identity, or truth label is passed
to the authority policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np


DTR = Path(__file__).resolve().parent
REPO = DTR.parents[2]
if str(DTR) not in sys.path:
    sys.path.insert(0, str(DTR))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import dtr_c27_persistent_point_support as c27  # noqa: E402
import dtr_c30_consensus_motion_authority as c30  # noqa: E402
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
from jrdb_rgb_bridge import require, sha256_file, write_json  # noqa: E402
from research_backend import (  # noqa: E402
    BackendCandidate,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)
from skydiscover_c29 import evaluator as base  # noqa: E402


SCHEMA = "blindassist-dtr-c31-temporal-component-authority-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c31-temporal-component-predictions-v1"
_MATCH_BACKEND = "numpy-cpu-component-matching"


@dataclass
class Component:
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    members: tuple[int, ...]
    support_offsets: np.ndarray


@dataclass
class Track:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    hits: int
    support_offsets: np.ndarray
    dynamic_score: float = 0.0
    missed_s: float = 0.0

    @property
    def mature(self) -> bool:
        return (
            self.hits >= 2
            and self.confidence >= 0.60
            and self.dynamic_score > 0.0
        )


def _num(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _vec(row: Mapping[str, Any], key: str) -> np.ndarray:
    value = row.get(key)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return np.asarray((float(value[0]), float(value[1])), dtype=np.float64)
    if key == "position":
        return np.asarray((_num(row, "forward_m"), _num(row, "left_m")))
    return np.asarray(
        (_num(row, "velocity_forward_mps"), _num(row, "velocity_left_mps"))
    )


def _pairwise(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.empty((len(first), len(second)), dtype=np.float64)
    if _MATCH_BACKEND == "torch-cuda-component-matching":
        import torch

        a = torch.as_tensor(first, dtype=torch.float32, device="cuda")
        b = torch.as_tensor(second, dtype=torch.float32, device="cuda")
        return torch.cdist(a, b).cpu().numpy().astype(np.float64, copy=False)
    delta = first[:, None, :] - second[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _select_matching_backend(receipt: Path) -> dict[str, Any]:
    rng = np.random.default_rng(31)
    first = rng.normal(size=(48, 2)).astype(np.float32)
    second = rng.normal(size=(12, 2)).astype(np.float32)

    def cpu_probe() -> np.ndarray:
        delta = first[:, None, :] - second[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=2))

    cpu = BackendCandidate(
        name="numpy-cpu-component-matching",
        expected_device_type="cpu",
        run_probe=cpu_probe,
        observe=lambda _output: DeviceObservation(
            "cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"
        ),
    )

    gpu: BackendCandidate | None = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_first = torch.as_tensor(first, device="cuda")
            gpu_second = torch.as_tensor(second, device="cuda")

            def gpu_probe() -> Any:
                return torch.cdist(gpu_first, gpu_second)

            gpu = BackendCandidate(
                name="torch-cuda-component-matching",
                expected_device_type="cuda",
                run_probe=gpu_probe,
                observe=lambda output: torch_observation(output=output),
                synchronize=torch.cuda.synchronize,
            )
    except (ImportError, OSError, RuntimeError):
        gpu = None

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=cpu,
        gpu=gpu,
        cpu_reason="GPU_BACKEND_UNAVAILABLE" if gpu is None else None,
        record_path=receipt,
        warmups=1,
        repeats=5,
    )


class TemporalComponentAuthority:
    """Maintain causal motion components and authorize only mature continuity."""

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.next_track_id = 1

    def _components(
        self, rows: Sequence[Mapping[str, Any]], indices: Sequence[int]
    ) -> list[Component]:
        if not indices:
            return []
        positions = np.stack([_vec(rows[index], "position") for index in indices])
        velocities = np.stack([_vec(rows[index], "velocity") for index in indices])
        distance = _pairwise(positions, positions)
        velocity_delta = _pairwise(velocities, velocities)
        parent = list(range(len(indices)))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(first: int, second: int) -> None:
            root_first, root_second = find(first), find(second)
            if root_first != root_second:
                parent[root_second] = root_first

        for first in range(len(indices)):
            for second in range(first + 1, len(indices)):
                speed = max(
                    0.2,
                    float(np.linalg.norm(velocities[first])),
                    float(np.linalg.norm(velocities[second])),
                )
                locality = 0.27 + 0.05 * max(
                    0.0, min(abs(positions[first, 0]), abs(positions[second, 0]))
                )
                if (
                    distance[first, second] <= locality
                    and velocity_delta[first, second] <= 0.18 + 0.28 * speed
                ):
                    union(first, second)

        groups: dict[int, list[int]] = {}
        for local_index in range(len(indices)):
            groups.setdefault(find(local_index), []).append(local_index)
        components = []
        for members in groups.values():
            source = tuple(int(indices[value]) for value in members)
            confidence = float(
                np.mean(
                    [
                        max(_num(rows[index], "q"), _num(rows[index], "quality"))
                        for index in source
                    ]
                )
            )
            components.append(
                Component(
                    position=(centroid := np.median(positions[members], axis=0)),
                    velocity=np.median(velocities[members], axis=0),
                    confidence=confidence,
                    members=source,
                    support_offsets=positions[members] - centroid,
                )
            )
        return components

    def choose(
        self, rows: Sequence[Mapping[str, Any]], *, delta_s: float
    ) -> tuple[list[int], list[dict[str, Any]], dict[str, int], list[int]]:
        delta_s = max(0.0, min(float(delta_s), 0.25))
        predicted = [track.position + track.velocity * delta_s for track in self.tracks]
        base_indices = c30.select_extension(rows)
        motion_indices = [
            index
            for index, row in enumerate(rows)
            if row.get("status") == "RAW_PD_RESIDUAL"
            and row.get("visibility")
            not in {"VISIBILITY_KNOWN_FREE", "VISIBILITY_UNSENSED"}
            and max(_num(row, "q"), _num(row, "quality")) >= 0.55
            and max(
                _num(row, "source_point_count"), _num(row, "support_count")
            )
            >= 2
            and max(_num(row, "flow_support"), _num(row, "motion_support")) > 0
        ]
        components = self._components(rows, motion_indices)
        matched_tracks: set[int] = set()
        matched_components: set[int] = set()
        authorized_components: set[int] = set()

        if predicted and components:
            track_positions = np.stack(predicted)
            track_velocities = np.stack([track.velocity for track in self.tracks])
            component_positions = np.stack([item.position for item in components])
            component_velocities = np.stack([item.velocity for item in components])
            distances = _pairwise(track_positions, component_positions)
            velocity_delta = _pairwise(track_velocities, component_velocities)
            pairs = sorted(
                (
                    float(distances[track_index, component_index]),
                    track_index,
                    component_index,
                )
                for track_index in range(len(self.tracks))
                for component_index in range(len(components))
            )
            for distance, track_index, component_index in pairs:
                if track_index in matched_tracks or component_index in matched_components:
                    continue
                track = self.tracks[track_index]
                component = components[component_index]
                speed = max(
                    0.2,
                    float(np.linalg.norm(track.velocity)),
                    float(np.linalg.norm(component.velocity)),
                )
                if (
                    distance <= 0.35 + speed * delta_s
                    and velocity_delta[track_index, component_index]
                    <= 0.20 + 0.30 * speed
                ):
                    static_residual = float(
                        np.linalg.norm(component.position - track.position)
                    )
                    dynamic_residual = distance
                    transport_evidence = component.confidence * (
                        static_residual * static_residual
                        - dynamic_residual * dynamic_residual
                    )
                    track.dynamic_score = (
                        0.8 * track.dynamic_score + transport_evidence
                    )
                    track.position = component.position
                    track.velocity = 0.65 * component.velocity + 0.35 * track.velocity
                    track.confidence = 0.65 * component.confidence + 0.35 * track.confidence
                    track.support_offsets = component.support_offsets
                    track.hits += 1
                    track.missed_s = 0.0
                    matched_tracks.add(track_index)
                    matched_components.add(component_index)
                    if track.mature:
                        authorized_components.add(component_index)

        prior_mature = [
            (index, self.tracks[index], predicted[index])
            for index in range(len(self.tracks))
            if self.tracks[index].mature
        ]
        selected = set(int(index) for index in base_indices)
        for component_index in authorized_components:
            selected.update(components[component_index].members)
        recovered_component_members = len(selected - set(base_indices))
        recovered_raw = 0
        recovered_occluded = 0

        for index, row in enumerate(rows):
            if index in selected or not prior_mature:
                continue
            status = str(row.get("status"))
            position = _vec(row, "position")
            velocity = _vec(row, "velocity")
            quality = max(_num(row, "q"), _num(row, "quality"))
            source_points = max(_num(row, "source_point_count"), _num(row, "support_count"))
            flow_support = max(_num(row, "flow_support"), _num(row, "motion_support"))
            for _track_index, track, predicted_position in prior_mature:
                speed = max(0.2, float(np.linalg.norm(track.velocity)))
                distance = float(np.linalg.norm(position - predicted_position))
                velocity_error = float(np.linalg.norm(velocity - track.velocity))
                if status == "RAW_PD_RESIDUAL":
                    if (
                        quality >= 0.55
                        and source_points >= 2
                        and flow_support > 0
                        and distance <= 0.30 + speed * delta_s
                        and velocity_error <= 0.16 + 0.26 * speed
                    ):
                        selected.add(index)
                        recovered_raw += 1
                        break
                elif status == "VISIBILITY_OCCLUDED":
                    age_s = _num(row, "age_s")
                    if (
                        0.0 < age_s <= 0.25
                        and track.missed_s + delta_s <= 0.25
                        and distance <= 0.25 + speed * age_s
                        and velocity_error <= 0.14 + 0.22 * speed
                    ):
                        selected.add(index)
                        recovered_occluded += 1
                        break

        known_free = [
            _vec(row, "position")
            for row in rows
            if row.get("status") == "VISIBILITY_KNOWN_FREE"
        ]
        occluded = [
            _vec(row, "position")
            for row in rows
            if row.get("status") == "VISIBILITY_OCCLUDED"
        ]
        synthesized: list[dict[str, Any]] = []
        survivors: list[Track] = []
        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.position = predicted[track_index]
                track.missed_s += delta_s
                track.confidence *= math.exp(-4.0 * delta_s)
            cleared = any(
                float(np.linalg.norm(point - track.position)) <= 0.30
                for point in known_free
            )
            occlusion_supported = any(
                float(np.linalg.norm(point - track.position))
                <= 0.35 + float(np.linalg.norm(track.velocity)) * delta_s
                for point in occluded
            )
            if (
                track_index not in matched_tracks
                and track.mature
                and not cleared
                and occlusion_supported
                and track.missed_s <= 0.25
                and track.confidence >= 0.55
            ):
                for support_index, offset in enumerate(track.support_offsets):
                    support_position = track.position + offset
                    synthesized.append(
                        {
                            "lineage_id": (
                                1_000_000 + track.track_id * 1_000 + support_index
                            ),
                            "status": "TEMPORAL_COMPONENT_OCCLUDED_PREDICTION",
                            "visibility": "OCCLUDED",
                            "emitted": True,
                            "age_s": track.missed_s,
                            "q": track.confidence,
                            "forward_m": float(support_position[0]),
                            "left_m": float(support_position[1]),
                            "velocity_forward_mps": float(track.velocity[0]),
                            "velocity_left_mps": float(track.velocity[1]),
                        }
                    )
            if not cleared and track.missed_s <= 0.25 and track.confidence >= 0.40:
                survivors.append(track)
        self.tracks = survivors
        for component_index, component in enumerate(components):
            if component_index not in matched_components:
                self.tracks.append(
                    Track(
                        track_id=self.next_track_id,
                        position=component.position,
                        velocity=component.velocity,
                        confidence=component.confidence,
                        hits=1,
                        support_offsets=component.support_offsets,
                    )
                )
                self.next_track_id += 1
        return (
            sorted(selected),
            synthesized,
            {
                "base": len(base_indices),
                "motion_vote_rows": len(motion_indices),
                "recovered_component_members": recovered_component_members,
                "recovered_raw": recovered_raw,
                "recovered_occluded": recovered_occluded,
                "synthesized_occluded_component": len(synthesized),
                "mature_tracks": sum(track.mature for track in self.tracks),
            },
            list(base_indices),
        )


def _public_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        public = {
            key: value
            for key, value in row.items()
            if key not in {"lineage_id", "emitted"}
        }
        public["position"] = (float(row["forward_m"]), float(row["left_m"]))
        public["velocity"] = (
            float(row["velocity_forward_mps"]),
            float(row["velocity_left_mps"]),
        )
        public["speed_mps"] = math.hypot(*public["velocity"])
        public["quality"] = float(row.get("q") or 0.0)
        public["motion_support"] = float(row.get("flow_support") or 0.0)
        public["support_count"] = int(row.get("source_point_count") or 0)
        output.append(public)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _MATCH_BACKEND
    selection = _select_matching_backend(args.backend_receipt.resolve())
    _MATCH_BACKEND = str(selection["selected_backend"])
    context = base._load_context()
    scores = []
    per_sequence = []
    predictions = []
    event_nonregression: list[bool] = []
    total_recovered = 0
    totals: Counter[str] = Counter()

    for sequence, data in context["sequences"].items():
        frames = data["frames"]
        trace_frames = data["trace"]["frames"]
        require(
            [int(row["frame"]) for row in trace_frames] == frames,
            f"c31_trace_frame_drift:{sequence}",
        )
        authority = TemporalComponentAuthority()
        selected_by_frame: dict[int, list[Mapping[str, Any]]] = {}
        base_selected_by_frame: dict[int, list[Mapping[str, Any]]] = {}
        previous_time: float | None = None
        sequence_totals: Counter[str] = Counter()
        for trace_frame in trace_frames:
            frame_time = float(trace_frame["frame_time_s"])
            delta_s = 0.0 if previous_time is None else frame_time - previous_time
            previous_time = frame_time
            rows = [dict(row) for row in trace_frame["rows"]]
            indices, synthesized, diagnostics, base_indices = authority.choose(
                _public_rows(rows), delta_s=delta_s
            )
            selected = [
                rows[index]
                for index in indices
                if rows[index]["status"] != "OBSERVED_PDC"
            ]
            selected.extend(synthesized)
            selected_by_frame[int(trace_frame["frame"])] = selected
            base_selected_by_frame[int(trace_frame["frame"])] = [
                rows[index]
                for index in base_indices
                if rows[index]["status"] != "OBSERVED_PDC"
            ]
            sequence_totals.update(diagnostics)
        totals.update(sequence_totals)
        extension = base._ledger_from_rows(frames, selected_by_frame, None)
        combined = base._ledger_from_rows(frames, selected_by_frame, data["pdc"])
        prediction = base._candidate_prediction(
            frames, data["timestamps"], extension, data["pdc_prediction"]
        )
        base_extension = base._ledger_from_rows(
            frames, base_selected_by_frame, None
        )
        base_prediction = base._candidate_prediction(
            frames, data["timestamps"], base_extension, data["pdc_prediction"]
        )
        prediction_delta = {
            field: sorted(
                set(int(value) for value in prediction[field])
                - set(int(value) for value in base_prediction[field])
            )
            for field in ("raw_alert_frames", "active_alert_frames", "urgent_frames")
        }
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
            baseline_event = pdc_events[event["event_id"]]
            event_nonregression.append(
                bool(event["recalled"] and baseline_event["recalled"])
                and float(event["first_alert_lead_s"]) + 1e-9
                >= float(baseline_event["first_alert_lead_s"])
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
        misses = [
            row
            for row in stress["rows"]
            if row.get("status") == "EVALUATED"
            and row.get("track_only_miss")
            and not row.get("m1_ct_recovered_track_only_miss")
        ]
        per_sequence.append(
            {
                "sequence": sequence,
                "score": score,
                "dropout_recovered": recovered,
                "remaining_dropout_misses": misses,
                "authority_counts": dict(sequence_totals),
                "new_route_risk_frames_over_c30": {
                    field: len(values) for field, values in prediction_delta.items()
                },
            }
        )
        predictions.append(
            {
                "sequence": sequence,
                "prediction": prediction,
                "c30_prediction": base_prediction,
                "new_route_risk_frames_over_c30": prediction_delta,
            }
        )

    aggregate = c27.aggregate_scores(scores)
    gate = (
        int(aggregate["bounded_contact_events_recalled"]) == 12
        and int(aggregate["false_alert_segments"]) <= 21
        and total_recovered >= 30
        and all(event_nonregression)
    )
    result = {
        "schema": SCHEMA,
        "terminal_status": (
            "DTR_C31_TEMPORAL_COMPONENT_AUTHORITY_DEVELOPMENT_GATE_MET"
            if gate
            else "DTR_C31_TEMPORAL_COMPONENT_AUTHORITY_DEVELOPMENT_GATE_NOT_MET"
        ),
        "truth_blind": True,
        "metrics": {
            "contact_recall": int(aggregate["bounded_contact_events_recalled"]),
            "contact_events": int(aggregate["bounded_contact_events"]),
            "false_alert_segments": int(aggregate["false_alert_segments"]),
            "event_f1": float(aggregate["bounded_contact_event_f1"]),
            "median_first_alert_lead_s": float(aggregate["median_first_alert_lead_s"]),
            "dropout_recovery": total_recovered,
            "dropout_trials": 36,
            "event_lead_nonregression": all(event_nonregression),
        },
        "authority_counts": dict(totals),
        "per_sequence": per_sequence,
        "backend": selection,
        "source": {
            "c30_trace": str(base.TRACE_PATH.resolve()),
            "c30_trace_sha256": sha256_file(base.TRACE_PATH.resolve()),
            "backend_receipt": str(args.backend_receipt.resolve()),
            "backend_receipt_sha256": sha256_file(args.backend_receipt.resolve()),
        },
        "evidence_boundary": (
            "Consumed C25 Development cohort; policy is causal and truth-blind, "
            "but selection and scoring are not fresh confirmation."
        ),
    }
    write_json(
        args.predictions.resolve(),
        {"schema": PREDICTION_SCHEMA, "sequences": predictions},
    )
    result["source"]["predictions"] = str(args.predictions.resolve())
    result["source"]["predictions_sha256"] = sha256_file(args.predictions.resolve())
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    root = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "temporal-component-authority"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "result.json")
    parser.add_argument("--predictions", type=Path, default=root / "predictions.json")
    parser.add_argument("--backend-receipt", type=Path, default=root / "component-backend.json")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"terminal_status": result["terminal_status"], **result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()

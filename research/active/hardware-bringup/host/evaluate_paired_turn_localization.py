"""Calibrate a causal lead on THÖR-UWB, then compare independent turn positions."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from causal_position_compensation import compare_heading_lead, compensate
from mobilio_turn_confirmation import write_json
from turn_path_odometry import route_distance_corrections


THOR_CALIBRATION_END_S = 55.0
THOR_SCORE_START_S = 65.0
ADVIO_SCORE_START_S = 30.0
ADVIO_SEQUENCES = ("02", "03", "10", "15")
LEAD_CHOICES_S = np.arange(-1.0, 1.01, 0.25)
REFERENCE_MAX_INTERVAL_S = 0.025
WINDOW_RADIUS_S = 1.5
TURN_SPACING_S = 6.0
TURN_MIN_ANGLE = 45.0
TURN_MIN_SEGMENT_M = 0.45
OUTBOUND_GATE_M = 1.0
ODOMETRY_SCAN_S = 5.0


@dataclass
class Trace:
    time_s: np.ndarray
    xy: np.ndarray

    @classmethod
    def from_csv(cls, path):
        rows = np.loadtxt(path, skiprows=1, delimiter=",", usecols=(0, 1, 2))
        times, xy = rows[:, 0], rows[:, 1:]
        if len(times) < 2 or not np.all(np.isfinite(rows)) or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid time-position data: {path}")
        return cls(times, xy)

    def at(self, query_s):
        """Interpolate only adjacent supported reference measurements."""
        times = self.time_s
        values = np.asarray(query_s, dtype=float)
        hi = np.searchsorted(times, values - 1e-9)
        bounded_hi = np.clip(hi, 0, len(times) - 1)
        exact = np.abs(times[bounded_hi] - values) <= 1e-9
        lo = np.where(exact, bounded_hi, np.maximum(0, bounded_hi - 1))
        intervals = times[bounded_hi] - times[lo]
        bracketed = (
            (hi > 0) & (hi < len(times)) & (values >= times[lo])
            & (values <= times[bounded_hi]) & (intervals <= REFERENCE_MAX_INTERVAL_S)
        )
        relative = np.divide(values - times[lo], intervals, out=np.zeros_like(values), where=intervals > 0)
        interpolated = self.xy[lo] + relative[..., None] * (self.xy[bounded_hi] - self.xy[lo])
        return np.where((exact | bracketed)[..., None], interpolated, np.nan)


def fit_lead(device, reference, start_s, end_s):
    max_lead = float(np.max(np.abs(LEAD_CHOICES_S)))
    subset = (device.time_s >= start_s + max_lead) & (device.time_s <= end_s - max_lead)
    fitted = reference.at(device.time_s[subset, None] - LEAD_CHOICES_S)
    common = np.isfinite(fitted).all(axis=(1, 2))
    points = device.xy[subset][common]
    if not len(points):
        return None
    errors = np.sqrt(np.mean(np.sum((points[:, None] - fitted[common]) ** 2, axis=2), axis=0))
    best = min(range(len(errors)), key=lambda i: (errors[i], abs(LEAD_CHOICES_S[i]), float(LEAD_CHOICES_S[i])))
    return {
        "pairs": len(points),
        "minimum_rmse_lead_s": float(LEAD_CHOICES_S[best]),
        "rmse_by_lead_s": {str(float(lead)): float(error) for lead, error in zip(LEAD_CHOICES_S, errors, strict=True)},
    }


def reference_turns(reference, start_s):
    first = max(start_s + 2, reference.time_s[0] + 2)
    grid = np.arange(np.ceil(first * 2) / 2, reference.time_s[-1] - 2, 0.5)
    coords = reference.at(grid[:, None] + [-2, -1, 0, 1, 2])
    incoming = coords[:, 1] - coords[:, 0]
    outgoing = coords[:, 4] - coords[:, 3]
    in_m = np.linalg.norm(incoming, axis=1)
    out_m = np.linalg.norm(outgoing, axis=1)
    usable = np.isfinite(coords).all(axis=(1, 2)) & (np.minimum(in_m, out_m) >= TURN_MIN_SEGMENT_M)
    unit_in = incoming[usable] / in_m[usable, None]
    unit_out = outgoing[usable] / out_m[usable, None]
    angles = np.degrees(np.arccos(np.clip(np.sum(unit_in * unit_out, axis=1), -1.0, 1.0)))
    candidates = [
        (float(time), float(angle), center, tangent)
        for time, angle, center, tangent in zip(grid[usable], angles, coords[usable, 2], unit_out, strict=True)
        if angle >= TURN_MIN_ANGLE
    ]
    accepted = []
    for candidate in sorted(candidates, key=lambda item: (-item[1], item[0])):
        if all(abs(candidate[0] - old[0]) >= TURN_SPACING_S for old in accepted):
            accepted.append(candidate)
    return sorted(accepted, key=lambda item: item[0])


def metrics(shifted, reference, raw):
    distance = np.linalg.norm(shifted - reference, axis=1)
    return {
        "pairs": len(distance),
        "rmse_m": float(np.sqrt(np.mean(distance**2))) if len(distance) else None,
        "median_m": float(np.median(distance)) if len(distance) else None,
        "p95_m": float(np.quantile(distance, .95)) if len(distance) else None,
        "adjusted": int(np.sum(np.any(shifted != raw, axis=1))),
    }


def window_metrics(arms, reference, directions, origins):
    rows = {}
    expected = np.sum((reference - origins) * directions, axis=1) >= OUTBOUND_GATE_M
    for name, points in arms.items():
        signed = np.sum((points - reference) * directions, axis=1)
        projected = np.sum((points - origins) * directions, axis=1) >= OUTBOUND_GATE_M
        rows[name] = {
            **metrics(points, reference, arms["raw"]),
            "outbound_mae_m": float(np.mean(np.abs(signed))) if len(signed) else None,
            "ahead_by_over_0_5m": int(np.sum(signed > 0.5)),
            "early_across_reference_line": int(np.sum(projected & ~expected)),
            "missed_reference_line": int(np.sum(expected & ~projected)),
        }
    return rows


def evaluate(device, reference, lead_s, start_s, yaw_deg=None):
    corrected, unguarded = compensate(device.time_s, device.xy, lead_s)
    truth = reference.at(device.time_s)
    valid = (device.time_s >= start_s) & np.isfinite(truth).all(axis=1)
    arms = {"raw": device.xy, "unguarded": device.xy + unguarded, "gated": device.xy + corrected}
    if yaw_deg is not None:
        same_gate, heading = compare_heading_lead(device.time_s, yaw_deg, corrected, lead_s)
        arms["gated_with_heading_check"] = device.xy + same_gate
        arms["heading_integration"] = device.xy + heading
    turns = []
    window_indices = []
    window_directions = []
    window_origins = []
    if yaw_deg is not None:
        along_only = []
    for center_s, angle, center, direction in reference_turns(reference, start_s):
        indices = np.flatnonzero(valid & (np.abs(device.time_s - center_s) <= WINDOW_RADIUS_S))
        if not len(indices):
            continue
        window_indices.extend(indices.tolist())
        window_directions.extend([direction] * len(indices))
        window_origins.extend([center] * len(indices))
        per_turn = {arm: pos[indices] for arm, pos in arms.items()}
        if yaw_deg is not None:
            along = per_turn["raw"] + (heading[indices] @ direction)[:, None] * direction
            per_turn["heading_projected_on_reference"] = along
            along_only.extend(along)
        turns.append({
            "reference_turn_s": center_s,
            "turn_angle_deg": angle,
            "reference_center_m": center.tolist(),
            "arms": window_metrics(per_turn, truth[indices], direction, center),
        })
    indices = np.asarray(window_indices, dtype=int)
    direction = np.asarray(window_directions).reshape(-1, 2)
    origin = np.asarray(window_origins).reshape(-1, 2)
    selected_arms = {name: item[indices] for name, item in arms.items()}
    if yaw_deg is not None:
        selected_arms["heading_projected_on_reference"] = np.asarray(along_only).reshape(-1, 2)
    return {
        "score_start_s": start_s,
        "all_supported": {name: metrics(item[valid], truth[valid], device.xy[valid]) for name, item in arms.items()},
        "turn_windows": window_metrics(selected_arms, truth[indices], direction, origin),
        "turns": turns,
    }


def odometry_map_turns(fixes, reference):
    for i in range(1, len(fixes) - 1):
        center_s = fixes[i, 0]
        if center_s < ADVIO_SCORE_START_S + 2 or center_s > reference.time_s[-1] - 2:
            continue
        triplet = fixes[i - 1:i + 2]
        if not np.isfinite(triplet[:, -1]).all() or not np.all(triplet[:, -1] == triplet[1, -1]):
            continue
        vertices = triplet[:, 1:3]
        legs = np.diff(vertices, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        if np.any(lengths < 2.5):
            continue
        directions = legs / lengths[:, None]
        mapped_turn = np.degrees(np.arccos(np.clip(directions[0] @ directions[1], -1, 1)))
        position = reference.at(center_s + np.array([-2, -1, 1, 2]))
        measured = np.diff(position.reshape(2, 2, 2), axis=1)[:, 0]
        n = np.linalg.norm(measured, axis=1)
        if not 45 <= mapped_turn <= 135 or not np.all(n >= TURN_MIN_SEGMENT_M):
            continue
        cosine = np.sum((measured / n[:, None]) * directions, axis=1)
        if np.all(cosine >= np.cos(np.radians(25))):
            yield float(center_s), vertices, directions[1]


def evaluate_odometry(device, reference, fixpoint_path):
    """Use fixed map checkpoints as a route proxy, and score the separate reference stream."""
    fixes = np.loadtxt(fixpoint_path, delimiter=",")
    cases = list(odometry_map_turns(fixes, reference))
    results = {}
    for bias_m in (0.0, -1.0, 1.0):
        raw_series = []
        corrected_series = []
        references = []
        directions = []
        origins = []
        records = []
        for center_s, vertices, direction in cases:
            xy = device.xy + bias_m * direction
            relative = device.time_s - center_s
            adjustments, anchor = route_distance_corrections(relative, xy, vertices, 0, ODOMETRY_SCAN_S)
            targets = np.arange(0.0, ODOMETRY_SCAN_S + 1e-6, 0.5)
            idx = np.searchsorted(relative, targets + 1e-9, side="right") - 1
            safe_idx = np.maximum(idx, 0)
            truth = reference.at(center_s + targets)
            usable = (idx >= 0) & (targets - relative[safe_idx] <= 0.250000001)
            usable &= np.isfinite(truth).all(axis=1)
            idx = idx[usable]
            truth = truth[usable]
            by_index = dict(adjustments)
            predicted = xy[idx] + np.array([by_index.get(i, np.zeros(2)) for i in idx]).reshape(-1, 2)
            raw_series.extend(xy[idx])
            corrected_series.extend(predicted)
            references.extend(truth)
            directions.extend([direction] * len(idx))
            origins.extend([vertices[1]] * len(idx))
            records.append({
                "center_s": center_s, "anchor_s_relative": anchor, "slots": len(idx),
                "changed": int(np.sum(np.any(predicted != xy[idx], axis=1))),
                "arms": window_metrics({"raw": xy[idx], "odometry": predicted}, truth, direction, vertices[1]),
            })
        results[str(bias_m)] = {
            "mapped_turns": len(cases),
            "arms": window_metrics(
                {"raw": np.asarray(raw_series).reshape(-1, 2), "odometry": np.asarray(corrected_series).reshape(-1, 2)},
                np.asarray(references).reshape(-1, 2), np.asarray(directions).reshape(-1, 2),
                np.asarray(origins).reshape(-1, 2),
            ),
            "each_turn": records,
        }
    return results


def run(root):
    thor = root / "ble-source/thor"
    advio = root / "walking-source/advio"
    sources = {
        "thor-uwb": (thor / "device_xy_m.csv", thor / "reference_xy_m.csv"),
        **{
            f"advio-{seq}": (advio / f"advio-{seq}/arcore_pair.csv", advio / f"advio-{seq}/gt_xy.csv")
            for seq in ADVIO_SEQUENCES
        },
    }
    device, reference = map(Trace.from_csv, sources["thor-uwb"])
    calibration = fit_lead(device, reference, 0.0, THOR_CALIBRATION_END_S)
    if calibration is None:
        raise ValueError("No calibration samples")
    lead_s = max(0.0, calibration["minimum_rmse_lead_s"])
    result = {
        "calibration": calibration,
        "fixed_lead_s": lead_s,
        "source_streams": {
            name: [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p.read_bytes()).hexdigest()} for p in paths]
            for name, paths in sources.items()
        },
        "independent_positions": {
            "thor-uwb": evaluate(device, reference, lead_s, THOR_SCORE_START_S),
        },
        "diagnostic_native_time_shift": {
            "thor-uwb": fit_lead(device, reference, THOR_SCORE_START_S, device.time_s[-1]),
        },
        "route_odometry": {},
    }
    for name in sources:
        if name == "thor-uwb":
            continue
        device, reference = map(Trace.from_csv, sources[name])
        yaw_deg = np.loadtxt(sources[name][0], skiprows=1, delimiter=",", usecols=(5,))
        result["independent_positions"][name] = evaluate(device, reference, lead_s, ADVIO_SCORE_START_S, yaw_deg)
        result["diagnostic_native_time_shift"][name] = fit_lead(device, reference, ADVIO_SCORE_START_S, device.time_s[-1])
        fixpoints = advio / name / "ground-truth/fixpoints.csv"
        result["route_odometry"][name] = {
            "fixpoints_sha256": sha256(fixpoints.read_bytes()).hexdigest(),
            **evaluate_odometry(device, reference, fixpoints),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    results = run(args.run_dir)
    write_json(args.run_dir / "external-replay/localization.json", results)
    print(f"Lead: {results['fixed_lead_s']} s")
    for name, source in results["independent_positions"].items():
        print(name, len(source["turns"]), json.dumps(source["turn_windows"]))


if __name__ == "__main__":
    main()

"""Offline CNH annotation scaffold: simple condition-weak or legacy strict geometry.

No serial, model, protected split or network access. Simple labels are approximate
condition evidence, never independent real-device truth. This task prepares the
code only; labeling the user's real capture remains a later task.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

QUERIES = {
    "HEAD": ((-.30, -.20, .30), (.30, .42, 3.00)),
    "BODY": ((-.30, .42, .30), (.30, .90, 3.00)),
}
PASS_HALF_WIDTH_M = .40
CONTACT_EPS_M = 1e-8  # Frozen cnh_sequence_observed_geometry.py EPS.


def box_mesh(low, high):
    """Six finite box surfaces, rather than occupied-volume intersection."""
    points = list(itertools.product(*zip(low, high)))
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return [[points[f[a]] for a in t] for f in faces for t in ((0, 1, 2), (0, 2, 3))]


def cylinder_mesh(center, radius, length, sides=64):
    """Closed vertical short rod, Y-down; 64 sides are a declared approximation."""
    rings = [[(center[0] + radius * math.cos(2 * math.pi * i / sides),
               center[1] + sign * length / 2,
               center[2] + radius * math.sin(2 * math.pi * i / sides))
              for i in range(sides)] for sign in (-1, 1)]
    cap_centers = [(center[0], center[1] + sign * length / 2, center[2]) for sign in (-1, 1)]
    triangles = []
    for i in range(sides):
        j = (i + 1) % sides
        triangles.extend(((rings[0][i], rings[1][i], rings[1][j]),
                          (rings[0][i], rings[1][j], rings[0][j]),
                          (cap_centers[0], rings[0][i], rings[0][j]),
                          (cap_centers[1], rings[1][j], rings[1][i])))
    return triangles


def surface_intersects(triangles, low, high):
    """Clip triangle polygons against the six closed query half-spaces."""
    for triangle in triangles:
        polygon = [tuple(p) for p in triangle]
        for axis in range(3):
            for boundary, greater in ((low[axis], True), (high[axis], False)):
                if not polygon:
                    break
                result = []
                for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                    a = start[axis] >= boundary if greater else start[axis] <= boundary
                    b = end[axis] >= boundary if greater else end[axis] <= boundary
                    if a != b:
                        t = (boundary - start[axis]) / (end[axis] - start[axis])
                        result.append(tuple(start[k] + t * (end[k] - start[k]) for k in range(3)))
                    if b:
                        result.append(end)
                polygon = result
        # Frozen geometry excludes zero-area tangency, even with closed planes.
        for i in range(1, len(polygon) - 1):
            a, b, c = polygon[0], polygon[i], polygon[i + 1]
            u, v = [b[k] - a[k] for k in range(3)], [c[k] - a[k] for k in range(3)]
            cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            if math.sqrt(sum(x * x for x in cross)) > 1e-15:
                return True
    return False


def _vector(value, name):
    if len(value) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
        raise ValueError(f"{name} requires three finite measured coordinates")
    return tuple(float(x) for x in value)


def object_realizations(obj):
    """Declared measured shapes plus corner translations; arbitrary pose is excluded.

    Measurement uncertainty here is bounded center translation plus cylinder
    radius/length intervals. A tilted/curved/irregular target requires a reviewed
    mesh and a different uncertainty model, otherwise UNKNOWN.
    """
    uncertainty = _vector(obj["translation_uncertainty_m"], "translation uncertainty")
    if min(uncertainty) < 0:
        raise ValueError("Uncertainty cannot be negative")
    offsets = list(itertools.product(*[sorted(set((-v, 0., v))) for v in uncertainty]))
    kind = obj["type"]
    if kind == "box":
        low, high = _vector(obj["low_m"], "box low"), _vector(obj["high_m"], "box high")
        if any(a >= b for a, b in zip(low, high)):
            raise ValueError("Finite box extents required; encode a panel with measured thickness")
        meshes = [box_mesh(low, high)]
    elif kind == "cylinder_y":
        center = _vector(obj["center_m"], "rod center")
        radii = sorted(set(map(float, obj["radius_interval_m"])))
        lengths = sorted(set(map(float, obj["length_interval_m"])))
        if not radii or not lengths or min(radii + lengths) <= 0 or not all(map(math.isfinite, radii + lengths)):
            raise ValueError("Positive finite measured rod radius and length bounds required")
        meshes = [cylinder_mesh(center, r, length) for r, length in itertools.product(radii, lengths)]
    else:
        raise ValueError("Scaffold supports axis-aligned box/panel or vertical short rod only")
    for mesh, offset in itertools.product(meshes, offsets):
        yield [[tuple(p[k] + offset[k] for k in range(3)) for p in triangle] for triangle in mesh]


def classify_scene(review):
    """Return POS/pass/clear/UNKNOWN per query, with measurement-bound sensitivity.

    POS needs at least one robust measured surface witness. pass/clear require
    reviewed scene completeness, since unlisted furniture can change the label.
    This is a finite sensitivity check, not a formal uncertainty certificate.
    """
    labels = {}
    reviewed = bool(review.get("photo_before") and review.get("photo_after")
                    and review.get("photos_before_after_reviewed") and review.get("stationary_verified")
                    and review.get("measurement_reviewed") and review.get("axis_alignment_reviewed"))
    for query, (low, high) in QUERIES.items():
        if not reviewed:
            labels[query] = {"label": "UNKNOWN", "reason": "measurement/photo/pose review pending"}
            continue
        object_states = []
        for obj in review.get("objects", []):
            states = set()
            for mesh in object_realizations(obj):
                if surface_intersects(mesh, (low[0] + CONTACT_EPS_M, low[1], low[2]),
                                      (high[0] - CONTACT_EPS_M, high[1], high[2])):
                    states.add("POS")
                elif surface_intersects(mesh, (-PASS_HALF_WIDTH_M, low[1], low[2]),
                                        (PASS_HALF_WIDTH_M, high[1], high[2])):
                    states.add("pass")
                else:
                    states.add("clear")
            object_states.append(states)
        complete = bool(review.get("query_completeness_reviewed", {}).get(query))
        if any(states == {"POS"} for states in object_states):
            label, reason = "POS", "measured object surface intersects in all declared realizations"
        elif any("POS" in states for states in object_states):
            label, reason = "UNKNOWN", "measurement bounds cross contact boundary"
        elif not complete:
            label, reason = "UNKNOWN", "query/background extent not fully reviewed"
        elif any(states == {"pass"} for states in object_states):
            label, reason = "pass", "no contact; robust surface in lateral 0-10 cm shell"
        elif all(states == {"clear"} for states in object_states):
            label, reason = "clear", "reviewed query has no measured surface in contact/pass boxes"
        else:
            label, reason = "UNKNOWN", "measurement bounds cross pass/clear boundary"
        labels[query] = {"label": label, "reason": reason}
    return labels


def review_template():
    return {
        "schema": "real-cnh-geometry-review-v1", "source_segment": "REPLACE_WITH_SEGMENT_ID",
        "capture_metadata_sha256": None, "photo_before": None, "photo_after": None,
        "photos_before_after_reviewed": False, "stationary_verified": False,
        "measurement_reviewed": False, "axis_alignment_reviewed": False,
        "query_completeness_reviewed": {"HEAD": False, "BODY": False},
        "measured": {"sensor_height_m": None, "physical_down_pitch_deg": None,
                     "target_axial_front_m": None, "target_center_height_m": None,
                     "target_center_x_m": None, "distance_reference": "CNH optical origin / measured uncertainty"},
        "assumptions": {"sensor_nominal_height_m": 1.65, "nominal_down_pitch_deg": 10.,
                        "rectangular_fov_deg": 45., "gravity_aligned_corridor": True},
        "objects": [],
        "instructions": "Insert measured panel/rod AND all query-relevant furniture/floor/support geometry; photos are supporting evidence, firmware range is never truth.",
    }


def annotate(frames_path, metadata_path, review_path, output):
    """Deferred offline entry point; output is exclusive to preserve old labels."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    expected = review.get("capture_metadata_sha256")
    actual = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("Human review must bind the exact capture metadata SHA256")
    labels = classify_scene(review)
    with frames_path.open(encoding="utf-8") as source, output.open("x", encoding="utf-8") as dest:
        for index, line in enumerate(source):
            frame = json.loads(line)
            # Live capture preserves Decoder envelopes; legacy flat rows remain
            # accepted for explicit offline fixtures. Motion review flags may be
            # attached to the envelope or its sensor record, never inferred from
            # firmware distance/status.
            sensor = frame.get("sensor", frame)
            if not isinstance(sensor, dict):
                raise ValueError("Frame sensor record must be a JSON object")
            invalid = any(bool(record.get(flag)) for record in (frame, sensor)
                          for flag in ("invalid", "motion_during_capture", "pose_changed"))
            for query in QUERIES:
                result = labels[query] if not invalid else {"label": "UNKNOWN", "reason": "frame movement/pose change"}
                row = {"frame_index": index, "seq": sensor.get("seq"), "ms": sensor.get("ms"),
                       "host_received_utc": frame.get("host_received_utc"),
                       "host_received_monotonic_ns": frame.get("host_received_monotonic_ns"),
                       "host_time": frame.get("host_received_utc", frame.get("host_unix_s", frame.get("host_timestamp"))),
                       "query": query, **result, "label_source": "measured-static-geometry+manual-photo-review",
                       "timeliness": "NOT_EVALUABLE", "firmware_distance_used_as_truth": False,
                       "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()}
                dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def condition_index(metadata):
    """Single10: 1..9 plus dark 4b. Legacy night10/11 remain readable."""
    index = metadata.get("condition_index")
    if isinstance(index, str) and index.strip().lower() == "4b":
        return "4b"
    if isinstance(index, int) and 1 <= index <= 11:
        return index
    text = str(metadata.get("segment_id", metadata.get("condition", "")))
    match = re.search(r"(?:^|[-_])(?:day|night)?[-_]?0?(4b|1[01]|[1-9])(?:$|[-_])", text.lower())
    return ("4b" if match.group(1) == "4b" else int(match.group(1))) if match else None


def _weak_geometry(metadata, frame):
    """Weak position intervals; defaults remain explicit estimates, not measurements."""
    reference = metadata.get("distance_reference") or {}
    distance = metadata.get("target_distance_m", reference.get("distance_m"))
    source = metadata.get("target_distance_source", reference.get("source_kind"))
    derived = frame.get("derived", {})
    known = derived.get("distance_known_mm", [])
    valid = [known[i] / 1000 for i in (27, 28, 35, 36)
             if i < len(known) and _finite(known[i]) and known[i] > 0]
    # A per-frame range may reveal an intent conflict, but cannot turn a failed
    # or missing stable probe into a confirmed far/empty segment.
    frame_distance = sorted(valid)[len(valid) // 2] if valid else None
    height = metadata.get("sensor_height_m")
    h_error = metadata.get("sensor_height_uncertainty_m", .15)
    distance_error = metadata.get("distance_uncertainty_m", max(.25, .20 * distance) if _finite(distance) else None)
    return {"sensor_height_m": height, "sensor_height_uncertainty_m": h_error,
            "distance_m": distance, "distance_uncertainty_m": distance_error,
            "frame_central_weak_distance_m": frame_distance,
            "distance_source": source, "distance_is_radial_weak_reference": bool(source and "firmware" in str(source)),
            "physical_down_pitch_deg": metadata.get("tilt_deg", 0.),
            "pitch_source": metadata.get("tilt_source", "nominal_assumption"),
            "frozen_nominal_down_pitch_deg": 10., "physically_calibrated": False}


def approximate_labels(metadata, frame=None, attempt_status=None):
    """Condition-weak labels for simple-dorm-single10; no tape/mesh review gate.

    Empty conditions support an intended-empty weak label. Target labels require
    approximate extents safely away from query boundaries; unrecorded pole height
    or horizontal-pole placement stays UNKNOWN. Negative labels are not geometry
    truth and never certify a pure-noise or pure-crosstalk exposure.
    """
    frame = frame or {}
    sensor = frame.get("sensor", frame)
    index = condition_index(metadata)
    geometry = _weak_geometry(metadata, frame)
    status = attempt_status or metadata.get("status")
    bad_status = status in ("failed", "interrupted", "interrupted_before_recording", "superseded", "skipped", "needs_review")
    quality_failed = bool(metadata.get("quality_failed") or metadata.get("quality_warnings")
                          or metadata.get("geometry_quality") == "failed")
    frame_bad = any(bool(record.get(flag)) for record in (frame, sensor if isinstance(sensor, dict) else {})
                    for flag in ("invalid", "motion_during_capture", "pose_changed", "quality_failed"))
    expected = {q: "UNKNOWN" for q in QUERIES}
    condition_label, expected_response = "UNKNOWN", "UNKNOWN"
    if index in (1, 4, "4b", 9, 10, 11):
        expected = {q: "corridor_empty" for q in QUERIES}
        condition_label, expected_response = "corridor_empty", "silent"
    elif index in (2, 3):
        expected = {q: "calibration_only" for q in QUERIES}
        condition_label, expected_response = "plane", "calibration"
    elif index in (5, 6):
        expected = {q: "target_surface_possible" for q in QUERIES}
        condition_label, expected_response = "corridor_target_present", "alarm"
    elif index == 7:
        expected = {q: "outside_target_no_strong_alert_expected" for q in QUERIES}
        condition_label, expected_response = "outside_target", "no_strong_alarm"
    elif index == 8:
        expected = {"HEAD": "intended_HEAD_target", "BODY": "UNKNOWN"}
        condition_label, expected_response = "corridor_target_present", "alarm"
    results = {}
    for query, (low, high) in QUERIES.items():
        label, reason = "UNKNOWN", "condition/approximate geometry unavailable"
        if bad_status or quality_failed or frame_bad:
            reason = "failed quality, superseded/skipped attempt or movement"
        elif index in (1, 4, "4b", 9, 10, 11):
            distance, de = geometry["distance_m"], geometry["distance_uncertainty_m"]
            observed = geometry["frame_central_weak_distance_m"]
            if (_finite(distance) and .30 <= distance <= 3.) or (_finite(observed) and .30 <= observed <= 3.):
                reason = "intended empty corridor conflicts with weak in-window surface (e.g. curtain)"
            elif index in (1, 10) and not _finite(distance):
                label, reason = "clear", "near-no-echo condition assumption; absence of central range is not truth"
            elif not _finite(distance):
                reason = "intended empty corridor has no usable weak distance reference"
            elif _finite(de) and distance - de > 3.:
                label, reason = "clear", "operator intended empty corridor plus weak far-surface reference"
            else:
                reason = "weak surface range overlaps or precedes query window; empty condition unresolved"
        elif index in (2, 3):
            reason = "plane is calibration data; condition supplies neither negative nor query truth"
        elif index in (5, 6, 7, 8):
            distance, de = geometry["distance_m"], geometry["distance_uncertainty_m"]
            height, he = geometry["sensor_height_m"], geometry["sensor_height_uncertainty_m"]
            target_low = metadata.get("target_height_low_m")
            target_high = metadata.get("target_height_high_m")
            center_height = metadata.get("target_height_m")
            vertical_length = metadata.get("rod_length_m") if metadata.get("rod_orientation") == "vertical" else None
            if not (_finite(target_low) and _finite(target_high)) and _finite(center_height):
                length = vertical_length if _finite(vertical_length) else metadata.get("rod_diameter_m")
                if _finite(length):
                    target_low, target_high = center_height - length / 2, center_height + length / 2
            x = metadata.get("target_x_m")
            xe = metadata.get("target_x_uncertainty_m", .15)
            radius = metadata.get("rod_diameter_m")
            target_he = metadata.get("target_height_uncertainty_m", .20)
            nums = (distance, de, height, he, target_low, target_high, x, xe, radius, target_he)
            if not all(_finite(v) for v in nums):
                reason = "target distance/height/extent unknown; condition name does not establish query POS"
            elif min(de, he, xe, radius, target_he) < 0 or target_low > target_high:
                reason = "invalid approximate extent or uncertainty"
            elif distance - de <= .30 or distance + de >= 3.:
                reason = "weak distance interval crosses the .30/3m query boundary"
            else:
                # A firmware radial range is not axial Z; broad intervals only
                # support coarse placement, and this fact travels with every row.
                y_low = height - he - (target_high + target_he)
                y_high = height + he - (target_low - target_he)
                nearest_x = max(0., abs(x) - radius / 2)
                x_low, x_high = max(0., nearest_x - xe), nearest_x + xe
                near_height_boundary = any(y_low <= boundary <= y_high for boundary in (low[1], high[1]))
                near_x_boundary = any(x_low <= boundary <= x_high for boundary in (.30, .40))
                if near_height_boundary or near_x_boundary:
                    reason = "estimated position interval crosses a height/contact/pass boundary"
                elif y_high < low[1] or y_low > high[1]:
                    reason = "target outside this height band; unknown background prevents query-clear inference"
                elif x_high < .30:
                    label, reason = "POS", "approximate target bounds inside query; condition-weak, not ground truth"
                elif x_low > .30 and x_high < .40:
                    label, reason = "pass", "approximate target in 0-10cm shell; condition-weak evidence"
                elif x_low > .40:
                    reason = "target likely outside; missing scene geometry prevents per-query clear inference"
        results[query] = {"condition_index": index, "lighting": metadata.get("lighting"), "attempt_status": status,
                          "noise_ambient_pair": "4/4b" if index in (4, "4b") else None,
                          "condition_label": condition_label, "expected_response": expected_response,
                          "expected_label": expected[query], "label": label,
                          "source": "CONDITION_WEAK_APPROXIMATE", "label_source": "CONDITION_WEAK_APPROXIMATE",
                          "evidence_quality": "weak_approximate", "reason": reason,
                          "geometry_reference": geometry, "firmware_distance_used_as_truth": False,
                          "independent_ground_truth": False, "pure_noise_or_bias_template": False,
                          "timeliness": "NOT_EVALUABLE"}
    return results


def approximate_session(session, output):
    """Future simple capture postprocessing; exclusive output keeps earlier versions."""
    manifest_path = session / "capture.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    counts = {q: {} for q in QUERIES}
    segment_rows = []
    with output.open("x", encoding="utf-8") as dest:
        for attempt in manifest.get("attempts", []):
            sid = attempt.get("segment_id", attempt.get("condition", "UNKNOWN"))
            number = attempt.get("attempt", attempt.get("attempt_number", 1))
            directory = session / attempt.get("directory", attempt.get("path", f"{sid}/attempt-{int(number):02d}"))
            metadata_path, frames_path = directory / "metadata.json", directory / "frames.jsonl"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else dict(attempt)
            metadata = {**manifest, **attempt, **metadata}
            labels = approximate_labels(metadata, attempt_status=attempt.get("status"))
            segment_rows.append({"segment_id": sid, "attempt": number, "directory": str(directory),
                                 "labels": labels, "frames_available": frames_path.exists()})
            if not frames_path.exists():
                continue  # Skips stay in the summary; never invent frame rows.
            with frames_path.open(encoding="utf-8") as source:
                for frame_index, line in enumerate(source):
                    frame = json.loads(line)
                    sensor = frame.get("sensor", frame)
                    labels = approximate_labels(metadata, frame, attempt.get("status"))
                    for query, label in labels.items():
                        counts[query][label["label"]] = counts[query].get(label["label"], 0) + 1
                        row = {"segment_id": sid, "attempt": number, "frame_index": frame_index,
                               "seq": sensor.get("seq"), "ms": sensor.get("ms"), "query": query, **label,
                               "host_received_utc": frame.get("host_received_utc"),
                               "host_received_monotonic_ns": frame.get("host_received_monotonic_ns"),
                               "manifest_sha256": manifest_hash}
                        dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {"source": "CONDITION_WEAK_APPROXIMATE", "counts": counts, "segments": segment_rows,
               "real_device_accuracy_claim": False, "timeliness": "NOT_EVALUABLE"}
    with output.with_suffix(".summary.json").open("x", encoding="utf-8") as dest:
        json.dump(summary, dest, ensure_ascii=False, indent=2, allow_nan=False)
        dest.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="Prepare an unapproved human-review template")
    template.add_argument("--output", type=Path, required=True)
    label = commands.add_parser("annotate", help="Future offline annotation after manual measurement/photo review")
    for name in ("frames", "metadata", "review", "output"):
        label.add_argument(f"--{name}", type=Path, required=True)
    approximate = commands.add_parser("approximate", help="Future simple-dorm condition-weak annotation; no mesh/tape gate")
    approximate.add_argument("--session", type=Path, required=True)
    approximate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "template":
        with args.output.open("x", encoding="utf-8") as dest:
            json.dump(review_template(), dest, ensure_ascii=False, indent=2, allow_nan=False)
            dest.write("\n")
    elif args.command == "annotate":
        annotate(args.frames, args.metadata, args.review, args.output)
    else:
        approximate_session(args.session, args.output)


if __name__ == "__main__":
    main()

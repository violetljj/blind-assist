"""Offline geometry-label scaffold for a future manually reviewed CNH capture.

No serial, firmware distance, model, protected split or network access. A scene
name alone never supplies a negative label. This task prepares the code only;
real frame annotation is deferred until measurement/photo review is available.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="Prepare an unapproved human-review template")
    template.add_argument("--output", type=Path, required=True)
    label = commands.add_parser("annotate", help="Future offline annotation after manual measurement/photo review")
    for name in ("frames", "metadata", "review", "output"):
        label.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "template":
        with args.output.open("x", encoding="utf-8") as dest:
            json.dump(review_template(), dest, ensure_ascii=False, indent=2, allow_nan=False)
            dest.write("\n")
    else:
        annotate(args.frames, args.metadata, args.review, args.output)


if __name__ == "__main__":
    main()

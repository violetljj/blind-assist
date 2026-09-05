"""Immutable, disclosed Development challenges; no reserved or held-out split.

Four paired mechanisms exercise temporal onset, stopping, obstacle height and
side-access geometry. Analytic controls are evaluator-only and are not method
performance. The 10 Hz source cadence supports the unchanged X24/X25 fit window.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import street_scenarios as street

SCHEMA = "street-discriminating-development-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _box(actor_id, kind, x, y, height, hx, hy):
    return street._actor(actor_id, kind, x, y, height, half_extents=(hx, hy))


def _person(points):
    return street._actor("pedestrian", "pedestrian", *points[0][1:], 1.75,
                         radius=.28, waypoints=points)


def _spec(pair, variant, actors, contact, contact_type=None):
    return {"schema": street.SCHEMA, "id": f"challenge_{pair}_{variant}",
            "family": pair, "pair_id": pair, "pair_variant": variant,
            "control": "collision" if contact else "near_miss",
            "duration_s": 8.0, "dt_s": .1, "ego_speed_mps": 1.0,
            "ego_start": {"x_m": 0.0, "y_m": 0.0}, "arms": list(street.ARMS),
            "visual_geometry_policy": "RENDER_BOUNDS_ENCLOSURE_V1",
            "actors": actors,
            "expected_open_loop_contact": contact,
            "expected_contact_type": contact_type if contact else None}


def scenario_catalog(pedestrian_radius_m=.70):
    if isinstance(pedestrian_radius_m, bool) or not .2 <= pedestrian_radius_m <= .95:
        raise ValueError("Pedestrian radius must be in [0.2, 0.95] m")
    specs = []
    for delayed in (False, True):
        # The crossing starts late in the episode, reaches the route at 5 s
        # or 7 s, and is paired by onset alone (identical speed/geometry).
        onset = 3.0 if not delayed else 5.0
        points = [[0, 5, -2.4], [onset, 5, -2.4], [onset + 4, 5, 2.4]]
        specs.append(_spec("late_crossing", "phase_miss" if delayed else "intercept",
            [_person(points), _box("occluder", "occluder", 3.8, -1.55, 2., .45, .4)],
            not delayed, "BODY_COLLISION_PROXY"))
    for stop in (True, False):
        points = ([[0, 1.8, 0], [3, 4.8, 0], [8, 4.8, 0]] if stop else
                  [[0, 1.8, 0], [8, 9.8, 0]])
        specs.append(_spec("late_stop", "stop" if stop else "continue",
                           [_person(points)], stop, "BODY_COLLISION_PROXY"))
    for raised in (True, False):
        # Same box, location and material kind: only its declared/rendered height
        # changes. Relief is not moved out of the wearer's path to make it easy.
        specs.append(_spec("height_relief", "trip" if raised else "relief",
            [_box("low_obstacle", "low_obstacle", 4.5, 0, .12 if raised else .004, .3, .32)],
            raised, "FOOT_TRIP_PROXY"))
    for blocked in (False, True):
        # Both straight paths hit the central obstacle. The left-offset witness
        # is clear only when the left wall is moved outward; right remains closed.
        spec = _spec("bypass_restriction", "closed" if blocked else "left_open", [
            _box("low_obstacle", "low_obstacle", 4.2, 0, .12, .3, .3),
            _box("right_wall", "barrier", 4.2, .9, 1.1, 1.3, .15),
            _box("left_wall", "barrier", 4.2, -.9 if blocked else -1.5, 1.1, 1.3, .15),
        ], True, "FOOT_TRIP_PROXY")
        spec["expected_left_bypass_contact"] = blocked
        spec["left_bypass_witness"] = [[0., 0., 0.], [2., 1., -.72], [8., 7., -.72]]
        specs.append(spec)
    for spec in specs:
        for actor in spec["actors"]:
            if actor["kind"] == "pedestrian":
                actor["radius_m"] = float(pedestrian_radius_m)
    return specs


def proxy_contacts(spec):
    start = spec["ego_start"]
    end = dict(start, x_m=start["x_m"] + spec["duration_s"] * spec["ego_speed_mps"])
    return street.contacts_for_step(spec, 0., spec["duration_s"], start, end)


def witness_contacts(spec):
    found = []
    for a, b in zip(spec["left_bypass_witness"], spec["left_bypass_witness"][1:]):
        found.extend(street.contacts_for_step(spec, a[0], b[0],
            {"x_m": a[1], "y_m": a[2]}, {"x_m": b[1], "y_m": b[2]}))
    return found


def validate_specs(specs):
    rows = []
    for spec in specs:
        hits = proxy_contacts(spec)
        passed = bool(hits) == spec["expected_open_loop_contact"] and (
            not hits or spec["expected_contact_type"] == hits[0]["contact_type"])
        witness = witness_contacts(spec) if "left_bypass_witness" in spec else None
        if witness is not None:
            passed = passed and bool(witness) == spec["expected_left_bypass_contact"]
        rows.append({"id": spec["id"], "passed": passed, "contacts": hits,
                     "left_bypass_contacts": witness})
    return {"passed": bool(rows) and all(r["passed"] for r in rows), "rows": rows,
            "claim": "Synthetic declared geometry controls only; not algorithm or safety performance"}


def _validate_profile(profile, radius):
    if profile is None:
        return  # Explicit geometry-only callers may omit an empirical profile.
    path = Path(profile["path"])
    raw = path.read_bytes()
    value = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != profile.get("sha256") or value != profile.get("value"):
        raise ValueError("Renderer profile file identity differs from frozen evidence")
    source_hash = value.get("source_sha256", "")
    if value.get("status") != "PASS" or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("Renderer profile must PASS and bind its capture source SHA-256")
    radii = [a["assessment"]["required_enclosing_radius_m"]
             for sample in value.get("samples", []) for a in sample.get("actors", [])
             if a.get("kind") == "pedestrian"]
    if not radii or any(isinstance(r, bool) or not isinstance(r, (int, float)) or
                       not math.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("Renderer profile requires finite positive pedestrian bounds samples")
    if max(radii) > radius + 1e-9:
        raise ValueError("Requested pedestrian radius does not enclose measured renderer bounds")


def build_manifest(pedestrian_radius_m=.70, geometry_profile=None):
    _validate_profile(geometry_profile, pedestrian_radius_m)
    splits = {"development": scenario_catalog(pedestrian_radius_m)}
    body = {"schema": SCHEMA,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry_source_sha256": hashlib.sha256(Path(street.__file__).read_bytes()).hexdigest(),
        "provenance": "HAND_SPECIFIED_SYNTHETIC_DEVELOPMENT; no fresh confirmation or safety authority",
        "pedestrian_radius_m": float(pedestrian_radius_m),
        "geometry_profile": copy.deepcopy(geometry_profile),
        "geometry_claim": "Conservative renderer bounds enclosure; not mesh contact or human safety truth",
        "cadence": {"dt_s": .1, "reason": "Four real samples fit unchanged 0.50 s X24/X25 window"},
        "split_sha256": {k: digest(v) for k, v in splits.items()}, "splits": splits}
    return dict(body, manifest_sha256=digest(body))


def freeze_manifest(path, pedestrian_radius_m=.70, geometry_profile=None):
    manifest = build_manifest(pedestrian_radius_m, geometry_profile)
    if not validate_specs(manifest["splits"]["development"])["passed"]:
        raise ValueError("Paired analytic controls fail at requested pedestrian radius")
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest != build_manifest(manifest.get("pedestrian_radius_m", .70), manifest.get("geometry_profile")):
        raise ValueError("Discriminating Development bank differs from immutable source definition")
    return manifest


def load_scenarios(path, split="development", *, allow_held_out=False):
    if split != "development" or allow_held_out:
        raise ValueError("This bank permits only disclosed development; no held_out admission")
    return copy.deepcopy(read_manifest(path)["splits"][split])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "list"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pedestrian-radius-m", type=float, default=.70,
                        help="Conservative enclosure radius; .28 is legacy geometry-only testing")
    parser.add_argument("--geometry-profile", help="Measured renderer profile JSON; file identity stored in manifest")
    args = parser.parse_args()
    if args.command == "freeze":
        profile = None
        if args.geometry_profile:
            path = Path(args.geometry_profile).resolve()
            profile = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "value": json.loads(path.read_text(encoding="utf-8"))}
        result = freeze_manifest(args.manifest, args.pedestrian_radius_m, profile)
        print(json.dumps({"manifest_sha256": result["manifest_sha256"], "development_count": 8}))
    else:
        specs = load_scenarios(args.manifest)
        result = validate_specs(specs) if args.command == "validate" else [s["id"] for s in specs]
        print(json.dumps(result, indent=2))
        if args.command == "validate" and not result["passed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

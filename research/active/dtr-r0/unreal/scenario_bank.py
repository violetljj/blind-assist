"""Frozen controlled Development bank; evaluator-side only, never model input.

Held-out entries are recipes only: no geometry truth is evaluated and execution requires explicit future admission and consumes access once. Each version is immutable; a changed bank requires a new version,
not a relabelled/repartitioned manifest. These are synthetic proxy controls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import street_scenarios as street

SCHEMA_V1 = "street-challenge-bank-v1"
SCHEMA = "street-challenge-bank-v2"
SUPPORTED_SCHEMAS = (SCHEMA_V1, SCHEMA)
SPLITS = ("regression", "development", "held_out")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def proxy_contacts(spec):
    start = dict(spec["ego_start"])
    end = dict(start, x_m=start["x_m"] + spec["duration_s"] * spec["ego_speed_mps"])
    return street.contacts_for_step(spec, 0.0, spec["duration_s"], start, end)


def _number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def _position(actor, position):
    if len(position) != 2:
        raise ValueError("position requires x,y")
    x, y = [_number(v, -12, 12, "position") for v in position]
    dx, dy = x - actor["x_m"], y - actor["y_m"]
    actor.update(x_m=x, y_m=y)
    actor["waypoints"] = [[t, px + dx, py + dy] for t, px, py in actor["waypoints"]]


def parameterize(source, *, pedestrian_speed_mps=None, onset_s=None,
                 occluder_position_m=None, occluder_half_extents_m=None,
                 obstacle_position_m=None, variant_id="custom_development"):
    """Change rendered proxy geometry and scripts together; recompute control truth.

    onset_s is movement start for crossing/passing, stop onset for sudden-stop
    collision. A sudden-stop near-miss has no stop; onset delays its movement.
    Speed changes segment timing, except a requested stop onset keeps that time
    fixed and derives the stop position from speed. All times are seconds.
    """
    spec = copy.deepcopy(source)
    actors = {a["id"]: a for a in spec["actors"]}
    speed = None if pedestrian_speed_mps is None else _number(pedestrian_speed_mps, 0.3, 2.5, "speed")
    onset = None if onset_s is None else _number(onset_s, 0, 4, "onset_s")
    if speed is not None or onset is not None:
        if "pedestrian" not in actors:
            raise ValueError("pedestrian parameters require a pedestrian")
        actor = actors["pedestrian"]
        points = actor["waypoints"]
        native = next(math.hypot(b[1]-a[1], b[2]-a[2])/(b[0]-a[0])
                      for a,b in zip(points, points[1:]) if a[1:] != b[1:])
        actual_speed = native if speed is None else speed
        if spec["family"] == "sudden_stop" and source["control"] == "collision":
            stop = points[1][0] * native / actual_speed if onset is None else onset
            if stop <= 0:
                raise ValueError("stop onset must be positive")
            x, y = points[0][1:]
            stop_x = x + actual_speed * stop
            actor["waypoints"] = [[0, x, y], [stop, stop_x, y],
                                  [max(spec["duration_s"], stop + 0.01), stop_x, y]]
        else:
            delay = 0.0 if onset is None else onset
            transformed = [[delay + t * native / actual_speed, x, y] for t,x,y in points]
            actor["waypoints"] = ([[0.0, *points[0][1:]]] if delay else []) + transformed
    if occluder_position_m is not None or occluder_half_extents_m is not None:
        if "occluder" not in actors:
            raise ValueError("occluder parameters require an occluder")
        actor = actors["occluder"]
        if occluder_position_m is not None:
            _position(actor, occluder_position_m)
        if occluder_half_extents_m is not None:
            if len(occluder_half_extents_m) != 2:
                raise ValueError("occluder requires two half extents")
            actor["half_extents_m"] = [_number(v, 0.1, 1.5, "occluder half extent") for v in occluder_half_extents_m]
    if obstacle_position_m is not None:
        if "low_obstacle" not in actors:
            raise ValueError("obstacle position requires a low obstacle")
        _position(actors["low_obstacle"], obstacle_position_m)
    spec["id"] = variant_id
    spec["source_scenario_id"] = source["id"]
    contacts = proxy_contacts(spec)
    spec["control"] = "collision" if contacts else "near_miss"
    spec["expected_open_loop_contact"] = bool(contacts)
    spec["expected_contact_type"] = contacts[0]["contact_type"] if contacts else None
    spec["control_derivation"] = "CONTINUOUS_SWEPT_DECLARED_PROXY_GEOMETRY"
    return spec


def _recipe(source, index, schema=SCHEMA):
    # Two disclosed Development settings, one disjoint reserved setting per seed.
    family = source["family"]
    if family == "low_obstacle":
        # V1 is retained exactly, including its duplicate low-obstacle scenes.
        if schema == SCHEMA and source["control"] == "near_miss":
            return {"obstacle_position_m": ([3.8, 0.75], [5.4, 0.25], [6.4, 0.8])[index]}
        return {"obstacle_position_m": ([3.5, 0.65], [5.0, 0.1], [6.0, 0.5])[index]}
    result = {"pedestrian_speed_mps": (0.75, 1.25, 1.6)[index],
              "onset_s": (0.5, 1.5, 2.75)[index]}
    if family == "occluded_crossing":
        result.update(occluder_position_m=([2.1,-1.4], [2.8,-1.8], [3.1,-1.2])[index],
                      occluder_half_extents_m=([0.5,0.4], [0.8,0.6], [0.6,0.7])[index])
    return result


def build_manifest(schema=SCHEMA):
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported frozen bank schema: {schema}")
    baseline = street.scenario_catalog()
    splits = {"regression": baseline, "development": [], "held_out": []}
    for source in baseline:
        for index in range(2):
            params = _recipe(source, index, schema)
            spec = parameterize(source, **params, variant_id=f"{source['id']}_dev_{index+1}")
            spec["parameters"] = params
            splits["development"].append(spec)
        # Do not call parameterize/proxy_contacts on reserved recipes.
        splits["held_out"].append({"id": source["id"] + "_reserved_1",
            "source_scenario_id": source["id"], "parameters": _recipe(source, 2, schema),
            "state": "RESERVED_NOT_EVALUATED", "execution_allowed": False})
    source_hash = hashlib.sha256(Path(street.__file__).read_bytes()).hexdigest()
    body = {"schema": schema, "source_sha256": source_hash,
            "baseline_sha256": digest(baseline), "split_sha256": {k: digest(v) for k,v in splits.items()},
            "provenance": "HAND_SPECIFIED_SYNTHETIC_DEVELOPMENT; no fresh confirmation or safety authority",
            "held_out_policy": "RESERVED_UNTIL_EXPLICIT_ADMISSION; single consumed access; preserve sidecars",
            "splits": splits}
    return dict(body, manifest_sha256=digest(body))


def freeze_manifest(path):
    """Exclusive create: never silently overwrite a previous freeze or consumption."""
    manifest = build_manifest()
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    # Comparing to the versioned definition also rejects a recomputed tamper hash,
    # repartition and reset/consumption state edits, beyond accidental corruption.
    if manifest != build_manifest(manifest.get("schema")):
        raise ValueError("Frozen bank/source mismatch: no mutation, repartition or consumption reset allowed")
    return manifest


def _sidecar(path, suffix):
    return Path(str(Path(path).resolve()) + suffix)


def release_holdout(manifest_path, reason):
    """Explicit future admission, without evaluating any held-out geometry.

    Sidecars are durable local evaluator evidence and must be preserved together
    with the frozen manifest. Copying/deleting them cannot create fresh authority.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Explicit admission reason is required")
    manifest = read_manifest(manifest_path)
    if _sidecar(manifest_path, ".heldout-consumed.json").exists():
        raise ValueError("Held-out split is already consumed")
    receipt = {"schema": manifest["schema"], "manifest_sha256": manifest["manifest_sha256"],
               "state": "ADMITTED_NOT_EVALUATED", "reason": reason.strip()}
    with _sidecar(manifest_path, ".heldout-admission.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def load_scenarios(manifest_path, split="regression", *, allow_held_out=False):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if split == "held_out" and not allow_held_out:
        raise ValueError("held_out is RESERVED_NOT_EVALUATED: explicit later admission required")
    manifest = read_manifest(manifest_path)
    if split != "held_out":
        return copy.deepcopy(manifest["splits"][split])
    admission_path = _sidecar(manifest_path, ".heldout-admission.json")
    if not admission_path.is_file():
        raise ValueError("Held-out admission receipt missing")
    receipt = json.loads(admission_path.read_text(encoding="utf-8"))
    if (receipt.get("manifest_sha256") != manifest["manifest_sha256"] or
            receipt.get("state") != "ADMITTED_NOT_EVALUATED" or not receipt.get("reason")):
        raise ValueError("Invalid held-out admission receipt")
    # Reserve consumption before geometry is exposed, including on interruption.
    # A failed execution stays consumed; it must not silently regain freshness.
    with _sidecar(manifest_path, ".heldout-consumed.json").open("x", encoding="utf-8") as stream:
        json.dump({"state": "CONSUMED_ON_ACCESS", "manifest_sha256": manifest["manifest_sha256"],
                   "admission_sha256": digest(receipt)}, stream, indent=2)
    sources = {s["id"]: s for s in manifest["splits"]["regression"]}
    return [parameterize(sources[r["source_scenario_id"]], **r["parameters"], variant_id=r["id"])
            for r in manifest["splits"]["held_out"]]


def validate_specs(specs):
    rows = []
    for spec in specs:
        hits = proxy_contacts(spec)
        rows.append({"id": spec["id"], "passed": bool(hits) == spec["expected_open_loop_contact"]
            and (not hits or spec["expected_contact_type"] in {h["contact_type"] for h in hits})
            and all(h["actor_id"] != "tactile_ground" for h in hits), "contacts": hits})
    return {"passed": all(r["passed"] for r in rows), "rows": rows,
            "claim": "Declared geometry consistency only; not algorithm performance"}


def validate_manifest(path, split="regression"):
    return dict(validate_specs(load_scenarios(path, split)), split=split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "list", "release-held-out"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=SPLITS, default="regression")
    parser.add_argument("--reason", help="Explicit future held-out admission reason")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_manifest(args.manifest)
        print(json.dumps({"manifest_sha256": result["manifest_sha256"],
                          "counts": {k:len(v) for k,v in result["splits"].items()}}))
    elif args.command == "release-held-out":
        print(json.dumps(release_holdout(args.manifest, args.reason)))
    elif args.command == "validate":
        result = validate_manifest(args.manifest, args.split)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    else:
        print(json.dumps([s["id"] for s in load_scenarios(args.manifest, args.split)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

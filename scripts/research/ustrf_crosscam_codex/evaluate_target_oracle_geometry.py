#!/usr/bin/env python3
"""Evaluate R1.1 target contact points against hash-bound per-frame polygons."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import sha256_file, write_json
    from .diagnostic_contract import ORACLE_SCHEMA, UNCERTAINTY_RATIOS, load_projection, load_target_ledger
    from .projected_corridor_geometry import classify_contact_point, robust_relation
except ImportError:
    from contract import sha256_file, write_json
    from diagnostic_contract import ORACLE_SCHEMA, UNCERTAINTY_RATIOS, load_projection, load_target_ledger
    from projected_corridor_geometry import classify_contact_point, robust_relation


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    ledger = load_target_ledger(args.target_ledger)
    projection = load_projection(args.projection_receipt, args.target_ledger, ledger)
    projection_events = {event["event_id"]: event for event in projection["events"]}
    source_results = []
    for event in ledger["events"]:
        target = event["target_instance"]
        projection_frames = {frame["frame_id"]: frame for frame in projection_events[event["event_id"]]["frames"]}
        frame_results = []
        for frame in target["frames"]:
            if frame["visibility"] != "visible":
                frame_results.append({"frame_id": frame["frame_id"], "timestamp_ms": frame["timestamp_ms"],
                                      "visibility": frame["visibility"], "robust_relation": None,
                                      "expected_relation_match": None})
                continue
            projection_frame = projection_frames[frame["frame_id"]]
            point_px = [frame["contact_xy_norm"][0] * frame["frame_width"],
                        frame["contact_xy_norm"][1] * frame["frame_height"]]
            profiles = [classify_contact_point(point_px, frame_width=frame["frame_width"], frame_height=frame["frame_height"],
                                               polygon_xy_norm=projection_frame["route_polygon_xy_norm"],
                                               uncertainty_frame_ratio=ratio) for ratio in UNCERTAINTY_RATIOS]
            robust = robust_relation([profile.relation for profile in profiles])
            frame_results.append({
                "frame_id": frame["frame_id"], "timestamp_ms": frame["timestamp_ms"], "visibility": "visible",
                "contact_xy_norm": frame["contact_xy_norm"], "route_polygon_xy_norm": projection_frame["route_polygon_xy_norm"],
                "profiles": [{"uncertainty_frame_ratio": ratio, "relation": result.relation,
                              "boundary_distance_px": result.boundary_distance_px,
                              "uncertainty_px": result.uncertainty_px, "nominal_inside": result.nominal_inside}
                             for ratio, result in zip(UNCERTAINTY_RATIOS, profiles)],
                "robust_relation": robust,
                "expected_relation_match": robust == target["expected_route_relation"],
            })
        visible = [frame for frame in frame_results if frame["visibility"] == "visible"]
        passed = bool(visible) and all(frame["expected_relation_match"] is True for frame in visible)
        source_results.append({"event_id": event["event_id"], "source_id": event["source_id"],
                               "target_instance_id": target["target_instance_id"],
                               "expected_route_relation": target["expected_route_relation"],
                               "projection_contract_valid": True, "oracle_geometry_passed": passed,
                               "frames": frame_results})
    report = {
        "schema": ORACLE_SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_set_role": ledger["diagnostic_set_role"],
        "target_ledger_sha256": sha256_file(args.target_ledger),
        "projection_receipt_sha256": sha256_file(args.projection_receipt),
        "uncertainty_frame_ratios": UNCERTAINTY_RATIOS, "sources": source_results,
        "training_performed": False, "held_out_claim_authorized": False,
    }
    write_json(args.output, report)
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ledger", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)); return 2
    print(json.dumps({"ok": True, "passed_sources": sum(row["oracle_geometry_passed"] for row in result["sources"])}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bridge THOR-MAGNI source-native route evidence to D7 review candidates.

The bridge is candidate discovery only.  It binds route-supervision samples to
existing THOR-MAGNI D7 windows by source session and QTM frame range, removes
proxy labels from reviewer-visible evidence, and leaves every candidate in the
NOT_EVALUABLE/assignment-only state until the normal five-role review chain.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


ROUTE_EVIDENCE_SCHEMA = "blindassist_hftf_stage_c_d8_thor_magni_local_route_supervision_v0"


def _load_route_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = load_jsonl(path)
    if not rows:
        raise ContractError(f"route supervision artifact is empty: {path}")
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        if row.get("schema") not in (None, ROUTE_EVIDENCE_SCHEMA):
            raise ContractError(f"unexpected route supervision schema: {row.get('schema')}")
        sample_id = str(row.get("sample_id") or "")
        session = str(row.get("source_session_id") or "")
        if not sample_id or sample_id in seen_ids or not session:
            raise ContractError(f"invalid or duplicate route sample identity: {sample_id}")
        if row.get("authority", {}).get("human_event_truth") is not False:
            raise ContractError(f"route sample is not truth-firewalled: {sample_id}")
        if row.get("authority", {}).get("promotion") is not False:
            raise ContractError(f"route sample is promotion-authorized: {sample_id}")
        target = row.get("target")
        if not isinstance(target, dict):
            raise ContractError(f"route sample lacks source-native target payload: {sample_id}")
        for key in ("future_corridor_intrusion", "future_proximity_le_1_25m"):
            if not isinstance(target.get(key), bool):
                raise ContractError(f"route sample lacks boolean discovery proxy: {sample_id}/{key}")
        seen_ids.add(sample_id)
        by_session[session].append(row)
    return rows, by_session


def _load_thor_frames(path: Path) -> dict[str, dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("dataset_id") != "THOR-MAGNI":
            continue
        frame_id = str(row.get("frame_id") or "")
        if not frame_id:
            raise ContractError("THOR-MAGNI frame row lacks frame_id")
        if frame_id in frames:
            raise ContractError(f"duplicate THOR-MAGNI frame_id: {frame_id}")
        timestamp_ns = row.get("timestamp_ns")
        session = str(row.get("source_session_id") or "")
        if not session or not isinstance(timestamp_ns, int):
            raise ContractError(f"THOR-MAGNI frame lacks session/timestamp: {frame_id}")
        frames[frame_id] = row
    if not frames:
        raise ContractError(f"no THOR-MAGNI frames in frame artifact: {path}")
    return frames


def _sanitized_route_evidence(sample: dict[str, Any]) -> dict[str, Any]:
    target = sample["target"]
    closest = target.get("closest")
    if not isinstance(closest, dict):
        raise ContractError(f"route sample closest geometry is missing: {sample['sample_id']}")
    # Do not carry the derived proxy booleans into reviewer-visible evidence.
    # The geometry reviewer receives only the source-native measurements from
    # which a human can independently reason about the route relation.
    return {
        "sample_id": sample["sample_id"],
        "anchor_scene_frame": sample.get("anchor_scene_frame"),
        "qtm_frame": sample.get("qtm_frame"),
        "qtm_time_seconds": sample.get("qtm_time_seconds"),
        "wearer_speed_mps": target.get("wearer_speed_mps"),
        "future_horizon_seconds": 2.0,
        "future_sample_seconds": 0.10,
        "future_minimum_synchronized_distance_m": target.get(
            "future_minimum_synchronized_distance_m"
        ),
        "closest": {
            key: closest.get(key)
            for key in (
                "body",
                "role",
                "time_offset_seconds",
                "distance_m",
                "longitudinal_m",
                "lateral_m",
            )
        },
        "observed_future_body_time_pairs": target.get(
            "observed_future_body_time_pairs"
        ),
        "occupancy_target": target.get("occupancy_target"),
        "source_native_geometry_only": True,
        "human_event_truth": False,
        "proxy_boolean_fields_withheld": [
            "future_corridor_intrusion",
            "future_proximity_le_1_25m",
        ],
    }


def _bridge_candidate(
    candidate: dict[str, Any],
    route_rows: list[dict[str, Any]],
    frames: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]] | None:
    if candidate.get("dataset_id") != "THOR-MAGNI":
        return None
    session = str(candidate.get("source_session_id") or "")
    metadata = candidate.get("source_metadata")
    if not session or not isinstance(metadata, dict):
        return None
    try:
        qtm_start = int(metadata["qtm_window_start_frame"])
        qtm_end = int(metadata["qtm_window_end_frame"])
    except (KeyError, TypeError, ValueError):
        return None
    if qtm_end < qtm_start:
        raise ContractError(f"non-monotone QTM range: {candidate.get('candidate_id')}")
    frame_ids = [str(value) for value in candidate.get("frame_ids", [])]
    frame_rows = [frames.get(frame_id) for frame_id in frame_ids]
    if not frame_rows or any(row is None for row in frame_rows):
        raise ContractError(f"candidate frame binding is incomplete: {candidate.get('candidate_id')}")
    bound_rows = [row for row in frame_rows if row is not None]
    if any(str(row.get("source_session_id")) != session for row in bound_rows):
        raise ContractError(f"candidate crosses source sessions: {candidate.get('candidate_id')}")
    hits = [
        sample
        for sample in route_rows
        if qtm_start <= int(sample["qtm_frame"]) <= qtm_end
    ]
    if not hits:
        return None
    enriched = copy.deepcopy(candidate)
    enriched["start_timestamp_ns"] = min(int(row["timestamp_ns"]) for row in bound_rows)
    enriched["end_timestamp_ns"] = max(int(row["timestamp_ns"]) for row in bound_rows)
    enriched["source_native_route_evidence"] = [
        _sanitized_route_evidence(sample) for sample in hits
    ]
    enriched["route_evidence_contract"] = {
        "schema": ROUTE_EVIDENCE_SCHEMA,
        "source_native_geometry_only": True,
        "human_event_truth": False,
        "promotion": False,
        "candidate_discovery_only": True,
    }
    enriched["candidate_selection"] = "MODEL_BLIND_SOURCE_NATIVE_ROUTE_GEOMETRY"
    enriched["model_output_visible_to_selector"] = False
    enriched["native_geometry_used_for_selection"] = True
    enriched["event_bucket"] = "NOT_EVALUABLE"
    enriched["truth_status"] = "NOT_EVALUABLE"
    enriched["parent_event_id"] = None
    enriched["parent_independence_status"] = "UNREVIEWED"
    enriched["_route_has_intrusion_proxy"] = any(
        bool(sample["target"]["future_corridor_intrusion"]) for sample in hits
    )
    enriched["_route_has_proximity_proxy"] = any(
        bool(sample["target"]["future_proximity_le_1_25m"]) for sample in hits
    )
    counts = {
        "sample_links": len(hits),
        "intrusion_sample_links": sum(
            bool(sample["target"]["future_corridor_intrusion"]) for sample in hits
        ),
        "proximity_sample_links": sum(
            bool(sample["target"]["future_proximity_le_1_25m"]) for sample in hits
        ),
    }
    return enriched, counts


def _select_round_robin(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(rows):
        return rows
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["source_session_id"])].append(row)
    for session_rows in by_session.values():
        session_rows.sort(
            key=lambda row: (
                not bool(row.get("_route_has_intrusion_proxy")),
                not bool(row.get("_route_has_proximity_proxy")),
                int(row["start_timestamp_ns"]),
                str(row["candidate_id"]),
            )
        )
    sessions = sorted(by_session)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < count:
        advanced = False
        for session in sessions:
            values = by_session[session]
            if index < len(values):
                selected.append(values[index])
                advanced = True
                if len(selected) >= count:
                    break
        if not advanced:
            break
        index += 1
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = Path(args.candidate_artifact).resolve()
    frame_path = Path(args.frame_artifact).resolve()
    route_path = Path(args.route_artifact).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise ContractError(f"refusing to overwrite route bridge: {output_root}")
    for path in (candidate_path, frame_path, route_path):
        if not path.is_file():
            raise ContractError(f"required bridge artifact is missing: {path}")
    route_rows, route_by_session = _load_route_samples(route_path)
    frames = _load_thor_frames(frame_path)
    candidates = load_jsonl(candidate_path)
    full_rows: list[dict[str, Any]] = []
    full_counts = Counter()
    for candidate in candidates:
        session = str(candidate.get("source_session_id") or "")
        result = _bridge_candidate(candidate, route_by_session.get(session, []), frames)
        if result is None:
            continue
        row, counts = result
        full_rows.append(row)
        full_counts["mapped_candidate_rows"] += 1
        full_counts.update(counts)
    if not full_rows:
        raise ContractError("no THOR-MAGNI D7 candidates could be bound to route samples")
    full_rows.sort(key=lambda row: (str(row["source_session_id"]), int(row["start_timestamp_ns"]), str(row["candidate_id"])))
    selected = _select_round_robin(full_rows, args.select_count)
    for row in full_rows:
        row.pop("_route_has_intrusion_proxy", None)
        row.pop("_route_has_proximity_proxy", None)
    # _select_round_robin returns object references, so the private discovery
    # keys have also been removed from selected rows.
    output_root.mkdir(parents=True, exist_ok=False)
    candidate_output = output_root / "candidate_route_bridge.jsonl"
    write_jsonl(candidate_output, selected)
    selected_sessions = {str(row["source_session_id"]) for row in selected}
    report = {
        "schema": "hftf_d7_public_real_thor_magni_route_bridge_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "status": "DEVELOPMENT_ONLY_ROUTE_EVIDENCE_BRIDGED",
        "route_artifact": {"path": str(route_path), "sha256": sha256_file(route_path)},
        "candidate_artifact": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "frame_artifact": {"path": str(frame_path), "sha256": sha256_file(frame_path)},
        "route_sample_count": len(route_rows),
        "route_source_session_count": len(route_by_session),
        "mapped_candidate_count": len(full_rows),
        "selected_candidate_count": len(selected),
        "selected_source_session_count": len(selected_sessions),
        "mapped_sample_links": int(full_counts["sample_links"]),
        "mapped_intrusion_proxy_sample_links": int(full_counts["intrusion_sample_links"]),
        "mapped_proximity_proxy_sample_links": int(full_counts["proximity_sample_links"]),
        "candidate_proxy_fields_withheld_from_reviewers": [
            "future_corridor_intrusion",
            "future_proximity_le_1_25m",
        ],
        "candidate_output": {"path": str(candidate_output), "sha256": sha256_file(candidate_output)},
        "authority": {
            "event_truth": False,
            "human_event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
        },
        "notes": [
            "Route samples are source-native geometric Development evidence only.",
            "Proxy booleans were used only for deterministic discovery priority and are withheld from reviewer-visible geometry.",
            "Every bridged candidate remains NOT_EVALUABLE until the D7 five-role review and final adjudication chain completes.",
            "The bridge does not create parent events, phase truth, or split assignments.",
        ],
    }
    write_json(output_root / "bridge_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-artifact", default=r"F:\ba-data\hftf-d7-public-real\candidates\candidate_index.jsonl")
    parser.add_argument("--frame-artifact", default=r"F:\ba-data\hftf-d7-public-real\canonical\frame_registry.jsonl")
    parser.add_argument("--route-artifact", required=True)
    parser.add_argument("--select-count", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from contract import sha256_file


EXPECTED_RELATIONS = {"inside": 3, "outside": 3}
FORBIDDEN_R11_SOURCE_IDS = {
    "wikimedia_commons_japan_rural_riverside_walk_2025",
    "youtube_cc_edmonton_city_construction_chaos_pov_2025",
    "wikimedia_commons_poptravel_london_westminster_piccadilly_2019",
    "wikimedia_commons_poptravel_ulm_germany",
    "youtube_cc_jakarta_car_free_reopening_2026",
    "youtube_cc_cape_town_waterfront_construction_walk_2026",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate(repo_root: Path, preregistration: Path) -> dict[str, Any]:
    prereg = _load(preregistration)
    if prereg.get("schema") != "blindassist_ustrf_crosscam_geometry_multisource_preregistration_v2":
        raise ValueError("unexpected preregistration schema")
    if prereg.get("dataset_role") != "new_held_out_unscored":
        raise ValueError("R1.2 must remain new_held_out_unscored")

    chronology = prereg["chronology"]
    candidate_path = repo_root / chronology["detector_candidate_contract"]
    if sha256_file(candidate_path) != chronology["detector_candidate_contract_sha256"]:
        raise ValueError("detector candidate contract hash mismatch")
    candidate = _load(candidate_path)
    audit_info = candidate["pre_preregistration_taxonomy_audit"]
    audit_path = repo_root / audit_info["report_path"]
    if sha256_file(audit_path) != audit_info["report_sha256"]:
        raise ValueError("taxonomy audit hash mismatch")
    audit = _load(audit_path)

    classes = set(candidate["model"]["frozen_static_classes"])
    summary = audit["summary"]
    if not audit_info["taxonomy_contract_passed"] or not summary["taxonomy_contract_passed"]:
        raise ValueError("taxonomy contract did not pass")
    if set(summary["candidate_classes"]) != classes or set(summary["matched_labels"]) != classes:
        raise ValueError("all frozen taxonomy labels must be observed in a target match")
    if audit["authority"].get("new_held_out_sources_read") is not False:
        raise ValueError("taxonomy audit must not read replacement held-out sources")
    if _utc(audit_info["completed_at_utc"]) >= _utc(prereg["frozen_at_utc"]):
        raise ValueError("taxonomy audit must precede source preregistration")
    if chronology != {
        "detector_candidate_contract": chronology["detector_candidate_contract"],
        "detector_candidate_contract_sha256": chronology["detector_candidate_contract_sha256"],
        "taxonomy_audit_completed_before_source_preregistration": True,
        "new_source_detector_inference_completed_before_freeze": False,
        "r11_diagnostic_sources_reused": False,
    }:
        raise ValueError("invalid chronology flags")

    events = prereg["held_out_events"]
    if len(events) != 6:
        raise ValueError("exactly six held-out events are required")
    event_ids = {event["event_id"] for event in events}
    source_ids = {event["source_id"] for event in events}
    provider_ids = {event["provider_video_id"] for event in events}
    if min(len(event_ids), len(source_ids), len(provider_ids)) != 6:
        raise ValueError("event, source and provider IDs must each be unique")
    if source_ids & FORBIDDEN_R11_SOURCE_IDS:
        raise ValueError("R1.1 diagnostic source reused")

    relation_counts = {relation: 0 for relation in EXPECTED_RELATIONS}
    for event in events:
        relation = event["expected_route_relation"]
        if relation not in relation_counts:
            raise ValueError(f"unexpected route relation for {event['event_id']}")
        relation_counts[relation] += 1
        if not set(event["detector_label_allowlist"]) <= classes:
            raise ValueError(f"unfrozen detector label for {event['event_id']}")
        video_path = repo_root / event["local_video_path"]
        if "artifacts.local" not in video_path.parts:
            raise ValueError(f"source must remain under artifacts.local: {event['event_id']}")
        if not video_path.is_file() or sha256_file(video_path) != event["video_sha256"]:
            raise ValueError(f"source video missing or hash mismatch: {event['event_id']}")
        start_ms, end_ms = event["window_ms"]
        freeze_ms = event["freeze_frame_ms"]
        duration_ms = event["video_geometry"]["duration_ms"]
        if not 0 <= start_ms <= freeze_ms <= end_ms <= duration_ms:
            raise ValueError(f"invalid window or freeze frame: {event['event_id']}")

    if relation_counts != EXPECTED_RELATIONS:
        raise ValueError("held-out split must contain three inside and three outside sources")
    if any(candidate["hard_stops"].values()):
        raise ValueError("candidate authority hard stops must all remain false")
    prohibited_prereg_authority = (
        "training_authorized",
        "android_runtime_change_authorized",
        "production_model_replacement_authorized",
    )
    if any(prereg["hard_stops"][key] for key in prohibited_prereg_authority):
        raise ValueError("preregistration grants prohibited authority")
    if any(key in prereg for key in ("results", "oracle_output", "detector_output")):
        raise ValueError("unscored preregistration may not contain results")

    return {
        "contract_id": prereg["contract_id"],
        "source_count": len(events),
        "relation_counts": relation_counts,
        "frozen_classes": sorted(classes),
        "taxonomy_visible_frames": summary["visible_frames"],
        "taxonomy_matched_frames": summary["matched_frames"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.repo_root.resolve(), args.preregistration.resolve())
    print(
        "USTRF_R12_PREREG_OK",
        result["source_count"],
        result["relation_counts"]["inside"],
        result["relation_counts"]["outside"],
        ",".join(result["frozen_classes"]),
    )


if __name__ == "__main__":
    main()

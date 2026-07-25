#!/usr/bin/env python3
"""Fail-closed validator for the R0 metadata-only source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = (
    ROOT
    / "artifacts.local"
    / "evidence"
    / "ustrf"
    / "egomotion_compensated_looming_r0"
    / "source_audit"
)
INVENTORY = AUDIT_DIR / "source_authority_inventory_r0.json"
FIREWALL = AUDIT_DIR / "old_window_admission_firewall_r0.json"
PRIORITY_SUMMARY = AUDIT_DIR / "priority_public_source_summary_r0.json"
R0_CLOSURE = AUDIT_DIR.parent / "r0_data_authority_closure_terminal.json"

EXPECTED_TERMINAL = "SOURCE_AUTHORITY_CANDIDATES_PRESENT_CELL_PRESCREEN_REQUIRED"
EXPECTED_ADT_CELL_COUNTS = {
    "PURE_EGO_ROTATION_NO_CLOSING": 0,
    "EGO_APPROACH_STATIC_SURFACE": 5,
    "STATIONARY_EGO_ACTIVE_TARGET_APPROACH": 0,
    "LATERAL_PASS_NO_SUSTAINED_CLOSING": 0,
}
EXPECTED_AV2_TERMINAL = "AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT"
EXPECTED_CODA_TERMINAL = "HOLD_CODA_BOUNDED_PRESCREEN"
EXPECTED_OLD_SOURCE_IDS = {
    "lilocbench_dynamics_0_front",
    "lilocbench_lt_changes_dynamics_0_front",
    "crowdbot_0410_mds",
    "crowdbot_0410_shared_control",
    "crowdbot_0424_shared_control",
    "crowdbot_1203_shared_control",
}


def _load(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"missing receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inventory(receipt: dict) -> None:
    assert receipt["evaluation_stage"] == "metadata_plus_bounded_groundtruth_prescreen"
    assert receipt["terminal"] == EXPECTED_TERMINAL
    assert receipt["candidate_rgb_payload_decoded"] is False
    assert receipt["candidate_signal_computed"] is False
    assert receipt["role_split_frozen"] is False
    assert receipt["source_admission_frozen"] is False
    assert receipt["admitted_real_source_count"] == 0
    assert len(receipt["required_counterfactual_cells"]) == 4

    authority = receipt["authority"]
    assert authority["may_fetch_metadata_or_bounded_preview"] is True
    for forbidden in (
        "may_decode_candidate_rgb_for_signal",
        "may_freeze_discovery_validation_holdout_split",
        "may_run_signal_comparison",
        "may_select_alarm_threshold",
        "may_connect_app_route_or_lifecycle",
    ):
        assert authority[forbidden] is False

    excluded = set(receipt["excluded_prior_source_families"])
    assert {"CROWDBOT", "LILOCBENCH", "JRDB", "THOR", "REVEL"} <= excluded

    real_candidates = receipt["real_source_candidates"]
    assert len(real_candidates) >= receipt["minimum_real_source_families"]
    assert all(candidate["r0_admission"] == "HOLD_R0_ADMISSION" for candidate in real_candidates)
    assert all(candidate["blocking_gaps"] for candidate in real_candidates)
    assert not any(candidate.get("r0_admission") == "ADMITTED" for candidate in real_candidates)
    assert all(
        source["counts_toward_real_source_minimum"] is False
        for source in receipt["diagnostic_only_sources"]
    )
    adt = next(
        candidate
        for candidate in real_candidates
        if candidate["source_id"] == "ARIA_DIGITAL_TWIN"
    )
    assert adt["metadata_probe"] == "PRESCREEN_INSUFFICIENT"

    for candidate in real_candidates:
        for local in candidate.get("local_metadata_receipts", []):
            local_path = AUDIT_DIR / local["path"]
            if "bytes" in local:
                assert local_path.stat().st_size == local["bytes"]
            assert _sha256(local_path) == local["sha256"]
            if "log_count" in local:
                nested = _load(local_path)
                assert nested["log_count"] == local["log_count"]
                assert nested["split_counts"] == local["split_counts"]
                assert nested["payload_downloaded"] is False
            if "sequence_count" in local:
                nested = _load(local_path)
                assert nested["sequence_count"] == local["sequence_count"]
                assert nested["download_entry_count"] == local["download_entry_count"]
                assert nested["payload_requested"] is False
            if "sequence_archive_count" in local:
                nested = _load(local_path)
                assert nested["sequence_archive_count"] == local["sequence_archive_count"]
                assert (
                    nested["capture_grouping"]["unique_capture_date_count"]
                    == local["unique_capture_date_count"]
                )
                assert nested["payload_decoded"] is False
            if "selected_sequence_count" in local:
                nested = _load(local_path)
                assert (
                    nested["selected_sequence_count"]
                    == local["selected_sequence_count"]
                )
                assert (
                    nested["selected_total_groundtruth_bytes"]
                    == local["selected_total_groundtruth_bytes"]
                )
                assert nested["rgb_payload_requested"] is False
            if "member_count" in local:
                nested = _load(local_path)
                assert nested["member_count"] == local["member_count"]
                assert nested["total_bytes"] == local["total_bytes"]
                assert nested["rgb_or_vrs_member_count"] == 0
                assert nested["candidate_signal_computed"] is False
            if local.get("terminal") == "ADT_CELL_PRESCREEN_INSUFFICIENT":
                nested = _load(local_path)
                assert nested["terminal"] == local["terminal"]
                assert nested["status"] == local["status"] == "VALID"
                assert (
                    nested["accepted_eligible_object_proposal_counts"]
                    == local["accepted_eligible_object_proposal_counts"]
                    == EXPECTED_ADT_CELL_COUNTS
                )
                assert nested["candidate_signal_read_count"] == 0
                assert nested["rgb_or_vrs_read_count"] == 0
                assert nested["source_admission"] == "HOLD_R0_ADMISSION"
                assert nested["authority"]["may_expand_adt_rgb_or_run_signal"] is False
            if local.get("terminal") == EXPECTED_AV2_TERMINAL:
                nested = _load(local_path)
                assert nested["terminal"] == EXPECTED_AV2_TERMINAL
                assert nested["status"] == local["status"] == "VALID"
                assert nested["join_audit"]["lidar_anchor_count"] == 3762
                assert nested["join_audit"]["unique_join_within_25ms_count"] == 3761
                assert nested["join_audit"]["tie_count"] == 0
                assert nested["join_audit"]["scope"] == (
                    "LIDAR_FILENAME_TO_CAMERA_FILENAME_ONLY"
                )
                assert nested["official_join_contract"][
                    "annotation_table_timestamp_coverage"
                ] == "NOT_EVALUATED"
                assert nested["geometry_table_preview"][
                    "annotation_to_camera_join_evaluated"
                ] is False
                assert nested["payload_get_count"] == 0
                assert nested["cell_mechanics"][
                    "pure_ego_rotation_no_closing_real_cell"
                ] == "STRUCTURALLY_ABSENT"
                assert nested["capture_authority"]["may_freeze_role_split"] is False
                assert nested["authority"][
                    "may_count_av2_toward_three_real_sources"
                ] is False
            if local.get("terminal") == EXPECTED_CODA_TERMINAL:
                nested = _load(local_path)
                assert nested["terminal"] == EXPECTED_CODA_TERMINAL
                assert nested["status"] == local["status"] == "VALID"
                bbox = nested["tacc_tiny"]["bbox_availability"]
                cam0 = nested["tacc_tiny"]["cam0_availability"]
                assert bbox["maximum_consecutive_frame_run"] == 3
                assert cam0["maximum_consecutive_frame_run"] == 3
                assert bbox[
                    "sequence_count_with_at_least_100_consecutive_frames"
                ] == 0
                assert cam0[
                    "sequence_count_with_at_least_100_consecutive_frames"
                ] == 0
                assert nested["payload_member_extraction_count"] == 0
                assert nested["tdr_tiny"]["metadata_snapshot_sha256"] == (
                    "016a860feb463fe18844d038180e44e88b811a8c9ee8674741f7ff88dc07060d"
                )
                assert nested["binding"]["tacc_tiny_continuity_applies_to_tdr_tiny"] is False
                proofs = nested["tacc_tiny"]["range_response_proofs"]
                assert len(proofs) == 3
                assert all(item["status"] == 206 for item in proofs)
                assert sum(item["bytes"] for item in proofs) == nested["tacc_tiny"][
                    "http_body_bytes_read"
                ] == 994820
                assert nested["authority"]["may_extract_tacc_members"] is False
                assert nested["authority"][
                    "may_count_coda_toward_three_real_sources"
                ] is False


def validate_firewall(receipt: dict) -> None:
    assert receipt["firewall_mode"] == "DENYLIST_ONLY_NO_OUTCOME_EXPOSURE"
    assert receipt["old_positive_window_count"] == 15
    assert receipt["old_negative_window_count"] == 15
    assert receipt["old_unique_frame_count"] == 4594
    assert receipt["old_window_source_ids"] == [
        "lilocbench_dynamics_0_front",
        "lilocbench_lt_changes_dynamics_0_front",
    ]
    assert len(receipt["old_canonical_crowdbot_source_ids"]) == 4
    assert set(receipt["deny_source_ids"]) == EXPECTED_OLD_SOURCE_IDS
    assert receipt["producer_visibility"]["old_frames"] is False
    assert receipt["producer_visibility"]["old_outcomes"] is False
    assert receipt["producer_visibility"]["old_thresholds"] is False
    assert receipt["producer_visibility"]["deny_receipt_only"] is True
    assert receipt["terminal"] == "OLD_WINDOW_ADMISSION_FIREWALL_READY"

    for source in receipt["authoritative_inputs"]:
        path = ROOT / source["path"]
        assert path.is_file(), path
        assert _sha256(path) == source["sha256"]


def validate_priority_summary(receipt: dict) -> None:
    assert receipt["artifact_role"] == "NON_TERMINAL_SOURCE_AUDIT_BOUNDARY_SUMMARY"
    assert receipt["parent_r0_execution_status"] == "NOT_EXECUTED"
    assert receipt["admitted_real_source_count"] == 0
    assert receipt["candidate_signal_computed"] is False
    assert receipt["authority"]["may_treat_summary_as_parent_r0_terminal"] is False
    assert receipt["authority"]["may_run_signal_comparison"] is False
    for source in receipt["source_boundary_receipts"]:
        path = AUDIT_DIR / source["receipt"]
        assert _sha256(path) == source["sha256"]
        nested = _load(path)
        assert nested["terminal"] == source["source_boundary_terminal"]
        assert nested["status"] == source["source_boundary_status"] == "VALID"


def validate_r0_closure(receipt: dict) -> None:
    assert receipt["terminal"] == "FAIL_CLOSED_NEW_DATA_OR_TRUTH_AUTHORITY_BLOCKED"
    assert receipt["status"] == "VALID"
    assert receipt["admitted_real_source_count"] == 0
    assert receipt["signal_arm_run_count"] == 0
    assert receipt["role_split_frozen"] is False
    summary_path = R0_CLOSURE.parent / receipt["source_boundary_summary"]["path"]
    assert _sha256(summary_path) == receipt["source_boundary_summary"]["sha256"]
    assert receipt["source_boundary_summary"]["artifact_role"] == (
        "NON_TERMINAL_SOURCE_AUDIT_BOUNDARY_SUMMARY"
    )
    assert receipt["authority"]["may_modify_r0_frozen_denominators"] is False
    assert receipt["authority"]["may_run_r0_signal_comparison"] is False


def main() -> None:
    inventory = _load(INVENTORY)
    firewall = _load(FIREWALL)
    priority_summary = _load(PRIORITY_SUMMARY)
    r0_closure = _load(R0_CLOSURE)
    validate_inventory(inventory)
    validate_firewall(firewall)
    validate_priority_summary(priority_summary)
    validate_r0_closure(r0_closure)
    print(
        json.dumps(
            {
                "status": "VALID",
                "terminal": r0_closure["terminal"],
                "admitted_real_source_count": inventory["admitted_real_source_count"],
                "old_window_firewall": firewall["terminal"],
                "priority_source_summary": priority_summary["boundary_summary_code"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

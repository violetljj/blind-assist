from __future__ import annotations

"""Render two separately shuffled, causal RGB-only P0 reviewer packets.

The command intentionally requires a passed admission receipt.  It never reads
source masks while rendering, and one opaque review item corresponds to one
anchor only, so later frames from the same screening event are not exposed in
the same item.
"""

import argparse
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Callable

from .common import ADMISSION_PASSED_STATUSES, ADMISSION_RECONCILIATION_SCHEMA, PROTOCOL_ID, read_json, sha256_file, sha256_json
from .freeze_screening_cohort import SCHEMA as SCREENING_COHORT_SCHEMA
from .materialize_screening_inputs import MATERIALIZED_SCHEMA


PACKET_SCHEMA = "blindassist.eval_validity_r0.p0_causal_rgb_packet.v1"
PRIVATE_MAP_SCHEMA = "blindassist.eval_validity_r0.p0_private_review_map.v1"
ADMISSION_STATUS = "EVAL_VALIDITY_DATA_ADMISSION_PASSED"


class P0PacketError(ValueError):
    """Raised on a disclosure, provenance, or output-path violation."""


def _ensure_distinct_output_roots(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise P0PacketError("packet output roots must be distinct")
    for left in resolved:
        for right in resolved:
            if left != right and left in right.parents:
                raise P0PacketError("packet output roots must not contain one another")
    if any(path.exists() for path in resolved):
        raise P0PacketError("refusing to overwrite an existing reviewer/private output root")


def _validate_inputs(cohort: dict[str, Any], admission: dict[str, Any], materialized_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if cohort.get("schema_version") != SCREENING_COHORT_SCHEMA or cohort.get("protocol_id") != PROTOCOL_ID:
        raise P0PacketError("screening cohort schema/protocol mismatch")
    if cohort.get("status") != "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN":
        raise P0PacketError("screening cohort continuous windows are not frozen")
    if cohort.get("candidate_outputs_opened") is not False or cohort.get("final_event_facts_frozen") is not False:
        raise P0PacketError("screening cohort output/event fact state is invalid")
    manifest_path = materialized_root / "manifest.json"
    if not manifest_path.is_file():
        raise P0PacketError("materialized native manifest is missing")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != MATERIALIZED_SCHEMA or manifest.get("protocol_id") != PROTOCOL_ID:
        raise P0PacketError("materialized native manifest schema/protocol mismatch")
    if manifest.get("candidate_outputs_opened") is not False:
        raise P0PacketError("materialized native manifest output state is invalid")
    cohort_hash, manifest_hash = sha256_json(cohort), sha256_file(manifest_path)
    if manifest.get("screening_cohort_sha256") != cohort_hash:
        raise P0PacketError("materialized native manifest cohort binding mismatch")
    admission_status = admission.get("status")
    if (
        admission.get("protocol_id") != PROTOCOL_ID
        or admission_status not in ADMISSION_PASSED_STATUSES
        or admission.get("screening_cohort_sha256") != cohort_hash
        or admission.get("materialized_manifest_sha256") != manifest_hash
        or admission.get("candidate_outputs_opened") is not False
    ):
        raise P0PacketError("admission receipt does not authorize reviewer packets")
    if admission_status == "EVAL_VALIDITY_DATA_ADMISSION_PASSED":
        if admission.get("schema_version") != "blindassist.eval_validity_r0.data_admission_receipt.v1":
            raise P0PacketError("direct admission receipt schema mismatch")
    else:
        checks, evidence = admission.get("checks"), admission.get("evidence")
        if (
            admission.get("schema_version") != ADMISSION_RECONCILIATION_SCHEMA
            or not isinstance(checks, dict)
            or checks.get("p_hash_manual_all_cases_distinct") is not True
            or not isinstance(evidence, dict)
            or not isinstance(evidence.get("held_admission_receipt_sha256"), str)
            or not isinstance(evidence.get("p_hash_manual_resolution_sha256"), str)
        ):
            raise P0PacketError("manual pHash admission reconciliation is malformed")
    cohort_items = cohort.get("items")
    materialized_items = manifest.get("items")
    if (
        not isinstance(cohort_items, list)
        or not isinstance(materialized_items, list)
        or len(cohort_items) != 48
        or len(materialized_items) != 48
        or len(cohort_items) != len(materialized_items)
    ):
        raise P0PacketError("screening/materialized coverage mismatch")
    source_by_event: dict[str, dict[str, Any]] = {}
    for row in materialized_items:
        event_id = row.get("screening_event_id") if isinstance(row, dict) else None
        if not isinstance(event_id, str) or event_id in source_by_event:
            raise P0PacketError("materialized event identity is invalid")
        source_by_event[event_id] = row
    cohort_by_event: dict[str, dict[str, Any]] = {}
    for item in cohort_items:
        event_id = item.get("screening_event_id") if isinstance(item, dict) else None
        if not isinstance(event_id, str) or event_id not in source_by_event or event_id in cohort_by_event:
            raise P0PacketError("screening event identity is invalid")
        cohort_by_event[event_id] = item
        source = source_by_event[event_id]
        if source.get("source_session_id") != item.get("source_session_id"):
            raise P0PacketError("materialized source session does not match screening cohort")
        frames = source.get("frames")
        window = item.get("source_window")
        if not isinstance(frames, list) or not isinstance(window, dict) or len(frames) != window.get("frame_count"):
            raise P0PacketError("materialized frame coverage mismatch")
        if [row.get("ordinal") for row in frames] != list(range(len(frames))):
            raise P0PacketError("materialized frame ordinals are not contiguous")
    return manifest, cohort_by_event


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _render_reviewer_packet(
    *,
    role: str,
    cohort_by_event: dict[str, dict[str, Any]],
    screening_cohort_sha256: str,
    source_items: dict[str, dict[str, Any]],
    materialized_root: Path,
    output_root: Path,
    make_opaque_id: Callable[[], str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    staged = output_root.with_name(f".{output_root.name}.{secrets.token_hex(8)}.staging")
    try:
        packet_items: list[dict[str, Any]] = []
        private_map: dict[str, dict[str, Any]] = {}
        for event_id, cohort_item in cohort_by_event.items():
            source_item = source_items[event_id]
            window = cohort_item["source_window"]
            frames = source_item["frames"]
            for anchor_ordinal in window["p0_anchor_offsets"]:
                opaque_id = make_opaque_id()
                while opaque_id in private_map:
                    opaque_id = make_opaque_id()
                asset_paths: list[str] = []
                for frame in frames[: anchor_ordinal + 1]:
                    source = materialized_root / str(frame["rgb_path"])
                    if not source.is_file() or sha256_file(source) != frame.get("rgb_sha256"):
                        raise P0PacketError("materialized RGB payload is missing or hash-mismatched")
                    target = staged / "assets" / opaque_id / f"{int(frame['ordinal']):03d}.png"
                    _link_or_copy(source, target)
                    asset_paths.append(target.relative_to(staged).as_posix())
                packet_items.append({
                    "review_item_id": opaque_id,
                    "current_frame_ordinal": anchor_ordinal,
                    "causal_rgb_frames": asset_paths,
                    "response_fields": {
                        "reminder_now": ["YES", "NO", "UNKNOWN"],
                        "cleared": ["YES", "NO", "UNKNOWN"],
                        "knownness": ["KNOWN", "UNKNOWN"],
                    },
                })
                private_map[opaque_id] = {
                    "screening_event_id": event_id,
                    "anchor_frame_index": anchor_ordinal,
                    "source_session_id": cohort_item["source_session_id"],
                }
        # Randomized opaque order is reviewer-specific and has no event grouping metadata.
        secrets.SystemRandom().shuffle(packet_items)
        packet = {
            "schema_version": PACKET_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "reviewer_role": role,
            "status": "P0_CAUSAL_RGB_REVIEW_PENDING",
            "disclosures": {
                "model_or_oracle_output_visible": False,
                "source_mask_visible": False,
                "source_session_or_event_identity_visible": False,
                "screening_stratum_or_bucket_visible": False,
                "other_reviewer_visible": False,
                "one_item_per_anchor": True,
                "future_frames_visible_in_item": False,
            },
            "items": packet_items,
            "submission_shape": {
                "schema_version": "blindassist.eval_validity_r0.action_review.v2",
                "protocol_id": PROTOCOL_ID,
                "reviewer_role": role,
                "screening_cohort_sha256": screening_cohort_sha256,
                "isolated_context": True,
                "other_review_visible_before_submission": False,
                "model_or_oracle_output_visible": False,
                "items": [{
                    "review_item_id": "copy from packet item",
                    "anchor": {"frame_index": "copy current_frame_ordinal", "reminder_now": "YES|NO|UNKNOWN", "cleared": "YES|NO|UNKNOWN", "knownness": "KNOWN|UNKNOWN"},
                }],
            },
        }
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, output_root)
        return packet, private_map
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def prepare_packets(
    *, cohort: dict[str, Any], admission: dict[str, Any], materialized_root: Path,
    reviewer_a_root: Path, reviewer_b_root: Path, private_root: Path,
) -> dict[str, Any]:
    _ensure_distinct_output_roots(reviewer_a_root, reviewer_b_root, private_root)
    manifest, cohort_by_event = _validate_inputs(cohort, admission, materialized_root)
    source_items = {item["screening_event_id"]: item for item in manifest["items"]}
    try:
        packet_a, map_a = _render_reviewer_packet(
            role="ACTION_REVIEW_A", cohort_by_event=cohort_by_event, source_items=source_items,
            screening_cohort_sha256=sha256_json(cohort),
            materialized_root=materialized_root, output_root=reviewer_a_root, make_opaque_id=lambda: f"a-{secrets.token_urlsafe(16)}",
        )
        packet_b, map_b = _render_reviewer_packet(
            role="ACTION_REVIEW_B", cohort_by_event=cohort_by_event, source_items=source_items,
            screening_cohort_sha256=sha256_json(cohort),
            materialized_root=materialized_root, output_root=reviewer_b_root, make_opaque_id=lambda: f"b-{secrets.token_urlsafe(16)}",
        )
        private_root.mkdir(parents=True, exist_ok=False)
        private = {
            "schema_version": PRIVATE_MAP_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "PRIVATE_REVIEW_MAP_FROZEN_BEFORE_SUBMISSIONS",
            "screening_cohort_sha256": sha256_json(cohort),
            "admission_receipt_sha256": sha256_json(admission),
            "packet_a_sha256": sha256_file(reviewer_a_root / "packet.json"),
            "packet_b_sha256": sha256_file(reviewer_b_root / "packet.json"),
            "reviewer_a_map": map_a,
            "reviewer_b_map": map_b,
            "sharing_rule": "Never disclose this directory or either map to a reviewer. Each reviewer receives only their own packet root.",
        }
        (private_root / "private-review-map.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "packet_a_item_count": len(packet_a["items"]), "packet_b_item_count": len(packet_b["items"]),
            "private_map": private,
        }
    except Exception:
        for path in (reviewer_a_root, reviewer_b_root, private_root):
            shutil.rmtree(path, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--reviewer-a-root", type=Path, required=True)
    parser.add_argument("--reviewer-b-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_packets(
        cohort=read_json(args.screening_cohort), admission=read_json(args.admission_receipt), materialized_root=args.materialized_root,
        reviewer_a_root=args.reviewer_a_root, reviewer_b_root=args.reviewer_b_root, private_root=args.private_root,
    )
    print(f"reviewer_a_items={result['packet_a_item_count']} reviewer_b_items={result['packet_b_item_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

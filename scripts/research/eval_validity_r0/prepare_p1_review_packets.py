from __future__ import annotations

"""Render the two fresh, full-event, RGB-only P1 review packets.

P1 is strictly downstream of passed P0 anchor consistency.  It exposes every
frame of each already-frozen native RGB sequence in causal order but never
opens source masks or any model/oracle/feedback output.  The two packets have
separate opaque identities and shuffled event order.
"""

import argparse
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Callable

from .common import P0_ANCHOR_AGREEMENT_SCHEMA, P1_ACTION_REVIEW_SCHEMA, PROTOCOL_ID, read_json, sha256_file, sha256_json
from .finalize_p0_anchor_agreement import P0_PASSED_STATUS
from .prepare_p0_review_packets import ADMISSION_STATUS, _ensure_distinct_output_roots, _link_or_copy, _validate_inputs


PACKET_SCHEMA = "blindassist.eval_validity_r0.p1_full_event_causal_rgb_packet.v1"
PRIVATE_MAP_SCHEMA = "blindassist.eval_validity_r0.p1_private_review_map.v1"


class P1PacketError(ValueError):
    """Raised for an invalid P0/data binding or a review-packet disclosure risk."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P1PacketError(message)


def _validate_p0(p0: dict[str, Any], *, cohort_sha256: str, admission_sha256: str) -> str:
    _require(p0.get("schema_version") == P0_ANCHOR_AGREEMENT_SCHEMA and p0.get("protocol_id") == PROTOCOL_ID, "P0 receipt schema/protocol mismatch")
    _require(p0.get("status") == P0_PASSED_STATUS, "P0 actionability consistency did not pass")
    _require(p0.get("screening_cohort_sha256") == cohort_sha256, "P0 screening cohort binding mismatch")
    _require(p0.get("admission_receipt_sha256") == admission_sha256, "P0 data admission binding mismatch")
    _require(p0.get("candidate_outputs_opened") is False, "P0 receipt records forbidden output access")
    agreement = p0.get("anchor_agreement")
    _require(isinstance(agreement, dict) and agreement.get("passed") is True, "P0 receipt has no passing anchor agreement")
    return sha256_json(p0)


def _render_packet(
    *, role: str, cohort_by_event: dict[str, dict[str, Any]], source_items: dict[str, dict[str, Any]],
    screening_cohort_sha256: str, p0_anchor_agreement_sha256: str,
    materialized_root: Path, output_root: Path, make_opaque_id: Callable[[], str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    staged = output_root.with_name(f".{output_root.name}.{secrets.token_hex(8)}.staging")
    try:
        packet_items: list[dict[str, Any]] = []
        private_map: dict[str, dict[str, Any]] = {}
        for event_id, cohort_item in cohort_by_event.items():
            source_item = source_items[event_id]
            frames = source_item["frames"]
            window = cohort_item["source_window"]
            _require(len(frames) == window["frame_count"], f"{event_id}: materialized full-event frame count mismatch")
            opaque_id = make_opaque_id()
            while opaque_id in private_map:
                opaque_id = make_opaque_id()
            assets: list[str] = []
            for frame in frames:
                source = materialized_root / str(frame["rgb_path"])
                _require(source.is_file() and sha256_file(source) == frame.get("rgb_sha256"), f"{event_id}: materialized RGB is missing or hash-mismatched")
                target = staged / "assets" / opaque_id / f"{int(frame['ordinal']):03d}.png"
                _link_or_copy(source, target)
                assets.append(target.relative_to(staged).as_posix())
            packet_items.append({
                "review_item_id": opaque_id,
                "frame_count": len(frames),
                "causal_rgb_frames": assets,
                "response_fields": {
                    "knownness": ["KNOWN", "UNKNOWN"],
                    "reminder_now_interval": "[first_frame,last_frame] or null",
                    "cleared_interval": "[first_frame,last_frame] or null",
                },
            })
            private_map[opaque_id] = {
                "screening_event_id": event_id,
                "source_session_id": cohort_item["source_session_id"],
                "frame_count": len(frames),
            }
        secrets.SystemRandom().shuffle(packet_items)
        packet = {
            "schema_version": PACKET_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "reviewer_role": role,
            "status": "P1_FULL_EVENT_CAUSAL_RGB_REVIEW_PENDING",
            "disclosures": {
                "model_or_oracle_output_visible": False,
                "source_mask_visible": False,
                "source_session_or_event_identity_visible": False,
                "screening_stratum_or_bucket_visible": False,
                "p0_review_or_consensus_visible": False,
                "other_reviewer_visible": False,
                "all_event_frames_visible_in_causal_order": True,
            },
            "items": packet_items,
            "submission_shape": {
                "schema_version": P1_ACTION_REVIEW_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "reviewer_role": role,
                "screening_cohort_sha256": screening_cohort_sha256,
                "p0_anchor_agreement_sha256": p0_anchor_agreement_sha256,
                "isolated_context": True,
                "reviewer_is_not_a_p0_reviewer": True,
                "other_review_visible_before_submission": False,
                "model_or_oracle_output_visible": False,
                "items": [{
                    "review_item_id": "copy from packet item",
                    "event_fact": {"knownness": "KNOWN|UNKNOWN", "reminder_now_interval": "[first,last]|null", "cleared_interval": "[first,last]|null"},
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
    *, cohort: dict[str, Any], admission: dict[str, Any], p0_agreement: dict[str, Any], materialized_root: Path,
    reviewer_a_root: Path, reviewer_b_root: Path, private_root: Path,
) -> dict[str, Any]:
    _ensure_distinct_output_roots(reviewer_a_root, reviewer_b_root, private_root)
    manifest, cohort_by_event = _validate_inputs(cohort, admission, materialized_root)
    cohort_sha256, admission_sha256 = sha256_json(cohort), sha256_json(admission)
    _require(admission.get("status") == ADMISSION_STATUS, "data admission did not pass")
    p0_sha256 = _validate_p0(p0_agreement, cohort_sha256=cohort_sha256, admission_sha256=admission_sha256)
    source_items = {item["screening_event_id"]: item for item in manifest["items"]}
    try:
        packet_a, map_a = _render_packet(
            role="P1_FULL_EVENT_REVIEW_A", cohort_by_event=cohort_by_event, source_items=source_items,
            screening_cohort_sha256=cohort_sha256, p0_anchor_agreement_sha256=p0_sha256,
            materialized_root=materialized_root, output_root=reviewer_a_root, make_opaque_id=lambda: f"p1a-{secrets.token_urlsafe(16)}",
        )
        packet_b, map_b = _render_packet(
            role="P1_FULL_EVENT_REVIEW_B", cohort_by_event=cohort_by_event, source_items=source_items,
            screening_cohort_sha256=cohort_sha256, p0_anchor_agreement_sha256=p0_sha256,
            materialized_root=materialized_root, output_root=reviewer_b_root, make_opaque_id=lambda: f"p1b-{secrets.token_urlsafe(16)}",
        )
        private_root.mkdir(parents=True, exist_ok=False)
        private = {
            "schema_version": PRIVATE_MAP_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "P1_PRIVATE_REVIEW_MAP_FROZEN_BEFORE_SUBMISSIONS",
            "screening_cohort_sha256": cohort_sha256,
            "admission_receipt_sha256": admission_sha256,
            "p0_anchor_agreement_sha256": p0_sha256,
            "packet_a_sha256": sha256_file(reviewer_a_root / "packet.json"),
            "packet_b_sha256": sha256_file(reviewer_b_root / "packet.json"),
            "reviewer_a_map": map_a,
            "reviewer_b_map": map_b,
            "sharing_rule": "Never disclose this directory, either map, or the P0 receipt to a P1 reviewer. Each reviewer receives only their own packet root.",
        }
        (private_root / "private-review-map.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"packet_a_item_count": len(packet_a["items"]), "packet_b_item_count": len(packet_b["items"]), "private_map": private}
    except Exception:
        for path in (reviewer_a_root, reviewer_b_root, private_root):
            shutil.rmtree(path, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--p0-agreement", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--reviewer-a-root", type=Path, required=True)
    parser.add_argument("--reviewer-b-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_packets(
        cohort=read_json(args.screening_cohort), admission=read_json(args.admission_receipt), p0_agreement=read_json(args.p0_agreement),
        materialized_root=args.materialized_root, reviewer_a_root=args.reviewer_a_root, reviewer_b_root=args.reviewer_b_root, private_root=args.private_root,
    )
    print(f"reviewer_a_items={result['packet_a_item_count']} reviewer_b_items={result['packet_b_item_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

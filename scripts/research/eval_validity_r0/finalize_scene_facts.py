from __future__ import annotations

"""Freeze deterministic source-mask scene facts after both action-review gates.

This merger is intentionally downstream of P0 and P1.  It opens only native
source masks that were already admitted, converts fixed SANPO panoptic labels
to canonical truth-mask/box facts, and applies a predeclared 12-12-12-12
bucket rule.  It never reads or materializes model, truth-adapter, oracle, or
feedback traces.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import COHORT_SCHEMA, FULL_EVENT_FACTS_SCHEMA, P0_ANCHOR_AGREEMENT_SCHEMA, P1_ACTION_FACTS_SCHEMA, PROTOCOL_ID, sha256_file, sha256_json
from .finalize_p0_anchor_agreement import P0_PASSED_STATUS, _screening_index, _validate_admission
from .finalize_p1_action_facts import P1_PASSED_STATUS
from .materialize_screening_inputs import MATERIALIZED_SCHEMA


SCENE_FACTS_SCHEMA = "blindassist.eval_validity_r0.source_scene_facts.v1"
RECEIPT_SCHEMA = "blindassist.eval_validity_r0.scene_event_finalization_receipt.v1"
FINAL_STATUS = "SCENE_AND_EVENT_FACTS_FROZEN_AFTER_P0_P1"
HOLD_STATUS = "HOLD_EVAL_VALIDITY_DATA"
WALKABLE_IDS = {1, 3, 5, 6, 17}
BOUNDARY_IDS = {2, 15}
OBSTACLE_IDS = {18, 20, 21, 24}
CANONICAL_IDS = BOUNDARY_IDS | OBSTACLE_IDS
EXPECTED_BUCKET_COUNTS = {
    "blocking_obstacle_positive": 12,
    "boundary_level_change_positive": 12,
    "parallel_curb_negative": 12,
    "normal_walkable_negative": 12,
}


class SceneFactFinalizationError(ValueError):
    """Raised for an invalid lineage, malformed native asset, or unsafe state."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SceneFactFinalizationError(message)


def _p0_sha(p0: dict[str, Any], *, screening_sha: str, admission_sha: str) -> str:
    _require(p0.get("schema_version") == P0_ANCHOR_AGREEMENT_SCHEMA and p0.get("protocol_id") == PROTOCOL_ID, "P0 receipt schema/protocol mismatch")
    _require(p0.get("status") == P0_PASSED_STATUS and p0.get("candidate_outputs_opened") is False, "P0 receipt did not pass cleanly")
    _require(p0.get("screening_cohort_sha256") == screening_sha and p0.get("admission_receipt_sha256") == admission_sha, "P0 receipt lineage binding mismatch")
    agreement = p0.get("anchor_agreement")
    _require(isinstance(agreement, dict) and agreement.get("passed") is True, "P0 agreement not passed")
    return sha256_json(p0)


def _p1_index(p1: dict[str, Any], *, cohort: dict[str, dict[str, Any]], screening_sha: str, admission_sha: str, p0_sha: str) -> dict[str, dict[str, Any]]:
    _require(p1.get("schema_version") == P1_ACTION_FACTS_SCHEMA and p1.get("protocol_id") == PROTOCOL_ID, "P1 action-facts schema/protocol mismatch")
    _require(p1.get("status") == P1_PASSED_STATUS and p1.get("candidate_outputs_opened") is False, "P1 action facts did not pass cleanly")
    _require(p1.get("screening_cohort_sha256") == screening_sha and p1.get("admission_receipt_sha256") == admission_sha and p1.get("p0_anchor_agreement_sha256") == p0_sha, "P1 action-facts lineage binding mismatch")
    evidence = p1.get("independent_full_review_evidence")
    _require(isinstance(evidence, dict) and evidence.get("agreement_passed") is True and evidence.get("unknown_or_disagreement_event_count") == 0, "P1 evidence is unresolved")
    rows = p1.get("items")
    _require(isinstance(rows, list) and len(rows) == len(cohort), "P1 action-facts coverage mismatch")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "P1 action-fact item malformed")
        event_id = row.get("screening_event_id")
        _require(event_id in cohort and event_id not in result and row.get("resolved") is True and row.get("p0_anchor_compatible") is True, "P1 action-fact item is not resolved")
        frames = cohort[event_id]["source_window"]["frame_count"]
        alertable, passed = row.get("alertable_interval_frames"), row.get("passed_interval_frames")
        if alertable is None:
            _require(passed is None, f"{event_id}: no-reminder event has passed interval")
        else:
            _require(isinstance(alertable, list) and len(alertable) == 2 and all(isinstance(value, int) for value in alertable), f"{event_id}: invalid alertable interval")
            _require(isinstance(passed, list) and len(passed) == 2 and all(isinstance(value, int) for value in passed), f"{event_id}: invalid passed interval")
            _require(0 <= alertable[0] <= alertable[1] < frames and 0 <= passed[0] <= passed[1] < frames and passed[0] > alertable[1], f"{event_id}: invalid temporal ordering")
        result[event_id] = {"alertable_interval_frames": alertable, "passed_interval_frames": passed}
    _require(set(result) == set(cohort), "P1 action-facts event coverage mismatch")
    return result


def _manifest_index(manifest: dict[str, Any], *, cohort: dict[str, dict[str, Any]], screening_sha: str, materialized_root: Path) -> dict[str, dict[str, Any]]:
    _require(manifest.get("schema_version") == MATERIALIZED_SCHEMA and manifest.get("protocol_id") == PROTOCOL_ID, "materialized manifest schema/protocol mismatch")
    _require(manifest.get("status") == "CONTINUOUS_NATIVE_RGB_AND_MASKS_MATERIALIZED_OUTPUT_BLIND" and manifest.get("candidate_outputs_opened") is False, "materialized manifest state is invalid")
    _require(manifest.get("screening_cohort_sha256") == screening_sha, "materialized manifest cohort binding mismatch")
    rows = manifest.get("items")
    _require(isinstance(rows, list) and len(rows) == len(cohort), "materialized manifest coverage mismatch")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "materialized manifest item malformed")
        event_id = row.get("screening_event_id")
        _require(event_id in cohort and event_id not in result and row.get("source_session_id") == cohort[event_id]["source_session_id"], "materialized manifest event/session mismatch")
        frames = row.get("frames")
        expected_count = cohort[event_id]["source_window"]["frame_count"]
        _require(isinstance(frames, list) and len(frames) == expected_count, f"{event_id}: materialized frame count mismatch")
        _require([frame.get("ordinal") for frame in frames] == list(range(expected_count)), f"{event_id}: materialized ordinal mismatch")
        for frame in frames:
            _require(isinstance(frame, dict), f"{event_id}: malformed materialized frame")
            mask = materialized_root / str(frame.get("source_mask_path", ""))
            _require(mask.is_file() and sha256_file(mask) == frame.get("source_mask_sha256"), f"{event_id}: source-mask payload is missing or hash-mismatched")
        result[event_id] = row
    _require(set(result) == set(cohort), "materialized manifest event coverage mismatch")
    return result


def _bucket_for(item: dict[str, Any], p1_fact: dict[str, Any]) -> str | None:
    profile = item.get("source_selection_profile")
    alertable = p1_fact["alertable_interval_frames"]
    if profile == "center_obstacle":
        return "blocking_obstacle_positive" if alertable is not None else None
    if profile == "step_curb":
        return "boundary_level_change_positive" if alertable is not None else "parallel_curb_negative"
    if profile == "strict_normal_walkable_source_mask_only":
        return "normal_walkable_negative" if alertable is None else None
    return None


def _classify_buckets(cohort: dict[str, dict[str, Any]], p1: dict[str, dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    buckets: dict[str, str] = {}
    reasons: list[str] = []
    for event_id, item in sorted(cohort.items()):
        bucket = _bucket_for(item, p1[event_id])
        if bucket is None:
            reasons.append(f"profile_action_mismatch:{event_id}:{item.get('source_selection_profile')}")
        else:
            buckets[event_id] = bucket
    counts = Counter(buckets.values())
    for bucket, expected in EXPECTED_BUCKET_COUNTS.items():
        if counts[bucket] != expected:
            reasons.append(f"bucket_count:{bucket}:{counts[bucket]}!= {expected}")
    return buckets, reasons


def _component_rows(ids: np.ndarray, canonical: np.ndarray) -> list[dict[str, Any]]:
    height, width = ids.shape[:2]
    instance = ids[:, :, 1].astype(np.uint32) * 256 + ids[:, :, 2].astype(np.uint32)
    classes = ids[:, :, 0].astype(np.uint32)
    positions = np.flatnonzero(canonical.reshape(-1))
    if not positions.size:
        return []
    flat_class, flat_instance = classes.reshape(-1)[positions], instance.reshape(-1)[positions]
    keys = flat_class * np.uint32(65536) + flat_instance
    order = np.argsort(keys, kind="stable")
    keys, positions = keys[order], positions[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
    rows: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        class_id, instance_id = divmod(int(keys[start]), 65536)
        ys, xs = divmod(positions[start:end], width)
        rows.append({
            "source_class_id": class_id,
            "canonical_kind": "boundary_level_change" if class_id in BOUNDARY_IDS else "blocking_obstacle",
            "source_instance_id": instance_id,
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "area_px": int(end - start),
        })
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finalize_scene_facts(
    *, screening_cohort: dict[str, Any], admission_receipt: dict[str, Any], p0_agreement: dict[str, Any], p1_action_facts: dict[str, Any],
    manifest: dict[str, Any], materialized_root: Path, output_root: Path,
) -> dict[str, Any]:
    try:
        cohort, screening_sha = _screening_index(screening_cohort)
    except ValueError as error:
        raise SceneFactFinalizationError(str(error)) from error
    try:
        admission_sha = _validate_admission(admission_receipt, screening_sha)
    except ValueError as error:
        raise SceneFactFinalizationError(str(error)) from error
    p0_sha = _p0_sha(p0_agreement, screening_sha=screening_sha, admission_sha=admission_sha)
    p1 = _p1_index(p1_action_facts, cohort=cohort, screening_sha=screening_sha, admission_sha=admission_sha, p0_sha=p0_sha)
    buckets, hold_reasons = _classify_buckets(cohort, p1)
    if output_root.exists():
        raise SceneFactFinalizationError(f"refusing to overwrite output root {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        if hold_reasons:
            receipt = {
                "schema_version": RECEIPT_SCHEMA, "protocol_id": PROTOCOL_ID, "status": HOLD_STATUS,
                "screening_cohort_sha256": screening_sha, "admission_receipt_sha256": admission_sha,
                "p0_anchor_agreement_sha256": p0_sha, "p1_action_facts_sha256": sha256_json(p1_action_facts),
                "candidate_outputs_opened": False, "hold_reasons": hold_reasons,
                "next_required_gate": "HOLD. Do not alter or replace events, intervals, buckets, or thresholds; do not materialize any trace.",
            }
            _write_json(staged / "scene-event-finalization-receipt.json", receipt)
            os.replace(staged, output_root)
            return receipt
        source = _manifest_index(manifest, cohort=cohort, screening_sha=screening_sha, materialized_root=materialized_root)
        scene_lines: list[str] = []
        for event_id, item in sorted(cohort.items()):
            source_item = source[event_id]
            for frame in source_item["frames"]:
                ordinal = int(frame["ordinal"])
                source_mask = materialized_root / str(frame["source_mask_path"])
                with Image.open(source_mask) as image:
                    panoptic = np.asarray(image.convert("RGB"), dtype=np.uint8)
                source_ids = panoptic[:, :, 0]
                canonical = np.isin(source_ids, list(CANONICAL_IDS))
                target = staged / "canonical-truth-masks" / event_id / f"{ordinal:03d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((canonical.astype(np.uint8) * 255), mode="L").save(target)
                components = _component_rows(panoptic, canonical)
                scene_lines.append(json.dumps({
                    "schema_version": SCENE_FACTS_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "parent_event_id": event_id,
                    "source_session_id": item["source_session_id"],
                    "frame_index": ordinal,
                    "source_frame_index": int(item["source_window"]["start_frame"]) + ordinal,
                    "source_mask_path": frame["source_mask_path"],
                    "source_mask_sha256": frame["source_mask_sha256"],
                    "frame_width": int(panoptic.shape[1]), "frame_height": int(panoptic.shape[0]),
                    "canonical_truth_mask_path": target.relative_to(staged).as_posix(),
                    "canonical_truth_mask_sha256": sha256_file(target),
                    "canonical_truth_area_px": int(canonical.sum()),
                    "objective_components": components,
                    "objective_class_area_px": {
                        "boundary_level_change": int(np.isin(source_ids, list(BOUNDARY_IDS)).sum()),
                        "blocking_obstacle": int(np.isin(source_ids, list(OBSTACLE_IDS)).sum()),
                        "walkable": int(np.isin(source_ids, list(WALKABLE_IDS)).sum()),
                    },
                }, ensure_ascii=False, sort_keys=True))
        scene_path = staged / "scene-facts.jsonl"
        scene_path.write_text("\n".join(scene_lines) + "\n", encoding="utf-8")
        scene_sha = sha256_file(scene_path)
        final_cohort = {
            "schema_version": COHORT_SCHEMA, "protocol_id": PROTOCOL_ID, "status": FINAL_STATUS,
            "candidate_outputs_opened": False, "screening_cohort_sha256": screening_sha,
            "scene_fact_manifest_sha256": scene_sha,
            "items": [{
                "parent_event_id": event_id, "source_session_id": item["source_session_id"], "bucket": buckets[event_id],
                "frame_indices": list(range(item["source_window"]["frame_count"])),
                "anchor_frame_indices": item["source_window"]["p0_anchor_offsets"],
                "scene_fact_manifest_sha256": scene_sha,
            } for event_id, item in sorted(cohort.items())],
        }
        cohort_path = staged / "cohort-v1.json"
        _write_json(cohort_path, final_cohort)
        cohort_sha = sha256_json(final_cohort)
        final_facts = {
            "schema_version": FULL_EVENT_FACTS_SCHEMA, "protocol_id": PROTOCOL_ID,
            "status": "FULL_EVENT_FACTS_FROZEN_AFTER_ANCHOR_CONSISTENCY", "cohort_sha256": cohort_sha,
            "anchor_consistency_sha256": sha256_json(p0_agreement["anchor_agreement"]),
            "p0_anchor_agreement_receipt_sha256": p0_sha,
            "p1_action_facts_sha256": sha256_json(p1_action_facts),
            "candidate_outputs_opened": False,
            "independent_full_review_evidence": {
                "review_a_sha256": p1_action_facts["independent_full_review_evidence"]["review_a_sha256"],
                "review_b_sha256": p1_action_facts["independent_full_review_evidence"]["review_b_sha256"],
                "reviewers_isolated": True, "model_or_oracle_output_visible": False,
                "agreement_passed": True, "unknown_anchor_or_frame_count": 0,
            },
            "items": [{
                "parent_event_id": event_id, "bucket": buckets[event_id],
                "alertable_interval_frames": p1[event_id]["alertable_interval_frames"],
                "passed_interval_frames": p1[event_id]["passed_interval_frames"],
            } for event_id in sorted(cohort)],
        }
        facts_path = staged / "full-event-facts-v1.json"
        _write_json(facts_path, final_facts)
        receipt = {
            "schema_version": RECEIPT_SCHEMA, "protocol_id": PROTOCOL_ID, "status": FINAL_STATUS,
            "screening_cohort_sha256": screening_sha, "admission_receipt_sha256": admission_sha,
            "p0_anchor_agreement_sha256": p0_sha, "p1_action_facts_sha256": sha256_json(p1_action_facts),
            "candidate_outputs_opened": False, "bucket_counts": dict(sorted(Counter(buckets.values()).items())),
            "scene_facts_sha256": scene_sha, "cohort_sha256": cohort_sha, "full_event_facts_sha256": sha256_file(facts_path),
            "next_required_gate": "Only now may a separately auditable same-decision-chain runner materialize the four arms and trace manifest.",
        }
        _write_json(staged / "scene-event-finalization-receipt.json", receipt)
        os.replace(staged, output_root)
        return receipt
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--p0-agreement", type=Path, required=True)
    parser.add_argument("--p1-action-facts", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.materialized_root / "manifest.json"
    result = finalize_scene_facts(
        screening_cohort=json.loads(args.screening_cohort.read_text(encoding="utf-8")),
        admission_receipt=json.loads(args.admission_receipt.read_text(encoding="utf-8")),
        p0_agreement=json.loads(args.p0_agreement.read_text(encoding="utf-8")),
        p1_action_facts=json.loads(args.p1_action_facts.read_text(encoding="utf-8")),
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")), materialized_root=args.materialized_root, output_root=args.output_root,
    )
    print(f"status={result['status']} output={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

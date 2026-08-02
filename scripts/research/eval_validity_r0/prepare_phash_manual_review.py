from __future__ import annotations

"""Prepare two isolated RGB-only packets for a frozen pHash HOLD.

pHash is deliberately a candidate screen, not proof that two images are from
the same natural capture.  This tool can only prepare a fixed review of every
candidate recorded in a completed admission receipt.  It never reads masks,
action facts, model output, oracle output, or feedback traces, and it cannot
change the candidate set or pHash threshold.
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .common import (
    PHASH_PRIVATE_MAP_SCHEMA,
    PHASH_REVIEW_PACKET_SCHEMA,
    PROTOCOL_ID,
    read_json,
    sha256_file,
    sha256_json,
)


PACKET_STATUS = "PHASH_RGB_REVIEW_PENDING"
PRIVATE_STATUS = "PHASH_PRIVATE_REVIEW_MAP_FROZEN_BEFORE_SUBMISSIONS"
ADMISSION_HOLD_STATUS = "HOLD_EVAL_VALIDITY_DATA"


class PHashReviewPacketError(ValueError):
    """Raised when frozen candidate evidence cannot be safely packaged."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PHashReviewPacketError(message)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"{path}:{line_number}: cache row must be an object")
            yield value


def _rgb_pixel_sha256(path: Path) -> str:
    with Image.open(path) as source:
        source.load()
        return hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _ensure_distinct_empty_roots(*roots: Path) -> None:
    resolved = [root.resolve() for root in roots]
    _require(len(set(resolved)) == len(resolved), "reviewer/private output roots must be distinct")
    for left in resolved:
        for right in resolved:
            _require(left == right or (left not in right.parents and right not in left.parents), "reviewer/private output roots must not contain one another")
    _require(not any(root.exists() for root in resolved), "refusing to overwrite an existing reviewer/private output root")


def _validate_hold(admission: dict[str, Any]) -> list[dict[str, Any]]:
    _require(admission.get("schema_version") == "blindassist.eval_validity_r0.data_admission_receipt.v1", "admission schema mismatch")
    _require(admission.get("protocol_id") == PROTOCOL_ID and admission.get("status") == ADMISSION_HOLD_STATUS, "admission is not a pHash HOLD")
    _require(admission.get("candidate_outputs_opened") is False, "admission records forbidden output access")
    checks = admission.get("checks")
    _require(isinstance(checks, dict), "admission checks are missing")
    expected_true = {
        "session_disjoint", "old_truth_session_disjoint", "parent_identity_disjoint", "exact_rgb_disjoint",
        "decoded_rgb_disjoint", "exact_source_mask_disjoint", "p_hash_prior_session_coverage_complete",
        "p_hash_prior_decode_complete", "p_hash_new_decode_complete",
    }
    _require(all(checks.get(field) is True for field in expected_true), "admission has a failure other than pHash candidate resolution")
    _require(checks.get("p_hash_no_unresolved_new_to_excluded_candidate") is False, "admission has no pHash HOLD to review")
    evidence = admission.get("evidence")
    _require(isinstance(evidence, dict) and evidence.get("p_hash_candidate_enumeration_complete") is True, "pHash evidence is incomplete and cannot be manually resolved")
    rows = evidence.get("p_hash_candidates")
    _require(isinstance(rows, list) and rows and evidence.get("p_hash_candidate_count_lower_bound") == len(rows), "pHash candidate evidence is malformed or truncated")
    for row in rows:
        _require(isinstance(row, dict), "pHash candidate row must be an object")
        _require(isinstance(row.get("new_event_id"), str) and isinstance(row.get("new_session_id"), str) and isinstance(row.get("new_ordinal"), int), "pHash candidate new identity is invalid")
        _require(isinstance(row.get("prior_path"), str) and isinstance(row.get("prior_sessions"), list), "pHash candidate prior identity is invalid")
        _require(isinstance(row.get("comparison"), str) and isinstance(row.get("hamming"), int), "pHash candidate comparison evidence is invalid")
    return rows


def _cache_pixel_hashes(cache_path: Path, paths: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _read_jsonl(cache_path):
        path = row.get("rel_path")
        if path not in paths:
            continue
        pixel_hash = row.get("rgb_pixel_sha256")
        _require(isinstance(pixel_hash, str) and len(pixel_hash) == 64 and not row.get("decode_error"), f"cache has no decoded RGB hash for {path}")
        result[str(path)] = pixel_hash
    missing = paths - set(result)
    _require(not missing, f"pHash cache is missing {len(missing)} candidate payload hashes")
    return result


def _candidate_cases(
    *, admission: dict[str, Any], materialized_root: Path, workspace_root: Path, prior_cache: Path,
) -> list[dict[str, Any]]:
    rows = _validate_hold(admission)
    cache_hashes = _cache_pixel_hashes(prior_cache, {str(row["prior_path"]) for row in rows})
    grouped: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in rows:
        event_id, ordinal, prior_relative = str(row["new_event_id"]), int(row["new_ordinal"]), str(row["prior_path"])
        new_path = (materialized_root / "events" / event_id / "rgb" / f"{ordinal:03d}.png").resolve()
        prior_path = (workspace_root / prior_relative).resolve()
        _require(materialized_root.resolve() in new_path.parents and new_path.is_file(), f"candidate new RGB payload is missing: {event_id}/{ordinal}")
        _require(workspace_root.resolve() in prior_path.parents and prior_path.is_file(), f"candidate prior RGB payload is missing: {prior_relative}")
        new_pixel_hash, prior_pixel_hash = _rgb_pixel_sha256(new_path), cache_hashes[prior_relative]
        key = (event_id, ordinal, new_pixel_hash, prior_pixel_hash)
        # Preserve the full frozen receipt row in the custodian-only map so
        # the finalizer can prove every enumerated pHash edge was reviewed.
        evidence = {
            "new_event_id": event_id,
            "new_session_id": str(row["new_session_id"]),
            "new_ordinal": ordinal,
            "prior_path": prior_relative,
            "prior_sessions": sorted({str(value) for value in row["prior_sessions"] if isinstance(value, str)}),
            "comparison": str(row["comparison"]),
            "hamming": int(row["hamming"]),
        }
        case = grouped.setdefault(key, {
            "new_event_id": event_id,
            "new_ordinal": ordinal,
            "new_rgb_path": new_path,
            "prior_rgb_path": prior_path,
            "new_rgb_pixel_sha256": new_pixel_hash,
            "prior_rgb_pixel_sha256": prior_pixel_hash,
            "candidate_evidence": [],
        })
        case["candidate_evidence"].append(evidence)
    cases = list(grouped.values())
    for case in cases:
        case["candidate_evidence"] = sorted(case["candidate_evidence"], key=lambda item: (item["hamming"], item["prior_path"], item["comparison"]))
        representative = case["candidate_evidence"][0]
        case["prior_rgb_path"] = (workspace_root / representative["prior_path"]).resolve()
        case["case_fingerprint"] = sha256_json({
            "new_event_id": case["new_event_id"], "new_ordinal": case["new_ordinal"],
            "new_rgb_pixel_sha256": case["new_rgb_pixel_sha256"], "prior_rgb_pixel_sha256": case["prior_rgb_pixel_sha256"],
            "candidate_evidence": case["candidate_evidence"],
        })
    cases.sort(key=lambda case: case["case_fingerprint"])
    _require(cases, "pHash HOLD has no reviewable unique RGB pairs")
    return cases


def _render_packet(
    *, role: str, cases: list[dict[str, Any]], output_root: Path, admission_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    staged = output_root.with_name(f".{output_root.name}.{secrets.token_hex(8)}.staging")
    try:
        mappings: dict[str, dict[str, Any]] = {}
        items: list[dict[str, Any]] = []
        for number, case in enumerate(cases, start=1):
            opaque_id = f"{role.lower()}-{secrets.token_urlsafe(16)}"
            while opaque_id in mappings:
                opaque_id = f"{role.lower()}-{secrets.token_urlsafe(16)}"
            left_is_new = bool(secrets.randbits(1))
            left, right = (case["new_rgb_path"], case["prior_rgb_path"]) if left_is_new else (case["prior_rgb_path"], case["new_rgb_path"])
            left_target = staged / "assets" / opaque_id / "left.png"
            right_target = staged / "assets" / opaque_id / "right.png"
            _link_or_copy(left, left_target)
            _link_or_copy(right, right_target)
            _require(sha256_file(left) == sha256_file(left_target) and sha256_file(right) == sha256_file(right_target), "review RGB copy/hash mismatch")
            items.append({
                "review_item_id": opaque_id,
                "rgb_pair": [left_target.relative_to(staged).as_posix(), right_target.relative_to(staged).as_posix()],
                "response_field": {"same_natural_capture": ["SAME_CAPTURE", "DISTINCT_CAPTURE", "UNKNOWN"]},
            })
            mappings[opaque_id] = {
                "case_fingerprint": case["case_fingerprint"], "left_is_new": left_is_new,
                "new_event_id": case["new_event_id"], "new_ordinal": case["new_ordinal"],
                "new_rgb_pixel_sha256": case["new_rgb_pixel_sha256"], "prior_rgb_pixel_sha256": case["prior_rgb_pixel_sha256"],
                "new_rgb_sha256": sha256_file(case["new_rgb_path"]), "prior_rgb_sha256": sha256_file(case["prior_rgb_path"]),
                "candidate_evidence": case["candidate_evidence"],
            }
        secrets.SystemRandom().shuffle(items)
        packet = {
            "schema_version": PHASH_REVIEW_PACKET_SCHEMA, "protocol_id": PROTOCOL_ID,
            "reviewer_role": role, "status": PACKET_STATUS,
            "disclosures": {
                "raw_rgb_only": True, "source_mask_visible": False, "action_fact_visible": False,
                "model_or_oracle_output_visible": False, "source_session_or_event_identity_visible": False,
                "other_reviewer_visible": False, "candidate_threshold_or_hamming_visible": False,
            },
            "items": items,
            "submission_shape": {
                "schema_version": "blindassist.eval_validity_r0.phash_manual_review.v1", "protocol_id": PROTOCOL_ID,
                "reviewer_role": role, "admission_receipt_sha256": admission_sha256,
                "isolated_context": True, "other_review_visible_before_submission": False,
                "model_or_oracle_output_visible": False, "items": [{
                    "review_item_id": "copy from packet item", "same_natural_capture": "SAME_CAPTURE|DISTINCT_CAPTURE|UNKNOWN",
                }],
            },
        }
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, output_root)
        return packet, mappings
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def prepare_packets(
    *, admission: dict[str, Any], materialized_root: Path, workspace_root: Path, prior_cache: Path,
    reviewer_a_root: Path, reviewer_b_root: Path, private_root: Path,
) -> dict[str, Any]:
    _ensure_distinct_empty_roots(reviewer_a_root, reviewer_b_root, private_root)
    cases = _candidate_cases(admission=admission, materialized_root=materialized_root, workspace_root=workspace_root, prior_cache=prior_cache)
    admission_sha256 = sha256_json(admission)
    try:
        packet_a, map_a = _render_packet(role="PHASH_RGB_REVIEW_A", cases=cases, output_root=reviewer_a_root, admission_sha256=admission_sha256)
        packet_b, map_b = _render_packet(role="PHASH_RGB_REVIEW_B", cases=cases, output_root=reviewer_b_root, admission_sha256=admission_sha256)
        private_root.mkdir(parents=True, exist_ok=False)
        private = {
            "schema_version": PHASH_PRIVATE_MAP_SCHEMA, "protocol_id": PROTOCOL_ID, "status": PRIVATE_STATUS,
            "admission_receipt_sha256": admission_sha256,
            "candidate_case_count": len(cases),
            "candidate_case_fingerprints_sha256": sha256_json([case["case_fingerprint"] for case in cases]),
            "packet_a_sha256": sha256_file(reviewer_a_root / "packet.json"), "packet_b_sha256": sha256_file(reviewer_b_root / "packet.json"),
            "reviewer_a_map": map_a, "reviewer_b_map": map_b,
            "sharing_rule": "Never disclose this directory, case source identities, or either map to a reviewer. Each reviewer receives only their own packet root.",
        }
        (private_root / "private-review-map.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"candidate_case_count": len(cases), "packet_a": packet_a, "packet_b": packet_b, "private": private}
    except Exception:
        for root in (reviewer_a_root, reviewer_b_root, private_root):
            shutil.rmtree(root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--prior-image-feature-cache", type=Path, required=True)
    parser.add_argument("--reviewer-a-root", type=Path, required=True)
    parser.add_argument("--reviewer-b-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_packets(
        admission=read_json(args.admission_receipt), materialized_root=args.materialized_root,
        workspace_root=args.workspace_root, prior_cache=args.prior_image_feature_cache,
        reviewer_a_root=args.reviewer_a_root, reviewer_b_root=args.reviewer_b_root, private_root=args.private_root,
    )
    print(f"candidate_cases={result['candidate_case_count']} reviewer_a_items={len(result['packet_a']['items'])} reviewer_b_items={len(result['packet_b']['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

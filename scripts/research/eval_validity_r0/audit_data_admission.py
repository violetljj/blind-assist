from __future__ import annotations

"""Fail-closed data admission audit for the fresh EVAL-VALIDITY cohort.

It binds the materialized native payload to the frozen screening cohort, checks
all excluded sessions and old truth hashes, and runs an exhaustive-on-clear
pHash/transform comparison against every decoded cached RGB asset from an
excluded session.  Any missing coverage, decode failure or candidate edge is
HOLD.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .common import EXCLUSION_SCHEMA, PROTOCOL_ID, read_json, sha256_file, sha256_json
from .freeze_screening_cohort import SCHEMA as SCREENING_COHORT_SCHEMA
from .materialize_screening_inputs import MATERIALIZED_SCHEMA
from .prepare_p0_review_packets import ADMISSION_STATUS


class AdmissionAuditError(ValueError):
    """Raised on malformed inputs; evidence insufficiency is a HOLD receipt."""


P_HASH_EVIDENCE_CAP = 200


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AdmissionAuditError(f"{path}:{line_number}: expected JSON object")
            yield value


def _dct_matrix() -> np.ndarray:
    n = 32
    rows = np.arange(n, dtype=np.float32)[:, None]
    cols = np.arange(n, dtype=np.float32)[None, :]
    matrix = np.cos((np.pi / n) * (cols + 0.5) * rows)
    matrix[0, :] *= 1.0 / np.sqrt(n)
    matrix[1:, :] *= np.sqrt(2.0 / n)
    return matrix


def _phash(image: Image.Image, matrix: np.ndarray) -> int:
    grey = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    array = np.asarray(grey, dtype=np.float32)
    coeff = matrix @ array @ matrix.T
    low = coeff[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    value = 0
    for item in low:
        value = (value << 1) | int(item > median)
    return value


def _variants(image: Image.Image) -> dict[str, Image.Image]:
    width, height = image.size
    result = {"original": image, "mirror_horizontal": image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)}
    for fraction, label in ((0.90, "crop_center_90"), (0.80, "crop_center_80")):
        crop_w, crop_h = max(32, int(width * fraction)), max(32, int(height * fraction))
        left, top = max(0, (width - crop_w) // 2), max(0, (height - crop_h) // 2)
        result[label] = image.crop((left, top, left + crop_w, top + crop_h))
    crop_w, crop_h = max(32, int(width * 0.85)), max(32, int(height * 0.85))
    result["crop_left_85"] = image.crop((0, 0, crop_w, crop_h))
    result["crop_right_85"] = image.crop((width - crop_w, height - crop_h, width, height))
    return result


def _hash_features(path: Path, matrix: np.ndarray) -> dict[str, Any]:
    with Image.open(path) as source:
        source.load()
        rgb = source.convert("RGB")
        return {
            "rgb_pixel_sha256": hashlib.sha256(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest(),
            "phash_variants": {name: f"{_phash(value, matrix):016x}" for name, value in _variants(rgb).items()},
        }


def _chunk_keys(value: int) -> list[tuple[int, int]]:
    """Nine chunks guarantee one exact chunk when Hamming distance is <= 8."""
    result: list[tuple[int, int]] = []
    offset = 0
    for index, width in enumerate((8, 7, 7, 7, 7, 7, 7, 7, 7)):
        result.append((index, (value >> offset) & ((1 << width) - 1)))
        offset += width
    return result


def _prior_cache(cache_path: Path, excluded: set[str]) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    rows: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    errors: list[str] = []
    for row in _read_jsonl(cache_path):
        sessions = {str(value) for value in row.get("session_ids", []) if isinstance(value, str)} & excluded
        if not sessions:
            continue
        seen_sessions.update(sessions)
        if row.get("decode_error"):
            errors.append(str(row.get("rel_path", "unknown")))
            continue
        variants = row.get("phash_variants")
        original = row.get("phash")
        if row.get("pixel_domain") != "rgb" or not isinstance(variants, dict) or not isinstance(original, str):
            continue
        if not all(isinstance(value, str) and len(value) == 16 for value in variants.values()):
            errors.append(str(row.get("rel_path", "unknown")))
            continue
        rows.append({
            "rel_path": row.get("rel_path"), "session_ids": sorted(sessions), "file_sha256": row.get("file_sha256"),
            "rgb_pixel_sha256": row.get("rgb_pixel_sha256"), "phash": original, "phash_variants": variants,
        })
    return rows, seen_sessions, errors


def _old_truth_hashes(path: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    rgb_hashes: set[str] = set()
    mask_hashes: set[str] = set()
    sessions: set[str] = set()
    parents: set[str] = set()
    for row in _read_jsonl(path):
        rgb_hashes.update(value for value in row.get("rgb_sha256s", []) if isinstance(value, str))
        mask_hashes.update(value for value in row.get("source_mask_sha256s", []) if isinstance(value, str))
        if isinstance(row.get("source_session_id"), str):
            sessions.add(row["source_session_id"])
        if isinstance(row.get("parent_event_id"), str):
            parents.add(row["parent_event_id"])
    return rgb_hashes, mask_hashes, sessions, parents


def _validated_new_rows(cohort: dict[str, Any], manifest: dict[str, Any], materialized_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    cohort_items = cohort.get("items")
    manifest_items = manifest.get("items")
    if not isinstance(cohort_items, list) or not isinstance(manifest_items, list) or len(cohort_items) != 48 or len(manifest_items) != 48:
        raise AdmissionAuditError("screening/materialized coverage must be exactly 48")
    cohort_by_id = {item.get("screening_event_id"): item for item in cohort_items if isinstance(item, dict)}
    if len(cohort_by_id) != 48:
        raise AdmissionAuditError("screening event identities are invalid")
    rgb_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    sessions: set[str] = set()
    for item in manifest_items:
        if not isinstance(item, dict) or item.get("screening_event_id") not in cohort_by_id:
            raise AdmissionAuditError("materialized event identity is invalid")
        cohort_item = cohort_by_id[item["screening_event_id"]]
        session = item.get("source_session_id")
        if session != cohort_item.get("source_session_id") or not isinstance(session, str):
            raise AdmissionAuditError("materialized source-session binding mismatch")
        sessions.add(session)
        frames = item.get("frames")
        if not isinstance(frames, list) or len(frames) != cohort_item.get("source_window", {}).get("frame_count"):
            raise AdmissionAuditError("materialized continuous frame coverage mismatch")
        if [row.get("ordinal") for row in frames] != list(range(len(frames))):
            raise AdmissionAuditError("materialized frame ordinals are not contiguous")
        for frame in frames:
            for kind, path_key, hash_key, target in (
                ("rgb", "rgb_path", "rgb_sha256", rgb_rows),
                ("source_mask", "source_mask_path", "source_mask_sha256", mask_rows),
            ):
                relative, expected = frame.get(path_key), frame.get(hash_key)
                if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
                    raise AdmissionAuditError(f"materialized {kind} provenance is invalid")
                path = (materialized_root / relative).resolve()
                if materialized_root.resolve() not in path.parents or not path.is_file() or sha256_file(path) != expected:
                    raise AdmissionAuditError(f"materialized {kind} payload is missing or hash-mismatched")
                target.append({"event_id": item["screening_event_id"], "session_id": session, "ordinal": frame["ordinal"], "path": path, "sha256": expected})
    if len(sessions) != 48:
        raise AdmissionAuditError("new cohort is not one-event-per-session")
    return rgb_rows, mask_rows, sessions


def _p_hash_candidates(
    new_rows: list[dict[str, Any]], prior_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], bool]:
    """Return pHash evidence, stopping only after HOLD is already inevitable.

    Degenerate, low-texture source frames can yield an enormous number of
    equivalent near-duplicate edges.  One edge already makes admission fail,
    so retain bounded, auditable evidence and record that it is a lower bound
    instead of risking memory exhaustion.  An empty result is exhaustive.
    """
    matrix = _dct_matrix()
    new_features: list[dict[str, Any]] = []
    decode_errors: list[str] = []
    for row in new_rows:
        try:
            features = _hash_features(row["path"], matrix)
            new_features.append({**row, **features})
        except Exception as error:
            decode_errors.append(f"{row['path']}: {type(error).__name__}: {error}")
    old_original_index: dict[tuple[int, int], list[int]] = {}
    old_variant_index: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(prior_rows):
        for key in _chunk_keys(int(row["phash"], 16)):
            old_original_index.setdefault(key, []).append(index)
        for value in row["phash_variants"].values():
            for key in _chunk_keys(int(value, 16)):
                old_variant_index.setdefault(key, []).append(index)
    edges: list[dict[str, Any]] = []
    def add_edge(edge: dict[str, Any]) -> bool:
        edges.append(edge)
        return len(edges) >= P_HASH_EVIDENCE_CAP

    for row in new_features:
        for transform, value in row["phash_variants"].items():
            candidate_indexes = {index for key in _chunk_keys(int(value, 16)) for index in old_original_index.get(key, [])}
            for old_index in candidate_indexes:
                distance = (int(value, 16) ^ int(prior_rows[old_index]["phash"], 16)).bit_count()
                if distance <= 8:
                    if add_edge({"new_event_id": row["event_id"], "new_session_id": row["session_id"], "new_ordinal": row["ordinal"], "prior_path": prior_rows[old_index]["rel_path"], "prior_sessions": prior_rows[old_index]["session_ids"], "comparison": f"new:{transform}:old:original", "hamming": distance}):
                        return new_features, edges, decode_errors, True
        value = row["phash_variants"]["original"]
        candidate_indexes = {index for key in _chunk_keys(int(value, 16)) for index in old_variant_index.get(key, [])}
        for old_index in candidate_indexes:
            for transform, old_value in prior_rows[old_index]["phash_variants"].items():
                distance = (int(value, 16) ^ int(old_value, 16)).bit_count()
                if distance <= 8:
                    if add_edge({"new_event_id": row["event_id"], "new_session_id": row["session_id"], "new_ordinal": row["ordinal"], "prior_path": prior_rows[old_index]["rel_path"], "prior_sessions": prior_rows[old_index]["session_ids"], "comparison": f"new:original:old:{transform}", "hamming": distance}):
                        return new_features, edges, decode_errors, True
    return new_features, edges, decode_errors, False


def audit_admission(
    *, cohort: dict[str, Any], manifest: dict[str, Any], materialized_root: Path, registry: dict[str, Any], truth_ledger: Path, cache_path: Path,
) -> dict[str, Any]:
    if cohort.get("schema_version") != SCREENING_COHORT_SCHEMA or cohort.get("protocol_id") != PROTOCOL_ID:
        raise AdmissionAuditError("screening cohort schema/protocol mismatch")
    if cohort.get("status") != "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN" or cohort.get("candidate_outputs_opened") is not False:
        raise AdmissionAuditError("screening cohort is not eligible for admission audit")
    if manifest.get("schema_version") != MATERIALIZED_SCHEMA or manifest.get("protocol_id") != PROTOCOL_ID:
        raise AdmissionAuditError("materialized manifest schema/protocol mismatch")
    manifest_path = materialized_root / "manifest.json"
    if manifest.get("screening_cohort_sha256") != sha256_json(cohort) or manifest.get("candidate_outputs_opened") is not False:
        raise AdmissionAuditError("materialized manifest cohort/output binding mismatch")
    if registry.get("schema_version") != EXCLUSION_SCHEMA or registry.get("protocol_id") != PROTOCOL_ID:
        raise AdmissionAuditError("exclusion registry schema/protocol mismatch")
    excluded = set(registry.get("excluded_source_sessions", []))
    if not excluded:
        raise AdmissionAuditError("exclusion registry has no sessions")
    rgb_rows, mask_rows, new_sessions = _validated_new_rows(cohort, manifest, materialized_root)
    old_rgb_hashes, old_mask_hashes, old_sessions, old_parents = _old_truth_hashes(truth_ledger)
    prior_rows, cached_sessions, cache_errors = _prior_cache(cache_path, excluded)
    exact_rgb = sorted({row["sha256"] for row in rgb_rows} & (old_rgb_hashes | {str(row.get("file_sha256")) for row in prior_rows}))
    exact_mask = sorted({row["sha256"] for row in mask_rows} & old_mask_hashes)
    new_parent_ids = {item["screening_event_id"] for item in cohort["items"]}
    new_features, phash_edges, new_decode_errors, phash_evidence_truncated = _p_hash_candidates(rgb_rows, prior_rows)
    decoded_rgb_overlap = sorted(
        {row["rgb_pixel_sha256"] for row in new_features}
        & {str(row.get("rgb_pixel_sha256")) for row in prior_rows}
    )
    checks = {
        "session_disjoint": not (new_sessions & excluded),
        "old_truth_session_disjoint": not (new_sessions & old_sessions),
        "parent_identity_disjoint": not (new_parent_ids & old_parents),
        "exact_rgb_disjoint": not exact_rgb,
        "decoded_rgb_disjoint": not decoded_rgb_overlap,
        "exact_source_mask_disjoint": not exact_mask,
        "p_hash_prior_session_coverage_complete": cached_sessions == excluded,
        "p_hash_prior_decode_complete": not cache_errors,
        "p_hash_new_decode_complete": not new_decode_errors,
        "p_hash_no_unresolved_new_to_excluded_candidate": not phash_edges,
    }
    passed = all(checks.values())
    return {
        "schema_version": "blindassist.eval_validity_r0.data_admission_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": ADMISSION_STATUS if passed else "HOLD_EVAL_VALIDITY_DATA",
        "screening_cohort_sha256": sha256_json(cohort),
        "materialized_manifest_sha256": sha256_file(manifest_path),
        "candidate_outputs_opened": False,
        "source_session_count": len(new_sessions),
        "frame_counts": {"rgb": len(rgb_rows), "source_mask": len(mask_rows)},
        "checks": checks,
        "evidence": {
            "excluded_source_session_count": len(excluded), "prior_p_hash_rgb_row_count": len(prior_rows),
            "prior_cache_sessions_missing": sorted(excluded - cached_sessions), "prior_cache_decode_errors": cache_errors[:50],
            "new_rgb_decode_errors": new_decode_errors[:50], "exact_rgb_overlap_hashes": exact_rgb[:50],
            "decoded_rgb_overlap_hashes": decoded_rgb_overlap[:50],
            "exact_source_mask_overlap_hashes": exact_mask[:50],
            "p_hash_candidate_count_lower_bound": len(phash_edges),
            "p_hash_candidate_enumeration_complete": not phash_evidence_truncated,
            "p_hash_candidates": phash_edges,
        },
        "next_required_gate": (
            "Generate two separate opaque causal RGB P0 reviewer packets; no model/oracle trace may be materialized."
            if passed else "HOLD. Resolve or replace every failed data-admission input before any reviewer packet or output trace is generated."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-cohort", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--old-truth-ledger", type=Path, required=True)
    parser.add_argument("--prior-image-feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = audit_admission(
        cohort=read_json(args.screening_cohort), manifest=read_json(args.materialized_root / "manifest.json"), materialized_root=args.materialized_root,
        registry=read_json(args.exclusion_registry), truth_ledger=args.old_truth_ledger, cache_path=args.prior_image_feature_cache,
    )
    result["input_sha256"] = {
        "screening_cohort": sha256_file(args.screening_cohort), "materialized_manifest": sha256_file(args.materialized_root / "manifest.json"),
        "exclusion_registry": sha256_file(args.exclusion_registry), "old_truth_ledger": sha256_file(args.old_truth_ledger),
        "prior_image_feature_cache": sha256_file(args.prior_image_feature_cache),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} p_hash_candidate_lower_bound={result['evidence']['p_hash_candidate_count_lower_bound']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

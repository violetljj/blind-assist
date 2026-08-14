#!/usr/bin/env python3
"""R36 outcome-blind fresh-parent confirmation for OpenLORIS corridor1-3..5.

The R31-v7 OpenLORIS regime expert is trained only on four consumed source
families.  Fresh candidate depth stays unopened until model predictions and
selected candidate identities have been sealed durably in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_multi_candidate_reliability_consistency as r31
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_openloris_home_frontdoor as openloris
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_openloris_corridor_fresh_parent_confirmation.v3"
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_R36_OPENLORIS_CORRIDOR_FRESH_PARENT_CONFIRMATION_LOCK_2026-08-15_R2.json"
PARENT_IDS = ("corridor1-3", "corridor1-4", "corridor1-5")
SOURCE_FAMILY = "OPENLORIS_SCENE_D435I_CORRIDOR"
MAX_REFERENCES_PER_PARENT = 6
MIN_REFERENCES = 18
MIN_PARENTS = 3
MIN_OPPORTUNITY_PARENTS = 3
PACKAGE_BYTES = 19_695_083_520
PACKAGE_SHA256 = "A2DE290B85BDEFC5388AAD27858125206A19B007D9B735DE5FAFF954CC473413"
PACKAGE_REPO_PATH = "package/corridor1-2_5-package.tar"
PACKAGE_URL = "https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/package/corridor1-2_5-package.tar?download=true"
EXPECTED_TAR_ENTRIES = {
    "corridor1-3.7z": {
        "header_offset": 6_075_167_232,
        "data_offset": 6_075_167_744,
        "bytes": 3_499_857_556,
    },
    "corridor1-4.7z": {
        "header_offset": 9_575_025_664,
        "data_offset": 9_575_026_176,
        "bytes": 3_227_354_165,
    },
    "corridor1-5.7z": {
        "header_offset": 12_802_380_800,
        "data_offset": 12_802_381_312,
        "bytes": 6_892_693_271,
    },
}
GROUNDTRUTH_BYTES = 11_076_559
GROUNDTRUTH_SHA256 = "07564D7ED3D6739585002AFA12BCF481CC0E9E358FC64EFD5E658E2C994BDC3B"


class R36Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R36Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_tar_header(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    require(len(payload) == 512, f"R36 TAR header byte count drift: {path}")
    require(payload[257:263] == b"ustar ", f"R36 TAR header magic drift: {path}")
    stored_checksum = int(payload[148:156].strip(b"\0 ") or b"0", 8)
    checksum_payload = payload[:148] + (b" " * 8) + payload[156:]
    require(sum(checksum_payload) == stored_checksum, f"R36 TAR header checksum drift: {path}")
    return {
        "name": payload[:100].split(b"\0", 1)[0].decode("utf-8"),
        "bytes": int(payload[124:136].strip(b"\0 ") or b"0", 8),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }


def _manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def verify_inputs(source_slice_manifest: Path, groundtruth_archive: Path) -> dict[str, Any]:
    require(LOCK_PATH.is_file(), "R36 confirmation lock absent")
    require(source_slice_manifest.is_file(), "R36 source slice manifest absent")
    require(groundtruth_archive.is_file(), "R36 groundtruth archive absent")
    require(groundtruth_archive.stat().st_size == GROUNDTRUTH_BYTES, "R36 groundtruth byte count drift")
    manifest = json.loads(source_slice_manifest.read_text(encoding="utf-8"))
    require(
        manifest.get("schema") == "blindassist.taro.openloris_tar_slice_manifest.v1",
        "R36 source slice manifest schema drift",
    )
    require(manifest.get("transport") == "HTTPS_BYTE_RANGES", "R36 source slice transport drift")
    source_object = manifest.get("source_object", {})
    require(source_object.get("path") == PACKAGE_REPO_PATH, "R36 source object path drift")
    require(source_object.get("url") == PACKAGE_URL, "R36 source object URL drift")
    require(source_object.get("bytes") == PACKAGE_BYTES, "R36 source object byte count drift")
    require(source_object.get("lfs_sha256") == PACKAGE_SHA256, "R36 source object SHA-256 drift")
    entries = manifest.get("tar_entries", [])
    require(len(entries) == len(EXPECTED_TAR_ENTRIES), "R36 source slice entry count drift")
    entry_by_name = {entry.get("name"): entry for entry in entries}
    require(set(entry_by_name) == set(EXPECTED_TAR_ENTRIES), "R36 source slice names drift")
    entry_receipts = []
    for name, expected in EXPECTED_TAR_ENTRIES.items():
        entry = entry_by_name[name]
        for field in ("header_offset", "data_offset", "bytes"):
            require(entry.get(field) == expected[field], f"R36 {name} {field} drift")
        expected_header_range = (
            f"bytes {expected['header_offset']}-{expected['header_offset'] + 511}/{PACKAGE_BYTES}"
        )
        expected_archive_range = (
            f"bytes {expected['data_offset']}-{expected['data_offset'] + expected['bytes'] - 1}/{PACKAGE_BYTES}"
        )
        require(entry.get("header_content_range") == expected_header_range, f"R36 {name} header range drift")
        require(entry.get("archive_content_range") == expected_archive_range, f"R36 {name} archive range drift")
        header_path = _manifest_path(source_slice_manifest, entry["header_path"])
        archive_path = _manifest_path(source_slice_manifest, entry["archive_path"])
        require(header_path.is_file(), f"R36 {name} TAR header absent")
        require(archive_path.is_file(), f"R36 {name} archive absent")
        header = _load_tar_header(header_path)
        require(header["name"] == name, f"R36 {name} TAR header name drift")
        require(header["bytes"] == expected["bytes"], f"R36 {name} TAR header size drift")
        require(header["sha256"] == entry.get("header_sha256"), f"R36 {name} TAR header SHA-256 drift")
        require(archive_path.stat().st_size == expected["bytes"], f"R36 {name} archive byte count drift")
        archive_sha = sha256_file(archive_path)
        require(archive_sha == entry.get("archive_sha256"), f"R36 {name} archive SHA-256 drift")
        entry_receipts.append(
            {
                **expected,
                "name": name,
                "header_path": str(header_path),
                "header_sha256": header["sha256"],
                "archive_path": str(archive_path),
                "archive_sha256": archive_sha,
            }
        )
    groundtruth_sha = sha256_file(groundtruth_archive)
    require(groundtruth_sha == GROUNDTRUTH_SHA256, "R36 groundtruth SHA-256 drift")
    return {
        "lock_path": str(LOCK_PATH),
        "lock_sha256": sha256_file(LOCK_PATH),
        "source_slice_manifest": {
            "path": str(source_slice_manifest),
            "bytes": source_slice_manifest.stat().st_size,
            "sha256": sha256_file(source_slice_manifest),
            "source_object": source_object,
            "tar_entries": entry_receipts,
        },
        "groundtruth_archive": {
            "path": str(groundtruth_archive),
            "bytes": groundtruth_archive.stat().st_size,
            "sha256": groundtruth_sha,
        },
    }


def _blind_gated_scores(
    records: Sequence[scorer.CandidateRecord],
    utility_ensemble: np.ndarray,
    opportunity_ensemble: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(
        utility_ensemble.ndim == 2 and utility_ensemble.shape[1] == len(records),
        "R36 utility ensemble shape drift",
    )
    require(opportunity_ensemble.shape == utility_ensemble.shape, "R36 opportunity ensemble shape drift")
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        require(record.target_gain is None and record.coverage is None, "fresh target opened before R36 seal")
        by_reference[record.reference_id].append(index)
    output = np.zeros(len(records), dtype=np.float64)
    utility_mean = np.mean(utility_ensemble, axis=0)
    receipts: list[dict[str, Any]] = []
    accepted = 0
    for reference_id, indices in sorted(by_reference.items()):
        generic = r27._generic_index(records, indices)
        opportunity_baseline = opportunity_ensemble[:, generic]
        opportunity_margins = {
            index: opportunity_ensemble[:, index] - opportunity_baseline for index in indices
        }
        opportunity_lcbs = {
            index: float(np.mean(margin) - r31.LCB_Z * np.std(margin, ddof=0))
            for index, margin in opportunity_margins.items()
        }
        utility_proposal = max(
            indices,
            key=lambda index: (utility_mean[index], records[index].pair.neighbor.frame_id),
        )
        utility_margin = utility_ensemble[:, utility_proposal] - utility_ensemble[:, generic]
        utility_lcb = float(np.mean(utility_margin) - r31.LCB_Z * np.std(utility_margin, ddof=0))
        opportunity_proposal = max(
            indices,
            key=lambda index: (opportunity_lcbs[index], records[index].pair.neighbor.frame_id),
        )
        opportunity_utility_margin = (
            utility_ensemble[:, opportunity_proposal] - utility_ensemble[:, generic]
        )
        opportunity_fallback = (
            opportunity_proposal != generic
            and opportunity_lcbs[opportunity_proposal] > 0.0
            and float(np.mean(opportunity_utility_margin)) >= 0.0
        )
        if utility_proposal != generic and utility_lcb > 0.0:
            selected = utility_proposal
            lane = "UTILITY_LCB"
        elif opportunity_fallback:
            selected = opportunity_proposal
            lane = "OPPORTUNITY_FALLBACK"
        else:
            selected = generic
            lane = "GENERIC_FALLBACK"
        output[selected] = 1.0
        accepted += int(selected != generic)
        selected_utility_margin = utility_ensemble[:, selected] - utility_ensemble[:, generic]
        receipts.append(
            {
                "reference_id": reference_id,
                "generic_neighbor_id": records[generic].pair.neighbor.frame_id,
                "selected_neighbor_id": records[selected].pair.neighbor.frame_id,
                "decision_lane": lane,
                "utility_margin_mean": float(np.mean(selected_utility_margin)),
                "utility_margin_std": float(np.std(selected_utility_margin, ddof=0)),
                "opportunity_margin_mean": float(np.mean(opportunity_margins[selected])),
                "opportunity_margin_std": float(np.std(opportunity_margins[selected], ddof=0)),
            }
        )
    return output, {
        "reference_count": len(by_reference),
        "accepted_override_count": accepted,
        "generic_fallback_count": len(by_reference) - accepted,
        "decision_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipts)).hexdigest().upper(),
    }


def _role_disjoint_references(
    selected: Sequence[bonn.ReferenceSupport],
) -> tuple[list[bonn.ReferenceSupport], dict[str, Any]]:
    reference_ids = {row.reference.frame_id for row in selected}
    filtered: list[bonn.ReferenceSupport] = []
    removed_rows: list[dict[str, Any]] = []
    removed_candidate_count = 0
    removed_micro_count = 0
    for row in selected:
        candidates = tuple(
            pair for pair in row.candidates if pair.neighbor.frame_id not in reference_ids
        )
        micro_candidates = tuple(
            pair for pair in row.micro_candidates if pair.neighbor.frame_id not in reference_ids
        )
        removed_candidates = sorted(
            pair.neighbor.frame_id
            for pair in row.candidates
            if pair.neighbor.frame_id in reference_ids
        )
        removed_micro = sorted(
            pair.neighbor.frame_id
            for pair in row.micro_candidates
            if pair.neighbor.frame_id in reference_ids
        )
        removed_candidate_count += len(removed_candidates)
        removed_micro_count += len(removed_micro)
        if removed_candidates or removed_micro:
            removed_rows.append(
                {
                    "reference_frame_id": row.reference.frame_id,
                    "removed_candidate_frame_ids": removed_candidates,
                    "removed_micro_candidate_frame_ids": removed_micro,
                }
            )
        require(candidates and micro_candidates, f"R36 role-disjoint support empty: {row.reference.frame_id}")
        filtered.append(bonn.ReferenceSupport(row.reference, candidates, micro_candidates))
    candidate_ids = {
        pair.neighbor.frame_id for row in filtered for pair in row.candidates
    }
    require(reference_ids.isdisjoint(candidate_ids), "R36 role-disjoint candidate filter failed")
    return filtered, {
        "policy": "EXCLUDE_ALL_SELECTED_REFERENCE_IDENTITIES_FROM_GLOBAL_CANDIDATE_ROLE",
        "reference_count": len(reference_ids),
        "removed_candidate_count": removed_candidate_count,
        "removed_micro_candidate_count": removed_micro_count,
        "minimum_remaining_candidate_count": min(len(row.candidates) for row in filtered),
        "minimum_remaining_micro_candidate_count": min(len(row.micro_candidates) for row in filtered),
        "removed_identity_sha256": hashlib.sha256(canonical_json_bytes(removed_rows)).hexdigest().upper(),
    }


def _build_feature_only_dataset(
    source_root: Path,
    groundtruth_root: Path,
) -> tuple[
    r31.SourceDataset,
    dict[str, scorer.ReferenceContext],
    openloris.PayloadStore,
    dict[str, Any],
    dict[str, Any],
    set[str],
    set[str],
]:
    frames, assets, source = openloris.load_outcome_blind_roster(
        source_root,
        groundtruth_root,
        PARENT_IDS,
        SOURCE_FAMILY,
        "FRESH_PARENT_CONFIRMATION",
    )
    selected, capability = openloris.balanced.select_pose_capable_references(
        frames, MAX_REFERENCES_PER_PARENT
    )
    require(capability["eligible_parent_count"] >= MIN_PARENTS, "R36 eligible parent count insufficient")
    require(capability["selected_reference_count"] >= MIN_REFERENCES, "R36 selected reference count insufficient")
    selected, role_disjoint_receipt = _role_disjoint_references(selected)
    capability = {**capability, "role_disjoint_candidate_filter": role_disjoint_receipt}
    proposals, candidate_identity_sha = openloris._candidate_identity(selected)
    reference_ids = {row.reference.frame_id for row in selected}
    candidate_ids = {
        pair.neighbor.frame_id for pairs in proposals.values() for pair in pairs
    }
    require(reference_ids.isdisjoint(candidate_ids), "R36 reference/candidate identity overlap")
    store = openloris.PayloadStore(assets)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    cell_rows: list[np.ndarray] = []
    query_receipts: list[dict[str, Any]] = []
    for row in selected:
        low, points, valid, _coverage = store.observation(row.reference.frame_id)
        asset = assets[row.reference.frame_id]
        low_intrinsics = bonn._scaled_intrinsics(
            asset.intrinsics, openloris.CROPPED_SIZE_WH, tum.LOW_SIZE_WH
        )
        queries, query_receipt = openloris._calibration_queries(
            row.reference, asset.camera_height_m
        )
        query_receipts.append(query_receipt)
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, low_intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        reference_planes = store.planes(row.reference)
        for pair in proposals[row.reference.frame_id]:
            detail = r31.reliability_consistency_features(
                context, pair, reference_planes, store.planes(pair.neighbor)
            )
            records.append(
                scorer.CandidateRecord(
                    row.reference.parent_id,
                    "FRESH_PARENT_CONFIRMATION",
                    row.reference.frame_id,
                    pair,
                    detail.global_features,
                    detail.analytic,
                )
            )
            cell_rows.append(detail.cell_features)
    require(records and cell_rows, "R36 feature-only candidate set empty")
    require(
        set(store.decoded_depth_frame_ids()) == reference_ids,
        "R36 candidate depth decoded before selection seal",
    )
    dataset = r31.SourceDataset(
        "OPENLORIS_CORRIDOR_FRESH",
        records,
        np.stack(cell_rows),
        {
            "source": source,
            "capability": capability,
            "candidate_identity_sha256": candidate_identity_sha,
            "query_plane_receipt_sha256": hashlib.sha256(canonical_json_bytes(query_receipts)).hexdigest().upper(),
        },
        0,
    )
    return dataset, contexts, store, source, capability, reference_ids, candidate_ids


def _confirmation_metrics(
    records: Sequence[scorer.CandidateRecord],
    scores: np.ndarray,
) -> dict[str, Any]:
    metrics = r21.fold_metrics(records, scores)
    opportunity_parents = int(metrics["opportunity_parent_count"])
    required_strict = max(
        r21.MIN_STRICT_WIN_PARENTS,
        int(np.ceil(r21.MIN_STRICT_WIN_FRACTION * opportunity_parents)),
    )
    macro = metrics["parent_macro"]
    checks = {
        "minimum_evaluated_references": metrics["reference_count"] >= MIN_REFERENCES,
        "minimum_evaluated_parents": metrics["parent_count"] >= MIN_PARENTS,
        "minimum_opportunity_parents": opportunity_parents >= MIN_OPPORTUNITY_PARENTS,
        "all_opportunity_parents_strict_win": metrics["strict_win_parent_count"] >= required_strict,
        "ranker_parent_macro_beats_passive": macro["ranker"] > macro["passive"],
        "ranker_parent_macro_beats_generic": macro["ranker"] > macro["generic"],
    }
    metrics["required_strict_win_parent_count"] = required_strict
    metrics["checks"] = checks
    return metrics


def evaluate(
    cache_root: Path,
    source_root: Path,
    groundtruth_root: Path,
    source_slice_manifest: Path,
    groundtruth_archive: Path,
) -> dict[str, Any]:
    input_receipt = verify_inputs(source_slice_manifest, groundtruth_archive)
    consumed = {source: r31.load_cache(cache_root, source) for source in r31.SOURCE_NAMES}
    transform = r31.SetTransform.fit([consumed[source] for source in r31.SOURCE_NAMES])
    models = []
    training_receipts = []
    for seed in r31.SEEDS:
        model, receipt = r31.train_ranker(
            [consumed[source] for source in r31.SOURCE_NAMES], transform, seed
        )
        models.append(model)
        training_receipts.append(receipt)
    (
        fresh,
        contexts,
        store,
        source,
        capability,
        reference_ids,
        candidate_ids,
    ) = _build_feature_only_dataset(source_root, groundtruth_root)
    utility, opportunity = r31.predict(fresh, transform, models)
    scores, selection = _blind_gated_scores(fresh.records, utility, opportunity)
    selected_rows = [
        {
            "reference_id": record.reference_id,
            "neighbor_id": record.pair.neighbor.frame_id,
        }
        for record, chosen in zip(fresh.records, scores, strict=True)
        if chosen > 0.0
    ]
    selection_seal = {
        "candidate_identity_sha256": fresh.receipt["candidate_identity_sha256"],
        "candidate_record_count": len(fresh.records),
        "reference_count": len(reference_ids),
        "training_source_names": list(r31.SOURCE_NAMES),
        "training_receipts": training_receipts,
        "normalizer_sha256": transform.receipt_sha256(),
        "selection": selection,
        "selected_identity_sha256": hashlib.sha256(canonical_json_bytes(selected_rows)).hexdigest().upper(),
        "candidate_depth_reads_before_selection_seal": len(
            set(store.decoded_depth_frame_ids()) & candidate_ids
        ),
    }
    require(
        selection_seal["candidate_depth_reads_before_selection_seal"] == 0,
        "R36 candidate depth read before selection seal",
    )
    selection_seal["seal_sha256"] = hashlib.sha256(
        canonical_json_bytes(selection_seal)
    ).hexdigest().upper()

    candidate_observations = {
        frame_id: store.observation(frame_id) for frame_id in sorted(candidate_ids)
    }
    scorer._attach_targets(fresh.records, contexts, candidate_observations)
    metrics = _confirmation_metrics(fresh.records, scores)
    passed = all(metrics["checks"].values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "FRESH_PARENT_DISJOINT_CONFIRMATION",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "input_receipt": input_receipt,
        "source": source,
        "pose_pair_capability": capability,
        "candidate_algorithm": {
            "name": "TARO_R35_OPENLORIS_REGIME_R31_V7_RELIABILITY_CONSISTENCY",
            "source_regime_policy": "R31_V7_RELIABILITY_CONSISTENCY",
            "ensemble_seeds": list(r31.SEEDS),
            "epochs": r31.EPOCHS,
            "lcb_z": r31.LCB_Z,
            "candidate_depth_in_scorer_input": False,
            "fresh_targets_in_fit": False,
        },
        "selection_seal_before_candidate_depth": selection_seal,
        "metrics": metrics,
        "payload_receipt": store.receipt(),
        "terminal": (
            "TARO_R36_OPENLORIS_CORRIDOR_FRESH_PARENT_CONFIRMATION_PASS"
            if passed
            else "STOP_TARO_R36_OPENLORIS_CORRIDOR_FRESH_PARENT_CONFIRMATION_FAIL"
        ),
        "fresh_parent_confirmation_pass": passed,
        "algorithm_breakthrough_supported": passed,
        "fresh_source_confirmation_pass": False,
        "read_boundary": {
            "source_selection_reads_task_outcome": False,
            "reference_rgb_and_depth_in_scorer_input": True,
            "candidate_rgb_in_scorer_input": True,
            "candidate_depth_in_scorer_input": False,
            "candidate_depth_opened_after_selection_seal": True,
            "candidate_depth_reads_before_selection_seal": 0,
            "fresh_parameters_fit_from_targets": 0,
            "network_requests_during_evaluation": 0,
        },
        "claim_ceiling": "A PASS supports parent-disjoint algorithm confirmation for the mapped OpenLORIS D435i corridor regime. It is not fresh-source-family or broad-domain generalization, collision correctness, Android, product, deployment, navigation, or safety authority.",
        "android_candidate_authorized": False,
        "product_authorized": False,
        "safety_authorized": False,
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--groundtruth-root", type=Path, required=True)
    parser.add_argument("--source-slice-manifest", type=Path, required=True)
    parser.add_argument("--groundtruth-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.cache_root.resolve(),
        args.source_root.resolve(),
        args.groundtruth_root.resolve(),
        args.source_slice_manifest.resolve(),
        args.groundtruth_archive.resolve(),
    )
    if args.output is not None:
        write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

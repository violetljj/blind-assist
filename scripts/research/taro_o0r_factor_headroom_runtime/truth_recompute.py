#!/usr/bin/env python3
"""Recompute dense FARO truth and verify every compact R3 commitment before join."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


class TruthRecomputeError(RuntimeError):
    """Stable error for an R3 commitment/recomputation mismatch."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise TruthRecomputeError(code, message, **context)


def recompute_committed_truth(
    decoded_source_frame: Mapping[str, Any],
    uncertainty_model: Any,
    compact_truth_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return dense verified truth frames only after all R3 hashes reproduce."""

    compact = materializer.validate_eval_truth_commitment_record(dict(compact_truth_record))
    require(isinstance(decoded_source_frame, Mapping), "TRUTH_SOURCE_FRAME_INVALID", "decoded source frame must be an object")
    required_frame_fields = {
        "source_frame_receipt",
        "bound_source_frame_envelope",
        "highres_faro_depth_mm",
        "confidence",
    }
    require(required_frame_fields <= set(decoded_source_frame), "TRUTH_SOURCE_FRAME_INVALID", "decoded source frame lacks required truth payloads")
    source = adapter._validate_base_receipt(dict(decoded_source_frame["source_frame_receipt"]))
    envelope = materializer.validate_bound_source_frame_envelope(dict(decoded_source_frame["bound_source_frame_envelope"]), source)
    require(adapter.canonical_sha256(source) == adapter.canonical_sha256(compact["source_frame_receipt"]), "TRUTH_SOURCE_RECEIPT_MISMATCH", "redecoded source receipt differs from the compact R3 commitment")
    require(adapter.canonical_sha256(envelope) == adapter.canonical_sha256(compact["bound_source_frame_envelope"]), "TRUTH_SOURCE_ENVELOPE_MISMATCH", "redecoded source envelope differs from the compact R3 commitment")
    require(uncertainty_model.content_sha256 == compact["uncertainty_model_sha256"], "TRUTH_UNCERTAINTY_MODEL_MISMATCH", "reloaded uncertainty model differs from the compact R3 commitment")

    highres = np.asarray(decoded_source_frame["highres_faro_depth_mm"])
    confidence = np.asarray(decoded_source_frame["confidence"])
    matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    geometry = adapter.derive_faro_geometry(highres, matrix, source["gravity_up_camera_xyz"], source)
    require(geometry.content_sha256 == compact["faro_geometry_sha256"], "TRUTH_FARO_GEOMETRY_MISMATCH", "recomputed FARO geometry differs from its R3 commitment")
    require(geometry.highres_depth_array_sha256 == compact["highres_depth_array_sha256"], "TRUTH_FARO_DEPTH_MISMATCH", "redecoded FARO depth differs from its R3 commitment")

    query_receipts = adapter.build_query_receipts(source, geometry)
    require(adapter.canonical_sha256(query_receipts) == adapter.canonical_sha256(compact["query_receipts"]), "TRUTH_QUERY_RECEIPT_MISMATCH", "recomputed query receipts differ from R3")
    lookups = [
        materializer.derive_query_uncertainty_lookup(highres, confidence, source, query)
        for query in query_receipts
    ]
    require(adapter.canonical_sha256(lookups) == adapter.canonical_sha256(compact["uncertainty_lookups"]), "TRUTH_UNCERTAINTY_LOOKUP_MISMATCH", "recomputed query uncertainty lookups differ from R3")
    factor_frames = adapter.build_truth_query_factor_frames(
        geometry,
        query_receipts,
        uncertainty_model,
        confidence_values=[lookup["confidence_value"] for lookup in lookups],
        ranges_m=[lookup["range_m"] for lookup in lookups],
    )
    require(len(factor_frames) == len(compact["factor_frame_commitments"]) == 9, "TRUTH_FACTOR_CARDINALITY_MISMATCH", "recomputed truth must contain exactly nine factor frames")
    persisted_results = {row["query_id"]: row for row in compact["query_bundle"]["results"]}
    for index, (frame, commitment, query) in enumerate(zip(factor_frames, compact["factor_frame_commitments"], query_receipts, strict=True)):
        validated = adapter.validate_query_factor_frame(frame)
        require(validated["content_sha256"] == commitment["factor_frame_sha256"], "TRUTH_FACTOR_FRAME_HASH_MISMATCH", "recomputed truth factor frame differs from R3", query_index=index)
        require(validated["base_geometry"]["content_sha256"] == commitment["base_geometry_sha256"], "TRUTH_BASE_GEOMETRY_HASH_MISMATCH", "recomputed truth base geometry differs from R3", query_index=index)
        require(validated["factor_identity"] == commitment["factor_identity"], "TRUTH_FACTOR_IDENTITY_MISMATCH", "recomputed truth factor identity differs from R3", query_index=index)
        require(validated["base_geometry"]["faro_factor_value_sha256s"] == commitment["faro_factor_value_sha256s"], "TRUTH_FACTOR_VALUE_HASH_MISMATCH", "recomputed FARO factor values differ from R3", query_index=index)
        for name in adapter.FACTOR_NAMES:
            metadata = commitment["factor_metadata"][name]
            block = validated["blocks"][name]
            require(
                adapter.canonical_sha256(block["validity"]) == metadata["validity_sha256"]
                and adapter.canonical_sha256(metadata["validity"]) == metadata["validity_sha256"]
                and adapter.canonical_sha256(block["uncertainty"]) == metadata["uncertainty_sha256"]
                and adapter.canonical_sha256(metadata["uncertainty"]) == metadata["uncertainty_sha256"],
                "TRUTH_FACTOR_METADATA_MISMATCH",
                "recomputed factor validity/uncertainty differs from R3",
                query_index=index,
                factor=name,
            )
        result = adapter.reduce_query_factor_frame(validated, query)
        persisted_result = persisted_results.get(validated["query_id"])
        require(
            isinstance(persisted_result, dict)
            and adapter.canonical_sha256(result) == commitment["query_result_sha256"]
            and adapter.canonical_sha256(result) == adapter.canonical_sha256(persisted_result),
            "TRUTH_QUERY_RESULT_MISMATCH",
            "recomputed truth reducer result differs from R3",
            query_index=index,
        )

    bundle = adapter.reduce_complete_query_bundle(factor_frames, query_receipts, source)
    require(adapter.canonical_sha256(bundle) == adapter.canonical_sha256(compact["query_bundle"]), "TRUTH_QUERY_BUNDLE_MISMATCH", "recomputed 9-query truth bundle differs from R3")
    return {
        "source_frame_receipt": copy.deepcopy(source),
        "faro_geometry": geometry,
        "query_receipts": query_receipts,
        "uncertainty_lookups": lookups,
        "truth_factor_frames": factor_frames,
        "truth_query_bundle": bundle,
        "compact_truth_record_sha256": compact["content_sha256"],
    }


__all__ = ["TruthRecomputeError", "recompute_committed_truth"]

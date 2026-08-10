#!/usr/bin/env python3
"""Rehydrate the exact R3 fit-only uncertainty model into a factory-bound model."""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


UNCERTAINTY_ARRAY_ARTIFACT_SCHEMA = "blindassist.taro.o0r.uncertainty_model_array_artifact.v1"
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")


class UncertaintyArtifactError(RuntimeError):
    """Stable error raised when persisted fit-only evidence cannot be trusted."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise UncertaintyArtifactError(code, message, **context)


def _hash(value: Any) -> str:
    return adapter.canonical_sha256(value).upper()


def load_factory_bound_uncertainty_model(
    hydrated_artifact: Mapping[str, Any],
    persisted_receipt: Mapping[str, Any],
    *,
    expected_artifact_canonical_sha256: str,
    expected_model_sha256: str,
) -> Any:
    """Validate every persisted cell and return an adapter-accepted immutable model.

    ``hydrated_artifact`` must first pass the materializer's content-addressed
    package/blob validation.  This second gate reconstructs the private model
    type and registers the exact fingerprint expected by the adapter.
    """

    require(isinstance(hydrated_artifact, Mapping), "UNCERTAINTY_ARTIFACT_INVALID", "hydrated uncertainty artifact must be an object")
    require(isinstance(persisted_receipt, Mapping), "UNCERTAINTY_RECEIPT_INVALID", "persisted uncertainty receipt must be an object")
    require(isinstance(expected_artifact_canonical_sha256, str) and bool(_SHA256.fullmatch(expected_artifact_canonical_sha256)), "UNCERTAINTY_EXPECTED_HASH_INVALID", "expected artifact hash is malformed")
    require(isinstance(expected_model_sha256, str) and bool(_SHA256.fullmatch(expected_model_sha256)), "UNCERTAINTY_EXPECTED_HASH_INVALID", "expected model hash is malformed")

    artifact = copy.deepcopy(dict(hydrated_artifact))
    require(_hash(artifact) == expected_artifact_canonical_sha256.upper(), "UNCERTAINTY_ARTIFACT_CANONICAL_HASH_MISMATCH", "hydrated uncertainty artifact differs from its bound canonical hash")
    expected_artifact_keys = {
        "schema",
        "uncertainty_model_receipt",
        "factory_bound_model_sha256",
        "cells",
        "content_sha256",
    }
    require(set(artifact) == expected_artifact_keys and artifact.get("schema") == UNCERTAINTY_ARRAY_ARTIFACT_SCHEMA, "UNCERTAINTY_ARTIFACT_INVALID", "uncertainty artifact key/schema drift")
    observed_artifact_seal = artifact.pop("content_sha256")
    require(isinstance(observed_artifact_seal, str) and bool(_SHA256.fullmatch(observed_artifact_seal)) and _hash(artifact) == observed_artifact_seal.upper(), "UNCERTAINTY_ARTIFACT_SEAL_MISMATCH", "uncertainty artifact self-seal drift")
    artifact["content_sha256"] = observed_artifact_seal.upper()

    receipt = copy.deepcopy(dict(persisted_receipt))
    expected_receipt_keys = {
        "schema",
        "content_sha256",
        "fit_parent_ids",
        "source_receipt_sha256s",
        "source_evidence_sha256",
        "source_frame_count",
        "support_frame_observations",
        "observation_counts",
        "cells_sha256",
        "cell_count",
    }
    require(set(receipt) == expected_receipt_keys and receipt.get("schema") == adapter.UNCERTAINTY_MODEL_SCHEMA, "UNCERTAINTY_RECEIPT_INVALID", "uncertainty model receipt key/schema drift")
    require(artifact["uncertainty_model_receipt"] == receipt, "UNCERTAINTY_ARTIFACT_RECEIPT_MISMATCH", "artifact-embedded and separately persisted receipts differ")
    require(receipt["content_sha256"] == expected_model_sha256.upper() and artifact["factory_bound_model_sha256"] == expected_model_sha256.upper(), "UNCERTAINTY_MODEL_BINDING_MISMATCH", "uncertainty model identity differs from the execution binding")

    expected_parents = tuple(parent_id for parent_id, _ in adapter.ADAPTER_FIT_ROSTER)
    require(tuple(receipt["fit_parent_ids"]) == expected_parents, "UNCERTAINTY_MODEL_ROSTER_MISMATCH", "persisted uncertainty model fit roster drift")
    source_hashes = tuple(receipt["source_receipt_sha256s"])
    require(
        len(source_hashes) == receipt["source_frame_count"] > 0
        and source_hashes == tuple(sorted(set(source_hashes)))
        and all(isinstance(value, str) and bool(_SHA256.fullmatch(value)) for value in source_hashes),
        "UNCERTAINTY_MODEL_SOURCE_RECEIPTS_INVALID",
        "persisted uncertainty source receipts are malformed, duplicated, or unordered",
    )
    require(isinstance(receipt["source_evidence_sha256"], str) and bool(_SHA256.fullmatch(receipt["source_evidence_sha256"])), "UNCERTAINTY_MODEL_SOURCE_EVIDENCE_INVALID", "persisted uncertainty source evidence hash is malformed")
    require(isinstance(receipt["support_frame_observations"], int) and not isinstance(receipt["support_frame_observations"], bool), "UNCERTAINTY_MODEL_SUPPORT_COUNT_INVALID", "persisted support observation count is invalid")
    counts = receipt["observation_counts"]
    require(isinstance(counts, dict) and set(counts) == set(adapter.UNCERTAINTY_TARGETS) and all(isinstance(counts[target], int) and not isinstance(counts[target], bool) and counts[target] >= 0 for target in adapter.UNCERTAINTY_TARGETS), "UNCERTAINTY_MODEL_OBSERVATION_COUNTS_INVALID", "persisted uncertainty observation counts drift")

    raw_cells = artifact["cells"]
    require(isinstance(raw_cells, list) and len(raw_cells) == receipt["cell_count"] > 0, "UNCERTAINTY_MODEL_CELLS_INVALID", "persisted uncertainty cells are empty or miscounted")
    require(_hash(raw_cells) == receipt["cells_sha256"], "UNCERTAINTY_MODEL_CELL_HASH_MISMATCH", "persisted uncertainty cell hash drift")
    target_rank = {target: index for index, target in enumerate(adapter.UNCERTAINTY_TARGETS)}
    parent_rank = {parent: index for index, parent in enumerate(expected_parents)}
    cells: list[Any] = []
    identities: list[tuple[int, int, int, int]] = []
    for index, raw in enumerate(raw_cells):
        require(isinstance(raw, Mapping) and set(raw) == {"target", "parent_id", "confidence", "range_bin", "values"}, "UNCERTAINTY_MODEL_CELL_INVALID", "persisted uncertainty cell fields drift", index=index)
        target, parent = raw["target"], raw["parent_id"]
        confidence, range_bin = raw["confidence"], raw["range_bin"]
        require(target in target_rank and parent in parent_rank, "UNCERTAINTY_MODEL_CELL_INVALID", "persisted uncertainty cell target/parent drift", index=index)
        require(isinstance(confidence, int) and not isinstance(confidence, bool) and confidence in (0, 1, 2), "UNCERTAINTY_MODEL_CELL_INVALID", "persisted uncertainty confidence drift", index=index)
        require(isinstance(range_bin, int) and not isinstance(range_bin, bool) and 0 <= range_bin < len(adapter.RANGE_EDGES_M) - 1, "UNCERTAINTY_MODEL_CELL_INVALID", "persisted uncertainty range bin drift", index=index)
        values = np.asarray(raw["values"])
        require(values.dtype == np.dtype(np.float64) and values.ndim == 1 and values.size > 0 and bool(np.all(np.isfinite(values))) and bool(np.all(values >= 0.0)), "UNCERTAINTY_MODEL_CELL_INVALID", "persisted uncertainty values must be non-empty finite non-negative float64", index=index)
        immutable = np.ascontiguousarray(values, dtype=np.float64).copy()
        immutable.setflags(write=False)
        cells.append(adapter._UncertaintyCell(str(target), str(parent), int(confidence), int(range_bin), immutable))
        identities.append((target_rank[str(target)], parent_rank[str(parent)], int(confidence), int(range_bin)))
    require(identities == sorted(identities) and len(identities) == len(set(identities)), "UNCERTAINTY_MODEL_CELL_ORDER_INVALID", "persisted uncertainty cells are duplicated or reordered")

    recomputed_counts = {
        target: sum(int(cell.values.size) for cell in cells if cell.target == target)
        for target in adapter.UNCERTAINTY_TARGETS
    }
    require(recomputed_counts == counts, "UNCERTAINTY_MODEL_OBSERVATION_COUNTS_INVALID", "persisted uncertainty counts do not equal cell values")
    model_payload = adapter._uncertainty_model_payload(
        cells,
        expected_parents,
        source_hashes,
        str(receipt["source_evidence_sha256"]),
        int(receipt["support_frame_observations"]),
        counts,
    )
    require(_hash(model_payload) == expected_model_sha256.upper(), "UNCERTAINTY_MODEL_HASH_MISMATCH", "persisted uncertainty model cannot reproduce its bound content hash")
    model = adapter._UncertaintyModel(
        tuple(cells),
        expected_parents,
        source_hashes,
        str(receipt["source_evidence_sha256"]),
        int(receipt["support_frame_observations"]),
        tuple((target, counts[target]) for target in adapter.UNCERTAINTY_TARGETS),
        expected_model_sha256.upper(),
        adapter._UNCERTAINTY_FACTORY_TOKEN,
    )
    adapter._UNCERTAINTY_FACTORY_FINGERPRINTS[model] = model.content_sha256
    return adapter._validate_uncertainty_model(model)


__all__ = [
    "UNCERTAINTY_ARRAY_ARTIFACT_SCHEMA",
    "UncertaintyArtifactError",
    "load_factory_bound_uncertainty_model",
]

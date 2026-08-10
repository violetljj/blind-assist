#!/usr/bin/env python3
"""Refit exact fit-source residuals after the candidate truth firewall closes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Mapping

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import (
    validate_candidate_phase_completion,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


class UncertaintyRefitError(RuntimeError):
    """Stable error for post-candidate exact uncertainty reconstruction."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise UncertaintyRefitError(code, message, **context)


class _ExactFitSequence(Sequence[dict[str, Any]]):
    def __init__(self, entries: list[tuple[Any, str]], decode_fn: Callable[[Any, str], Mapping[str, Any]]) -> None:
        self._entries = entries
        self._decode_fn = decode_fn
        self.decoded_indices: list[int] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        require(index not in self.decoded_indices, "UNCERTAINTY_REFIT_FRAME_REUSED", "fit source frame was decoded more than once", index=index)
        prepared, token = self._entries[index]
        frame = self._decode_fn(prepared, token)
        self.decoded_indices.append(index)
        return materializer.adapter_fit_source_frame(frame)


def refit_and_verify_uncertainty_model(
    prepared_parents: Sequence[Any],
    *,
    candidate_phase_completion: Mapping[str, Any],
    persisted_receipt: Mapping[str, Any],
    persisted_artifact_canonical_sha256: str,
    decode_fn: Callable[[Any, str], Mapping[str, Any]],
) -> Any:
    """Recreate the pre-rounding fit model and prove it matches the R3 evidence."""

    validate_candidate_phase_completion(dict(candidate_phase_completion))
    require(isinstance(prepared_parents, Sequence) and len(prepared_parents) == 24, "UNCERTAINTY_REFIT_PARENT_PLAN_INVALID", "uncertainty refit requires the exact 24-parent prepared plan")
    entries: list[tuple[Any, str]] = []
    observed_fit_roster: list[tuple[str, str]] = []
    for prepared in prepared_parents:
        parent = prepared.parent
        if parent["role"] != "ADAPTER_FIT":
            continue
        identity = (str(parent["visit_id"]), str(parent["video_id"]))
        adapter._validate_roster_identity("ADAPTER_FIT", *identity)
        observed_fit_roster.append(identity)
        tokens = prepared.frame_plan["exact_timestamp_tokens"]
        require(isinstance(tokens, list) and bool(tokens), "UNCERTAINTY_REFIT_PARENT_PLAN_INVALID", "fit parent has no exact timestamps", parent_id=identity[0])
        entries.extend((prepared, str(token)) for token in tokens)
    require(observed_fit_roster == list(adapter.ADAPTER_FIT_ROSTER) and bool(entries), "UNCERTAINTY_REFIT_ROSTER_DRIFT", "uncertainty refit roster/order drift")
    fit_sequence = _ExactFitSequence(entries, decode_fn)
    model = adapter.fit_uncertainty_model(fit_sequence)
    require(fit_sequence.decoded_indices == list(range(len(entries))), "UNCERTAINTY_REFIT_INCOMPLETE", "uncertainty refit did not decode each exact fit frame once")
    receipt = materializer.uncertainty_model_receipt(model)
    require(adapter.canonical_sha256(receipt) == adapter.canonical_sha256(persisted_receipt), "UNCERTAINTY_REFIT_RECEIPT_MISMATCH", "refitted uncertainty model receipt differs from R3")
    artifact = materializer.uncertainty_model_artifact(model)
    require(adapter.canonical_sha256(artifact) == str(persisted_artifact_canonical_sha256).upper(), "UNCERTAINTY_REFIT_ARTIFACT_MISMATCH", "refitted uncertainty model artifact differs from R3's canonical artifact")
    require(model.content_sha256 == persisted_receipt["content_sha256"], "UNCERTAINTY_REFIT_MODEL_HASH_MISMATCH", "refitted uncertainty model content hash differs from R3")
    model._assert_integrity()
    return model


__all__ = ["UncertaintyRefitError", "refit_and_verify_uncertainty_model"]

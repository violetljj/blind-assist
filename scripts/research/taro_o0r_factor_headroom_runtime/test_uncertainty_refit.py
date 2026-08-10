from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import seal_record
from scripts.research.taro_o0r_factor_headroom_runtime.uncertainty_refit import (
    UncertaintyRefitError,
    refit_and_verify_uncertainty_model,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime import test_source_adapter as fixtures
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


def _completion() -> dict[str, object]:
    runtime = {"runtime": "test"}
    return seal_record(
        {
            "schema": "blindassist.taro.o0r.depthart_candidate_phase_completion.v1",
            "candidate_frame_count": 16,
            "candidate_frame_sequence_sha256": "A" * 64,
            "candidate_frame_record_hashes_sha256": "B" * 64,
            "parent_frame_counts": {parent: 1 for parent, _ in adapter.O0R_EVAL_CANDIDATE_ROSTER},
            "runtime_identity": runtime,
            "runtime_identity_sha256": adapter.canonical_sha256(runtime),
            "truth_frame_packages_opened_before_completion": 0,
            "truth_payload_read_by_candidate_phase": False,
            "truth_alignment_used_by_candidate_phase": False,
            "candidate_outputs_sealed_before_truth_join": True,
        }
    )


class UncertaintyRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = list(fixtures.fit_sources(1))
        cls.original = fixtures.fitted_model(1)
        cls.receipt = materializer.uncertainty_model_receipt(cls.original)
        cls.artifact_hash = adapter.canonical_sha256(materializer.uncertainty_model_artifact(cls.original))
        cls.prepared: list[SimpleNamespace] = []
        cls.source_by_parent: dict[str, dict[str, object]] = {}
        for source in cls.sources:
            receipt = source["source_frame_receipt"]
            parent_id, video_id = receipt["parent_id"], receipt["session_id"]
            cls.source_by_parent[parent_id] = source
            cls.prepared.append(SimpleNamespace(parent={"role": "ADAPTER_FIT", "visit_id": parent_id, "video_id": video_id}, frame_plan={"exact_timestamp_tokens": [receipt["sensor_timestamp"]["decimal_token"]]}))
        for parent_id, video_id in adapter.O0R_EVAL_CANDIDATE_ROSTER:
            cls.prepared.append(SimpleNamespace(parent={"role": "O0R_EVAL_CANDIDATE", "visit_id": parent_id, "video_id": video_id}, frame_plan={"exact_timestamp_tokens": ["1.000000000"]}))

    def test_refit_reproduces_persisted_receipt_and_artifact(self) -> None:
        model = refit_and_verify_uncertainty_model(
            self.prepared,
            candidate_phase_completion=_completion(),
            persisted_receipt=self.receipt,
            persisted_artifact_canonical_sha256=self.artifact_hash,
            decode_fn=lambda prepared, _token: self.source_by_parent[prepared.parent["visit_id"]],
        )
        self.assertEqual(model.content_sha256, self.original.content_sha256)

    def test_truth_firewall_and_artifact_swap_fail_closed(self) -> None:
        breached = copy.deepcopy(_completion())
        breached["truth_payload_read_by_candidate_phase"] = True
        breached.pop("content_sha256")
        breached = seal_record(breached)
        with self.assertRaises(Exception):
            refit_and_verify_uncertainty_model(
                self.prepared,
                candidate_phase_completion=breached,
                persisted_receipt=self.receipt,
                persisted_artifact_canonical_sha256=self.artifact_hash,
                decode_fn=lambda prepared, _token: self.source_by_parent[prepared.parent["visit_id"]],
            )
        with self.assertRaises(UncertaintyRefitError) as swapped:
            refit_and_verify_uncertainty_model(
                self.prepared,
                candidate_phase_completion=_completion(),
                persisted_receipt=self.receipt,
                persisted_artifact_canonical_sha256="0" * 64,
                decode_fn=lambda prepared, _token: self.source_by_parent[prepared.parent["visit_id"]],
            )
        self.assertEqual(swapped.exception.code, "UNCERTAINTY_REFIT_ARTIFACT_MISMATCH")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Focused assembly tests for the factor one-shot runner."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime import run_factor_headroom as runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime.test_source_adapter import (
    eval_geometry,
    source_receipt,
)


class FactorRunnerAssemblyTests(unittest.TestCase):
    def test_descriptive_frame_accounts_for_all_queries_and_retains_extractor_failure(self) -> None:
        source = source_receipt()
        queries = adapter.build_query_receipts(source, eval_geometry())
        truth_frames = [{"query_id": query["query_id"], "base_geometry": {}} for query in queries]
        lookups = [
            {"query_id": query["query_id"], "confidence_value": 2, "range_m": 1.5}
            for query in queries
        ]
        verified_truth = {
            "source_frame_receipt": source,
            "query_receipts": queries,
            "truth_factor_frames": truth_frames,
            "uncertainty_lookups": lookups,
            "compact_truth_record_sha256": "B" * 64,
        }
        output_receipt = {"content_sha256": "C" * 64}
        rejected_query = queries[3]["query_id"]

        def candidate_builder(*args: object, **kwargs: object) -> dict[str, object]:
            query = args[4]
            if query["query_id"] == rejected_query:
                raise adapter.AdapterError("CANDIDATE_COMMON_SUPPORT_INSUFFICIENT", "synthetic failure")
            return {"query_id": query["query_id"]}

        def canary_builder(parent_id: str, truth: dict[str, object], candidate: dict[str, object], *_: object) -> dict[str, object]:
            self.assertEqual(parent_id, source["parent_id"])
            self.assertEqual(truth["query_id"], candidate["query_id"])
            return {"query_id": truth["query_id"], "synthetic_validated_record": True}

        with mock.patch.object(runner.adapter, "build_candidate_query_factor_frame", side_effect=candidate_builder), mock.patch.object(
            runner,
            "build_factor_canary_record",
            side_effect=canary_builder,
        ):
            frame = runner._build_descriptive_factor_canary_frame(
                {"highres_faro_depth_mm": np.zeros(adapter.HIGHRES_SHAPE_HW, dtype=np.float32)},
                verified_truth,
                np.zeros(adapter.HIGHRES_SHAPE_HW, dtype=np.float32),
                output_receipt,
                object(),
            )

        self.assertEqual(frame["query_attempt_count"], 9)
        self.assertEqual(frame["record_count"], 8)
        self.assertEqual(frame["failure_count"], 1)
        self.assertEqual(frame["candidate_extractor_failures"][0]["query_id"], rejected_query)
        self.assertEqual(frame["candidate_extractor_failures"][0]["error_code"], "CANDIDATE_COMMON_SUPPORT_INSUFFICIENT")
        observed_hash = frame.pop("content_sha256")
        self.assertEqual(observed_hash, adapter.canonical_sha256(frame))


if __name__ == "__main__":
    unittest.main()

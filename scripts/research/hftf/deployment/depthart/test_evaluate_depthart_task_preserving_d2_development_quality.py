#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import unittest

from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d2_development_quality import (
    PROTOCOL_ID,
    SCHEMA,
    evaluate,
)
from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_development_chunk import (
    chunk_schedule,
)


def decision(state: str) -> dict[str, object]:
    return {"state": state, "unknown_reasons": []}


def clearance(value: float) -> dict[str, object]:
    return {"clearance_valid": True, "clearance_m": value, "unknown_reasons": []}


class D2DevelopmentQualityTest(unittest.TestCase):
    def fixture(self) -> tuple[dict, dict]:
        scope, rows = [], []
        for session_index in range(4):
            visit_id = f"visit-{session_index}"
            video_id = f"video-{session_index}"
            stems = [f"frame_{session_index}_{index:03d}" for index in range(300)]
            scope.append({
                "visit_id": visit_id,
                "video_id": video_id,
                "frame_count": 300,
                "frame_stems_sha256": hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest().upper(),
            })
            for frame_index, stem in enumerate(stems):
                state = "OCCUPIED" if frame_index % 2 else "CLEAR"
                frame_bands = []
                for band in ("left", "center", "right"):
                    frame_bands.append({
                        "band": band,
                        "truth": clearance(1.0),
                        "reference": clearance(1.0),
                        "candidate": clearance(1.0),
                        "cells": [
                            {
                                "horizon_m": horizon,
                                "truth": decision(state),
                                "reference": decision(state),
                                "candidate": decision(state),
                            }
                            for horizon in (1.0, 1.5, 2.0)
                        ],
                    })
                rows.append({
                    "parent_id": visit_id, "session_id": video_id,
                    "frame_index": frame_index, "frame_id": stem,
                    "timestamp_ns": session_index * 1_000_000_000_000 + frame_index * 100_000_000,
                    "orientation": "portrait", "bands": frame_bands,
                })
        gates = {
            "known_coverage_min": 0.9, "clearance_mae_m_max": 0.2,
            "false_clear_all_known_max": 0.08, "false_block_given_clear_max": 0.02,
            "temporal_clearance_delta_mae_m_max": 0.15,
            "geometry_transition_agreement_min": 0.9, "valid_to_unknown_rate_max": 0.1,
            "worst_parent_false_clear_all_known_max": 0.12,
            "noninferiority_against_baseline": {
                "known_coverage_decrease_max": 0.02, "clearance_mae_m_increase_max": 0.025,
                "false_clear_all_known_increase_max": 0.01,
                "false_clear_given_occupied_increase_max": 0.02,
                "false_block_given_clear_increase_max": 0.01,
                "temporal_clearance_delta_mae_m_increase_max": 0.025,
                "geometry_transition_agreement_decrease_max": 0.02,
                "valid_to_unknown_rate_increase_max": 0.02,
            },
        }
        return (
            {"protocol_id": PROTOCOL_ID, "development_scope": scope, "quality_gates": gates},
            {"schema": SCHEMA, "protocol_id": PROTOCOL_ID, "rows": rows},
        )

    def test_identical_candidate_passes_all_gates(self) -> None:
        protocol, payload = self.fixture()
        result = evaluate(protocol, payload)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["gates"]["absolute"].values()))
        self.assertTrue(all(result["gates"]["noninferiority"].values()))
        self.assertFalse(result["authority"]["r2_candidate_lock"])

    def test_order_drift_fails_closed(self) -> None:
        protocol, payload = self.fixture()
        payload["rows"][0]["frame_index"] = 9
        with self.assertRaisesRegex(ValueError, "ordered frame indices drift"):
            evaluate(protocol, payload)

    def test_chunk_schedule_is_exactly_24_by_50(self) -> None:
        protocol, _ = self.fixture()
        protocol["execution"] = {"chunk_size_frames": 50}
        chunks = chunk_schedule(protocol)
        self.assertEqual(len(chunks), 24)
        self.assertTrue(all(chunk["frame_stop"] - chunk["frame_start"] == 50 for chunk in chunks))
        self.assertEqual(chunks[6]["session_index"], 1)


if __name__ == "__main__":
    unittest.main()

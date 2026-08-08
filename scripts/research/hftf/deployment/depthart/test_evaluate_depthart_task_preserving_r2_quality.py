import copy
import json
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_r2_quality import (
    SCHEMA,
    evaluate,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL = json.loads((
    REPO_ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json"
).read_text(encoding="utf-8"))


def decision(state: str, clearance: float | None) -> dict:
    return {"state": state, "clearance_m": clearance}


def payload(candidate_shift: float = 0.01) -> dict:
    rows = []
    for parent_index in range(2):
        for frame_index, states in enumerate((
            ("CLEAR", "OCCUPIED", "CLEAR"),
            ("OCCUPIED", "CLEAR", "OCCUPIED"),
            ("CLEAR", "OCCUPIED", "CLEAR"),
        )):
            items = []
            for band_index, state in enumerate(states):
                truth_clearance = 2.0 if state == "CLEAR" else 1.0
                items.append({
                    "band": ("left", "center", "right")[band_index],
                    "truth": decision(state, truth_clearance),
                    "reference": decision(state, truth_clearance + 0.005),
                    "candidate": decision(state, truth_clearance + candidate_shift),
                })
            rows.append({
                "parent_id": f"parent-{parent_index}",
                "session_id": f"session-{parent_index}",
                "frame_id": f"frame-{frame_index}",
                "timestamp_ns": frame_index * 100_000_000,
                "decisions": items,
            })
    return {"schema": SCHEMA, "protocol_id": PROTOCOL["protocol_id"], "rows": rows}


class EvaluateDepthArtTaskPreservingR2QualityTest(unittest.TestCase):
    def test_task_preserving_candidate_passes(self) -> None:
        result = evaluate(PROTOCOL, payload())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["downstream"]["candidate_partition_performance"], "ELIGIBLE")

    def test_false_clear_fails_closed(self) -> None:
        data = payload()
        for row in data["rows"]:
            for item in row["decisions"]:
                if item["truth"]["state"] == "OCCUPIED":
                    item["candidate"] = decision("CLEAR", 2.0)
        result = evaluate(PROTOCOL, data)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["absolute"]["false_clear_all_known"])

    def test_unknown_is_not_negative_and_coverage_fails(self) -> None:
        data = payload()
        for row in data["rows"]:
            for item in row["decisions"]:
                item["candidate"] = decision("UNKNOWN_GROUND", None)
        result = evaluate(PROTOCOL, data)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["candidate"]["pooled"]["known_coverage"], 0.0)
        self.assertFalse(result["gates"]["aggregation_complete"])

    def test_gate_drift_changes_decision_only_through_protocol(self) -> None:
        strict = copy.deepcopy(PROTOCOL)
        strict["gates"]["clearance_mae_m_max"] = 0.001
        result = evaluate(strict, payload())
        self.assertFalse(result["gates"]["absolute"]["clearance_mae_m"])


if __name__ == "__main__":
    unittest.main()

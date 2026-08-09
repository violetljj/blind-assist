import copy
import hashlib
import unittest

from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d1_quality import (
    SCHEMA,
    PROTOCOL_ID,
    evaluate,
)


GATES = {
    "known_coverage_min": .9, "known_coverage_decrease_max": .02,
    "clearance_mae_m_max": .2, "clearance_mae_m_increase_max": .025,
    "false_clear_all_known_max": .08, "false_clear_all_known_increase_max": .01,
    "false_clear_given_occupied_increase_max": .02,
    "false_block_given_clear_max": .02, "false_block_given_clear_increase_max": .01,
    "temporal_clearance_delta_mae_m_max": .15,
    "temporal_clearance_delta_mae_m_increase_max": .025,
    "geometry_transition_agreement_min": .9,
    "geometry_transition_agreement_decrease_max": .02,
    "valid_to_unknown_rate_max": .1, "valid_to_unknown_rate_increase_max": .02,
    "worst_parent_false_clear_all_known_max": .12,
}
PROTOCOL = {"protocol_id": PROTOCOL_ID, "gates": GATES,
            "quality_pass_terminal": "PASS", "quality_fail_terminal": "FAIL"}


def decision(state):
    return {"state": state}


def payload(candidate_shift=.01):
    rows = []
    for parent in range(8):
        for frame in range(300):
            bands = []
            for band_index, band in enumerate(("left", "center", "right")):
                truth_clearance = (0.8 if frame % 2 else 2.2) + .01 * band_index
                truth = {"clearance_valid": True, "clearance_m": truth_clearance}
                reference = {"clearance_valid": True, "clearance_m": truth_clearance + .005}
                candidate = {"clearance_valid": True, "clearance_m": truth_clearance + candidate_shift}
                cells = []
                for horizon in (1.0, 1.5, 2.0):
                    state = "OCCUPIED" if truth_clearance <= horizon else "CLEAR"
                    cells.append({"horizon_m": horizon, "truth": decision(state),
                                  "reference": decision(state), "candidate": decision(state)})
                bands.append({"band": band, "truth": truth, "reference": reference,
                              "candidate": candidate, "cells": cells})
            rows.append({"parent_id": f"p{parent}", "session_id": f"s{parent}",
                         "frame_id": f"f{frame}", "frame_index": frame,
                         "timestamp_ns": frame * 10_000_000,
                         "orientation": "portrait", "bands": bands})
    protocol_sessions = []
    for parent in range(8):
        stems = [f"f{frame}" for frame in range(300)]
        protocol_sessions.append({"visit_id": f"p{parent}", "video_id": f"s{parent}",
                                  "frame_stems_sha256": hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest().upper()})
    PROTOCOL["cohort"] = {"ordered_sessions": protocol_sessions}
    return {"schema": SCHEMA, "protocol_id": PROTOCOL_ID, "rows": rows}


class EvaluateD1QualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good = payload()

    def test_exact_8x300_nine_cell_payload_passes(self):
        result = evaluate(PROTOCOL, self.good)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["counts"]["cells"], 21600)
        self.assertEqual(len(result["candidate"]["by_grid"]), 9)

    def test_false_clear_fails(self):
        data = copy.deepcopy(self.good)
        for row in data["rows"]:
            for band in row["bands"]:
                for cell in band["cells"]:
                    if cell["truth"]["state"] == "OCCUPIED":
                        cell["candidate"] = decision("CLEAR")
        self.assertEqual(evaluate(PROTOCOL, data)["status"], "FAIL")

    def test_unknown_is_not_negative_and_fails_coverage(self):
        data = copy.deepcopy(self.good)
        for row in data["rows"]:
            for band in row["bands"]:
                band["candidate"] = {"clearance_valid": False, "clearance_m": None}
                for cell in band["cells"]:
                    cell["candidate"] = decision("UNKNOWN_GROUND")
        result = evaluate(PROTOCOL, data)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["candidate"]["pooled"]["known_coverage"], 0.0)

    def test_missing_horizon_fails_schema(self):
        data = copy.deepcopy(self.good)
        data["rows"][0]["bands"][0]["cells"].pop()
        with self.assertRaisesRegex(ValueError, "three horizons"):
            evaluate(PROTOCOL, data)


if __name__ == "__main__":
    unittest.main()

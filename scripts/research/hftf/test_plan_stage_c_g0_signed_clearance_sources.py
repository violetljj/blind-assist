from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_g0_signed_clearance_sources import (  # noqa: E402
    _resolve_parent,
    _validate_outcome_and_freshness_chain,
    _validate_role_sets,
)


def _records(prefix: str, count: int) -> list[dict[str, str]]:
    return [{"session_id": f"{prefix}-{index}"} for index in range(count)]


def _chain() -> tuple[dict, ...]:
    f0_records: list[dict] = []
    for index in range(12):
        f0_records.append(
            {
                "session_id": f"f0-{index}",
                "inventory_eligible": True,
                "inventory_eligible_rank": index + 1,
                "role": (
                    "train"
                    if index < 6
                    else "dev"
                    if index < 9
                    else "heldout"
                ),
            }
        )
    f0_1_records = [
        {
            "session_id": (
                f"f0-{index}" if index < 9 else f"test-{index - 9}"
            ),
            "role": (
                "train" if index < 6 else "dev" if index < 9 else "heldout"
            ),
            "official_split": "train" if index < 9 else "test",
        }
        for index in range(12)
    ]
    f0 = {
        "inventory_candidates": f0_records,
        "geometry_outcome_read": False,
        "teacher_outcome_read": False,
        "student_outcome_read": False,
    }
    f0_1 = {
        "sources": f0_1_records,
        "geometry_outcome_read": False,
        "teacher_outcome_read": False,
        "student_outcome_read": False,
    }
    source_lock = {"sources": [dict(item) for item in f0_1_records]}
    acquisition = {"sources": [dict(item) for item in f0_1_records]}
    cohort = {"sources": [dict(item) for item in f0_1_records]}
    opportunity = {
        "source_results": [dict(item) for item in f0_1_records]
    }
    result = {
        "burn_and_authorization": {
            "official_test_parent_sessions_consumed_for_f0_1_effect": [
                "test-0",
                "test-1",
                "test-2",
            ]
        }
    }
    return (
        f0,
        f0_1,
        source_lock,
        acquisition,
        cohort,
        opportunity,
        result,
    )


class StageCG0SignedClearanceSourcePlanTest(unittest.TestCase):
    def test_exact_disjoint_role_counts_pass(self) -> None:
        _validate_role_sets(
            _records("train", 9),
            _records("dev", 3),
            _records("heldout", 3),
        )

    def test_cross_role_parent_overlap_fails(self) -> None:
        train = _records("train", 9)
        dev = _records("dev", 3)
        heldout = _records("heldout", 3)
        heldout[0]["session_id"] = dev[0]["session_id"]
        with self.assertRaisesRegex(ValueError, "multiple roles"):
            _validate_role_sets(train, dev, heldout)

    def test_wrong_role_count_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "count"):
            _validate_role_sets(
                _records("train", 8),
                _records("dev", 3),
                _records("heldout", 3),
            )

    def test_parent_resolution_routes_artifacts_to_repo_root(self) -> None:
        protocol = (
            Path(__file__).resolve().parents[3]
            / "docs"
            / "research"
            / "hftf"
            / "protocol.json"
        )
        path = _resolve_parent(
            protocol,
            {"path": "artifacts.local/evidence/example.json"},
        )
        self.assertEqual(
            (
                Path(__file__).resolve().parents[3]
                / "artifacts.local"
                / "evidence"
                / "example.json"
            ).resolve(),
            path,
        )

    def test_complete_outcome_chain_preserves_three_fresh_sources(
        self,
    ) -> None:
        documents = _chain()
        _, fresh = _validate_outcome_and_freshness_chain(
            *documents, historical_burned=set()
        )
        self.assertEqual(["f0-9", "f0-10", "f0-11"], fresh)

    def test_authority_source_drift_fails_closed(self) -> None:
        documents = list(_chain())
        documents[4]["sources"][0]["session_id"] = "drift"
        with self.assertRaisesRegex(ValueError, "authority-cohort"):
            _validate_outcome_and_freshness_chain(
                *documents, historical_burned=set()
            )

    def test_open_f0_outcome_firewall_fails_closed(self) -> None:
        documents = list(_chain())
        documents[0]["teacher_outcome_read"] = True
        with self.assertRaisesRegex(ValueError, "firewall"):
            _validate_outcome_and_freshness_chain(
                *documents, historical_burned=set()
            )

    def test_historically_burned_fresh_source_fails_closed(self) -> None:
        documents = _chain()
        with self.assertRaisesRegex(ValueError, "outcome-open"):
            _validate_outcome_and_freshness_chain(
                *documents, historical_burned={"f0-10"}
            )


if __name__ == "__main__":
    unittest.main()

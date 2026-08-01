from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_f0_sanpo_inventory import (
    _role,
    _select_timeline,
    _validate_frozen_inputs,
    _validate_split,
)


def _objects(count: int = 50) -> dict[int, dict[str, str]]:
    return {
        index: {"name": f"{index:06d}.png"}
        for index in range(count)
    }


class StageCF0SanpoInventoryPlanTest(unittest.TestCase):
    def test_physical_timeline_is_fps_specific(self) -> None:
        objects = _objects()
        target_5, frames_5 = _select_timeline(
            objects, objects, objects, 5.0
        )
        target_20, frames_20 = _select_timeline(
            objects, objects, objects, 20.0
        )
        self.assertEqual(5.0, target_5)
        self.assertEqual(list(range(25)), frames_5)
        self.assertEqual(10.0, target_20)
        self.assertEqual(list(range(0, 50, 2)), frames_20)

    def test_missing_source_frame_or_wrong_fps_fails_closed(self) -> None:
        incomplete = _objects(49)
        with self.assertRaisesRegex(ValueError, "0..49"):
            _select_timeline(
                incomplete, incomplete, incomplete, 5.0
            )
        objects = _objects()
        with self.assertRaisesRegex(ValueError, "exactly 5 or 20"):
            _select_timeline(objects, objects, objects, 10.0)

    def test_roles_are_exact_and_bounded(self) -> None:
        self.assertEqual(
            ["train"] * 6 + ["dev"] * 3 + ["heldout"] * 3,
            [_role(rank) for rank in range(1, 13)],
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            _role(13)

    def test_split_drift_fails_closed(self) -> None:
        split = "a\nb\n"
        source = {
            "split_object_generation": "7",
            "split_text_sha256": hashlib.sha256(
                split.encode()
            ).hexdigest(),
        }
        _validate_split(source, "7", split)
        with self.assertRaisesRegex(ValueError, "generation drift"):
            _validate_split(source, "8", split)

    def test_parent_and_additional_burn_union_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs/research/hftf"
            docs.mkdir(parents=True)
            artifacts = (
                root
                / "artifacts.local/evidence/hftf/r4-lock/cohort.json"
            )
            artifacts.parent.mkdir(parents=True)
            for name in ("r4.md", "e0.md", "mechanics.json"):
                (docs / name).write_text(name, encoding="utf-8")
            parent_ledger_path = docs / "r4-ledger.json"
            parent_ledger_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "blindassist_hftf_r4_source_pool_burn_ledger"
                        ),
                        "status": "FROZEN_BEFORE_R4_OUTCOME",
                        "burned_session_ids": ["old"],
                    }
                ),
                encoding="utf-8",
            )
            artifacts.write_text(
                json.dumps(
                    {
                        "schema": (
                            "blindassist_hftf_r4_obstacle_opportunity_"
                            "cohort_lock"
                        ),
                        "terminal": (
                            "R4_OBSTACLE_OPPORTUNITY_COHORT_QUALIFIED"
                        ),
                        "required_sessions": [
                            {"source_session_id": "new"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sha = lambda path: hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            ledger_path = docs / "ledger.json"
            ledger_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "blindassist_hftf_stage_c_sanpo_body_head_"
                            "source_pool_burn_ledger_f0"
                        ),
                        "status": "FROZEN_BEFORE_F0_SOURCE_OUTCOME",
                        "parent_r4_burn_ledger": {
                            "path": "r4-ledger.json",
                            "sha256": sha(parent_ledger_path),
                            "burned_session_count": 1,
                        },
                        "r4_obstacle_cohort_lock": {
                            "path": (
                                "artifacts.local/evidence/hftf/r4-lock/"
                                "cohort.json"
                            ),
                            "sha256": sha(artifacts),
                        },
                        "additional_r4_outcome_open_session_ids": [
                            "new"
                        ],
                        "effective_burned_session_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            protocol_path = docs / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "blindassist_hftf_stage_c_sanpo_body_head_"
                            "temporal_student_canary_f0"
                        ),
                        "status": "FROZEN_BEFORE_F0_SOURCE_OUTCOME",
                        "parents": {
                            "stage_b_r4_result": {
                                "path": "r4.md",
                                "sha256": sha(docs / "r4.md"),
                            },
                            "egowalk_route_closure": {
                                "path": "e0.md",
                                "sha256": sha(docs / "e0.md"),
                            },
                            "swept_envelope_mechanics": {
                                "path": "mechanics.json",
                                "sha256": sha(docs / "mechanics.json"),
                            },
                            "source_pool_burn_ledger": {
                                "path": "ledger.json"
                            },
                        },
                        "source_selection": {
                            "exclude_effective_burned_session_count": 2
                        },
                    }
                ),
                encoding="utf-8",
            )
            _, burned = _validate_frozen_inputs(
                protocol_path, ledger_path, root
            )
            self.assertEqual({"old", "new"}, burned)


if __name__ == "__main__":
    unittest.main()

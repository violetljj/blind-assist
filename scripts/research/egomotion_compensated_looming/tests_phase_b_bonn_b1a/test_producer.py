from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_b1a import (
    geometry,
    protocol,
)
from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_b1a import (
    validator,
)


class B1AProducerContractTest(unittest.TestCase):
    def test_atomic_publish_works_on_frozen_windows_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "receipt.json"
            protocol.atomic_write_json(target, {"version": 1})
            protocol.atomic_write_json(target, {"version": 2})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"version": 2},
            )

    def test_geometry_and_independent_validator_parse_same_fixture(self) -> None:
        index = b"# comment\r\n1.00 rgb/a.png\n1.03 rgb/b.png\n"
        pose = (
            b"1.00 0 0 0 0 0 0 1\n"
            b"1.03 0 0 0 0 0 0 1\n"
        )
        producer_index = geometry.parse_index_text(index)
        validator_index = validator.parse_index_text(index)
        self.assertEqual(
            [(row.timestamp, row.path, row.source_row_rank) for row in producer_index],
            [(row.timestamp, row.path, row.source_row_rank) for row in validator_index],
        )
        producer_pose = geometry.parse_pose_text(pose)
        validator_pose = validator.parse_pose_text(pose)
        self.assertEqual(
            [row.timestamp for row in producer_pose],
            [row.timestamp for row in validator_pose],
        )

    def test_candidate_and_depth_join_match_independent_validator(self) -> None:
        rgb_raw = b"1.000 rgb/a.png\n1.019 rgb/b.png\n1.040 rgb/c.png\n"
        depth_raw = (
            b"0.999 depth/a.png\n1.020 depth/b.png\n1.041 depth/c.png\n"
        )
        p_rgb = geometry.parse_index_text(rgb_raw)
        p_depth = geometry.parse_index_text(depth_raw)
        v_rgb = validator.parse_index_text(rgb_raw)
        v_depth = validator.parse_index_text(depth_raw)
        p_pairs = geometry.adjacent_pairs(p_rgb, Decimal("1"), Decimal("2"))
        v_pairs = validator.adjacent_pairs(v_rgb, Decimal("1"), Decimal("2"))
        self.assertEqual(
            [(row["dt"], row["candidate"]) for row in p_pairs],
            [(row["dt"], row["candidate"]) for row in v_pairs],
        )
        p_join = geometry.assign_depth_rows(
            p_rgb, p_depth, Decimal("1"), Decimal("2")
        )
        v_join = validator.assign_depth_rows(
            v_rgb, v_depth, Decimal("1"), Decimal("2")
        )
        self.assertEqual(
            {
                key: None if value is None else value.source_row_rank
                for key, value in p_join.items()
            },
            v_join,
        )

    def test_relative_geometry_matches_independent_validator(self) -> None:
        identity = np.asarray([0.0, 0.0, 0.0, 1.0])
        previous = (np.asarray([0.0, 0.0, 0.0]), identity)
        current = (np.asarray([0.01, 0.0, 0.0]), identity)
        produced = geometry.relative_geometry(previous, current, 0.05)
        checked = validator.relative_geometry(previous, current, 0.05)
        np.testing.assert_allclose(
            produced["R_current_from_previous"],
            checked["R_current_from_previous"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            produced["t_current_from_previous"],
            checked["t_current_from_previous"],
            atol=1e-12,
        )
        self.assertAlmostEqual(
            produced["translation_speed_m_s"],
            checked["translation_speed"],
            places=12,
        )
        self.assertAlmostEqual(
            produced["angular_rate_rad_s"],
            checked["angular_rate_rad_s"],
            places=12,
        )

    def test_truth_and_roles_match_independent_validator(self) -> None:
        previous = np.full((480, 640), 5000, dtype=np.uint16)
        current = previous.copy()
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        produced = geometry.evaluate_truth(
            previous, current, rotation, translation, 0.04
        )
        checked = validator.evaluate_truth(
            previous, current, rotation, translation, 0.04
        )
        for left, right in zip(produced, checked, strict=True):
            self.assertEqual(
                {
                    key: value
                    for key, value in left.items()
                    if key != "c_truth_grid"
                },
                {
                    key: value
                    for key, value in right.items()
                    if key != "c_truth_grid"
                },
            )
            self.assertAlmostEqual(
                float(left["c_truth_grid"]),
                float(right["c_truth_grid"]),
                places=12,
            )

    def test_implementation_lock_is_fail_closed_before_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = root / "pre.md"
            design = root / "design.json"
            lock_path = root / "lock.json"
            source = root / "source.py"
            for path in (prereg, design, source):
                path.write_text("fixture", encoding="utf-8")
            lock_path.write_text(
                json.dumps(
                    {
                        "design_lock_sha256": protocol.DESIGN_LOCK_SHA256,
                        "preregistration_sha256": protocol.PREREGISTRATION_SHA256,
                        "canonical_execution_authorized": False,
                        "source_files": {"source.py": protocol.sha256_file(source)},
                    }
                ),
                encoding="utf-8",
            )
            paths = {
                "preregistration": prereg,
                "design_lock": design,
                "implementation_lock": lock_path,
            }
            with mock.patch.object(
                protocol,
                "sha256_file",
                side_effect=lambda path: {
                    prereg: protocol.PREREGISTRATION_SHA256,
                    design: protocol.DESIGN_LOCK_SHA256,
                }.get(path, __import__("hashlib").sha256(path.read_bytes()).hexdigest()),
            ):
                with self.assertRaisesRegex(
                    ValueError, "CANONICAL_EXECUTION_NOT_AUTHORIZED"
                ):
                    protocol.validate_implementation_lock(root, paths)

    def test_runner_claim_contains_machine_required_fields(self) -> None:
        runner = (
            Path(__file__).parents[1] / "run_phase_b_bonn_b1a.py"
        ).read_text(encoding="utf-8")
        for token in (
            '"application_data_operations_before_claim": 0',
            '"claim_permanently_retained": True',
            '"delete_replace_or_rewrite_claim": "FORBIDDEN"',
            "os.O_CREAT | os.O_EXCL | os.O_WRONLY",
            'argv == ["--validate-existing"]',
        ):
            self.assertIn(token, runner)
        self.assertLess(
            runner.index("_create_permanent_claim(runner_sha256)"),
            runner.index("_run_formal(claim)"),
        )

    def test_protocol_never_writes_claim_or_reads_rgb_member_bytes(self) -> None:
        source = Path(protocol.__file__).read_text(encoding="utf-8")
        self.assertIn("B1A_PROTOCOL_MUST_NOT_WRITE_RUN_CLAIM", source)
        self.assertNotIn("bundle.read(rgb_path)", source)
        self.assertNotIn("bundle.read(previous.path)", source)
        self.assertIn('"rgb_member_bytes_read": 0', source)


if __name__ == "__main__":
    unittest.main()

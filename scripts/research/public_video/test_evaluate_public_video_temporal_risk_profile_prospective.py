import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import evaluate_public_video_temporal_risk_profile_prospective as subject


class TemporalRiskProfileProspectiveTest(unittest.TestCase):
    def test_relative_peak_formula_is_per_horizon_then_mean(self) -> None:
        logits = np.zeros((3, 2, 2), dtype=np.float32)
        logits[:, 0, 0] = np.asarray([2.0, 1.0, 0.5])
        logits[:, 1, 1] = np.asarray([1.0, -1.0, 0.5])
        mask = np.asarray([[False, False], [False, True]])
        actual = subject.relative_peak_readout(logits, mask)
        expected = [np.exp(-1.0), np.exp(-2.0), 1.0]
        np.testing.assert_allclose(expected, actual["per_horizon_relative_peak"], rtol=1e-7)
        self.assertAlmostEqual(float(np.mean(expected)), actual["frame_score"])

    def test_no_marker_is_exactly_zero(self) -> None:
        actual = subject.relative_peak_readout(np.ones((3, 2, 2)), np.zeros((2, 2), dtype=bool))
        self.assertEqual([0.0, 0.0, 0.0], actual["per_horizon_relative_peak"])
        self.assertEqual(0.0, actual["frame_score"])

    def test_fixed_threshold_boundary_belongs_only_to_positive(self) -> None:
        self.assertTrue(subject.threshold_check(0.68, True, 0.68))
        self.assertFalse(subject.threshold_check(0.68, False, 0.68))
        self.assertTrue(subject.threshold_check(0.679999, False, 0.68))

    def test_rejects_existing_r754_to_r765_source(self) -> None:
        old = "wikimedia_commons_poptravel_london_westminster_piccadilly_2019"
        with self.assertRaisesRegex(ValueError, "r7.54-r7.65 derivation source"):
            subject.ensure_prospective_source(old, {old})

    def test_derivation_lineage_collects_ids_and_video_hashes(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            report = root / "features.json"
            report.write_text(
                '{"sources":[{"source_id":"old","video_sha256":"ABC"}]}',
                encoding="utf-8",
            )
            contract = root / "contract.json"
            contract.write_text(
                '{"feature_reports":{"old":{"path":"features.json","sha256":"unused"}}}',
                encoding="utf-8",
            )
            with mock.patch.object(subject.Path, "cwd", return_value=root), \
                    mock.patch.object(subject.common, "sha256_file", return_value="unused"):
                source_ids, video_hashes = subject.forbidden_lineage_from_derivation_contract(contract)
        self.assertEqual({"old"}, source_ids)
        self.assertEqual({"abc"}, video_hashes)

    def test_lifecycle_replay_isolates_the_review_bound_event(self) -> None:
        selected = {
            "event_entry_timestamp_ms": 328000,
            "last_active_timestamp_ms": 339000,
            "radial_approach_passed": True,
        }
        other = {
            "event_entry_timestamp_ms": 300000,
            "last_active_timestamp_ms": 311000,
            "radial_approach_passed": True,
        }
        expected = {"intervals": [], "reminder_timestamps_ms": [328000]}
        with mock.patch.object(
            subject.gap,
            "radial_entry_lifecycle",
            return_value=expected,
        ) as replay:
            actual = subject.replay_selected_event_lifecycle(
                [{"timestamp_ms": 0}],
                {"unused": True},
                selected,
            )
        self.assertEqual(expected, actual)
        replay.assert_called_once_with(
            [{"timestamp_ms": 0}],
            {"unused": True},
            [selected],
            clear_absent_samples=9,
        )
        self.assertNotIn(other, replay.call_args.args[2])


if __name__ == "__main__":
    unittest.main()

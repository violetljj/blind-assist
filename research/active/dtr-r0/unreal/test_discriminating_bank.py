"""Paired geometry and unchanged predictor cadence checks; no UE or held-out use."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import discriminating_bank as bank
import scenario_bank as historical
import street_bank_loader as loader
import ue_dtr_replay as replay


class DiscriminatingBankTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4] / "artifacts.local" / "tmp"
        root.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="discriminating-bank-test-", dir=root)
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "bank.json"

    def test_eight_unique_supported_scenes_in_four_pairs(self):
        specs = bank.scenario_catalog()
        self.assertEqual(8, len({s["id"] for s in specs}))
        self.assertEqual(8, len({bank.digest(s["actors"]) for s in specs}))
        self.assertEqual(4, len({s["pair_id"] for s in specs}))
        for pair in {s["pair_id"] for s in specs}:
            self.assertEqual(2, sum(s["pair_id"] == pair for s in specs))
        for spec in specs:
            self.assertEqual(.1, spec["dt_s"])
            for actor in spec["actors"]:
                self.assertIn(actor["kind"], ("pedestrian", "occluder", "barrier", "low_obstacle"))
                self.assertEqual([actor["x_m"], actor["y_m"]], actor["waypoints"][0][1:])

    def test_contacts_and_pair_contrasts_remain_valid_across_render_radius(self):
        for radius in (.2, .28, .6, .9, .95):
            with self.subTest(radius=radius):
                specs = bank.scenario_catalog(radius)
                self.assertTrue(bank.validate_specs(specs)["passed"])
                self.assertEqual([True, False, True, False, True, False, True, True],
                                 [bool(bank.proxy_contacts(s)) for s in specs])
                self.assertEqual([False, True], [bool(bank.witness_contacts(s)) for s in specs[-2:]])

    def test_relief_pair_changes_only_height_not_location_or_material(self):
        a, b = [copy.deepcopy(s["actors"][0]) for s in bank.scenario_catalog()[4:6]]
        self.assertEqual(.12, a.pop("height_m"))
        self.assertEqual(.004, b.pop("height_m"))
        self.assertEqual(a, b)

    def test_pair_contact_is_late_not_initial_overlap(self):
        for spec in bank.scenario_catalog(.95)[:4]:
            self.assertTrue(all(h["time_s"] > 3 for h in bank.proxy_contacts(spec)))

    def test_freeze_dispatch_and_immutable_geometry(self):
        manifest = bank.freeze_manifest(self.path, .6)
        self.assertEqual(manifest["splits"]["development"], loader.load_scenarios(self.path, "development"))
        with self.assertRaises(FileExistsError):
            bank.freeze_manifest(self.path)
        manifest["splits"]["development"][0]["actors"][0]["radius_m"] = .3
        manifest["split_sha256"]["development"] = bank.digest(manifest["splits"]["development"])
        manifest["manifest_sha256"] = bank.digest({k:v for k,v in manifest.items() if k != "manifest_sha256"})
        self.path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "immutable"):
            loader.load_scenarios(self.path, "development")

    def test_new_bank_rejects_reserved_access_before_reading(self):
        with patch.object(bank, "read_manifest", side_effect=AssertionError("must not read")):
            with self.assertRaises(ValueError):
                bank.load_scenarios(self.path, "held_out", allow_held_out=True)

    def profile(self, status="PASS"):
        value = {"status": status, "source_sha256": "a" * 64,
                 "samples": [{"actors": [{"kind": "pedestrian", "assessment": {
                     "required_enclosing_radius_m": .660534659}}]}]}
        path = self.path.with_name("profile.json")
        path.write_text(json.dumps(value), encoding="utf-8")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "value": value}

    def test_profile_rejects_undersized_radius_and_failed_probe(self):
        with self.assertRaisesRegex(ValueError, "does not enclose"):
            bank.freeze_manifest(self.path, .6, self.profile())
        self.assertFalse(self.path.exists())
        with self.assertRaisesRegex(ValueError, "must PASS"):
            bank.freeze_manifest(self.path, .7, self.profile("FAIL"))
        manifest = bank.freeze_manifest(self.path, .7, self.profile())
        self.assertEqual(manifest, bank.read_manifest(self.path))

    def test_profile_changed_file_rejected_on_read(self):
        profile = self.profile()
        bank.freeze_manifest(self.path, .7, profile)
        Path(profile["path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file identity"):
            bank.read_manifest(self.path)

    def test_historical_dispatch_retains_own_loader(self):
        # Envelope-only routing check: no reserved geometry or historical builder.
        for schema in historical.SUPPORTED_SCHEMAS:
            self.path.write_text(json.dumps({"schema": schema}), encoding="utf-8")
            with patch.object(historical, "load_scenarios", return_value=["unchanged"]) as called:
                self.assertEqual(["unchanged"], loader.load_scenarios(self.path, "regression"))
                called.assert_called_once_with(self.path, "regression", allow_held_out=False)

    def test_ten_hz_supports_unchanged_fit_where_five_hz_does_not(self):
        for dt, supported in ((.1, True), (.2, False)):
            history = [(i * dt, np.array([2 + i * dt, .3 * i * dt])) for i in range(12)]
            fitted = replay.x24.robust_motion(history, history[-1][0])
            self.assertEqual(supported, fitted is not None)
            if fitted is not None:
                np.testing.assert_allclose(fitted[1], [1., .3])
        self.assertEqual(.5, replay.x24.VELOCITY_WINDOW_S)
        self.assertEqual(4, replay.x24.MINIMUM_FIT_SAMPLES)
        self.assertEqual(1., replay.x24.TRACK_HISTORY_S)


if __name__ == "__main__":
    unittest.main()

"""Narrow geometry/split contract checks. No UE or held-out evaluation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import scenario_bank as bank
import street_scenarios as street


class ScenarioBankTests(unittest.TestCase):
    def setUp(self):
        self.sources = {s["id"]: s for s in street.scenario_catalog()}
        root = Path(__file__).resolve().parents[4] / "artifacts.local" / "tmp"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="scenario-bank-test-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bank.json"

    def test_regression_specs_unchanged_and_copies_independent(self):
        bank.freeze_manifest(self.path)
        specs = bank.load_scenarios(self.path)
        self.assertEqual(specs, street.scenario_catalog())
        specs[0]["actors"][0]["x_m"] = 100
        self.assertEqual(bank.load_scenarios(self.path), street.scenario_catalog())
        self.assertTrue(bank.validate_manifest(self.path)["passed"])
        self.assertTrue(bank.validate_manifest(self.path, "development")["passed"])

    def test_speed_and_delayed_onset_affect_actual_trajectory(self):
        source = self.sources["occluded_crossing_collision"]
        fast = bank.parameterize(source, pedestrian_speed_mps=2)
        delayed = bank.parameterize(source, pedestrian_speed_mps=2, onset_s=1)
        self.assertAlmostEqual(street.actors_at(fast, 1)[0]["y_m"], -2)
        self.assertAlmostEqual(street.actors_at(delayed, 1)[0]["y_m"], -4)
        self.assertAlmostEqual(street.actors_at(delayed, 2)[0]["y_m"], -2)
        self.assertFalse(fast["expected_open_loop_contact"])
        self.assertEqual(fast["control"], "near_miss")
        self.assertEqual(source, self.sources["occluded_crossing_collision"])

    def test_stop_onset_derives_stop_position_from_speed(self):
        spec = bank.parameterize(self.sources["sudden_stop_collision"],
                                 pedestrian_speed_mps=1.5, onset_s=3)
        self.assertAlmostEqual(street.actors_at(spec, 2)[0]["x_m"], 5.4)
        self.assertAlmostEqual(street.actors_at(spec, 4)[0]["x_m"], 6.9)

    def test_static_positions_and_extents_change_render_and_truth(self):
        source = self.sources["low_obstacle_collision"]
        moved = bank.parameterize(source, obstacle_position_m=[5, 1.5])
        self.assertEqual(street.actors_at(moved, 4)[0]["x_m"], 5)
        self.assertEqual(street.actors_at(moved, 4)[0]["y_m"], 1.5)
        self.assertFalse(moved["expected_open_loop_contact"])
        self.assertIsNone(moved["expected_contact_type"])
        occluded = bank.parameterize(self.sources["occluded_crossing_near_miss"],
                                    occluder_position_m=[3, 0], occluder_half_extents_m=[0.6, 0.4])
        actor = next(a for a in street.actors_at(occluded, 5) if a["id"] == "occluder")
        self.assertEqual(actor["half_extents_m"], [0.6, 0.4])
        self.assertEqual(actor["y_m"], 0)
        self.assertTrue(occluded["expected_open_loop_contact"])
        self.assertIn("occluder", {h["actor_id"] for h in bank.proxy_contacts(occluded)})

    def test_reserved_not_evaluated_or_selected(self):
        original = bank.parameterize
        with mock.patch.object(bank, "parameterize", wraps=original) as called:
            manifest = bank.freeze_manifest(self.path)
        self.assertEqual(called.call_count, 16)
        self.assertEqual([len(manifest["splits"][k]) for k in bank.SPLITS], [8,16,8])
        for row in manifest["splits"]["held_out"]:
            self.assertNotIn("expected_open_loop_contact", row)
            self.assertNotIn("actors", row)
            self.assertEqual(row["state"], "RESERVED_NOT_EVALUATED")
        with mock.patch.object(bank, "read_manifest", side_effect=AssertionError("protected read")):
            with self.assertRaisesRegex(ValueError, "RESERVED"):
                bank.load_scenarios(self.path, "held_out")
        self.assertTrue(all("reserved" not in s["id"] for s in bank.load_scenarios(self.path, "development")))

    def test_no_overwrite_repartition_or_consumption_reset(self):
        original = bank.freeze_manifest(self.path)
        with self.assertRaises(FileExistsError):
            bank.freeze_manifest(self.path)
        for change in ("partition", "consumed", "geometry"):
            edited = copy.deepcopy(original)
            if change == "partition":
                edited["splits"]["development"].append(edited["splits"]["held_out"].pop())
            elif change == "consumed":
                edited["splits"]["held_out"][0]["state"] = "CONSUMED"
            else:
                edited["splits"]["development"][0]["actors"][0]["radius_m"] = 0.9
            edited["split_sha256"] = {k:bank.digest(v) for k,v in edited["splits"].items()}
            edited["manifest_sha256"] = bank.digest({k:v for k,v in edited.items() if k != "manifest_sha256"})
            self.path.write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                bank.read_manifest(self.path)

    def test_explicit_future_admission_and_single_consumption_without_truth_access(self):
        bank.freeze_manifest(self.path)
        with self.assertRaisesRegex(ValueError, "admission receipt missing"):
            bank.load_scenarios(self.path, "held_out", allow_held_out=True)
        # This is a disposable unit-test manifest; mock protected materialization.
        # No real reserved geometry or algorithm outcome is evaluated in this test.
        original = bank.parameterize
        def guard(source, **kwargs):
            if "reserved" in kwargs.get("variant_id", ""):
                return {"id": kwargs["variant_id"], "mock_only": True}
            return original(source, **kwargs)
        bank.release_holdout(self.path, "Unit test of admission; geometry mocked")
        with self.assertRaises(FileExistsError):
            bank.release_holdout(self.path, "Repeated release")
        with mock.patch.object(bank, "parameterize", side_effect=guard):
            rows = bank.load_scenarios(self.path, "held_out", allow_held_out=True)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["mock_only"] for row in rows))
            with self.assertRaises(FileExistsError):
                bank.load_scenarios(self.path, "held_out", allow_held_out=True)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            bank.release_holdout(self.path, "Reset")

    def test_v2_development_has_sixteen_distinct_physical_definitions(self):
        manifest = bank.build_manifest()
        self.assertEqual(manifest["schema"], "street-challenge-bank-v2")
        self.assertEqual(manifest["splits"]["regression"], street.scenario_catalog())
        def rounded(value):
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                return round(float(value), 9)
            if isinstance(value, list):
                return [rounded(v) for v in value]
            if isinstance(value, dict):
                return {k: rounded(v) for k,v in value.items()}
            return value
        def physical(spec):
            geometry = {k:spec[k] for k in ("duration_s", "dt_s", "ego_speed_mps", "ego_start")}
            geometry["actors"] = [{k:a[k] for k in ("shape", "x_m", "y_m", "base_m",
                "height_m", "yaw_deg", "waypoints", "radius_m", "half_extents_m") if k in a}
                for a in spec["actors"]]
            return bank.digest(rounded(geometry))
        definitions = [physical(spec) for spec in manifest["splits"]["development"]]
        self.assertEqual(len(definitions), 16)
        self.assertEqual(len(set(definitions)), 16)
        low_reserved = [r["parameters"] for r in manifest["splits"]["held_out"]
                        if r["source_scenario_id"].startswith("low_obstacle_")]
        self.assertEqual(len({bank.digest(r) for r in low_reserved}), 2)

    def test_v1_read_compatibility_preserves_original_split_fingerprints(self):
        legacy = bank.build_manifest("street-challenge-bank-v1")
        self.assertEqual(legacy["split_sha256"], {
            "regression": "bfdd07f7c06c1bc2c7f1048600628d7c20955d26d03c35b0ffd721758055fca6",
            "development": "8c29b4c52a2633d7a51c88955b8ffbbaa909ceaef044f71525d5bdaecb5bf3f2",
            "held_out": "e86c6911ac6018ca51c4c43dd84d3e1f1d65a35c2ed95a9eef034428391fedd0"})
        self.path.write_text(json.dumps(legacy), encoding="utf-8")
        self.assertEqual(bank.read_manifest(self.path), legacy)
        self.assertEqual(bank.load_scenarios(self.path, "development"), legacy["splits"]["development"])
        self.assertEqual(bank.load_scenarios(self.path), street.scenario_catalog())
        with self.assertRaises(ValueError):
            bank.build_manifest("street-challenge-bank-v0")

    def test_invalid_parameters_rejected(self):
        for speed in (0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                bank.parameterize(self.sources["sudden_stop_collision"], pedestrian_speed_mps=speed)
        with self.assertRaises(ValueError):
            bank.parameterize(self.sources["low_obstacle_collision"], onset_s=1)


if __name__ == "__main__":
    unittest.main()

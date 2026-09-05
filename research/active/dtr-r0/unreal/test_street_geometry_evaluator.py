"""Evaluator independently verifies frame-bound native visual evidence."""
import copy
import unittest

import discriminating_bank as bank
import evaluate_street_closed_loop as evaluator
import street_scenarios as street
import visual_geometry as geometry


class GeometryEvaluatorTests(unittest.TestCase):
    def fixture(self):
        spec = bank.scenario_catalog()[4]
        frames, audits = [], []
        for index, t in enumerate((0., .1)):
            frames.append({"sample_index": index, "time_s": t, "ego": {"x_m": 0., "y_m": 0.},
                           "contacts_since_previous": [], "applied_command": {"vx_mps": 0., "vy_mps": 0.}})
            rows = []
            for actor in street.actors_at(spec, t):
                center = [actor["x_m"], actor["y_m"], actor["base_m"] + actor["height_m"] / 2]
                extent = [*actor["half_extents_m"], actor["height_m"] / 2]
                rows.append(dict(geometry.assess_bounds(actor, center, extent, center[:2]), actor_id=actor["id"]))
            audits.append({"time_s": t, "passed": True, "policy": geometry.POLICY, "rows": rows})
        return spec, {"episode_id": "geometry-test", "frames": frames, "visual_geometry": audits,
                      "completed": True, "goal_reached": False, "goal_forward_m": 8.}

    def test_valid_native_bounds_are_recomputed_at_all_frame_times(self):
        spec, episode = self.fixture()
        result = evaluator.evaluate_episode(spec, episode)
        self.assertEqual("COMPLETE", result["status"])
        self.assertEqual(2, result["visual_geometry"]["actor_checks"])

    def test_missing_duplicate_timestamp_actor_or_false_native_bounds_are_invalid(self):
        spec, original = self.fixture()
        for mutation in ("missing", "time", "actor", "bounds", "reported"):
            with self.subTest(mutation=mutation):
                episode = copy.deepcopy(original)
                audit = episode["visual_geometry"][1]
                if mutation == "missing": episode["visual_geometry"].pop()
                elif mutation == "time": audit["time_s"] = 0.
                elif mutation == "actor": audit["rows"][0]["actor_id"] = "wrong"
                elif mutation == "bounds": audit["rows"][0]["native_extent_m"][0] += .1
                else: audit["rows"][0]["passed"] = False
                result = evaluator.evaluate_episode(spec, episode)
                self.assertEqual("INVALID", result["status"])
                self.assertFalse(result["visual_geometry"]["passed"])


if __name__ == "__main__":
    unittest.main()

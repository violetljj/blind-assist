from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("p3_planner", SCRIPTS / "plan_sanpo_p3_session_split.py")
assert SPEC and SPEC.loader
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class SanpoP3SessionSplitPlannerTest(unittest.TestCase):
    def make_recipe(self, root: Path, sessions_per_scene: int = 6) -> Path:
        sequences = []
        native_for_target = (1, 2, 4, 0)
        for scene_index, scene in enumerate(planner.SCENES):
            for session_index in range(sessions_per_scene):
                session_id = f"s{scene_index}_{session_index}"
                package = root / session_id
                masks = package / "source_masks"
                masks.mkdir(parents=True)
                # Every original-resolution mask contains all four mapped classes.
                # The rotating tail makes its raw SHA and class histogram session-specific.
                values = list(native_for_target) * 4
                values[(scene_index + session_index) % len(values)] = native_for_target[
                    (scene_index + 2 * session_index) % 4
                ]
                array = np.asarray(values, dtype=np.uint8).reshape(4, 4)
                mask = masks / "000.png"
                Image.fromarray(array, mode="L").save(mask)
                manifest = package / "manifest.draft.jsonl"
                manifest.write_text(json.dumps({
                    "session_id": session_id,
                    "frame_index": 0,
                    "source_mask_path": "source_masks/000.png",
                    "source": {"official_split": "train", "session_id": session_id},
                }) + "\n", encoding="utf-8")
                sequences.append({
                    "source_id": "sanpo_real_v0",
                    "package_root": str(package),
                    "manifest_path": "manifest.draft.jsonl",
                    "native_session_id": session_id,
                    "official_split": "train",
                    "expected_frame_count": 1,
                    "scene_bucket": scene,
                })
        recipe = root / "candidates.json"
        recipe.write_text(json.dumps({
            "coverage_policy": {
                "format": "legacy_fixture_policy",
                "blind_sequence_count": 2,
                "blind_sequence_frame_count": 60,
            },
            "sources": [{
                "source_id": "sanpo_real_v0",
                "adapter_id": "sanpo_v0",
                "fixture_evidence": "preserve-me",
            }],
            "fixture_top_level_evidence": {"sha256": "a" * 64},
            "sequences": sequences,
        }), encoding="utf-8")
        return recipe

    def test_exact_plan_meets_scene_constraints_and_reports_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            recipe = self.make_recipe(Path(temp))
            plan, report = planner.plan_split(recipe)
            self.assertEqual("not_accessed", report["blind_access"])
            self.assertGreater(report["search"]["search_space"], 0)
            for counts in report["scene_session_counts"].values():
                self.assertEqual((4, 2, 0), (counts["train"], counts["dev"], counts["reserve"]))
            self.assertEqual(16, report["splits"]["train"]["session_count"])
            self.assertEqual(8, report["splits"]["dev"]["session_count"])
            self.assertTrue(all(value > 0 for value in report["splits"]["dev"]["class_pixels"].values()))
            self.assertEqual(report["plan_sha256"], planner.canonical_sha256(plan))
            self.assertEqual("preserve-me", plan["sources"][0]["fixture_evidence"])
            self.assertEqual("a" * 64, plan["fixture_top_level_evidence"]["sha256"])
            self.assertNotIn("reserve", {item["split"] for item in plan["sequences"]})
            self.assertEqual({"train", "dev"}, {item["split"] for item in plan["sequences"]})
            policy = plan["coverage_policy"]
            self.assertEqual("blindassist_sanpo_v4_coverage_policy_v1", policy["format"])
            self.assertEqual(16, policy["min_train_sessions"])
            self.assertEqual(8, policy["min_dev_sessions"])
            self.assertEqual(1, policy["sequence_frame_count"])
            self.assertEqual(2, policy["blind_sequence_count"])
            self.assertEqual(60, policy["blind_sequence_frame_count"])
            self.assertEqual(
                {"train": 4, "dev": 2, "total": 6},
                policy["required_scene_sessions"][planner.SCENES[0]],
            )

    def test_assignment_is_stable_when_candidate_order_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = self.make_recipe(root)
            first, _ = planner.plan_split(recipe)
            payload = json.loads(recipe.read_text(encoding="utf-8"))
            payload["sequences"].reverse()
            reversed_recipe = root / "reversed.json"
            reversed_recipe.write_text(json.dumps(payload), encoding="utf-8")
            second, _ = planner.plan_split(reversed_recipe)
            self.assertEqual(first["assignment_sha256"], second["assignment_sha256"])
            first_assignment = {
                item["native_session_id"]: item["split"] for item in first["sequences"]
            }
            second_assignment = {
                item["native_session_id"]: item["split"] for item in second["sequences"]
            }
            self.assertEqual(first_assignment, second_assignment)

    def test_insufficient_candidates_fail_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = self.make_recipe(root, sessions_per_scene=5)
            plan_path, report_path = root / "plan.json", root / "report.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "planner", "--candidate-recipe", str(recipe),
                    "--output-plan", str(plan_path), "--report", str(report_path),
                ]
                self.assertEqual(1, planner.main())
            finally:
                sys.argv = old_argv
            self.assertFalse(plan_path.exists())
            self.assertFalse(report_path.exists())

    def test_official_test_is_rejected_before_any_manifest_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = self.make_recipe(root)
            payload = json.loads(recipe.read_text(encoding="utf-8"))
            payload["sequences"].append({
                "source_id": "sanpo_real_v0",
                "package_root": str(root / "blind_should_not_be_opened"),
                "manifest_path": "does-not-exist.jsonl",
                "native_session_id": "blind",
                "official_split": "test",
                "scene_bucket": planner.SCENES[0],
            })
            recipe.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(planner.PlanningError, "official test/blind"):
                planner.collect_sessions(recipe)

    def test_duplicate_raw_mask_cannot_cross_train_and_dev(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = self.make_recipe(root)
            payload = json.loads(recipe.read_text(encoding="utf-8"))
            # Force every session to reference the same package-local mask bytes.
            common = root / "common.png"
            Image.fromarray(np.asarray([[1, 2], [4, 0]], dtype=np.uint8), mode="L").save(common)
            for sequence in payload["sequences"]:
                package = Path(sequence["package_root"])
                target = package / "source_masks" / "000.png"
                target.write_bytes(common.read_bytes())
            recipe.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(planner.PlanningError, "no leakage-free|no global leakage-free"):
                planner.plan_split(recipe)

    def test_search_fails_closed_when_no_candidate_passes_distribution_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = self.make_recipe(root)
            payload = json.loads(recipe.read_text(encoding="utf-8"))
            # Leave boundary pixels in only two sessions globally.  No dev split
            # can reach the hard minimum of three independent contributors.
            for sequence in payload["sequences"][2:]:
                mask = Path(sequence["package_root"]) / "source_masks" / "000.png"
                with Image.open(mask) as image:
                    values = np.asarray(image.convert("L"), dtype=np.uint8).copy()
                values[values == 2] = 1
                Image.fromarray(values, mode="L").save(mask)
            with self.assertRaisesRegex(planner.PlanningError, "P3 distribution gates"):
                planner.plan_split(recipe)

    def test_original_resolution_thin_boundary_pixel_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mask = Path(temp) / "thin.png"
            array = np.ones((32, 32), dtype=np.uint8)
            array[0, 0] = 2  # native boundary: one pixel in the original mask
            Image.fromarray(array, mode="L").save(mask)
            counts, _ = planner.native_mask_counts(mask)
            self.assertEqual(1, counts[1])
            self.assertEqual(1023, counts[0])

    def test_distribution_gate_accepts_exact_ratio_and_concentration_boundaries(self) -> None:
        def metrics(shares: dict[str, float], boundary_concentration: float, boundary_sessions: int) -> dict:
            return {
                "class_shares": shares,
                "max_session_contribution": {
                    name: boundary_concentration if name == "boundary_step_curb" else 0.60
                    for name in planner.CLASS_NAMES
                },
                "contributing_session_count": {
                    name: boundary_sessions if name == "boundary_step_curb" else 2
                    for name in planner.CLASS_NAMES
                },
            }

        train = metrics({name: 0.25 for name in planner.CLASS_NAMES}, 0.60, 2)
        dev_shares = {name: 0.25 for name in planner.CLASS_NAMES}
        dev_shares["boundary_step_curb"] = 0.125  # exact symmetric ratio 2.0
        dev = metrics(dev_shares, 0.50, 3)
        ratios, _, failures = planner.distribution_gate(train, dev)
        self.assertEqual(2.0, ratios["boundary_step_curb"])
        self.assertEqual([], failures)

    def test_distribution_gate_rejects_ratio_concentration_and_session_count_overrun(self) -> None:
        shares = {name: 0.25 for name in planner.CLASS_NAMES}
        train = {
            "class_shares": shares,
            "max_session_contribution": {name: 0.60 for name in planner.CLASS_NAMES},
            "contributing_session_count": {name: 2 for name in planner.CLASS_NAMES},
        }
        dev = {
            "class_shares": dict(shares, boundary_step_curb=0.124),
            "max_session_contribution": dict(
                {name: 0.60 for name in planner.CLASS_NAMES}, boundary_step_curb=0.5001,
            ),
            "contributing_session_count": dict(
                {name: 2 for name in planner.CLASS_NAMES}, boundary_step_curb=2,
            ),
        }
        _, _, failures = planner.distribution_gate(train, dev)
        self.assertTrue(any("ratio" in item for item in failures), failures)
        self.assertTrue(any("max session contribution" in item for item in failures), failures)
        self.assertTrue(any("contributing session count" in item for item in failures), failures)


if __name__ == "__main__":
    unittest.main()

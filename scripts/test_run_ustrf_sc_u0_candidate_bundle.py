from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import run_ustrf_sc_u0_candidate_bundle as subject


SCRIPTS = Path(__file__).resolve().parent
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "u0_runner_fixture",
    SCRIPTS / "test_evaluate_ustrf_sc_u0_teacher_upper_bound.py",
)
assert FIXTURE_SPEC and FIXTURE_SPEC.loader
fixture_module = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixture_module)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def rerun(root: Path) -> Path:
    return subject.run_bundle(
        contract_path=root / "u0-contract.json",
        truth_config_path=root / "truth-config.json",
        truth_manifest_path=root / "truth-manifest.json",
        registry_path=root / "registry/registry.json",
        output_dir=root / "prediction-bundle",
        synthetic_fixture=True,
    )


class UstrfU0CandidateRunnerTest(unittest.TestCase):
    def test_subprocess_bundle_uses_only_sanitized_policy_specific_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, truth, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            bundle = root / "prediction-bundle"
            by_truth = {row["episode_id"]: row for row in truth["episodes"]}
            for arm in predictions["arms"]:
                for episode in arm["episodes"]:
                    request = load(bundle / episode["adapter_request_path"])
                    inference = load(bundle / episode["sanitized_inference_manifest_path"])
                    self.assertNotIn("input_video_path", request)
                    self.assertNotIn("source_route_intent_path", request)
                    self.assertFalse(inference["review_fields_present"])
                    self.assertFalse(inference["adjudication_fields_present"])
                    self.assertFalse(inference["event_label_fields_present"])
                    self.assertEqual(500, request["decision_cadence"]["canonical_step_ms"])
                    policy = arm["route_input_policy"]
                    if policy == "no_route_input_v1":
                        self.assertIsNone(inference["adapter_route_input_path"])
                    elif policy == "episode_explicit_causal_route_v1":
                        self.assertEqual(episode["episode_id"], inference["adapter_route_source_episode_id"])
                        self.assertEqual(by_truth[episode["episode_id"]]["route_intent_sha256"], inference["adapter_route_input_sha256"])
                    elif policy == "uniform_full_frame_equal_weight_v1":
                        self.assertIsNone(inference["adapter_route_source_episode_id"])
                        uniform = load(bundle / inference["adapter_route_input_path"])
                        self.assertFalse(uniform["uses_labels"])
                    else:
                        source_id = inference["adapter_route_source_episode_id"]
                        self.assertNotEqual(episode["episode_id"], source_id)
                        self.assertEqual(by_truth[source_id]["route_intent_sha256"], inference["adapter_route_input_sha256"])

    def test_registry_inventory_and_loso_leakage_fail_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            missing = base / "missing"
            fixture_module.UstrfU0TeacherUpperBoundTest.fixture(missing)
            shutil.rmtree(missing / "prediction-bundle")
            registry_path = missing / "registry/registry.json"
            registry = load(registry_path)
            registry["arms"].pop()
            save(registry_path, registry)
            with self.assertRaisesRegex(subject.RunnerError, "every preregistered arm"):
                rerun(missing)

            leak = base / "leak"
            fixture_module.UstrfU0TeacherUpperBoundTest.fixture(leak)
            shutil.rmtree(leak / "prediction-bundle")
            registry_path = leak / "registry/registry.json"
            registry = load(registry_path)
            fitted = next(row for row in registry["arms"] if row["fit_policy"] == "leave_one_session_out_fit_v1")
            fold = fitted["folds"][0]
            manifest_path = leak / "registry" / fold["training_input_manifest"]["path"]
            manifest = load(manifest_path)
            manifest["training_session_ids"] = [manifest["held_out_session_id"]]
            save(manifest_path, manifest)
            fold["training_input_manifest"]["sha256"] = subject.sha256_file(manifest_path)
            save(registry_path, registry)
            with self.assertRaisesRegex(subject.RunnerError, "exact LOSO inventory"):
                rerun(leak)

    def test_nonzero_process_and_frame_drop_never_write_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            failed = base / "failed"
            fixture_module.UstrfU0TeacherUpperBoundTest.fixture(failed)
            shutil.rmtree(failed / "prediction-bundle")
            adapter_path = failed / "registry/adapter.py"
            adapter_path.write_text("raise SystemExit(7)\n", encoding="utf-8")
            registry_path = failed / "registry/registry.json"
            registry = load(registry_path)
            for arm in registry["arms"]:
                arm["implementation"]["sha256"] = subject.sha256_file(adapter_path)
            save(registry_path, registry)
            with self.assertRaisesRegex(subject.RunnerError, "adapter exited 7"):
                rerun(failed)
            self.assertFalse((failed / "prediction-bundle/predictions.json").exists())

            dropped = base / "dropped"
            fixture_module.UstrfU0TeacherUpperBoundTest.fixture(dropped)
            shutil.rmtree(dropped / "prediction-bundle")
            adapter_path = dropped / "registry/adapter.py"
            source = adapter_path.read_text(encoding="utf-8")
            adapter_path.write_text(source.replace("'frames':frames})", "'frames':frames[:-1]})"), encoding="utf-8")
            registry_path = dropped / "registry/registry.json"
            registry = load(registry_path)
            for arm in registry["arms"]:
                arm["implementation"]["sha256"] = subject.sha256_file(adapter_path)
            save(registry_path, registry)
            with self.assertRaisesRegex(subject.RunnerError, "every requested truth frame"):
                rerun(dropped)
            self.assertFalse((dropped / "prediction-bundle/predictions.json").exists())


if __name__ == "__main__":
    unittest.main()

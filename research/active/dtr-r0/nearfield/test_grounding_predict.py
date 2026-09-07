"""Input-boundary tests; no fitting, inference, labels or evaluator reads."""
import json
from pathlib import Path
import tempfile
import unittest

from grounding_predict import load_model_inputs


def project_temporary_directory():
    base = Path(__file__).resolve().parents[4] / 'artifacts.local/nearfield/grounding-20260907/unit-tmp'
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


class GroundingInputTests(unittest.TestCase):
    def fixture(self, root):
        (root / "model").mkdir()
        (root / "verification.json").write_text(json.dumps(dict(status="PASS")))
        dataset = dict(calibration=dict(width=640, height=360, horizontal_fov_degrees=100),
            frames=[dict(sample_index=i, rgb_path=f"{i}.png", time_s=i*.2,
                clip_id="g004_both", frame_in_clip=i) for i in range(3)],
            samples=[dict(sample_id="g004_both_t2", clip_id="g004_both", group_id="g004",
                split="test", frame_indices=[0, 1, 2])])
        self.save(root, dataset)
        return dataset

    def save(self, root, dataset):
        (root / "model/dataset.json").write_text(json.dumps(dataset))

    def test_all_test_without_training_directory(self):
        with project_temporary_directory() as directory:
            root = Path(directory)
            self.fixture(root)
            dataset, paths, hashes = load_model_inputs(root)
            self.assertEqual(len(paths), 3)
            self.assertEqual(dataset["samples"][0]["sample_id"], "g004_both_t2")
            self.assertEqual(set(hashes), {"model/dataset.json", "verification.json"})

    def test_rejects_pose_train_split_and_path_escape(self):
        with project_temporary_directory() as directory:
            root = Path(directory)
            dataset = self.fixture(root)
            dataset["frames"][0]["camera_transform"] = {}
            self.save(root, dataset)
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_model_inputs(root)
            del dataset["frames"][0]["camera_transform"]
            dataset["samples"][0]["split"] = "train"
            self.save(root, dataset)
            with self.assertRaisesRegex(ValueError, "all-test"):
                load_model_inputs(root)
            dataset["samples"][0]["split"] = "test"
            dataset["frames"][0]["rgb_path"] = "../outside.png"
            self.save(root, dataset)
            with self.assertRaisesRegex(ValueError, "contained"):
                load_model_inputs(root)

    def test_verification_required(self):
        with project_temporary_directory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "verification.json").write_text(json.dumps(dict(status="FAIL")))
            with self.assertRaisesRegex(ValueError, "verification"):
                load_model_inputs(root)


if __name__ == "__main__":
    unittest.main()

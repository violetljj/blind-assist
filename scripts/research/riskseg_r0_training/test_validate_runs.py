from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.riskseg_r0_training.validate_runs import (
    EXPECTED_SEEDS,
    build_receipt,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidateRunsTest(unittest.TestCase):
    def make_run(self, root: Path, seed: int) -> Path:
        run = root / f"seed-{seed}"
        run.mkdir()
        checkpoint = run / "best_checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        history = run / "epoch_metrics.jsonl"
        history.write_text(
            "\n".join(
                json.dumps({"epoch": epoch, "dev": {"mean_iou": 0.2}})
                for epoch in range(1, 41)
            )
            + "\n",
            encoding="utf-8",
        )
        recipe = {
            "minimum_epochs": 40,
            "max_epochs": 200,
            "class_order": [
                "walkable",
                "blocking_obstacle",
                "boundary_level_change",
                "unknown_nonwalkable",
            ],
        }
        report = {
            "schema_version": "blindassist.riskseg_r0.pidnet_training.v1",
            "protocol_id": "RISKSEG-R0",
            "seed": seed,
            "status": "TRAINING_COMPLETE_DEV_SELECTED",
            "decision_seed": seed == 20260801,
            "event_eval_outcome_accessed_by_trainer": False,
            "stop_reason": "DEV_MIOU_EARLY_STOPPING",
            "epochs_completed": 40,
            "best_epoch": 20,
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
            "history_path": history.name,
            "history_sha256": sha256(history),
            "implementation_sha256": "implementation",
            "pretrained_sha256": "pretrained",
            "official_repo_commit": "commit",
            "recipe": recipe,
            "data": {
                "manifest_sha256": "manifest",
                "train_sessions": ["train-a"],
                "dev_sessions": ["dev-a"],
                "session_overlap": [],
            },
            "best_dev_metrics": {
                "mean_iou": 0.2 + seed % 3 / 100,
                "boundary_f1_tolerance_1px": 0.1,
                "worst_session_mean_iou": 0.05,
                "per_class_iou": {
                    "walkable": 0.4,
                    "blocking_obstacle": 0.2,
                    "boundary_level_change": 0.1,
                    "unknown_nonwalkable": 0.2,
                },
            },
        }
        (run / "training_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return run

    def test_freezes_exact_fixed_seed_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [self.make_run(root, seed) for seed in EXPECTED_SEEDS]
            receipt = build_receipt(runs)
        self.assertEqual(
            "THREE_FIXED_SEED_TRAINING_ARTIFACTS_FROZEN", receipt["status"]
        )
        self.assertEqual(list(EXPECTED_SEEDS), receipt["seed_order"])
        self.assertTrue(receipt["runs"][0]["decision_seed"])
        self.assertFalse(receipt["runs"][1]["decision_seed"])

    def test_rejects_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [self.make_run(root, seed) for seed in EXPECTED_SEEDS]
            path = runs[-1] / "training_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["data"]["manifest_sha256"] = "drift"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract drift"):
                build_receipt(runs)


if __name__ == "__main__":
    unittest.main()

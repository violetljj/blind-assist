from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_p3_temporal_development_screen_r0 import evaluate_records, validate_training_result, validate_activation_bindings


def protocol() -> dict:
    return {"development_screen": {"decision_rules": {"minimum_delta_mae_improvement_fraction": 0.10, "minimum_parent_improvements": 2, "maximum_raw_abs_rel_increase": 0.02, "maximum_clearance_mae_increase_m": 0.05, "maximum_false_clear_rate_increase": 0.02, "maximum_valid_to_unknown_rate": 0.20}}}


def rows() -> list[dict]:
    result = []
    values = (2.0, 1.0, 2.0, 1.0)
    for parent in range(3):
        for index, value in enumerate(values):
            state = "OCCUPIED" if value <= 1.5 else "CLEAR"
            truth = {"clearance_m": [value] * 3, "state": [state] * 3}
            metric = {"metric_abs_rel_median": .10, "scale_aligned_abs_rel_median": .05}
            a2 = {"depth": metric, "clearance_m": [value + (.2 if index % 2 else 0)] * 3, "state": [state] * 3, "external_abstain": [False] * 3}
            p3 = {"depth": metric, "clearance_m": [value] * 3, "state": [state] * 3, "external_abstain": [False] * 3}
            if index:
                prior = values[index - 1]
                delta = value - prior
                p3["head_from_previous"] = {"clearance_delta_m": [delta] * 3, "transition": [f"{'OCCUPIED' if prior <= 1.5 else 'CLEAR'}_TO_{state}"] * 3}
            result.append({"clip_id": f"clip-{parent}", "parent_id": f"parent-{parent}", "frame_index": index, "truth": truth, "a2": a2, "p3": p3})
    return result


class DevelopmentScreenTest(unittest.TestCase):
    def _protocol_for_receipt(self, protocol_sha: str) -> dict:
        return {"_path": Path("missing-protocol.json"), "a2": {"checkpoint": {"sha256": "A" * 64}}, "teacher_cache": {"depth": {"sha256": "B" * 64}}, "training": {"seed": 20260805}}

    def _activation(self, protocol_sha: str) -> dict:
        return {"schema": "blindassist_p3_temporal_development_screen_r0_activation_bindings", "protocol_sha256": protocol_sha, "claim_ceiling": "DEVELOPMENT_SIGNAL_ONLY", "train_manifest": {"path": "train", "sha256": "C" * 64}, "validation_manifest": {"path": "valid", "sha256": "D" * 64}, "class_weights": {"path": "weights", "sha256": "E" * 64}, "disagreement_cache": {"path": "disagreement", "sha256": "F" * 64}, "runtime_state": {"bonn_sealed_bundle_read": False, "holdout_outcomes_opened": False, "p3_model_constructed": False, "optimizer_constructed": False, "training_started": False, "a2_loaded_only_for_frozen_disagreement": True}, "terminal": "P3_TEMPORAL_DEVELOPMENT_ASSETS_MATERIALIZED_DEVELOPMENT_SIGNAL_ONLY"}

    def _receipt(self, checkpoint: Path, protocol_sha: str, activation_sha: str, epochs: int = 3) -> dict:
        from evaluate_p3_temporal_development_screen_r0 import sha256_file
        return {"schema": "blindassist_p3_temporal_development_screen_r0_training_result", "data_role": "DEVELOPMENT_SIGNAL_ONLY", "claim_ceiling": "DEVELOPMENT_SIGNAL_ONLY", "protocol_sha256": protocol_sha, "activation_bindings_sha256": activation_sha, "train_manifest_sha256": "C" * 64, "validation_manifest_sha256": "D" * 64, "a2_checkpoint_sha256": "A" * 64, "teacher_depth_sha256": "B" * 64, "seed": 20260805, "epochs_completed": epochs, "best_epoch": 2, "best_validation_composite_total": 0.1, "history": [{}, {}, {}], "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}, "training_duration_s": 1.0, "sealed_holdout_opened": False, "terminal": "P3_TEMPORAL_DEVELOPMENT_SCREEN_TRAINING_COMPLETE_EVALUATION_PENDING"}

    def test_training_result_binds_checkpoint_to_activation_not_protocol_candidate_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "p3.pth"
            checkpoint.write_bytes(b"hash-bound checkpoint")
            from evaluate_p3_temporal_development_screen_r0 import sha256_file
            protocol_sha = "9" * 64
            activation_sha = "8" * 64
            protocol = self._protocol_for_receipt(protocol_sha)
            protocol["_path"] = Path(temporary) / "protocol.json"
            protocol["_path"].write_text("protocol")
            # The receipt's protocol SHA is compared to protocol bytes in the real
            # preflight; use a monkey-equivalent digest here.
            protocol_sha = sha256_file(protocol["_path"])
            receipt = self._receipt(checkpoint, protocol_sha, activation_sha)
            actual, digest = validate_training_result(receipt, protocol, self._activation(protocol_sha), activation_sha)
            self.assertEqual(actual, checkpoint.resolve())
            self.assertEqual(digest, sha256_file(checkpoint))

    def test_training_result_rejects_wrong_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protocol_path = Path(temporary) / "protocol.json"; protocol_path.write_text("protocol")
            checkpoint = Path(temporary) / "nope"; checkpoint.write_bytes(b"checkpoint")
            from evaluate_p3_temporal_development_screen_r0 import sha256_file
            protocol = self._protocol_for_receipt(sha256_file(protocol_path)); protocol["_path"] = protocol_path
            receipt = self._receipt(checkpoint, sha256_file(protocol_path), "8" * 64, epochs=2)
            activation = self._activation(sha256_file(protocol_path))
            with self.assertRaisesRegex(ValueError, "three epochs"):
                validate_training_result(receipt, protocol, activation, "8" * 64)

    def test_activation_rejects_runtime_state_drift(self) -> None:
        activation = self._activation("7" * 64)
        activation["runtime_state"]["training_started"] = True
        with self.assertRaisesRegex(ValueError, "runtime-state"):
            validate_activation_bindings(activation, "6" * 64, "7" * 64)

    def test_supported_signal_is_development_only(self) -> None:
        result = evaluate_records(protocol(), rows())
        self.assertEqual(result["terminal"], "P3_TEMPORAL_DEVELOPMENT_SIGNAL_SUPPORTED")
        self.assertEqual(result["data_role"], "DEVELOPMENT_SIGNAL_ONLY")
        self.assertFalse(result["bootstrap_used"])

    def test_not_supported_for_no_delta_improvement(self) -> None:
        value = rows()
        for row in value:
            row["p3"]["clearance_m"] = list(row["a2"]["clearance_m"])
            if "head_from_previous" in row["p3"]:
                row["p3"]["head_from_previous"]["clearance_delta_m"] = [0.0] * 3
        result = evaluate_records(protocol(), value)
        self.assertEqual(result["terminal"], "P3_TEMPORAL_DEVELOPMENT_SIGNAL_NOT_SUPPORTED")

    def test_mixed_when_delta_signal_has_safety_regression(self) -> None:
        value = rows()
        for row in value:
            row["p3"]["external_abstain"] = [True] * 3
        result = evaluate_records(protocol(), value)
        self.assertEqual(result["terminal"], "P3_TEMPORAL_DEVELOPMENT_SIGNAL_MIXED")

    def test_requires_exactly_three_parents(self) -> None:
        value = [row for row in rows() if row["parent_id"] != "parent-2"]
        with self.assertRaisesRegex(ValueError, "exactly three"):
            evaluate_records(protocol(), value)


if __name__ == "__main__":
    unittest.main()

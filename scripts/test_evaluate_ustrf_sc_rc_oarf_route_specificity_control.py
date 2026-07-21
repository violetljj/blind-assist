from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("evaluate_ustrf_sc_rc_oarf_route_specificity_control.py")
SPEC = importlib.util.spec_from_file_location("route_specificity", SCRIPT)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RouteSpecificityControlTest(unittest.TestCase):
    def test_within_image_derangements_pass_without_refit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, repo = self.fixture(root)

            report = subject.evaluate(contract, repo_root=repo)

            self.assertTrue(report["gate_passed"])
            self.assertTrue(report["mechanism_signal_observed"])
            self.assertEqual(6, report["image_count"])
            self.assertEqual(2, report["parent_source_count"])
            self.assertEqual(1.0, report["true_route"]["balanced_accuracy"])
            self.assertTrue(all(
                value["balanced_accuracy"] < 1.0
                for value in report["wrong_route_controls"].values()
            ))

    def test_missing_triplet_or_changed_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, repo = self.fixture(root)
            rows_path = repo / contract["bound_route_examples"]["path"]
            rows = rows_path.read_text(encoding="utf-8").splitlines()
            rows_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            contract["bound_route_examples"]["sha256"] = sha(rows_path)
            contract["bound_r816_report"]["example_count"] -= 1
            with self.assertRaisesRegex(subject.ContractError, "prediction, route-row"):
                subject.evaluate(contract, repo_root=repo)

            contract, repo = self.fixture(root / "second")
            report_path = repo / contract["bound_r816_report"]["path"]
            report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(subject.ContractError, "SHA256 changed"):
                subject.evaluate(contract, repo_root=repo)

    def test_missing_prediction_identity_binding_blocks_formal_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, repo = self.fixture(root)
            report_path = repo / contract["bound_r816_report"]["path"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            del report["evaluation"]["route_conditioned_readout"]["example_ids"]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            contract["bound_r816_report"]["sha256"] = sha(report_path)

            result = subject.evaluate(contract, repo_root=repo)

            self.assertTrue(result["mechanism_signal_observed"])
            self.assertFalse(result["gate_passed"])
            self.assertEqual("BLOCKED_ON_PREDICTION_IDENTITY_BINDING", result["decision"])

    def test_duplicate_wrong_route_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract, repo = self.fixture(Path(temp))
            controls = contract["permutation_policy"]["controls"]
            controls["reverse"] = dict(controls["forward"])
            with self.assertRaisesRegex(subject.ContractError, "distinct derangements"):
                subject.evaluate(contract, repo_root=repo)

    def test_legacy_prediction_replay_requires_exact_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract, repo = self.fixture(Path(temp))
            current_path = repo / contract["bound_r816_report"]["path"]
            legacy_path = repo / "artifacts.local" / "legacy.json"
            legacy = json.loads(current_path.read_text(encoding="utf-8"))
            legacy["evaluation"]["route_conditioned_readout"].pop("example_ids")
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            contract["legacy_prediction_replay"] = {
                "path": "artifacts.local/legacy.json",
                "sha256": sha(legacy_path),
                "schema": "test-r816",
                "prediction_path": "evaluation.route_conditioned_readout.predictions",
                "require_exact_predictions": True,
                "require_exact_evaluation_except_example_ids": True,
            }
            contract["frozen_execution"] = {
                "checkpoint_sha256": "c" * 64,
                "input_size": 224,
                "layer_index": 11,
                "teacher_target": "bbox_distance",
                "distance_sigma_patches": 1.5,
                "seed": 20260719,
                "teacher_ridge": 10.0,
                "head_ridge": 1.0,
                "route_balanced_accuracy_gte": 0.80,
                "each_class_recall_gte": 0.70,
                "balanced_accuracy_gain_over_global_gte": 0.10,
            }
            legacy_contract_path = repo / "configs" / "legacy-control.json"
            legacy_contract_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_scope = {
                key: json.loads(json.dumps(contract[key]))
                for key in ("bound_route_examples", "permutation_policy", "gate", "authority")
            }
            legacy_contract_path.write_text(json.dumps(legacy_scope), encoding="utf-8")
            contract["frozen_at_utc"] = "2026-07-20T19:01:00Z"
            contract["post_run_binding_disclosure"] = {
                "identity_report_was_created_before_this_hash_binding": True,
                "legacy_preregistered_contract": {
                    "path": "configs/legacy-control.json",
                    "sha256": sha(legacy_contract_path),
                },
                "unchanged_from_legacy_contract": [
                    "route rows", "wrong-route permutations", "gate thresholds", "authority",
                ],
                "new_binding_only": "test identity replay",
            }
            stability_contract_path = repo / "configs" / "stability.json"
            stability_contract = {
                "bound_frame_probe": {"sha256": contract["legacy_prediction_replay"]["sha256"]},
                "gate": {
                    "minimum_worst_seed_balanced_accuracy": 0.85,
                    "minimum_worst_seed_each_class_recall": 0.80,
                    "minimum_mean_balanced_accuracy": 0.90,
                    "maximum_seed_balanced_accuracy_stddev": 0.03,
                    "minimum_worst_seed_parent_source_balanced_accuracy": 0.80,
                    "repeat_exact": True,
                },
            }
            stability_contract_path.write_text(json.dumps(stability_contract), encoding="utf-8")
            stability_report_path = repo / "artifacts.local" / "stability-report.json"
            stability_report = {
                "schema": "test-stability",
                "contract_sha256": sha(stability_contract_path),
                "repeat_exact": True,
                "stability": {
                    "mean_balanced_accuracy": 0.87,
                    "stddev_balanced_accuracy": 0.01,
                    "worst_seed_balanced_accuracy": 0.86,
                    "worst_seed_candidate_no_alert_recall": 0.79,
                    "worst_seed_candidate_alert_recall": 0.90,
                    "worst_seed_parent_source_balanced_accuracy": 0.84,
                },
                "prototype_bootstrap_gate": {"passed": False},
            }
            stability_report_path.write_text(json.dumps(stability_report), encoding="utf-8")
            contract["bound_stability_gate"] = {
                "contract": {"path": "configs/stability.json", "sha256": sha(stability_contract_path)},
                "report": {
                    "path": "artifacts.local/stability-report.json",
                    "sha256": sha(stability_report_path),
                    "schema": "test-stability",
                },
                "required_for_student_training": True,
            }

            result = subject.evaluate(contract, repo_root=repo)
            self.assertTrue(result["legacy_prediction_replay"]["exact"])
            self.assertFalse(result["stability_gate"]["passed"])
            self.assertEqual("BLOCKED_ON_R818_STABILITY", result["combined_readiness_decision"])

            stability_report["prototype_bootstrap_gate"]["passed"] = True
            stability_report_path.write_text(json.dumps(stability_report), encoding="utf-8")
            contract["bound_stability_gate"]["report"]["sha256"] = sha(stability_report_path)
            with self.assertRaisesRegex(subject.ContractError, "differs from recomputed frozen checks"):
                subject.evaluate(contract, repo_root=repo)
            stability_report["prototype_bootstrap_gate"]["passed"] = False
            stability_report_path.write_text(json.dumps(stability_report), encoding="utf-8")
            contract["bound_stability_gate"]["report"]["sha256"] = sha(stability_report_path)

            original_gate = contract["gate"]["minimum_true_route_balanced_accuracy"]
            contract["gate"]["minimum_true_route_balanced_accuracy"] = 0.79
            with self.assertRaisesRegex(subject.ContractError, "inherited legacy scope: gate"):
                subject.evaluate(contract, repo_root=repo)
            contract["gate"]["minimum_true_route_balanced_accuracy"] = original_gate

            current = json.loads(current_path.read_text(encoding="utf-8"))
            original_accuracy = current["evaluation"]["global_readout"]["metrics"]["accuracy"]
            current["frozen_risk_representation"]["head_ridge"] = 2.0
            current_path.write_text(json.dumps(current), encoding="utf-8")
            contract["bound_r816_report"]["sha256"] = sha(current_path)
            with self.assertRaisesRegex(subject.ContractError, "execution changed: head_ridge"):
                subject.evaluate(contract, repo_root=repo)

            current["frozen_risk_representation"]["head_ridge"] = 1.0
            current["evaluation"]["global_readout"]["metrics"]["accuracy"] = 0.5
            current_path.write_text(json.dumps(current), encoding="utf-8")
            contract["bound_r816_report"]["sha256"] = sha(current_path)
            with self.assertRaisesRegex(subject.ContractError, "evaluation differs"):
                subject.evaluate(contract, repo_root=repo)

            current["evaluation"]["global_readout"]["metrics"]["accuracy"] = original_accuracy
            current_path.write_text(json.dumps(current), encoding="utf-8")
            contract["bound_r816_report"]["sha256"] = sha(current_path)

            legacy["evaluation"]["route_conditioned_readout"]["predictions"][0] ^= 1
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            contract["legacy_prediction_replay"]["sha256"] = sha(legacy_path)
            with self.assertRaisesRegex(subject.ContractError, "differ from the frozen legacy"):
                subject.evaluate(contract, repo_root=repo)

    @staticmethod
    def fixture(root: Path) -> tuple[dict, Path]:
        repo = root / "repo"
        report_path = repo / "artifacts.local" / "report.json"
        rows_path = repo / "artifacts.local" / "rows.jsonl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        predictions = []
        for source_index, source in enumerate(("source-a", "source-b")):
            for image_index in range(3):
                image_id = f"{source}-{image_index}"
                blocked_choice = (image_index + source_index) % 3
                for choice_index, choice in enumerate(("LEFT", "STRAIGHT", "RIGHT")):
                    label = int(choice_index == blocked_choice)
                    rows.append({
                        "example_id": f"{image_id}-{choice}",
                        "image_id": image_id,
                        "parent_source_id": source,
                        "route_choice": choice,
                        "route_blocked": label,
                        "train_only": True,
                    })
                    predictions.append(label)
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        metrics = subject.binary_metrics([row["route_blocked"] for row in rows], predictions)
        report = {
            "schema": "test-r816",
            "created_at_utc": "2026-07-20T19:00:00Z",
            "dataset_build_receipt_sha256": "a" * 64,
            "dataset_manual_review_sha256": "b" * 64,
            "evaluation": {
                "global_readout": {
                    "predictions": predictions,
                    "example_ids": [row["example_id"] for row in rows],
                    "metrics": metrics,
                },
                "route_conditioned_readout": {
                    "predictions": predictions,
                    "example_ids": [row["example_id"] for row in rows],
                    "metrics": metrics,
                },
                "exact_field_linear_head": {
                    "predictions": predictions,
                    "example_ids": [row["example_id"] for row in rows],
                    "metrics": metrics,
                },
            },
            "frozen_risk_representation": {
                "checkpoint_sha256": "c" * 64,
                "input_size": 224,
                "layer_index": 11,
                "teacher_target": "bbox_distance",
                "distance_sigma_patches": 1.5,
                "seed": 20260719,
                "teacher_ridge": 10.0,
                "head_ridge": 1.0,
            },
            "route_interaction_gate": {"thresholds": {
                "route_balanced_accuracy_gte": 0.80,
                "each_class_recall_gte": 0.70,
                "balanced_accuracy_gain_over_global_gte": 0.10,
            }},
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        contract = {
            "schema": subject.CONTRACT_SCHEMA,
            "contract_id": "test-control",
            "bound_r816_report": {
                "path": "artifacts.local/report.json",
                "sha256": sha(report_path),
                "schema": "test-r816",
                "prediction_path": "evaluation.route_conditioned_readout.predictions",
                "prediction_example_id_path": "evaluation.route_conditioned_readout.example_ids",
                "example_count": len(rows),
            },
            "bound_route_examples": {
                "path": "artifacts.local/rows.jsonl",
                "sha256": sha(rows_path),
                "dataset_build_receipt_sha256": "a" * 64,
                "manual_review_sha256": "b" * 64,
                "required_route_choices": ["LEFT", "STRAIGHT", "RIGHT"],
            },
            "permutation_policy": {
                "kind": "within_image_cyclic_derangements_only",
                "seed": None,
                "threshold_or_model_refit": False,
                "controls": {
                    "forward": {"LEFT": "STRAIGHT", "STRAIGHT": "RIGHT", "RIGHT": "LEFT"},
                    "reverse": {"LEFT": "RIGHT", "STRAIGHT": "LEFT", "RIGHT": "STRAIGHT"},
                },
            },
            "gate": {
                "minimum_true_route_balanced_accuracy": 0.80,
                "minimum_true_route_each_class_recall": 0.70,
                "minimum_true_minus_each_wrong_route_balanced_accuracy": 0.10,
                "every_parent_source_true_route_must_exceed_every_wrong_route": True,
                "formal_gate_requires_identity_bound_predictions": True,
            },
            "authority": {
                "train_only_synthetic_mechanism_evidence": True,
                "real_event_truth": False,
                "route_provider_evaluation": False,
                "student_training": False,
                "calibration": False,
                "blind": False,
                "android_runtime_change": False,
                "production_model_replacement": False,
            },
        }
        return contract, repo


if __name__ == "__main__":
    unittest.main()

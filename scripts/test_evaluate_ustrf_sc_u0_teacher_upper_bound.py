from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import evaluate_ustrf_sc_u0_teacher_upper_bound as evaluator
import run_ustrf_sc_u0_candidate_bundle as runner


SCRIPTS = Path(__file__).resolve().parent
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "route_truth_fixture",
    SCRIPTS / "test_validate_route_conditioned_event_truth.py",
)
assert FIXTURE_SPEC and FIXTURE_SPEC.loader
fixture_module = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixture_module)

PILOT_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "route_pilot_fixture",
    SCRIPTS / "test_validate_ustrf_sc_route_conditioned_event_pilot.py",
)
assert PILOT_FIXTURE_SPEC and PILOT_FIXTURE_SPEC.loader
pilot_fixture_module = importlib.util.module_from_spec(PILOT_FIXTURE_SPEC)
PILOT_FIXTURE_SPEC.loader.exec_module(pilot_fixture_module)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class UstrfU0TeacherUpperBoundTest(unittest.TestCase):
    def test_hash_contract_is_stable_across_json_layout_and_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            json_a = root / "a.json"
            json_b = root / "b.json"
            text_a = root / "a.py"
            text_b = root / "b.py"
            json_a.write_text('{"b": 2, "a": 1}\n', encoding="utf-8", newline="")
            json_b.write_text('{\r\n  "a": 1,\r\n  "b": 2\r\n}\r\n', encoding="utf-8", newline="")
            text_a.write_text("one\ntwo\n", encoding="utf-8", newline="")
            text_b.write_text("one\r\ntwo\r\n", encoding="utf-8", newline="")
            self.assertEqual(evaluator.canonical_json_sha256(json_a), evaluator.canonical_json_sha256(json_b))
            self.assertEqual(evaluator.normalized_text_sha256(text_a), evaluator.normalized_text_sha256(text_b))

    def test_perfect_dense_route_and_causal_arms_pass_frozen_u0_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, truth_config, truth_manifest, predictions, config_sha, manifest_sha = self.fixture(root)

            report = evaluator.evaluate(
                contract,
                truth_config,
                truth_manifest,
                predictions,
                truth_root=root,
                prediction_root=root / "prediction-bundle",
                truth_config_sha256=config_sha,
                truth_manifest_sha256=manifest_sha,
            )

            self.assertTrue(report["truth_gate"]["route_conditioned_truth_eligible"])
            self.assertEqual("GO_TO_S0_RECOMMENDED", report["decision"])
            self.assertTrue(report["u0_passed"])
            self.assertTrue(report["s0_probe_eligible"])
            self.assertFalse(report["student_training_authorized"])
            self.assertFalse(report["production_model_replacement_authorized"])

    def test_valid_synthetic_bundle_never_authorizes_official_u0(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, truth_config, truth_manifest, predictions, config_sha, manifest_sha = self.fixture(root)
            contract["authority"]["synthetic_fixture_can_authorize_u0"] = False
            report = evaluator.evaluate(
                contract,
                truth_config,
                truth_manifest,
                predictions,
                truth_root=root,
                prediction_root=root / "prediction-bundle",
                truth_config_sha256=config_sha,
                truth_manifest_sha256=manifest_sha,
            )
            self.assertTrue(report["prediction_evidence_gate"]["all_alerts_trace_derived"])
            self.assertFalse(report["u0_passed"])
            self.assertFalse(report["s0_probe_eligible"])
            self.assertIn("authority:synthetic_fixture_cannot_authorize_u0", report["failed_checks"])

    def test_independent_json_round_trip_and_duplicate_arm_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, truth_config, truth_manifest, predictions, config_sha, manifest_sha = self.fixture(root)
            independent_contract = json.loads(json.dumps(contract))
            independent_predictions = json.loads(json.dumps(predictions))
            report = evaluator.evaluate(
                independent_contract,
                truth_config,
                truth_manifest,
                independent_predictions,
                truth_root=root,
                prediction_root=root / "prediction-bundle",
                truth_config_sha256=config_sha,
                truth_manifest_sha256=manifest_sha,
            )
            self.assertTrue(report["u0_passed"])
            independent_predictions["arms"].append(json.loads(json.dumps(independent_predictions["arms"][0])))
            with self.assertRaisesRegex(evaluator.ContractError, "exactly match"):
                evaluator.evaluate(
                    independent_contract,
                    truth_config,
                    truth_manifest,
                    independent_predictions,
                    truth_root=root,
                    prediction_root=root / "prediction-bundle",
                    truth_config_sha256=config_sha,
                    truth_manifest_sha256=manifest_sha,
                )

    def test_future_input_hash_drift_and_missing_arm_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, truth_config, truth_manifest, predictions, config_sha, manifest_sha = self.fixture(root)
            predictions["future_inputs_used"] = True
            with self.assertRaisesRegex(evaluator.ContractError, "future-input-free"):
                evaluator.evaluate(
                    contract,
                    truth_config,
                    truth_manifest,
                    predictions,
                    truth_root=root,
                    prediction_root=root / "prediction-bundle",
                    truth_config_sha256=config_sha,
                    truth_manifest_sha256=manifest_sha,
                )

            _, _, _, predictions, _, _ = self.fixture(root)
            predictions["truth_manifest_sha256"] = "f" * 64
            with self.assertRaisesRegex(evaluator.ContractError, "truth_manifest_sha256"):
                evaluator.evaluate(
                    contract,
                    truth_config,
                    truth_manifest,
                    predictions,
                    truth_root=root,
                    prediction_root=root / "prediction-bundle",
                    truth_config_sha256=config_sha,
                    truth_manifest_sha256=manifest_sha,
                )

            _, _, _, predictions, _, _ = self.fixture(root)
            predictions["arms"].pop()
            with self.assertRaisesRegex(evaluator.ContractError, "exactly match"):
                evaluator.evaluate(
                    contract,
                    truth_config,
                    truth_manifest,
                    predictions,
                    truth_root=root,
                    prediction_root=root / "prediction-bundle",
                    truth_config_sha256=config_sha,
                    truth_manifest_sha256=manifest_sha,
                )

            contract, truth_config, truth_manifest, predictions, config_sha, manifest_sha = self.fixture(root)
            causal = next(row for row in predictions["arms"] if row["arm_id"] == "teacher_dense_explicit_route_causal")
            positive = next(row for row in causal["episodes"] if row["alert_timestamps_ms"])
            positive["alert_timestamps_ms"] = [1000]
            with self.assertRaisesRegex(evaluator.ContractError, "prediction evidence gate failed"):
                evaluator.evaluate(
                    contract,
                    truth_config,
                    truth_manifest,
                    predictions,
                    truth_root=root,
                    prediction_root=root / "prediction-bundle",
                    truth_config_sha256=config_sha,
                    truth_manifest_sha256=manifest_sha,
                )

    def test_current_empty_official_manifest_cannot_enter_u0(self) -> None:
        repo = SCRIPTS.parent
        contract_path = repo / "configs/ustrf_sc_u0_teacher_upper_bound_v1.json"
        truth_config_path = repo / "configs/ustrf_sc_route_conditioned_event_collection_v1.json"
        truth_manifest_path = repo / "configs/ustrf_sc_route_conditioned_event_manifest_template_v1.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        truth_config = json.loads(truth_config_path.read_text(encoding="utf-8"))
        truth_manifest = json.loads(truth_manifest_path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(evaluator.ContractError, "truth gate failed"):
            evaluator.evaluate(
                contract,
                truth_config,
                truth_manifest,
                {},
                truth_root=truth_manifest_path.parent,
                prediction_root=truth_manifest_path.parent,
                truth_config_sha256=evaluator.canonical_json_sha256(truth_config_path),
                truth_manifest_sha256=evaluator.canonical_json_sha256(truth_manifest_path),
            )

    @staticmethod
    def fixture(root: Path) -> tuple[dict, dict, dict, dict, str, str]:
        root.mkdir(parents=True, exist_ok=True)
        for child in (root / "prediction-bundle", root / "registry"):
            if child.exists():
                shutil.rmtree(child)
        truth_config, truth_manifest = pilot_fixture_module.RouteConditionedEventPilotTest.package(root)
        truth_manifest.update({
            "schema": "blindassist_sanpo_counterfactual_episode_manifest_v1",
            "contract_id": truth_config["contract_id"],
            "collection_scope": "official_full_matrix",
            "collection_status": "complete",
            "pilot": False,
            "training_eligible": False,
            "android_runtime_change_authorized": False,
        })
        for receipt in truth_manifest["source_receipts"]:
            receipt["origin_scope"] = "official_full_matrix_capture"
        for episode in truth_manifest["episodes"]:
            episode["origin_scope"] = "official_full_matrix_capture"
        contract = json.loads(
            (SCRIPTS.parent / "configs/ustrf_sc_u0_teacher_upper_bound_v1.json").read_text(encoding="utf-8")
        )
        contract["truth_contract_id"] = truth_config["contract_id"]
        contract["truth_requirements"].update({
            "episode_count": 2,
            "matched_pair_count": 1,
            "route_bound_episode_count": 2,
            "minimum_critical_events_each_fold": 1,
        })
        contract["evaluation"]["mechanism_thresholds_aggregate"]["minimum_sessions_with_unknown_low_obstacle_gain"] = 1
        contract["authority"]["synthetic_fixture_can_authorize_u0"] = True
        truth_config["scenes"] = [{"scene_id": "unknown_low_obstacle"}]
        for episode in truth_manifest["episodes"]:
            episode["scene_id"] = "unknown_low_obstacle"
            episode["risk_profile"]["primary_hazard_type"] = "unknown_low_obstacle"
        truth_config_path = root / "truth-config.json"
        truth_manifest_path = root / "truth-manifest.json"
        truth_config_path.write_text(json.dumps(truth_config, sort_keys=True), encoding="utf-8")
        truth_manifest_path.write_text(json.dumps(truth_manifest, sort_keys=True), encoding="utf-8")
        config_sha = evaluator.canonical_json_sha256(truth_config_path)
        manifest_sha = evaluator.canonical_json_sha256(truth_manifest_path)
        contract["truth_requirements"].update({
            "collection_scope": "official_full_matrix",
            "hash_contract": {
                "json": "utf8_canonical_sorted_keys_compact",
                "validator_text": "utf8_lf_normalized",
            },
            "official_truth_config_sha256": config_sha,
            "truth_validator_implementation_sha256": evaluator.normalized_text_sha256(SCRIPTS / "validate_sanpo_counterfactual_episodes.py"),
            "truth_validator_dependency_sha256": {
                filename: evaluator.normalized_text_sha256(SCRIPTS / filename)
                for filename in (
                    "validate_explicit_route_intent_episode.py",
                    "validate_ustrf_sc_capture_frame_ledger.py",
                    "validate_ai_review_receipt.py",
                )
            },
        })
        contract["prediction_evidence_contract"]["validator_implementation_sha256"] = evaluator.normalized_text_sha256(
            SCRIPTS / "validate_ustrf_sc_u0_prediction_bundle.py"
        )
        contract["prediction_evidence_contract"]["runner_implementation_sha256"] = evaluator.normalized_text_sha256(
            SCRIPTS / "run_ustrf_sc_u0_candidate_bundle.py"
        )
        contract["prediction_evidence_contract"]["kernel_execution_backend_id"] = "synthetic_protocol_fixture_v1"
        registry_root = root / "registry"
        registry_root.mkdir()
        kernel_path = registry_root / "synthetic-kernel.txt"
        kernel_path.write_text("synthetic protocol fixture; not Android AssistDecisionKernel\n", encoding="utf-8")
        contract["prediction_evidence_contract"]["shared_decision_kernel_implementation_sha256"] = evaluator.normalized_text_sha256(kernel_path)
        kernel_dependency_path = registry_root / "synthetic-engine.txt"
        kernel_dependency_path.write_text("synthetic kernel dependency fixture\n", encoding="utf-8")
        contract["prediction_evidence_contract"]["shared_decision_kernel_dependency_sha256"] = {
            kernel_dependency_path.name: evaluator.normalized_text_sha256(kernel_dependency_path),
        }
        adapter_path = registry_root / "adapter.py"
        adapter_path.write_text(
            """import argparse, json
p=argparse.ArgumentParser()
p.add_argument('--request', required=True); p.add_argument('--inference-manifest', required=True)
p.add_argument('--inference-root', required=True); p.add_argument('--artifact', required=True)
p.add_argument('--threshold-config', required=True); p.add_argument('--output', required=True)
a=p.parse_args(); r=json.load(open(a.request, encoding='utf-8')); t=json.load(open(a.threshold_config, encoding='utf-8'))
triggered=False; frames=[]
for f in r['frames']:
  hit=(r['episode_id'] in t['trigger_episode_ids'] and not triggered and f['video_pts_ms'] >= t['trigger_at_ms'])
  triggered = triggered or hit
  event_required=r['event_identity_policy']=='kernel_native_required_v1' and hit
  frames.append({**f, 'decision': {'raw_risk_level': 'MEDIUM' if hit else 'NONE', 'stable_risk_level': 'MEDIUM' if hit else 'NONE', 'event_id': 'seg-1' if event_required else None, 'event_state': 'ALERTED' if event_required else None, 'candidate_adapter_id': r['candidate_adapter_id'], 'feedback_receipt': {'outcome': 'TRIGGERED' if hit else 'NO_ALERT', 'kernel_feedback_reason': 'TRIGGERED' if hit else 'NO_FEEDBACK_RISK', 'delivered': hit, 'adapter_id': r['feedback_adapter_id'], 'kernel_contract_id': r['shared_decision_kernel_contract_id']}}})
o={k:v for k,v in r.items() if k not in ('schema','frames')}
o.update({'schema':'blindassist_ustrf_sc_u0_candidate_adapter_output_v1','kernel_trace_order':r['kernel_trace_order'],'execution_completed':True,'failure_count':0,'abstained':False,'frames':frames})
open(a.output,'w',encoding='utf-8').write(json.dumps(o,sort_keys=True))
""",
            encoding="utf-8",
        )
        sessions = sorted({episode["session_id"] for episode in truth_manifest["episodes"]})
        episodes_by_session = {
            session_id: sorted(episode["episode_id"] for episode in truth_manifest["episodes"] if episode["session_id"] == session_id)
            for session_id in sessions
        }
        positive_ids = [episode["episode_id"] for episode in truth_manifest["episodes"] if episode["pair_role"] == "positive"]
        registry = {
            "schema": "blindassist_ustrf_sc_u0_candidate_adapter_registry_v1",
            "contract_id": contract["contract_id"],
            "synthetic_fixture": True,
            "blind_accessed": False,
            "future_inputs_used": False,
            "production_model_replacement_authorized": False,
            "shared_decision_kernel_implementation": {"path": kernel_path.name, "sha256": sha(kernel_path)},
            "shared_decision_kernel_dependencies": [{
                "name": kernel_dependency_path.name,
                "path": kernel_dependency_path.name,
                "sha256": sha(kernel_dependency_path),
            }],
            "kernel_execution_backend_id": "synthetic_protocol_fixture_v1",
            "arms": [],
        }
        passing = {"teacher_dense_explicit_route", "teacher_dense_explicit_route_causal"}
        for arm_contract in contract["required_arms"]:
            arm_id = arm_contract["arm_id"]
            arm_root = registry_root / arm_id
            arm_root.mkdir()
            threshold_path = arm_root / "thresholds.json"
            threshold_path.write_text(json.dumps({
                "trigger_episode_ids": positive_ids if arm_id in passing else [],
                "trigger_at_ms": 500,
            }), encoding="utf-8")
            folds = []
            for held_out in sessions:
                fold_root = arm_root / held_out
                fold_root.mkdir()
                artifact_path = fold_root / "artifact.bin"
                artifact_path.write_bytes(f"synthetic artifact {arm_id} {held_out}".encode())
                fit_policy = arm_contract["fit_policy"]
                training_sessions = [] if fit_policy == "fixed_no_fit_v1" else [value for value in sessions if value != held_out]
                training_episodes = [] if fit_policy == "fixed_no_fit_v1" else sorted(
                    episode_id for session_id in training_sessions for episode_id in episodes_by_session[session_id]
                )
                training_manifest_path = fold_root / "training-input-manifest.json"
                training_manifest_path.write_text(json.dumps({
                    "schema": "blindassist_ustrf_sc_u0_fold_training_input_manifest_v1",
                    "contract_id": contract["contract_id"], "arm_id": arm_id,
                    "candidate_adapter_id": arm_contract["candidate_adapter_id"], "fit_policy": fit_policy,
                    "held_out_session_id": held_out, "truth_manifest_sha256": manifest_sha,
                    "training_session_ids": training_sessions, "training_episode_ids": training_episodes,
                    "held_out_inputs_used": False, "blind_accessed": False, "future_inputs_used": False,
                }, sort_keys=True), encoding="utf-8")
                training_receipt_path = fold_root / "training-receipt.json"
                training_receipt_path.write_text(json.dumps({
                    "schema": "blindassist_ustrf_sc_u0_fold_training_receipt_v1",
                    "contract_id": contract["contract_id"], "arm_id": arm_id,
                    "candidate_adapter_id": arm_contract["candidate_adapter_id"], "fit_policy": fit_policy,
                    "held_out_session_id": held_out, "training_input_manifest_sha256": sha(training_manifest_path),
                    "artifact_sha256": sha(artifact_path), "fit_executed": fit_policy == "leave_one_session_out_fit_v1",
                    "held_out_inputs_used": False, "blind_accessed": False, "future_inputs_used": False,
                    "provenance_completed": True, "failure_count": 0,
                }, sort_keys=True), encoding="utf-8")
                folds.append({
                    "held_out_session_id": held_out,
                    "artifact": {"path": str(artifact_path.relative_to(registry_root)), "sha256": sha(artifact_path)},
                    "training_input_manifest": {"path": str(training_manifest_path.relative_to(registry_root)), "sha256": sha(training_manifest_path)},
                    "training_receipt": {"path": str(training_receipt_path.relative_to(registry_root)), "sha256": sha(training_receipt_path)},
                })
            registry["arms"].append({
                **{key: arm_contract[key] for key in ("arm_id", "candidate_adapter_id", "fit_policy", "event_identity_policy", "route_input_policy")},
                "runtime_id": "python_subprocess_v1",
                "implementation": {"path": adapter_path.name, "sha256": sha(adapter_path)},
                "threshold_config": {"path": str(threshold_path.relative_to(registry_root)), "sha256": sha(threshold_path)},
                "folds": folds,
            })
        registry_path = registry_root / "registry.json"
        registry_path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
        contract_path = root / "u0-contract.json"
        contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
        predictions_path = runner.run_bundle(
            contract_path=contract_path,
            truth_config_path=truth_config_path,
            truth_manifest_path=truth_manifest_path,
            registry_path=registry_path,
            output_dir=root / "prediction-bundle",
            synthetic_fixture=True,
        )
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        return contract, truth_config, truth_manifest, predictions, config_sha, manifest_sha


if __name__ == "__main__":
    unittest.main()

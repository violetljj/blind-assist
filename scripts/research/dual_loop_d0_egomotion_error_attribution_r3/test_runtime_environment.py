from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import runner
from runtime_environment import (
    RuntimeEnvironmentError,
    canonical_json_bytes,
    explicit_third_party_import_roots,
    prestart_dependency_smoke,
    probe_designated_vicon_message,
    sha256_file,
    validate_runtime_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_MANIFEST = (
    REPO_ROOT
    / "artifacts.local/evidence/dual-loop/"
    "d0-egomotion-error-attribution-r3/runtime-freeze/"
    "runtime_environment_manifest.json"
)
REVEL_BAG = (
    REPO_ROOT
    / "artifacts.local/evidence/datasets/"
    "revel-dynamic-bag-v1-20260720/dynamic.bag"
)


class RuntimeEnvironmentTest(unittest.TestCase):
    def assert_typed_equal(self, left: object, right: object) -> None:
        self.assertIs(type(left), type(right))
        if isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_typed_equal(left[key], right[key])
        elif isinstance(left, list):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_typed_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_live_manifest_and_designated_probe_are_exact(self) -> None:
        result = validate_runtime_manifest(RUNTIME_MANIFEST)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(
            result["manifest_sha256"],
            "86ebe10fffd37c4454fc42a0d21fd695a8dd8cddee58d178bc54de2486afb7db",
        )
        probe = probe_designated_vicon_message(REVEL_BAG)
        self.assertEqual(
            probe["status"], "VALID_INHERITED_R2_OPERATIONAL_PROBE"
        )
        self.assertEqual(probe["deserialized_message_count"], 0)
        self.assertEqual(probe["inherited_deserialized_message_count"], 1)
        self.assertFalse(probe["vicon_bag_messages_opened"])
        self.assertFalse(probe["pose_values_retained"])
        self.assertFalse(probe["d0_metrics_computed"])

    def test_manifest_semantic_mutation_is_rejected(self) -> None:
        payload = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
        payload["python"]["version"] = "mutated"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_bytes(canonical_json_bytes(payload) + b"\n")
            with self.assertRaisesRegex(
                RuntimeEnvironmentError, "semantic drift"
            ):
                validate_runtime_manifest(path)

    def test_dependency_smoke_and_ast_import_closure_are_exact(self) -> None:
        self.assertEqual(
            explicit_third_party_import_roots(),
            ["numpy", "rosbags", "yaml"],
        )
        smoke = prestart_dependency_smoke()
        self.assertEqual(smoke["status"], "VALID_SYNTHETIC_RUNTIME_SMOKE")
        self.assertTrue(smoke["yaml_safe_load_called"])
        self.assertEqual(smoke["synthetic_calibration_shape"], [4, 4])
        for key in (
            "real_calibration_opened",
            "bag_messages_opened",
            "truth_opened",
            "event_rows_built",
            "d0_metrics_computed",
        ):
            self.assertFalse(smoke[key])

    def test_yaml_module_source_drift_is_rejected(self) -> None:
        payload = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
        payload["module_sources"]["yaml"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_bytes(canonical_json_bytes(payload) + b"\n")
            with self.assertRaisesRegex(
                RuntimeEnvironmentError, "semantic drift"
            ):
                validate_runtime_manifest(path)

    def test_scientific_modules_are_byte_identical_to_r1(self) -> None:
        r1 = REPO_ROOT / (
            "scripts/research/"
            "dual_loop_d0_egomotion_error_attribution_r1"
        )
        r3 = REPO_ROOT / (
            "scripts/research/"
            "dual_loop_d0_egomotion_error_attribution_r3"
        )
        for name in ("analysis.py", "bindings.py", "producer.py"):
            with self.subTest(name=name):
                self.assertEqual(sha256_file(r3 / name), sha256_file(r1 / name))

    def test_r3_scientific_contract_is_type_exact_with_r2(self) -> None:
        protocol_root = REPO_ROOT / "docs/research/dual-loop"
        r2 = json.loads(
            (
                protocol_root
                / "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R2_PROTOCOL_2026-07-30.json"
            ).read_text(encoding="utf-8")
        )
        r3 = json.loads(
            (
                protocol_root
                / "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
            ).read_text(encoding="utf-8")
        )
        fields = (
            "question",
            "estimand",
            "non_goals",
            "data_role",
            "frozen_inputs",
            "forbidden_inputs",
            "data_join_contract",
            "source_motion_derivation",
            "roi_and_temporal_derivation",
            "summary_math",
            "dependence_contract",
            "missingness_contract",
            "error_partition",
            "preselected_routing_metrics",
            "diagnostic_only_metrics",
            "diagnostic_metric_route_authority",
            "robust_support_definition",
            "person_competing_rule",
            "material_contradiction_rule",
            "routing_booleans",
            "scientific_exit_rule",
            "event_table_contract",
            "conditional_successors",
        )
        for field in fields:
            with self.subTest(field=field):
                self.assert_typed_equal(r2[field], r3[field])
        probe = r3["runtime_environment"]["designated_prestart_probe"]
        self.assertEqual(probe["timestamp_ns"], 1708490365692128652)

    def test_runtime_prestart_failure_does_not_create_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / runner.FORMAL_OUTPUT_RELATIVE_PATH
            activation_path = root / "activation.json"
            lock_path = root / "implementation_lock.json"
            lock_path.write_text("{}\n", encoding="utf-8")
            activation = {
                "formal_execution_authorized": True,
                "authority": {
                    "formal_execution_authorized": True,
                    "successor_execution_authorized": False,
                    "confirmation_authorized": False,
                    "product_or_safety_authorized": False,
                },
                "repository": {
                    "head": "a" * 40,
                    "origin_master": "a" * 40,
                },
            }
            activation_path.write_bytes(
                runner.canonical_json_bytes(activation) + b"\n"
            )
            protocol = {
                "runtime_environment": {
                    "manifest": {
                        "path": "missing.json",
                        "sha256": "b" * 64,
                        "tree_sha256": "c" * 64,
                    }
                }
            }
            with (
                mock.patch.object(runner, "load_protocol", return_value=protocol),
                mock.patch.object(
                    runner,
                    "validate_implementation_lock",
                    return_value={"status": "VALID"},
                ),
                mock.patch.object(runner, "_load_json", return_value=activation),
                mock.patch.object(
                    runner,
                    "validate_activation_identity",
                    return_value={
                        "status": "VALID_IDENTITY",
                        "failures": [],
                    },
                ),
                mock.patch.object(
                    runner, "_validate_runtime_invocation", return_value=None
                ),
                mock.patch.object(
                    runner,
                    "_git_clean_and_at_origin",
                    return_value=activation["repository"],
                ),
                mock.patch.object(
                    runner,
                    "validate_runtime_manifest",
                    side_effect=RuntimeEnvironmentError("synthetic drift"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeEnvironmentError, "synthetic drift"
                ):
                    runner.run_producer(
                        repo_root=root,
                        activation_path=activation_path,
                        implementation_lock_path=lock_path,
                        output_root=output,
                    )
            self.assertFalse(output.exists())

    def test_formal_argv_identity_is_exact(self) -> None:
        activation_path = REPO_ROOT / (
            "artifacts.local/evidence/dual-loop/"
            "d0-egomotion-error-attribution-r3/implementation/activation.json"
        )
        lock_path = activation_path.with_name("implementation_lock.json")
        adapter = REPO_ROOT / (
            "scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py"
        )
        python = Path(runner.sys.executable).resolve()
        activation = {
            "runtime_execution": {
                "python_executable": {
                    "path": str(python).replace("\\", "/"),
                    "sha256": sha256_file(python),
                },
                "argv_contract": {
                    "adapter": adapter.relative_to(REPO_ROOT).as_posix(),
                    "command": "produce",
                    "activation_path": activation_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "implementation_lock_path": lock_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                },
            }
        }
        exact = [
            str(python),
            "-I",
            "-B",
            str(adapter),
            "produce",
            "--activation",
            str(activation_path),
            "--implementation-lock",
            str(lock_path),
        ]
        with mock.patch.object(runner.sys, "orig_argv", exact):
            runner._validate_runtime_invocation(
                activation,
                REPO_ROOT,
                activation_path,
                lock_path,
            )
        base_entrypoint = [*exact]
        base_entrypoint[0] = str(
            Path(
                getattr(
                    runner.sys,
                    "_base_executable",
                    runner.sys.executable,
                )
            ).resolve()
        )
        with mock.patch.object(runner.sys, "orig_argv", base_entrypoint):
            runner._validate_runtime_invocation(
                activation,
                REPO_ROOT,
                activation_path,
                lock_path,
            )
        wrong_entrypoint = [*exact]
        wrong_entrypoint[0] = str(REPO_ROOT / "wrong-python.exe")
        with (
            mock.patch.object(runner.sys, "orig_argv", wrong_entrypoint),
            self.assertRaisesRegex(runner.RunnerError, "argv identity drift"),
        ):
            runner._validate_runtime_invocation(
                activation,
                REPO_ROOT,
                activation_path,
                lock_path,
            )
        mutated = [*exact]
        mutated[4] = "validate-execution"
        with (
            mock.patch.object(runner.sys, "orig_argv", mutated),
            self.assertRaisesRegex(runner.RunnerError, "argv identity drift"),
        ):
            runner._validate_runtime_invocation(
                activation,
                REPO_ROOT,
                activation_path,
                lock_path,
            )
        mismatched_activation = json.loads(json.dumps(activation))
        mismatched_activation["runtime_execution"]["argv_contract"][
            "activation_path"
        ] = "artifacts.local/wrong-activation.json"
        with (
            mock.patch.object(runner.sys, "orig_argv", exact),
            self.assertRaisesRegex(
                runner.RunnerError,
                "activation-declared formal argv path drift",
            ),
        ):
            runner._validate_runtime_invocation(
                mismatched_activation,
                REPO_ROOT,
                activation_path,
                lock_path,
            )


if __name__ == "__main__":
    unittest.main()

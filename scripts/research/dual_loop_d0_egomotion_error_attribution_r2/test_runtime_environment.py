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
    probe_designated_vicon_message,
    sha256_file,
    validate_runtime_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_MANIFEST = (
    REPO_ROOT
    / "artifacts.local/evidence/dual-loop/"
    "d0-egomotion-error-attribution-r2/runtime-freeze/"
    "runtime_environment_manifest.json"
)
REVEL_BAG = (
    REPO_ROOT
    / "artifacts.local/evidence/datasets/"
    "revel-dynamic-bag-v1-20260720/dynamic.bag"
)


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_live_manifest_and_designated_probe_are_exact(self) -> None:
        result = validate_runtime_manifest(RUNTIME_MANIFEST)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(
            result["manifest_sha256"],
            "0faceae2077e87a90bc96da1a9e953dd81bd5c4baeec75779b23fd2f783e823a",
        )
        probe = probe_designated_vicon_message(REVEL_BAG)
        self.assertEqual(probe["status"], "VALID_OPERATIONAL_PROBE")
        self.assertEqual(probe["deserialized_message_count"], 1)
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

    def test_scientific_modules_are_byte_identical_to_r1(self) -> None:
        r1 = REPO_ROOT / (
            "scripts/research/"
            "dual_loop_d0_egomotion_error_attribution_r1"
        )
        r2 = REPO_ROOT / (
            "scripts/research/"
            "dual_loop_d0_egomotion_error_attribution_r2"
        )
        for name in ("analysis.py", "bindings.py", "producer.py"):
            with self.subTest(name=name):
                self.assertEqual(sha256_file(r2 / name), sha256_file(r1 / name))

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
            "d0-egomotion-error-attribution-r2/implementation/activation.json"
        )
        lock_path = activation_path.with_name("implementation_lock.json")
        adapter = REPO_ROOT / (
            "scripts/run_dual_loop_d0_egomotion_error_attribution_r2.py"
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

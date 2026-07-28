from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "rgb_segment_confirmation_mvsec_r2"))
runner = importlib.import_module("run_mvsec_identity_r2")


class DummyCore:
    class IdentityFailure(RuntimeError):
        pass


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def binding(repo: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(repo).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class AuthorityR2Tests(unittest.TestCase):
    def make_valid_fixture(self, directory: str):
        repo = Path(directory)
        config_path = repo / "config.json"
        runner_path = repo / "scripts" / "runner.py"
        runner_path.parent.mkdir(parents=True)
        runner_path.write_text("BOUND_RUNNER = True\n", encoding="utf-8")
        write_json(config_path, {"fixture": True})
        bound = {}
        for name in (
            "capture_runner",
            "diagnostics",
            "core_runner",
            "parser",
            "tests",
        ):
            path = repo / "bound" / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            bound[name] = binding(repo, path)
        candidate_path = repo / runner.CANDIDATE_PATH
        candidate = {
            "decision": "R2_IDENTITY_EXTRACTION_NOT_AUTHORIZED",
            "execution_authority": False,
            "bindings": {
                "config": binding(repo, config_path),
                "runner": binding(repo, runner_path),
                **bound,
            },
        }
        write_json(candidate_path, candidate)
        candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        review_path = repo / runner.REVIEW_PATH
        write_json(
            review_path,
            {
                "decision": "MVSEC_RGB_IDENTITY_R2_CANDIDATE_REVIEW_PASS",
                "candidate_sha256": candidate_sha,
                "execution_authority": False,
            },
        )
        authorization_path = repo / runner.AUTHORIZATION_PATH
        write_json(
            authorization_path,
            {
                "decision": (
                    "MVSEC_RGB_IDENTITY_R2_NEW_NETWORK_CLAIM_AUTHORIZED"
                ),
                "candidate_sha256": candidate_sha,
                "new_network_claim_authority": True,
                "retry_or_resume_r1_authority": False,
                "full_bag_authority": False,
            },
        )
        activation_path = repo / "activation.json"
        runtime_bindings = {
            "config": binding(repo, config_path),
            "runner": binding(repo, runner_path),
        }
        write_json(
            activation_path,
            {
                "schema_version": "rcle_mvsec_rgb_identity_r2_activation.v1",
                "decision": "MVSEC_RGB_IDENTITY_R2_ONE_SHOT_AUTHORIZED",
                "execution_authority": True,
                "bindings": {
                    "candidate": {
                        "path": runner.CANDIDATE_PATH,
                        "sha256": candidate_sha,
                    },
                    "review": binding(repo, review_path),
                    "new_network_claim_authorization": binding(
                        repo,
                        authorization_path,
                    ),
                    **runtime_bindings,
                    **bound,
                },
                "authority": {
                    "bag_index_and_target_chunks": True,
                    "image_metadata_ledger": True,
                    "selected_mono8_payload_materialization": True,
                    "image_decode": False,
                    "rectification": False,
                    "rgb_algorithm": False,
                    "android": False,
                },
            },
        )
        return (
            repo,
            config_path,
            runner_path,
            candidate_path,
            review_path,
            authorization_path,
            activation_path,
        )

    def test_missing_review_or_authorization_fails_before_activation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_valid_fixture(directory)
            fixture[5].unlink()
            with self.assertRaisesRegex(
                DummyCore.IdentityFailure,
                "R2_REVIEW_GATE_MISSING",
            ):
                runner.validate_authority(
                    fixture[0],
                    fixture[1],
                    fixture[6],
                    core=DummyCore,
                    runner_path=fixture[2],
                )

    def test_mismatched_network_authorization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_valid_fixture(directory)
            authorization = json.loads(
                fixture[5].read_text(encoding="utf-8")
            )
            authorization["candidate_sha256"] = "0" * 64
            write_json(fixture[5], authorization)
            with self.assertRaisesRegex(
                DummyCore.IdentityFailure,
                "R2_REVIEW_GATE",
            ):
                runner.validate_authority(
                    fixture[0],
                    fixture[1],
                    fixture[6],
                    core=DummyCore,
                    runner_path=fixture[2],
                )

    def test_mismatched_activation_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_valid_fixture(directory)
            activation = json.loads(
                fixture[6].read_text(encoding="utf-8")
            )
            activation["authority"]["rgb_algorithm"] = True
            write_json(fixture[6], activation)
            with self.assertRaisesRegex(
                DummyCore.IdentityFailure,
                "R2_ACTIVATION_IDENTITY",
            ):
                runner.validate_authority(
                    fixture[0],
                    fixture[1],
                    fixture[6],
                    core=DummyCore,
                    runner_path=fixture[2],
                )

    def test_exact_review_authorization_and_activation_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_valid_fixture(directory)
            candidate = runner.validate_authority(
                fixture[0],
                fixture[1],
                fixture[6],
                core=DummyCore,
                runner_path=fixture[2],
            )
            self.assertEqual(
                candidate["decision"],
                "R2_IDENTITY_EXTRACTION_NOT_AUTHORIZED",
            )

    def test_bootstrap_hash_mismatch_fails_before_import_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            bindings = {}
            for name in (
                "capture_runner",
                "evidence_module",
                "core_runner",
                "parser",
            ):
                path = repo / f"{name}.py"
                path.write_text(name, encoding="utf-8")
                bindings[name] = binding(repo, path)
            runner.verify_bootstrap_bindings(
                repo,
                {"bootstrap_bindings": bindings},
            )
            (repo / bindings["parser"]["path"]).write_text(
                "changed",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                runner.BootstrapFailure,
                "R2_BOOTSTRAP_BINDING",
            ):
                runner.verify_bootstrap_bindings(
                    repo,
                    {"bootstrap_bindings": bindings},
                )


if __name__ == "__main__":
    unittest.main()

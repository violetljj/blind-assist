from __future__ import annotations

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, render_structured

from .hard_benchmark import evaluate_instance, load_benchmark
from .protocol_b4a import PAIRED_IDENTITIES, build_protocol_manifest
from .run_b4a import _balanced_prompt
from .run_b4a import run as run_b4a


class B4AProtocolTest(unittest.TestCase):
    def test_fresh_pair_grid_and_budget_are_frozen(self) -> None:
        manifest = build_protocol_manifest(Path("."))
        identities = [int(row["paired_identity"]) for row in PAIRED_IDENTITIES]

        self.assertEqual(len(identities), 9)
        self.assertEqual(len(set(identities)), 9)
        self.assertEqual(manifest["execution"]["planned_model_calls"], 144)
        self.assertFalse(manifest["anti_post_hoc"]["generation_budget_reduced_to_two"])
        self.assertEqual(
            manifest["harder_cohort"]["model_calls_used_to_construct_or_qualify"], 0
        )

    def test_search_prompt_does_not_expose_hidden_instance_outcomes(self) -> None:
        instance = load_benchmark()["instances"][0]
        result = evaluate_instance(INITIAL_SPEC, instance)
        prompt = _balanced_prompt(519302, 1, INITIAL_SPEC, result, [])

        self.assertNotIn("amber", prompt)
        self.assertNotIn("accepted_actions", prompt)
        self.assertNotIn("episode_ledger", prompt)

    def test_v2_resolves_worker_mounts_before_provider_dispatch(self) -> None:
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            protocol_path = temporary_path / "protocol.json"
            protocol_path.write_text("{}", encoding="utf-8")
            observed: list[Path] = []

            def fake_provider(*args, **kwargs):
                workdir = args[4]
                self.assertTrue(workdir.is_absolute())
                observed.append(workdir)
                return render_structured(INITIAL_SPEC), 0, ""

            try:
                os.chdir(temporary_path)
                with (
                    patch("scripts.research.l10m_b4.run_b4a._validate_protocol", return_value={}),
                    patch("scripts.research.l10m_b4.run_b4a._validate_transport", return_value={}),
                    patch("scripts.research.l10m_b4.run_b4a.provider_preflight_docker", return_value={}),
                    patch("scripts.research.l10m_b4.run_b4a.docker_isolation_canary", return_value={"status": "PASS"}),
                    patch("scripts.research.l10m_b4.run_b4a.run_provider_docker", side_effect=fake_provider),
                ):
                    run_dir = run_b4a(
                        repo_root=original,
                        output_root=Path("relative-output"),
                        protocol_path=protocol_path,
                        transport_path=temporary_path / "transport.json",
                        docker=Path("docker.exe"),
                        docker_image="test-image",
                        auth_path=Path("auth.json"),
                        proxy_bind="127.0.0.1",
                        timeout_seconds=1,
                    )
            finally:
                os.chdir(original)

            self.assertTrue(run_dir.is_absolute())
            self.assertEqual(len(observed), 144)


if __name__ == "__main__":
    unittest.main()

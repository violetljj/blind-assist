from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, render_structured

from .protocol_b4a import PAIRED_IDENTITIES as B4_PAIRED_IDENTITIES
from .protocol_b5a import PAIRED_IDENTITIES, PROTOCOL_ID, build_protocol_manifest
from .run_b5a import run as run_b5a


class B5AProtocolTest(unittest.TestCase):
    def test_fresh_pair_grid_budget_and_unchanged_operator_are_frozen(self) -> None:
        manifest = build_protocol_manifest(Path("."))
        identities = {int(row["paired_identity"]) for row in PAIRED_IDENTITIES}
        b4_identities = {int(row["paired_identity"]) for row in B4_PAIRED_IDENTITIES}

        self.assertEqual(len(identities), 9)
        self.assertTrue(identities.isdisjoint(b4_identities))
        self.assertEqual(manifest["execution"]["planned_model_calls"], 144)
        self.assertFalse(manifest["balanced_operator"]["algorithm_change_from_b4a"])
        self.assertFalse(manifest["balanced_operator"]["progress_conditioning"])
        self.assertEqual(manifest["fresh_harder_cohort"]["model_calls_used_to_construct_or_qualify"], 0)

    def test_runner_emits_b5_protocol_identity_and_absolute_worker_paths(self) -> None:
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
                    patch("scripts.research.l10m_b4.run_b5a._validate_protocol", return_value={}),
                    patch("scripts.research.l10m_b4.run_b5a._validate_transport", return_value={}),
                    patch("scripts.research.l10m_b4.run_b5a.provider_preflight_docker", return_value={}),
                    patch("scripts.research.l10m_b4.run_b5a.docker_isolation_canary", return_value={"status": "PASS"}),
                    patch("scripts.research.l10m_b4.run_b4a.run_provider_docker", side_effect=fake_provider),
                ):
                    run_dir = run_b5a(
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

            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(observed), 144)
            self.assertTrue(all(row["protocol_id"] == PROTOCOL_ID for row in events))


if __name__ == "__main__":
    unittest.main()

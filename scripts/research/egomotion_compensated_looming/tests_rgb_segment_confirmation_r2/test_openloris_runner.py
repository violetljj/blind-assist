from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[4]
MODULE_DIR = (
    REPO
    / "scripts"
    / "research"
    / "egomotion_compensated_looming"
    / "rgb_segment_confirmation_r2"
)
MODULE_PATH = MODULE_DIR / "run_openloris_one_shot.py"
CONFIG_PATH = (
    REPO
    / "artifacts.local"
    / "evidence"
    / "rcle_rgb_segment_confirmation_r2"
    / "openloris_runner_config.v1.json"
)
LOCK_PATH = (
    REPO
    / "artifacts.local"
    / "evidence"
    / "rcle_rgb_segment_confirmation_r1"
    / "preaccess_lock.v11.json"
)


def load_module():
    sys.path.insert(0, str(MODULE_DIR))
    spec = importlib.util.spec_from_file_location("run_openloris_one_shot", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRemote:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.length = len(data)
        self.position = 0

    def seek(self, target: int) -> int:
        if not 0 <= target <= self.length:
            raise ValueError
        self.position = target
        return target

    def read(self, size: int) -> bytes:
        body = self.data[self.position : self.position + size]
        self.position += len(body)
        return body


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        content_range: str,
        final_url: str = "https://huggingface.co/object",
        content_encoding: str | None = None,
    ) -> None:
        self.body = body
        self.status = 206
        self.headers = {"Content-Range": content_range}
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding
        self.final_url = final_url
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, cap: int) -> bytes:
        return self.body[:cap]

    def geturl(self) -> str:
        return self.final_url

    def close(self) -> None:
        self.closed = True


class OpenLorisRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_canonical_config_preflight_passes(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.module.validate_config(REPO, config)

    def test_adapter_supports_seek_modes_and_read_all(self) -> None:
        adapter = self.module.RemoteFileAdapter(FakeRemote(b"abcdefghij"))
        self.assertTrue(adapter.readable())
        self.assertTrue(adapter.seekable())
        self.assertFalse(adapter.writable())
        self.assertEqual(adapter.read(3), b"abc")
        self.assertEqual(adapter.tell(), 3)
        self.assertEqual(adapter.seek(-1, io.SEEK_CUR), 2)
        self.assertEqual(adapter.read(2), b"cd")
        self.assertEqual(adapter.seek(-3, io.SEEK_END), 7)
        with self.assertRaises(io.UnsupportedOperation):
            adapter.read(-1)
        self.assertEqual(adapter.read(3), b"hij")
        self.assertEqual(adapter.read(1), b"")

    def test_identity_opener_limits_same_range_across_claim(self) -> None:
        calls = 0

        def opener(request, *, timeout):
            nonlocal calls
            calls += 1
            return FakeResponse(b"abcd", "bytes 0-3/10")

        bound = self.module.IdentityBoundOpener(
            allowed_final_hosts={"huggingface.co"},
            maximum_attempts_per_identical_range=3,
            opener=opener,
        )
        request = self.module.urllib.request.Request(
            "https://huggingface.co/object", headers={"Range": "bytes=0-3"}
        )
        for _ in range(3):
            bound(request, timeout=1)
        with self.assertRaises(OSError):
            bound(request, timeout=1)
        self.assertEqual(calls, 3)

    def test_candidate_receipt_cannot_activate_runner(self) -> None:
        candidate = (
            REPO
            / "artifacts.local"
            / "evidence"
            / "rcle_rgb_segment_confirmation_r2"
            / "openloris_activation_candidate.v1.json"
        )
        with self.assertRaises(self.module.ActivationPreflightFailure):
            self.module.validate_activation(
                REPO, CONFIG_PATH, candidate, MODULE_PATH
            )

    def test_target_manifest_rejects_duplicate(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(self.module.targets_from_lock(lock)), 302)
        changed = copy.deepcopy(lock)
        changed["segments"][0]["frame_inventory"][1]["rgb_member_path"] = changed[
            "segments"
        ][0]["frame_inventory"][0]["rgb_member_path"]
        with self.assertRaises(self.module.ActivationPreflightFailure):
            self.module.targets_from_lock(changed)

    def test_target_manifest_rejects_order_and_metadata_mutation(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        changed_order = copy.deepcopy(lock)
        inventory = changed_order["segments"][0]["frame_inventory"]
        inventory[0], inventory[1] = inventory[1], inventory[0]
        with self.assertRaises(self.module.ActivationPreflightFailure):
            self.module.targets_from_lock(changed_order)
        changed_crc = copy.deepcopy(lock)
        changed_crc["segments"][0]["guard_before"]["crc32"] = "00000000"
        with self.assertRaises(self.module.ActivationPreflightFailure):
            self.module.targets_from_lock(changed_crc)

    def test_identity_opener_rejects_host_and_content_encoding(self) -> None:
        for response in (
            FakeResponse(
                b"abcd",
                "bytes 0-3/10",
                final_url="https://evil.invalid/object",
            ),
            FakeResponse(
                b"abcd",
                "bytes 0-3/10",
                content_encoding="gzip",
            ),
        ):
            with self.subTest(final=response.final_url, headers=response.headers):
                bound = self.module.IdentityBoundOpener(
                    allowed_final_hosts={"huggingface.co"},
                    maximum_attempts_per_identical_range=3,
                    opener=lambda *_args, value=response, **_kwargs: value,
                )
                request = self.module.urllib.request.Request(
                    "https://huggingface.co/object",
                    headers={"Range": "bytes=0-3"},
                )
                with self.assertRaises(OSError):
                    bound(request, timeout=1)
                self.assertTrue(response.closed)

    def test_orphan_audit_consumes_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            namespace = Path(temporary)
            (namespace / "claim.json").write_text(
                '{"claim":"consumed"}\n', encoding="utf-8"
            )
            result = self.module.audit_orphan(namespace)
            self.assertEqual(
                result["decision"], "ORPHAN_CONSUMED_NOT_EVALUABLE"
            )
            self.assertTrue(result["mutated"])
            again = self.module.audit_orphan(namespace)
            self.assertEqual(again["decision"], "TERMINAL_ALREADY_PRESENT")
            self.assertFalse(again["mutated"])

    def test_phase_rows_bind_status_budget_and_range_head(self) -> None:
        class Remote:
            last_request = {"row_sha256": "a" * 64}

            @staticmethod
            def snapshot():
                return {"budget_consumed": 123, "budget_limit": 456}

        with tempfile.TemporaryDirectory() as temporary:
            ledger = self.module.AppendOnlyLedger(
                Path(temporary) / "phase.jsonl"
            )
            row = self.module.append_phase(
                ledger,
                "TARGET_IDENTITY_VALIDATION",
                "BEGIN",
                7,
                Remote(),
            )
            self.assertEqual(row["status"], "BEGIN")
            self.assertEqual(row["transport"]["budget_consumed"], 123)
            self.assertEqual(row["range_ledger_head_sha256"], "a" * 64)

    def test_real_local_solid_7z_extracts_only_requested_target(self) -> None:
        import py7zr

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            archive_path = root / "fixture.7z"
            for name, body in (
                ("before.bin", b"a" * 10_000),
                ("target.bin", b"b" * 20_000),
                ("after.bin", b"c" * 10_000),
            ):
                (inputs / name).write_bytes(body)
            with py7zr.SevenZipFile(archive_path, "w") as archive:
                for path in sorted(inputs.iterdir()):
                    archive.write(path, arcname=f"capture/{path.name}")
            raw = archive_path.read_bytes()

            def opener(request, *, timeout):
                match = re.fullmatch(
                    r"bytes=(\d+)-(\d+)", request.get_header("Range")
                )
                assert match is not None
                start, end = map(int, match.groups())
                body = raw[start : end + 1]
                return FakeResponse(
                    body,
                    f"bytes {start}-{end}/{len(raw)}",
                )

            bound = self.module.IdentityBoundOpener(
                allowed_final_hosts={"huggingface.co"},
                maximum_attempts_per_identical_range=3,
                opener=opener,
            )
            remote = self.module.DiagnosticRemoteRange(
                url="https://huggingface.co/object",
                length=len(raw),
                budget=len(raw) * 4,
                ledger_path=root / "range.jsonl",
                progress_path=root / "progress.json",
                failure_receipt_path=root / "failure.json",
                phase="LOCAL_SOLID_FIXTURE",
                user_agent="R2-test",
                opener=bound,
                resource_probe=lambda: {
                    "peak_rss_bytes": 1,
                    "peak_commit_bytes": 1,
                },
            )
            output = root / "output"
            output.mkdir()
            with py7zr.SevenZipFile(
                self.module.RemoteFileAdapter(remote), "r"
            ) as archive:
                archive.extract(
                    path=output,
                    targets=["capture/target.bin"],
                )
            files = sorted(
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            )
            self.assertEqual(files, ["capture/target.bin"])
            self.assertEqual(
                (output / "capture" / "target.bin").read_bytes(),
                b"b" * 20_000,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


from carla_storage import (
    GIB,
    _os_path,
    acquire_storage_lease,
    apply_dedupe_plan,
    build_dedupe_plan,
    check_storage_lease,
    clone_tree,
    guard_storage,
    release_storage_lease,
    StorageError,
    storage_maintenance_lock,
    storage_accounting,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_seal(root: Path, paths: list[Path]) -> None:
    (root / "result.json").write_text(
        json.dumps({"status": "TEST_TERMINAL"}) + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
        for path in paths
    ]
    (root / "sealed_evidence_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )


class CarlaStorageTest(unittest.TestCase):
    def test_accounting_counts_hardlinks_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"payload" * 100)
            os.link(first, second)

            result = storage_accounting(root)

            self.assertEqual(result["file_count"], 2)
            self.assertEqual(result["unique_file_count"], 1)
            self.assertEqual(result["logical_bytes"], first.stat().st_size * 2)
            self.assertEqual(result["unique_bytes"], first.stat().st_size)

    @unittest.skipUnless(os.name == "nt", "Windows MAX_PATH behavior")
    def test_accounting_handles_windows_long_paths(self) -> None:
        root = Path(tempfile.mkdtemp())
        try:
            nested = root.joinpath(*(["long-segment-0123456789"] * 12))
            os.makedirs(_os_path(nested))
            payload = nested / "payload.png"
            with open(_os_path(payload), "wb") as handle:
                handle.write(b"long-path")
            self.assertGreater(len(str(payload)), 260)

            result = storage_accounting(root)

            self.assertEqual(result["file_count"], 1)
            self.assertEqual(result["unique_bytes"], len(b"long-path"))
        finally:
            shutil.rmtree(_os_path(root), ignore_errors=True)

    def test_dedupe_requires_frozen_plan_and_preserves_paths_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            source = root / "shards" / "payload" / "000000.png"
            model = root / "model" / "rgb" / "000000.png"
            evaluator = root / "evaluator" / "witness" / "000000.png"
            source.parent.mkdir(parents=True)
            model.parent.mkdir(parents=True)
            evaluator.parent.mkdir(parents=True)
            source.write_bytes(os.urandom(256 * 1024))
            shutil.copy2(source, model)
            shutil.copy2(source, evaluator)
            _write_seal(root, [source, model, evaluator])
            before_hash = _digest(source)
            before_mtime = source.stat().st_mtime_ns

            plan = build_dedupe_plan(root, frozenset({".png"}), 0)
            self.assertEqual(plan["action_path_count"], 2)
            receipt = Path(temporary) / "receipt"
            result = apply_dedupe_plan(
                plan,
                receipt,
                str(plan["plan_sha256"]),
            )

            self.assertEqual(result["linked_path_count"], 2)
            self.assertTrue(os.path.samefile(source, model))
            self.assertTrue(os.path.samefile(source, evaluator))
            for path in (source, model, evaluator):
                self.assertTrue(path.is_file())
                self.assertEqual(_digest(path), before_hash)
                self.assertEqual(path.stat().st_mtime_ns, before_mtime)
            saved = json.loads((receipt / "dedupe-result.json").read_text())
            self.assertFalse(saved["content_or_path_deleted"])

    def test_dedupe_does_not_merge_equal_content_with_different_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same")
            shutil.copy2(first, second)
            os.utime(second, ns=(second.stat().st_atime_ns, time.time_ns() - 5_000_000_000))
            _write_seal(root, [first, second])

            plan = build_dedupe_plan(root, frozenset({".png"}), 0)

            self.assertEqual(plan["action_path_count"], 0)

    def test_dedupe_requires_manifest_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same")
            shutil.copy2(first, second)
            _write_seal(root, [first])

            plan = build_dedupe_plan(root, frozenset({".png"}), 0)

            self.assertEqual(plan["action_path_count"], 0)
            self.assertEqual(plan["scan"]["skipped_unlisted_file_count"], 1)

    @unittest.skipUnless(os.name == "nt", "NTFS alternate streams")
    def test_dedupe_refuses_named_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same")
            shutil.copy2(first, second)
            with open(_os_path(second) + ":private", "wb") as stream:
                stream.write(b"must-remain")
            _write_seal(root, [first, second])

            plan = build_dedupe_plan(root, frozenset({".png"}), 0)

            self.assertEqual(plan["action_path_count"], 0)
            self.assertEqual(plan["scan"]["skipped_named_stream_file_count"], 1)
            with open(_os_path(second) + ":private", "rb") as stream:
                self.assertEqual(stream.read(), b"must-remain")

    def test_clone_tree_uses_hardlinks_on_one_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            destination = base / "destination"
            payload = source / "nested" / "frame.png"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"frame")

            result = clone_tree(source, destination)

            clone = destination / "nested" / "frame.png"
            self.assertEqual(result["linked_file_count"], 1)
            self.assertEqual(result["copied_file_count"], 0)
            self.assertTrue(os.path.samefile(payload, clone))

    def test_clone_tree_copy_fallback_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            (source / "payload.bin").write_bytes(b"fallback")
            destination = base / "destination"

            with mock.patch("carla_storage.os.link", side_effect=OSError("no links")):
                result = clone_tree(source, destination)

            self.assertEqual(result["linked_file_count"], 0)
            self.assertEqual(result["copied_file_count"], 1)
            self.assertEqual((destination / "payload.bin").read_bytes(), b"fallback")

    def test_clone_tree_refuses_descendant_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "frame.png").write_bytes(b"frame")

            with self.assertRaises(StorageError):
                clone_tree(source, source / "clone")

            self.assertFalse((source / "clone").exists())

    def test_clone_tree_does_not_remove_racing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            destination = base / "destination"
            source.mkdir()
            (source / "frame.png").write_bytes(b"frame")
            real_mkdir = os.mkdir

            def create_rival(path: str, *args: object, **kwargs: object) -> None:
                real_mkdir(path, *args, **kwargs)
                (destination / "rival.txt").write_text("owned elsewhere")
                raise FileExistsError(path)

            with mock.patch("carla_storage.os.mkdir", side_effect=create_rival):
                with self.assertRaises(StorageError):
                    clone_tree(source, destination)

            self.assertEqual((destination / "rival.txt").read_text(), "owned elsewhere")

    def test_clone_tree_refuses_links_outside_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            outside = base / "outside"
            outside.mkdir()
            link = source / "outside-link"
            try:
                os.symlink(outside, link, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are unavailable on this host")

            destination = base / "destination"
            with self.assertRaises(StorageError):
                clone_tree(source, destination)

            self.assertFalse(destination.exists())

    def test_guard_fails_closed_on_projected_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"x" * 1024)
            policy = {
                "policy_path": "test",
                "policy_sha256": "0" * 64,
                "maximum_experiment_unique_bytes": 1024,
                "minimum_volume_free_bytes": 0,
                "overflow_action": "REFUSE_NEW_RUN",
                "automatic_payload_deletion": False,
            }

            blocked = guard_storage(root, policy, 1)
            allowed = guard_storage(root, policy, 0)

            self.assertEqual(blocked["status"], "REFUSE_NEW_RUN")
            self.assertIn("EXPERIMENT_UNIQUE_BYTE_CAP", blocked["reasons"])
            self.assertEqual(allowed["status"], "PASS")
            with self.assertRaises(StorageError):
                guard_storage(root, policy, -1)

    def test_lease_is_atomic_and_blocks_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            policy = {
                "policy_path": "test",
                "policy_sha256": "0" * 64,
                "maximum_experiment_unique_bytes": 1024 * 1024,
                "minimum_volume_free_bytes": 0,
            }

            lease = acquire_storage_lease(root, policy, 4096, os.getpid(), "test", output)
            token = str(lease["lease_token"])
            try:
                checked = check_storage_lease(root, policy, token, output)
                self.assertEqual(checked["unique_bytes"], 0)
                with self.assertRaises(StorageError):
                    with storage_maintenance_lock(root):
                        pass
            finally:
                released = release_storage_lease(root, token)

            self.assertEqual(released["status"], "RELEASED")
            self.assertFalse((root / ".carla-storage-leases").exists())

    def test_lease_check_preserves_other_active_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = {
                "policy_path": "test",
                "policy_sha256": "0" * 64,
                "maximum_experiment_unique_bytes": 6000,
                "minimum_volume_free_bytes": 0,
            }
            first = acquire_storage_lease(
                root, policy, 1000, os.getpid(), "first", root / "first"
            )
            second = acquire_storage_lease(
                root, policy, 1000, os.getpid(), "second", root / "second"
            )
            try:
                (root / "payload.bin").write_bytes(b"x" * 5500)
                with self.assertRaises(StorageError):
                    check_storage_lease(
                        root,
                        policy,
                        str(first["lease_token"]),
                        root / "first",
                    )
            finally:
                release_storage_lease(root, str(first["lease_token"]))
                release_storage_lease(root, str(second["lease_token"]))

    def test_apply_recomputes_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same")
            shutil.copy2(first, second)
            _write_seal(root, [first, second])
            plan = build_dedupe_plan(root, frozenset({".png"}), 0)
            frozen_hash = str(plan["plan_sha256"])
            plan["groups"][0]["canonical"] = "../escape.png"

            with self.assertRaises(StorageError):
                apply_dedupe_plan(plan, Path(temporary) / "receipt", frozen_hash)


if __name__ == "__main__":
    unittest.main()

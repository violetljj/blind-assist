"""Offline geometry and RGB-firewall tests for CID-SIMS R2."""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer,
    validator,
)


def _depth_png(value: int = 2000) -> bytes:
    stream = BytesIO()
    Image.fromarray(np.full((480, 640), value, dtype=np.uint16)).save(
        stream, format="PNG"
    )
    return stream.getvalue()


def _archive(path: Path, *, frame_count: int = 102, missing_depth: int | None = None) -> None:
    poses: list[str] = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(frame_count):
            timestamp = f"{index * 0.1:.6f}"
            poses.append(
                f"{timestamp} 0 0 {index * 0.02:.6f} 0 0 0 1\n"
            )
            archive.writestr(
                f"floor3_1/color/{timestamp}.png", b"RGB_MUST_NOT_BE_READ"
            )
            if index != missing_depth:
                archive.writestr(
                    f"floor3_1/depth/{timestamp}.png", _depth_png()
                )
        archive.writestr("floor3_1/pose.txt", "".join(poses))


class GeometryTest(unittest.TestCase):
    def test_producer_and_independent_replay_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "floor3_1.zip"
            _archive(path)
            with producer.CidSimsArchive(path) as archive:
                produced_window, produced_pairs, produced_source = producer._evaluate(
                    archive, 1
                )
            replay_window, replay_pairs, replay_source = validator.replay(path)
            self.assertTrue(producer.canonical_json_sha(produced_window))
            self.assertTrue(validator.close_enough(produced_window, replay_window))
            self.assertTrue(validator.close_enough(produced_pairs, replay_pairs))
            self.assertEqual(
                produced_source["depth_members_read"],
                replay_source["depth_members_read"],
            )
            self.assertNotIn("read_rgb", dir(producer.CidSimsArchive))

    def test_missing_depth_is_retained_as_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "floor3_1.zip"
            _archive(path, missing_depth=3)
            with producer.CidSimsArchive(path) as archive:
                window, pairs, _ = producer._evaluate(archive, 1)
            self.assertEqual(window["candidate_pair_count"], 99)
            self.assertEqual(window["evaluable_pair_count"], 97)
            self.assertEqual(
                sum(row.get("reason") == "ASSOCIATED_MEMBER_MISSING" for row in pairs),
                2,
            )

    def test_incomplete_first_window_is_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "floor3_1.zip"
            _archive(path, frame_count=80)
            with producer.CidSimsArchive(path) as archive:
                window, _, _ = producer._evaluate(archive, 1)
            self.assertFalse(window["window_complete"])
            self.assertFalse(window["admitted"])

    def test_validator_does_not_import_producer(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(any("producer" in name for name in imported))


if __name__ == "__main__":
    unittest.main()

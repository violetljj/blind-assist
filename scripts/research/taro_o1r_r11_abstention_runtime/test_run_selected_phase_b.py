from __future__ import annotations

import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.research.taro_o1r_r11_abstention_runtime import run_selected_phase_b as runner


def synthetic_inventory() -> dict:
    parents = []
    identities = runner._load_json(runner._repo_path(runner.TOP24_SELECTION_RELATIVE))["selected_parent_identities"]
    scores = runner._load_json(runner._repo_path(runner.TOP24_SELECTION_RELATIVE))["selected_parent_scores"]
    for identity, score in zip(identities, scores, strict=True):
        tokens = [f"{index:06d}" for index in range(score["frame_count"])]
        parents.append(
            {
                "visit_id": identity[0],
                "video_id": identity[1],
                "frame_plan": {"exact_timestamp_tokens": tokens},
                "container_bindings": {
                    "upsampling": {
                        "path": f"fixture/{identity[1]}.zip",
                        "bytes": 1,
                        "sha256": "A" * 64,
                        "declared_uncompressed_bytes": 1,
                        "recognized_member_index_sha256": "B" * 64,
                    }
                },
            }
        )
    return {"parents": parents}


class R11SelectedPhaseBRunnerTests(unittest.TestCase):
    def test_frozen_selected_cohort_is_exact_24_674_6066(self) -> None:
        _scores, selection, formal = runner._load_top24()
        self.assertEqual(len(selection["selected_parent_identities"]), 24)
        self.assertEqual(sum(row["frame_count"] for row in selection["selected_parent_scores"]), 674)
        self.assertEqual(sum(row["query_count"] for row in selection["selected_parent_scores"]), 6066)
        self.assertEqual(formal["status"], "TARO_O1R_R11_SOURCE_ONLY_TOP24_INDEPENDENT_VALIDATION_PASS")

    def test_selected_frame_derivation_never_admits_unselected_parent(self) -> None:
        inventory = synthetic_inventory()
        selection = runner._load_json(runner._repo_path(runner.TOP24_SELECTION_RELATIVE))

        class Member:
            role = "highres_depth"
            source_member_path = "highres_depth/frame.png"
            bytes = 1
            declared_crc32 = "00000000"

        index = {role: {} for role in ("color", "highres_depth", "lowres_depth", "confidence")}
        for parent in inventory["parents"]:
            for token in parent["frame_plan"]["exact_timestamp_tokens"]:
                for role in index:
                    value = copy.copy(Member())
                    value.role = role
                    index[role][token] = value
        with (
            mock.patch.object(runner.run_pool_inventory, "validate_inventory", side_effect=lambda value: value),
            mock.patch.object(runner.phase_a, "_verify_container"),
            mock.patch.object(runner, "_repo_path", side_effect=lambda relative: Path(relative)),
            mock.patch.object(runner.run_pool_inventory, "index_upsampling_archive_metadata_only", return_value=(index, 1)),
            mock.patch.object(runner.phase_a, "_member_index_sha256", return_value="B" * 64),
        ):
            frames = runner.derive_selected_frames(inventory, selection)
        selected = {tuple(row) for row in selection["selected_parent_identities"]}
        self.assertEqual(len(frames), 674)
        self.assertTrue({(row.parent_id, row.video_id) for row in frames}.issubset(selected))

    def test_faro_reader_rejects_non_highres_before_zip_lookup(self) -> None:
        frame = runner.SelectedFrame(
            "p", "v", "t", "v:t", Path("fixture.zip"),
            runner.phase_a.PhaseAMemberRef("color", "color.png", 1, "00000000"), {},
        )
        bundle = mock.Mock()
        with self.assertRaises(runner.R11PhaseBError):
            runner._read_faro_member(bundle, frame, runner.Counter())
        bundle.getinfo.assert_not_called()
        bundle.read.assert_not_called()

    def test_faro_reader_counts_exact_attempt_completed_and_bytes(self) -> None:
        frame = runner.SelectedFrame(
            "p", "v", "t", "v:t", Path("fixture.zip"),
            runner.phase_a.PhaseAMemberRef("highres_depth", "highres.png", 3, "1234ABCD"), {},
        )
        info = mock.Mock(file_size=3, CRC=int("1234ABCD", 16))
        bundle = mock.Mock()
        bundle.getinfo.return_value = info
        bundle.read.return_value = b"abc"
        ledger = runner.Counter()
        self.assertEqual(runner._read_faro_member(bundle, frame, ledger), b"abc")
        self.assertEqual(dict(ledger), {"attempt:highres_depth": 1, "completed:highres_depth": 1, "bytes:highres_depth": 3})

    def test_label_builder_reuses_one_geometry_for_all_nine_queries(self) -> None:
        source = {
            "input_bindings": {
                "intrinsics_highres_sha256": runner.adapter.canonical_sha256(np.eye(3)),
                "gravity_up_camera_xyz_sha256": runner.adapter.canonical_sha256(
                    runner.adapter._normalize_vector([0.0, 1.0, 0.0], "fixture")
                ),
            },
            "parent_id": "p", "video_id": "v", "timestamp_token": "t", "physical_frame_id": "v:t",
            "content_sha256": "S" * 64,
            "query_features": [
                {"grid_index": index, "query_id": f"q{index}", "query_receipt": {"fixture": index}}
                for index in range(9)
            ],
        }
        validated = {"fixture": "label"}
        with (
            mock.patch.object(runner.r7_canary, "validate_source_frame_record", return_value=source),
            mock.patch.object(runner.r7_canary, "_matrix", return_value=np.eye(3)),
            mock.patch.object(runner.prospective, "_fit_depth_plane", return_value={"evaluable": True}),
            mock.patch.object(runner.prospective, "_build_geometry", return_value=object()) as geometry,
            mock.patch.object(
                runner.r7_canary,
                "_truth_query_label",
                return_value={
                    "state": "UNKNOWN", "obstacle_pixel_count": 0, "minimum_truth_obstacle_pixels": 4,
                    "query_support_points": 0, "observed_forward_m": None, "local_valid_fraction": 0.0,
                    "reason_codes": ["fixture"],
                },
            ) as query_label,
            mock.patch.object(runner.r7_canary, "validate_label_frame_record", return_value=validated),
        ):
            self.assertIs(
                runner.build_label_frame_record(
                    source, np.zeros(runner.adapter.HIGHRES_SHAPE_HW, dtype=np.uint16), np.eye(3), [0.0, 1.0, 0.0]
                ),
                validated,
            )
        geometry.assert_called_once()
        self.assertEqual(query_label.call_count, 9)

    def test_actual_argv_and_missing_lock_fail_before_root_creation(self) -> None:
        with mock.patch.object(runner.sys, "orig_argv", ["python.exe", *runner.EXPECTED_ARGV]):
            runner._validate_actual_argv()
        with (
            mock.patch.object(runner.sys, "orig_argv", ["python.exe", *runner.EXPECTED_ARGV, "--extra"]),
            self.assertRaises(runner.R11PhaseBError),
        ):
            runner._validate_actual_argv()
        self.assertFalse(runner._repo_path(runner.OUTPUT_ROOT).exists())

    def test_code_binding_must_equal_implementation_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "code.py"
            path.write_bytes(b"locked\n")
            binding = {"role": "CODE", "path": "code.py", "bytes": 7, "sha256": runner.materializer.sha256_file(path)}
            lock = {"implementation_commit": "a" * 40, "bindings": [binding]}
            with (
                mock.patch.object(runner, "EXPECTED_BINDINGS", {"CODE": "code.py"}),
                mock.patch.object(runner, "ARTIFACT_BINDING_ROLES", set()),
                mock.patch.object(runner, "_repo_path", return_value=path),
                mock.patch.object(runner, "_git_bytes", return_value=b"locked\n"),
            ):
                runner._validate_bindings(lock)
            with (
                mock.patch.object(runner, "EXPECTED_BINDINGS", {"CODE": "code.py"}),
                mock.patch.object(runner, "ARTIFACT_BINDING_ROLES", set()),
                mock.patch.object(runner, "_repo_path", return_value=path),
                mock.patch.object(runner, "_git_bytes", return_value=b"drift\n"),
                self.assertRaises(runner.R11PhaseBError),
            ):
                runner._validate_bindings(lock)

    def test_partial_root_is_created_once_then_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            partial = parent / "phase-b.partial-fixture"
            target = parent / "phase-b"
            writer = runner.FactorEvidenceWriter(partial, runner.TERMINAL_RESERVE_BYTES + 4096)
            self.assertFalse(partial.exists())
            writer.activate({"schema": "fixture.execution.v1"})
            runner._allocate_reserve(writer)
            runner._release_reserve(writer)
            writer.write_json("terminal.json", {"schema": "fixture.terminal.v1"})
            runner.os.replace(partial, target)
            self.assertFalse(partial.exists())
            self.assertTrue((target / "terminal.json").is_file())

    def test_runner_has_no_r10_or_model_network_import(self) -> None:
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        joined = "\n".join(imports).lower()
        for forbidden in ("taro_o1r_r10", "depthart_runner", "requests", "urllib"):
            self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main()

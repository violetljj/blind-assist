#!/usr/bin/env python3
"""Tests for the static P3 R0.1 activation receipt gate."""

from __future__ import annotations

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

import p3_r0_1_activation_receipt as gate


SHA = "A" * 64
HEAD = "1" * 40


class ActivationReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = {
            "schema": gate.PROTOCOL_SCHEMA,
            "class_weight_derivation": {"beta": 0.999},
        }
        self.files: dict[str, dict[str, object]] = {}
        self._write("r0_1_protocol", {
            "schema": "blindassist_dav2_temporal_392_student_p3_r0_1_protocol",
            "current_terminal": "P3_R0_1_PRE_ACTIVATION_CORRECTION_COMPLETE_BINDINGS_PENDING_HOLDOUT_UNOPENED",
            "implementation": {},
        })
        for key in ("training_module", "clip_p1_evaluator", "module_tests", "evaluator_tests"):
            self._write_bytes(key, key.encode())
        self._write_bytes("activation_generator", b"static generator")
        self._write_bytes("activation_tests", b"static tests")
        self.protocol["implementation"] = {
            "generator": {"sha256": self.files["activation_generator"]["expected_sha256"]},
            "tests": {"sha256": self.files["activation_tests"]["expected_sha256"]},
        }
        r0 = self._json("r0_1_protocol")
        r0["implementation"] = {
            "evidence_loss_manifest_module_sha256": self.files["training_module"]["expected_sha256"],
            "clip_p1_evaluator_sha256": self.files["clip_p1_evaluator"]["expected_sha256"],
            "module_tests_sha256": self.files["module_tests"]["expected_sha256"],
            "evaluator_tests_sha256": self.files["evaluator_tests"]["expected_sha256"],
        }
        self._rewrite("r0_1_protocol", r0)
        self._write("a2_protocol", {"schema": "blindassist_dav2_392_distillation_a2_r0_protocol"})
        self._write("a2_result", {"schema": "blindassist_dav2_model_variant_gate_r0_result"})
        self._write_bytes("a2_checkpoint", b"opaque checkpoint bytes")
        self._write("a2_training_receipt", {
            "schema": "blindassist_dav2_392_distillation_a2_r0_training_result",
            "terminal": "A2_DISTILLATION_TRAINING_COMPLETE_P1_UNOPENED",
            "truth_inputs_opened": False,
            "epochs_completed": 3,
            "protocol_sha256": self.files["a2_protocol"]["expected_sha256"],
            "checkpoint": {"sha256": self.files["a2_checkpoint"]["expected_sha256"]},
        })
        self._write_bytes("disagreement_cache", b"frozen cache")
        self._write_bytes("disagreement_producer", b"producer")
        self._write("disagreement_manifest", {
            "schema": "blindassist_p3_r0_1_frozen_a2_disagreement_manifest",
            "status": "FROZEN_PARENT_A2_ONLY",
            "current_student_used": False,
            "a2_checkpoint_sha256": self.files["a2_checkpoint"]["expected_sha256"],
            "cache_sha256": self.files["disagreement_cache"]["expected_sha256"],
            "producer_sha256": self.files["disagreement_producer"]["expected_sha256"],
        })
        self._write("legacy_p1_exclusion_ledger", {
            "schema": "blindassist_p3_r0_1_legacy_p1_ancestry_exclusion_ledger",
            "status": "FROZEN_CONSUMED_EXCLUSION_ONLY",
            "source_p1_roster_sha256": SHA,
            "parent_ids": ["old-parent"],
        })
        self._write_bytes("legacy_p1_roster", b"consumed P1 roster")
        self.files["legacy_p1_roster"]["expected_sha256"] = SHA
        self._path("legacy_p1_roster").write_bytes(b"")
        # Use the real hash while retaining a single fixture source value.
        roster_sha = gate.sha256_file(self._path("legacy_p1_roster"))
        self.files["legacy_p1_roster"]["expected_sha256"] = roster_sha
        ledger = self._json("legacy_p1_exclusion_ledger")
        ledger["source_p1_roster_sha256"] = roster_sha
        self._rewrite("legacy_p1_exclusion_ledger", ledger)
        self._write("train_manifest", self._role("train", "train-parent"))
        self._write("validation_manifest", self._role("validation", "validation-parent"))
        self._write("public_holdout_manifest", self._role("public_holdout", "holdout-parent"))
        self._write_bytes("sealed_target_bundle", b"not JSON; must never be parsed")
        self._write_bytes("coverage_producer", b"coverage producer")
        self._write("sealed_coverage_receipt", {
            "schema": "blindassist_dav2_temporal_392_student_p3_r0_1_sealed_coverage_receipt",
            "status": "SEALED_COVERAGE_VERIFIED",
            "label_rows_disclosed": False,
            "created_before_training_activation": True,
            "evaluable_clip_count": 32,
            "video_parent_count": 8,
            "sealed_bundle_sha256": self.files["sealed_target_bundle"]["expected_sha256"],
            "coverage_producer_sha256": self.files["coverage_producer"]["expected_sha256"],
            "identity_manifest_sha256": self.files["public_holdout_manifest"]["expected_sha256"],
            "key_transition_counts": {
                "CLEAR_TO_OCCUPIED": 8,
                "OCCUPIED_TO_CLEAR": 8,
                "KNOWN_TO_UNKNOWN_GROUND": 8,
                "UNKNOWN_GROUND_TO_KNOWN": 8,
            },
            "geometry_transition_counts": {name: 1 for name in gate.TRANSITIONS},
        })
        self.bindings = {
            "schema": gate.BINDINGS_SCHEMA,
            "files": self.files,
            "a2": {
                "selected_checkpoint_sha256": self.files["a2_checkpoint"]["expected_sha256"],
                "training_receipt_sha256": self.files["a2_training_receipt"]["expected_sha256"],
                "protocol_sha256": self.files["a2_protocol"]["expected_sha256"],
                "result_sha256": self.files["a2_result"]["expected_sha256"],
            },
            "legacy_p1": {"roster_sha256": roster_sha},
            "runtime_state_assertions": {
                "model_loaded": False,
                "optimizer_constructed": False,
                "training_started": False,
                "holdout_bundle_parsed": False,
            },
            "candidate_output_path": "candidate-output",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.asset"

    def _write_bytes(self, key: str, value: bytes) -> None:
        path = self._path(key)
        path.write_bytes(value)
        self.files[key] = {"path": path.name, "expected_sha256": gate.sha256_file(path)}

    def _write(self, key: str, value: dict) -> None:
        self._write_bytes(key, json.dumps(value, sort_keys=True).encode())

    def _rewrite(self, key: str, value: dict) -> None:
        self._path(key).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        self.files[key]["expected_sha256"] = gate.sha256_file(self._path(key))

    def _json(self, key: str) -> dict:
        return json.loads(self._path(key).read_text(encoding="utf-8"))

    def _role(self, role: str, parent: str) -> dict:
        states = list(gate.STATES)
        clips = []
        # Three clips cover all 3x3 transitions, one transition per band per clip.
        for clip_index in range(3):
            frames = []
            for frame_index in range(4):
                common = {
                    "frame_id": f"{role}-{clip_index}-{frame_index}",
                    "video_id": f"{role}-video",
                    "parent_id": parent,
                    "timestamp_ns": 1_000_000_000 + frame_index * 100_000_000,
                    "rgb_identity": f"rgb-{clip_index}-{frame_index}",
                    "rgb_sha256": SHA,
                }
                if role == "public_holdout":
                    common["sealed_target_id"] = f"sealed-{clip_index}-{frame_index}"
                else:
                    # Across adjacent pairs, each left state transitions to every right state.
                    common.update({
                        "teacher_depth_ref": f"depth-{clip_index}-{frame_index}",
                        "teacher_depth_sha256": SHA,
                        "teacher_timestamp_ns": common["timestamp_ns"],
                        "teacher_valid": True,
                        "tof_valid": True,
                        "frozen_a2_mean_abs_log_depth_disagreement": 0.1,
                        "clearance_m": [1.0, 1.5, 2.0],
                        "geometry_state": [
                            states[(band + frame_index) % 3]
                            if frame_index == 0
                            else states[(band + clip_index) % 3]
                            for band in range(3)
                        ],
                        "geometry_target_valid": [True, True, True],
                    })
                frames.append(common)
            clips.append({"clip_id": f"{role}-{clip_index}", "video_id": f"{role}-video", "parent_id": parent, "frames": frames})
        result = {
            "schema": "blindassist_dav2_temporal_392_student_p3_r0_1_role_manifest",
            "protocol_sha256": self.files["r0_1_protocol"]["expected_sha256"],
            "role": role,
            "clips": clips,
        }
        if role == "public_holdout":
            result["outcomes_opened"] = False
        return result

    def _build(self, bindings: dict | None = None) -> dict:
        return gate.build_receipt(
            self.root,
            self.protocol,
            bindings or self.bindings,
            current_git_commit=HEAD,
            activation_protocol_sha256=SHA,
        )

    def test_ready_and_exact_reproduction_without_parsing_sealed_bundle(self) -> None:
        receipt = self._build()
        self.assertEqual(gate.READY, receipt["terminal"])
        validation = gate.verify_receipt(
            receipt, self.root, self.protocol, self.bindings,
            current_git_commit=HEAD, activation_protocol_sha256=SHA,
        )
        self.assertTrue(validation["valid"])

    def test_missing_binding_fails_closed(self) -> None:
        bindings = copy.deepcopy(self.bindings)
        bindings["files"]["disagreement_cache"]["expected_sha256"] = None
        self.assertEqual(gate.INVALID, self._build(bindings)["terminal"])

    def test_parent_overlap_fails_closed(self) -> None:
        manifest = self._json("validation_manifest")
        for clip in manifest["clips"]:
            clip["parent_id"] = "train-parent"
            for frame in clip["frames"]:
                frame["parent_id"] = "train-parent"
        self._rewrite("validation_manifest", manifest)
        self.assertEqual(gate.INVALID, self._build()["terminal"])

    def test_candidate_output_and_runtime_mutation_fail_closed(self) -> None:
        (self.root / "candidate-output").mkdir()
        self.assertEqual(gate.INVALID, self._build()["terminal"])
        (self.root / "candidate-output").rmdir()
        bindings = copy.deepcopy(self.bindings)
        bindings["runtime_state_assertions"]["model_loaded"] = True
        self.assertEqual(gate.INVALID, self._build(bindings)["terminal"])

    def test_implementation_is_static_stdlib_only(self) -> None:
        source_path = Path(gate.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        forbidden = {"torch", "tensorflow", "onnxruntime", "cv2", "numpy"}
        self.assertTrue(imported.isdisjoint(forbidden))
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("torch.load", source)


if __name__ == "__main__":
    unittest.main()

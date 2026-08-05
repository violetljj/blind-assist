#!/usr/bin/env python3
"""Fail-closed contract tests for all five P3 R0.1 asset producers."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import build_p3_r0_1_legacy_p1_exclusion_ledger as exclusion
import build_p3_r0_1_role_manifests as roles
import materialize_p3_r0_1_frozen_a2_disagreement as disagreement
import produce_p3_r0_1_sealed_coverage_receipt as coverage
import produce_p3_r0_1_sealed_target_bundle as sealing
from p3_r0_1_asset_common import ROLE_MANIFEST_SCHEMA, STATES, sha256_file


class FrozenAssetProducerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = self._json("protocol.json", {"schema": "blindassist_dav2_temporal_392_student_p3_r0_1_protocol"})
        self.rgb = self._bytes("rgb.bin", b"rgb")
        self.teacher = self._bytes("teacher.npy", b"teacher")
        self.catalog_value, self.assignment_value = self._catalog_assignment()
        self.catalog = self._json("catalog.json", self.catalog_value)
        self.assignment = self._json("assignment.json", self.assignment_value)
        self.exclusion_value = {
            "schema": exclusion.LEDGER_SCHEMA,
            "status": "FROZEN_CONSUMED_EXCLUSION_ONLY",
            "parent_ids": ["legacy"],
            "outcomes_read": False,
        }
        self.exclusion = self._json("exclusion.json", self.exclusion_value)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _bytes(self, name: str, data: bytes) -> dict[str, str]:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": name.replace("\\", "/"), "sha256": sha256_file(path)}

    def _json(self, name: str, value: dict) -> dict[str, str]:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return {"path": name.replace("\\", "/"), "sha256": sha256_file(path)}

    def _producer_sha(self, module) -> str:
        return sha256_file(Path(module.__file__))

    def _catalog_assignment(self) -> tuple[dict, dict]:
        training, holdout, clips = [], [], []
        for role_index, role in enumerate(("train", "validation")):
            parent = f"{role}-parent"
            for clip_index in range(3):
                frame_ids = []
                for frame_index in range(4):
                    frame_id = f"{role}-{clip_index}-{frame_index}"
                    frame_ids.append(frame_id)
                    states = [
                        STATES[band] if frame_index == 0 else STATES[(band + clip_index) % 3]
                        for band in range(3)
                    ]
                    training.append({
                        "frame_id": frame_id, "video_id": f"{role}-video", "parent_id": parent,
                        "timestamp_ns": 1_000_000_000 + frame_index * 100_000_000,
                        "rgb_identity": frame_id, "rgb_path": self.rgb["path"], "rgb_sha256": self.rgb["sha256"],
                        "teacher_depth_ref": "npy-index:0", "teacher_depth_path": self.teacher["path"],
                        "teacher_depth_sha256": self.teacher["sha256"], "teacher_timestamp_ns": 1_000_000_000 + frame_index * 100_000_000,
                        "teacher_valid": True, "tof_valid": True, "clearance_m": [1.0, 1.5, 2.0],
                        "geometry_state": states, "geometry_target_valid": [True, True, True],
                    })
                clips.append({"clip_id": f"{role}-{clip_index}", "role": role, "video_id": f"{role}-video", "parent_id": parent, "frame_ids": frame_ids})
        for clip_index in range(3):
            frame_ids = []
            for frame_index in range(4):
                frame_id = f"holdout-{clip_index}-{frame_index}"
                frame_ids.append(frame_id)
                holdout.append({
                    "frame_id": frame_id, "video_id": "holdout-video", "parent_id": "holdout-parent",
                    "timestamp_ns": 2_000_000_000 + frame_index * 100_000_000,
                    "rgb_identity": frame_id, "rgb_path": self.rgb["path"], "rgb_sha256": self.rgb["sha256"],
                })
            clips.append({"clip_id": f"holdout-{clip_index}", "role": "public_holdout", "video_id": "holdout-video", "parent_id": "holdout-parent", "frame_ids": frame_ids})
        return (
            {"schema": roles.CATALOG_SCHEMA, "train_validation_frames": training, "public_holdout_frames": holdout},
            {"schema": roles.ASSIGNMENT_SCHEMA, "clip_length": 4, "clips": clips},
        )

    def _role_request(self, operation: str) -> dict:
        return {
            "schema": roles.REQUEST_SCHEMA, "operation": operation, "protocol": self.protocol,
            "source_catalog": self.catalog, "assignment": self.assignment, "exclusion_ledger": self.exclusion,
            "identity_lock": None, "disagreement_cache": None,
            "producer_sha256": self._producer_sha(roles),
            "outputs": {
                "identity_lock": "out/identity-lock.json", "public_holdout_manifest": "out/public.json", "receipt": "out/freeze-receipt.json"
            },
        }

    def test_exclusion_producer_rejects_outcomes_and_overwrite(self) -> None:
        roster_value = {
            "schema": "blindassist_dav2_model_variant_gate_r0_roster",
            "data_role": "consumed_development_engineering_regression_only",
            "source_manifest_sha256": "A" * 64,
            "sequence_counts": {"legacy": 1},
            "rows": [{
                "depth_path": "depth", "depth_sha256": "A" * 64, "frame_id": "legacy-frame", "index": 0,
                "intrinsics_fx_fy_cx_cy": [1, 1, 1, 1], "rgb_path": "rgb", "rgb_sha256": "B" * 64,
                "sequence_id": "legacy", "sequence_root": "root", "timestamp": 1.0,
            }],
        }
        roster = self._json("roster.json", roster_value)
        request = {
            "schema": exclusion.REQUEST_SCHEMA, "protocol": self.protocol, "legacy_p1_roster": roster,
            "producer_sha256": self._producer_sha(exclusion),
            "outputs": {"ledger": "ledger.json", "receipt": "ledger-receipt.json"},
        }
        exclusion.build(self.root, request, Path(exclusion.__file__))
        with self.assertRaisesRegex(ValueError, "overwrite forbidden"):
            exclusion.build(self.root, request, Path(exclusion.__file__))
        bad = copy.deepcopy(roster_value)
        bad["rows"][0]["outcome"] = "PASS"
        request["legacy_p1_roster"] = self._json("bad-roster.json", bad)
        request["outputs"] = {"ledger": "bad-ledger.json", "receipt": "bad-receipt.json"}
        with self.assertRaisesRegex(ValueError, "forbidden outcome"):
            exclusion.build(self.root, request, Path(exclusion.__file__))

    def test_role_producer_freezes_then_finalizes_and_rejects_overlap(self) -> None:
        freeze = self._role_request("freeze_identity")
        roles.build(self.root, freeze, Path(roles.__file__))
        public = json.loads((self.root / "out/public.json").read_text())
        self.assertEqual({"schema", "protocol_sha256", "role", "outcomes_opened", "clips"}, set(public))
        self.assertFalse(public["outcomes_opened"])
        identity = self._bytes("identity-copy.json", (self.root / "out/identity-lock.json").read_bytes())
        cache_rows = []
        for row in self.catalog_value["train_validation_frames"]:
            cache_rows.append(json.dumps({"schema": roles.DISAGREEMENT_SCHEMA, "frame_id": row["frame_id"], "mean_abs_log_depth_disagreement": 0.1}, sort_keys=True))
        cache = self._bytes("cache.jsonl", ("\n".join(cache_rows) + "\n").encode())
        final = self._role_request("finalize_training")
        final["identity_lock"] = identity
        final["disagreement_cache"] = cache
        final["outputs"] = {"train_manifest": "final/train.json", "validation_manifest": "final/validation.json", "class_weight_receipt": "final/weights.json", "receipt": "final/receipt.json"}
        roles.build(self.root, final, Path(roles.__file__))
        self.assertEqual(9, len(json.loads((self.root / "final/weights.json").read_text())["weights"]))
        bad_assignment = copy.deepcopy(self.assignment_value)
        bad_assignment["clips"][3]["parent_id"] = "train-parent"
        for frame_id in bad_assignment["clips"][3]["frame_ids"]:
            for row in self.catalog_value["train_validation_frames"]:
                if row["frame_id"] == frame_id:
                    row["parent_id"] = "train-parent"
        bad_catalog = self._json("bad-catalog.json", self.catalog_value)
        bad_assignment_binding = self._json("bad-assignment.json", bad_assignment)
        bad_request = self._role_request("freeze_identity")
        bad_request["source_catalog"], bad_request["assignment"] = bad_catalog, bad_assignment_binding
        bad_request["outputs"] = {"identity_lock": "bad/lock.json", "public_holdout_manifest": "bad/public.json", "receipt": "bad/receipt.json"}
        with self.assertRaisesRegex(ValueError, "overlap"):
            roles.build(self.root, bad_request, Path(roles.__file__))

    def test_disagreement_producer_uses_locked_training_only_and_fails_sha(self) -> None:
        freeze = self._role_request("freeze_identity")
        freeze["outputs"] = {"identity_lock": "d/lock.json", "public_holdout_manifest": "d/public.json", "receipt": "d/freeze.json"}
        roles.build(self.root, freeze, Path(roles.__file__))
        lock = self._bytes("d/lock-bound.json", (self.root / "d/lock.json").read_bytes())
        checkpoint = self._bytes("a2.pth", b"opaque")
        training_receipt = self._json("a2-receipt.json", {
            "schema": "blindassist_dav2_392_distillation_a2_r0_training_result", "truth_inputs_opened": False,
            "checkpoint": {"sha256": checkpoint["sha256"]},
        })
        dpt = self._bytes("dpt.py", b"source")
        request = {
            "schema": disagreement.REQUEST_SCHEMA, "protocol": self.protocol, "a2_checkpoint": checkpoint,
            "a2_training_receipt": training_receipt, "identity_lock": lock, "source_catalog": self.catalog,
            "dav2_repo_path": ".", "dav2_dpt_source": dpt, "input_size": 392, "device": "cpu",
            "producer_sha256": self._producer_sha(disagreement),
            "outputs": {"cache": "d/cache.jsonl", "manifest": "d/manifest.json", "receipt": "d/receipt.json"},
        }
        disagreement.build(self.root, request, Path(disagreement.__file__), infer_factory=lambda _root, _request: lambda _row: 0.125)
        manifest = json.loads((self.root / "d/manifest.json").read_text())
        self.assertFalse(manifest["p3_model_constructed"])
        bad = copy.deepcopy(request)
        bad["a2_checkpoint"]["sha256"] = "0" * 64
        bad["outputs"] = {"cache": "x/cache", "manifest": "x/manifest", "receipt": "x/receipt"}
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            disagreement.build(self.root, bad, Path(disagreement.__file__), infer_factory=lambda *_: None)

    def _sealed_fixture(self, clip_count: int = 32) -> tuple[dict, dict]:
        public_clips, private_clips = [], []
        for clip_index in range(clip_count):
            parent = f"sealed-parent-{clip_index % 8}"
            sequence = (["CLEAR", "OCCUPIED", "UNKNOWN_GROUND", "CLEAR"] if clip_index % 2 == 0 else ["OCCUPIED", "CLEAR", "UNKNOWN_GROUND", "OCCUPIED"])
            public_frames, private_frames = [], []
            for frame_index, state in enumerate(sequence):
                target_id = f"sealed-{clip_index}-{frame_index}"
                base = {"frame_id": target_id, "video_id": parent, "parent_id": parent, "timestamp_ns": 3_000_000_000 + frame_index * 100_000_000, "sealed_target_id": target_id, "rgb_identity": target_id, "rgb_sha256": "A" * 64}
                public_frames.append(base)
                private_frames.append(base | {"clearance_m": [1.0, 1.0, 1.0], "geometry_state": [state, state, state], "geometry_target_valid": [True, True, True], "teacher_valid": True, "tof_valid": True, "truth_depth_ref": target_id, "truth_depth_sha256": "B" * 64})
            public_clips.append({"clip_id": f"sealed-clip-{clip_index}", "video_id": parent, "parent_id": parent, "frames": public_frames})
            private_clips.append({"clip_id": f"sealed-clip-{clip_index}", "video_id": parent, "parent_id": parent, "frames": private_frames})
        return (
            {"schema": ROLE_MANIFEST_SCHEMA, "protocol_sha256": self.protocol["sha256"], "role": "public_holdout", "outcomes_opened": False, "clips": public_clips},
            {"schema": sealing.PRIVATE_SCHEMA, "clips": private_clips},
        )

    def test_sealing_and_coverage_producers_fail_closed(self) -> None:
        public_value, private_value = self._sealed_fixture()
        public, private = self._json("s/public.json", public_value), self._json("s/private.json", private_value)
        key = self._bytes("s/key.bin", bytes(range(32)))
        seal_request = {
            "schema": sealing.REQUEST_SCHEMA, "protocol": self.protocol, "public_holdout_manifest": public,
            "private_targets": private, "encryption_key": key, "producer_sha256": self._producer_sha(sealing),
            "outputs": {"bundle": "s/bundle.bin", "receipt": "s/bundle-receipt.json"},
        }
        sealing.build(self.root, seal_request, Path(sealing.__file__))
        self.assertTrue((self.root / "s/bundle.bin").read_bytes().startswith(sealing.MAGIC))
        with self.assertRaisesRegex(ValueError, "overwrite forbidden"):
            sealing.build(self.root, seal_request, Path(sealing.__file__))
        bundle = self._bytes("s/bundle-copy.bin", (self.root / "s/bundle.bin").read_bytes())
        bundle_receipt = self._bytes("s/bundle-receipt-copy.json", (self.root / "s/bundle-receipt.json").read_bytes())
        coverage_request = {
            "schema": coverage.REQUEST_SCHEMA, "protocol": self.protocol, "public_holdout_manifest": public,
            "private_targets": private, "sealed_target_bundle": bundle, "bundle_production_receipt": bundle_receipt,
            "producer_sha256": self._producer_sha(coverage),
            "outputs": {"coverage_receipt": "s/coverage.json", "materialization_receipt": "s/coverage-materialization.json"},
        }
        coverage.build(self.root, coverage_request, Path(coverage.__file__))
        self.assertFalse(json.loads((self.root / "s/coverage.json").read_text())["label_rows_disclosed"])
        small_public, small_private = self._sealed_fixture(8)
        small_public_binding = self._json("small/public.json", small_public)
        small_private_binding = self._json("small/private.json", small_private)
        bad = copy.deepcopy(coverage_request)
        bad["public_holdout_manifest"], bad["private_targets"] = small_public_binding, small_private_binding
        bad["outputs"] = {"coverage_receipt": "small/coverage.json", "materialization_receipt": "small/receipt.json"}
        with self.assertRaisesRegex(ValueError, "32 evaluable"):
            coverage.build(self.root, bad, Path(coverage.__file__))

    def test_no_producer_constructs_p3_or_optimizer(self) -> None:
        for module in (exclusion, roles, disagreement, sealing, coverage):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for forbidden in ("torch.optim", "AdamW(", "DecoupledTemporalStateHead", "optimizer ="):
                self.assertNotIn(forbidden, source, f"{module.__name__} contains {forbidden}")


if __name__ == "__main__":
    unittest.main()

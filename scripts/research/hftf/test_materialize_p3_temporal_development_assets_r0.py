#!/usr/bin/env python3
"""Tests for the development-only P3 temporal asset producer."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import materialize_p3_temporal_development_assets_r0 as subject


class DevelopmentAssetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.rgb = self.root / "rgb.png"; self.rgb.write_bytes(b"rgb")
        self.teacher_depth = self.root / "teacher.npy"; np.save(self.teacher_depth, np.ones((64, 2, 2), np.float32))
        self.depth_png = self.root / "truth.png"; self.depth_png.write_bytes(b"depth")
        self.intrinsics = self.root / "intrinsics.pincam"; self.intrinsics.write_text("256 192 100 101 128 96", encoding="utf8")
        clearance = np.full((64, 3), 2.0, np.float32)
        validity = np.ones((64, 3), bool)
        # The first 13 clips collectively enumerate all 3x3 state transitions.
        for clip in range(13):
            source, target = clip % 3, (clip // 3) % 3
            for offset, state in enumerate((source, target, source, target)):
                index = clip * 4 + offset
                if state == 0:
                    clearance[index] = 2.0
                elif state == 1:
                    clearance[index] = 1.0
                else:
                    validity[index] = False
        self.arrays = self.root / "arrays.npz"; np.savez(self.arrays, truth_clearance=clearance, truth_valid=validity)
        self.protocol = self._json("protocol.json", {"schema": subject.PROTOCOL_SCHEMA, "status": "FROZEN_BEFORE_MODEL_LOAD_OR_TRAINING", "claim_ceiling": "DEVELOPMENT_SIGNAL_ONLY"})
        records = []
        for i in range(64): records.append({"frame_id": f"f{i}", "index": i, "rgb_path": "rgb.png", "rgb_sha256": self._sha(self.rgb)})
        self.teacher = self._json("teacher.json", {"schema": subject.TEACHER_SCHEMA, "teacher_depth": {"sha256": self._sha(self.teacher_depth)}, "records": records})
        self.spatial = self._json("spatial.json", {"schema": subject.SPATIAL_SCHEMA, "arrays": {"sha256": self._sha(self.arrays)}, "records": [{"frame_stem": f"f{i}"} for i in range(64)]})
        self.media = self._json("media.json", {"videos": [{"extracted": {"lowres_depth": [{"path": str(self.depth_png), "sha256": self._sha(self.depth_png)} | {"path": str(self.depth_png).replace("truth", f"f{i}")} for i in []], "lowres_wide_intrinsics": []}}]})
        # The production index uses path stems, so materialize tiny per-frame fixtures.
        depth, intrinsics = [], []
        for i in range(64):
            d=self.root/f"f{i}.png"; d.write_bytes(b"depth"); depth.append({"path":str(d),"sha256":self._sha(d)})
            p=self.root/f"f{i}.pincam"; p.write_text("256 192 100 101 128 96",encoding="utf8"); intrinsics.append({"path":str(p),"sha256":self._sha(p)})
        self.media=self._json("media.json", {"videos":[{"extracted":{"lowres_depth":depth,"lowres_wide_intrinsics":intrinsics}}]})
        self.a2 = self._bytes("a2.pth", b"opaque"); self.a2receipt = self._json("a2result.json", {"schema": "blindassist_dav2_392_distillation_a2_r0_training_result", "checkpoint": {"sha256": self.a2["sha256"]}}); self.dpt = self._bytes("dpt.py", b"source")
        self.train = self._identity("train", 13, 0); self.validation = self._identity("validation", 3, 52)

    def tearDown(self): self.tmp.cleanup()
    def _sha(self, p): return subject._sha_file(p)
    def _bytes(self, n, b):
        p=self.root/n; p.write_bytes(b); return {"path": n, "sha256": self._sha(p)}
    def _json(self, n, v):
        p=self.root/n; p.write_text(json.dumps(v), encoding="utf8"); return {"path": n,"sha256":self._sha(p)}
    def _identity(self, role, parents, start):
        clips=[]
        for p in range(parents):
            frames=[]
            for j in range(4):
                i=start+p*4+j; frames.append({"frame_id":f"f{i}","video_id":f"v{p}","parent_id":f"{role}{p}","timestamp_ns":1_000_000_000+i})
            clips.append({"clip_id":f"{role}{p}","video_id":f"v{p}","parent_id":f"{role}{p}","frames":frames})
        return self._json(f"{role}.json", {"schema":subject.IDENTITY_SCHEMA,"role":role,"clips":clips})
    def _request(self):
        return {"schema":subject.REQUEST_SCHEMA,"protocol":self.protocol,"train_identity":self.train,"validation_identity":self.validation,"teacher_manifest":self.teacher,"teacher_depth":{"path":"teacher.npy","sha256":self._sha(self.teacher_depth)},"spatial_manifest":self.spatial,"spatial_arrays":{"path":"arrays.npz","sha256":self._sha(self.arrays)},"arkit_media_manifest":self.media,"a2_checkpoint":self.a2,"a2_training_receipt":self.a2receipt,"dav2_repo_path":".","dav2_dpt_source":self.dpt,"device":"cpu","producer_sha256":self._sha(Path(subject.__file__)),"outputs":{"train_manifest":"out/train.json","validation_manifest":"out/val.json","disagreement_cache":"out/disagreement.jsonl","class_weights":"out/weights.json","activation_bindings":"out/activation.json","receipt":"out/receipt.json"}}
    def test_materializes_only_after_static_preflight(self):
        req=self._request(); subject.build(self.root, req, Path(subject.__file__), infer_factory=lambda *_: lambda _row:.1)
        train=json.loads((self.root/'out/train.json').read_text()); row=train['clips'][0]['frames'][0]
        self.assertIn(row['geometry_state'][0], ('CLEAR', 'OCCUPIED', 'UNKNOWN_GROUND')); self.assertEqual('CLEAR',row['geometry_state'][1]); self.assertTrue(row['tof_valid']); self.assertEqual(row['timestamp_ns'],row['teacher_timestamp_ns'])
        self.assertEqual(9,len(json.loads((self.root/'out/weights.json').read_text())['weights']))
    def test_rejects_sealed_input_and_overwrite(self):
        req=self._request(); req['outputs']['receipt']='sealing-attempt-02/receipt.json'
        with self.assertRaisesRegex(ValueError,'sealed Bonn'):
            subject.build(self.root,req,Path(subject.__file__),infer_factory=lambda *_:None)
        req=self._request(); subject.build(self.root,req,Path(subject.__file__),infer_factory=lambda *_:lambda _:.1)
        with self.assertRaisesRegex(ValueError,'overwrite'):
            subject.build(self.root,req,Path(subject.__file__),infer_factory=lambda *_:lambda _:.1)

    def test_lexical_repo_boundary_accepts_artifacts_junction_path(self):
        root = Path(r"E:\repo")
        expected = Path(os.path.abspath(root / "artifacts.local/evidence/value.json"))
        self.assertEqual(expected, subject._inside(root, "artifacts.local/evidence/value.json"))
        with self.assertRaisesRegex(ValueError, "path leaves repository"):
            subject._inside(root, "../escape.json")

    def test_frozen_input_schema_names_match_materialized_assets(self):
        self.assertEqual("blindassist_dav2_distillation_teacher_r0", subject.TEACHER_SCHEMA)
        self.assertEqual("blindassist_spatial_calibration_head_r1_cache", subject.SPATIAL_SCHEMA)
        self.assertEqual("blindassist_p3_r0_2_identity_manifest", subject.IDENTITY_SCHEMA)

    def test_activation_binding_contract_fields(self):
        self.assertEqual("DEVELOPMENT_SIGNAL_ONLY", "DEVELOPMENT_SIGNAL_ONLY")
        self.assertEqual("P3_TEMPORAL_DEVELOPMENT_ASSETS_MATERIALIZED_DEVELOPMENT_SIGNAL_ONLY", "P3_TEMPORAL_DEVELOPMENT_ASSETS_MATERIALIZED_DEVELOPMENT_SIGNAL_ONLY")

if __name__ == '__main__': unittest.main()

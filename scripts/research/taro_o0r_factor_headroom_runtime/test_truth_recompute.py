from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime.truth_recompute import (
    TruthRecomputeError,
    recompute_committed_truth,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime import test_source_adapter as fixtures
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o0r_truth_materializer_runtime.test_materializer import reseal


def _source_frame() -> dict[str, object]:
    receipt = copy.deepcopy(fixtures.source_receipt())
    container_sha = "A" * 64
    token = receipt["sensor_timestamp"]["decimal_token"]
    video_id = receipt["session_id"]
    for role in materializer.DECODED_ROLES:
        receipt["asset_bindings"][role]["container_id"] = f"sha256:{container_sha}"
    receipt = reseal(receipt)
    members: dict[str, dict[str, object]] = {}
    for role in materializer.DECODED_ROLES:
        asset = receipt["asset_bindings"][role]
        decoded = receipt["decoded_payload_bindings"][role]
        members[role] = {
            "source_container_sha256": container_sha,
            "source_member_path": "lowres_wide.traj" if role == "trajectory" else f"official/{role}/{video_id}_{token}.bin",
            "source_member_bytes": asset["bytes"],
            "source_member_sha256": asset["sha256"],
            "source_member_crc32": asset["crc32"],
            "canonical_member_path": asset["member_path"],
            "decoded_content_sha256": decoded["decoded_content_sha256"],
        }
    envelope = materializer.seal_record(
        {
            "schema": materializer.BOUND_SOURCE_FRAME_SCHEMA,
            "source_role": receipt["source_role"],
            "visit_id": receipt["site_id"],
            "video_id": video_id,
            "timestamp_token": token,
            "physical_frame_id": receipt["physical_frame_id"],
            "source_frame_receipt_sha256": receipt["content_sha256"],
            "members": members,
            "mapping_rule": "OFFICIAL_VIDEO_ID_TIMESTAMP_TO_EXPLICIT_TIMESTAMP_ONLY_CANONICAL_MEMBER",
            "silent_rename_allowed": False,
        }
    )
    faro = fixtures.synthetic_faro_depth(True)
    confidence = np.full(adapter.APPLE_SHAPE_HW, 2, dtype=np.uint8)
    return {
        "source_frame_receipt": receipt,
        "bound_source_frame_envelope": envelope,
        "highres_faro_depth_mm": faro,
        "confidence": confidence,
    }


class TruthRecomputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = _source_frame()
        cls.model = fixtures.fitted_model(1)
        compact = materializer.build_eval_truth_record(cls.frame, cls.model)
        package, blobs = materializer.package_content_addressed_artifact(compact)
        persisted_package = json.loads(materializer.canonical_json_bytes(package))
        cls.compact = materializer.hydrate_content_addressed_artifact(persisted_package, lambda path: blobs[path])

    def test_recomputes_all_dense_frames_and_matches_compact_commitments(self) -> None:
        recomputed = recompute_committed_truth(self.frame, self.model, self.compact)
        self.assertEqual(len(recomputed["truth_factor_frames"]), 9)
        self.assertEqual(recomputed["compact_truth_record_sha256"], self.compact["content_sha256"])
        self.assertEqual(
            [frame["content_sha256"] for frame in recomputed["truth_factor_frames"]],
            [row["factor_frame_sha256"] for row in self.compact["factor_frame_commitments"]],
        )

    def test_rejects_redecoded_source_mismatch(self) -> None:
        frame = dict(self.frame)
        frame["confidence"] = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
        with self.assertRaises((TruthRecomputeError, materializer.MaterializerError)):
            recompute_committed_truth(frame, self.model, self.compact)


if __name__ == "__main__":
    unittest.main()

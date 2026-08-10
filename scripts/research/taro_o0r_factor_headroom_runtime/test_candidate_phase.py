from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import (
    CandidatePhaseError,
    expected_candidate_keys,
    load_sealed_candidate_frame,
    run_candidate_phase,
)
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


def _frame_plan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, roster in (
        ("ADAPTER_FIT", adapter.ADAPTER_FIT_ROSTER),
        ("O0R_EVAL_CANDIDATE", adapter.O0R_EVAL_CANDIDATE_ROSTER),
    ):
        for rank, (parent_id, video_id) in enumerate(roster):
            token = f"{rank + 1}.000"
            rows.append(
                {
                    "parent": {"role": role, "visit_id": parent_id, "video_id": video_id},
                    "frame_plan": {"exact_timestamp_tokens": [token]},
                    "container_receipts": {},
                }
            )
    return rows


class CandidatePhaseTests(unittest.TestCase):
    def test_seals_every_frozen_candidate_before_completion(self) -> None:
        plan = _frame_plan()
        keys = expected_candidate_keys(plan)
        native = np.ones((448, 608), dtype=np.float32)
        native_sha = adapter.canonical_sha256(native)

        def iterator(*_args: object, **_kwargs: object):
            for parent_id, video_id, token in keys:
                yield {
                    "candidate_input_receipt": {
                        "parent_id": parent_id,
                        "video_id": video_id,
                        "timestamp_token": token,
                        "content_sha256": "A" * 64,
                    },
                    "color_rgb_u8": np.zeros((1,), dtype=np.uint8),
                }

        def inference(_model: object, *, candidate_input_receipt: dict[str, object], **_kwargs: object) -> dict[str, object]:
            return {
                "native_depth_m": native,
                "inference_receipt": {
                    "parent_id": candidate_input_receipt["parent_id"],
                    "video_id": candidate_input_receipt["video_id"],
                    "timestamp_token": candidate_input_receipt["timestamp_token"],
                    "candidate_input_receipt_sha256": candidate_input_receipt["content_sha256"],
                    "native_output_array_sha256": native_sha,
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "factor"
            writer = FactorEvidenceWriter(root, 64 * 1024 * 1024)
            writer.activate({"schema": "test"})
            with (
                patch(
                    "scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase.validate_candidate_input_receipt",
                    side_effect=lambda value: value,
                ),
                patch(
                    "scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase.validate_depthart_inference_receipt",
                    side_effect=lambda value: value,
                ),
            ):
                completion = run_candidate_phase(
                    plan,
                    Path(temporary),
                    writer=writer,
                    model=object(),
                    runtime_identity={"runtime": "test"},
                    candidate_iterator_fn=iterator,
                    inference_fn=inference,
                )
            self.assertEqual(completion["candidate_frame_count"], 16)
            self.assertEqual(completion["truth_frame_packages_opened_before_completion"], 0)
            self.assertTrue((root / "candidate-phase-completion.json").is_file())
            self.assertEqual(len(list((root / "candidates").rglob("*.record.json"))), 16)
            self.assertEqual(len(list((root / "candidates").rglob("*.depth.npy.gz"))), 16)
            with (
                patch(
                    "scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase.validate_candidate_input_receipt",
                    side_effect=lambda value: value,
                ),
                patch(
                    "scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase.validate_depthart_inference_receipt",
                    side_effect=lambda value: value,
                ),
            ):
                loaded = load_sealed_candidate_frame(root, *keys[0])
            self.assertEqual(loaded["native_depth_m"].shape, (448, 608))

    def test_rejects_candidate_sequence_drift(self) -> None:
        plan = _frame_plan()
        keys = list(reversed(expected_candidate_keys(plan)))

        def iterator(*_args: object, **_kwargs: object):
            parent_id, video_id, token = keys[0]
            yield {
                "candidate_input_receipt": {"parent_id": parent_id, "video_id": video_id, "timestamp_token": token},
                "color_rgb_u8": np.zeros((1,), dtype=np.uint8),
            }

        with tempfile.TemporaryDirectory() as temporary:
            writer = FactorEvidenceWriter(Path(temporary) / "factor", 1024 * 1024)
            writer.activate({"schema": "test"})
            with patch(
                "scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase.validate_candidate_input_receipt",
                side_effect=lambda value: value,
            ):
                with self.assertRaises(CandidatePhaseError) as error:
                    run_candidate_phase(
                        plan,
                        Path(temporary),
                        writer=writer,
                        model=object(),
                        runtime_identity={"runtime": "test"},
                        candidate_iterator_fn=iterator,
                    )
            self.assertEqual(error.exception.code, "CANDIDATE_SEQUENCE_DRIFT")


if __name__ == "__main__":
    unittest.main()

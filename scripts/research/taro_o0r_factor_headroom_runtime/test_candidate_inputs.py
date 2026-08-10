#!/usr/bin/env python3
"""Focused proof that candidate input materialization opens only RGB and K."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_inputs import iter_candidate_inputs
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _png() -> bytes:
    value = np.zeros((*adapter.HIGHRES_SHAPE_HW, 3), dtype=np.uint8)
    value[..., 0] = 12
    value[..., 1] = 34
    value[..., 2] = 56
    buffer = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class CandidateInputTests(unittest.TestCase):
    def test_only_rgb_and_intrinsics_are_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows: list[dict[str, object]] = []
            expected_reads: list[tuple[str, str]] = []
            for index, (visit_id, video_id) in enumerate(adapter.O0R_EVAL_CANDIDATE_ROSTER):
                token = f"{index + 1}.100000000"
                up_path = root / "upsampling" / "Training" / f"{video_id}.zip"
                intr_path = root / "raw" / "Training" / video_id / "lowres_wide_intrinsics.zip"
                up_path.parent.mkdir(parents=True, exist_ok=True)
                intr_path.parent.mkdir(parents=True, exist_ok=True)
                color_member = f"{video_id}/wide/{video_id}_{token}.png"
                intr_member = f"lowres_wide_intrinsics/{video_id}_{token}.pincam"
                with zipfile.ZipFile(up_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    bundle.writestr(color_member, _png())
                    bundle.writestr(f"{video_id}/highres_depth/{video_id}_{token}.png", b"FORBIDDEN_FARO_PAYLOAD")
                    bundle.writestr(f"{video_id}/lowres_depth/{video_id}_{token}.png", b"FORBIDDEN_APPLE_PAYLOAD")
                    bundle.writestr(f"{video_id}/confidence/{video_id}_{token}.png", b"FORBIDDEN_CONFIDENCE_PAYLOAD")
                with zipfile.ZipFile(intr_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    bundle.writestr(intr_member, b"256 192 200.0 200.0 127.5 95.5\n")
                rows.append(
                    {
                        "parent": {"role": "O0R_EVAL_CANDIDATE", "visit_id": visit_id, "video_id": video_id, "official_fold": "Training"},
                        "frame_plan": {"exact_timestamp_tokens": [token]},
                        "container_receipts": {
                            "upsampling.zip": {"bytes": up_path.stat().st_size, "sha256": _sha(up_path)},
                            "lowres_wide_intrinsics.zip": {"bytes": intr_path.stat().st_size, "sha256": _sha(intr_path)},
                        },
                    }
                )
                expected_reads.extend((("color", color_member), ("intrinsics", intr_member)))
            for visit_id, video_id in adapter.ADAPTER_FIT_ROSTER:
                rows.insert(
                    0,
                    {
                        "parent": {"role": "ADAPTER_FIT", "visit_id": visit_id, "video_id": video_id, "official_fold": "Training"},
                        "frame_plan": {"exact_timestamp_tokens": ["1.0"]},
                        "container_receipts": {},
                    },
                )
            reads: list[tuple[str, str]] = []
            outputs = list(iter_candidate_inputs(rows, root, read_observer=lambda role, member: reads.append((role, member))))
            self.assertEqual(16, len(outputs))
            self.assertEqual(expected_reads, reads)
            self.assertEqual({"color", "intrinsics"}, {role for role, _ in reads})
            self.assertTrue(all(item["candidate_input_receipt"]["truth_payloads_opened"] is False for item in outputs))


if __name__ == "__main__":
    unittest.main()

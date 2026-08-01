"""End-to-end canary for the batch correspondence contract."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.dual_loop_segmentation_instance_correspondence.batch import PROTOCOL_ID, run_batch


def _pack(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(np.uint8), bitorder="big").tobytes()).decode("ascii")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BatchCanaryTest(unittest.TestCase):
    def test_depth_flow_and_temporal_evidence_reach_batch_output(self) -> None:
        shape = (16, 16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "analysis_shape": list(shape),
                        "thresholds": {
                            "minimum_pair_score": 0.45,
                            "minimum_score_margin": 0.05,
                            "minimum_present_evidence": 2,
                        },
                        "class_compatibility": {
                            "yolo_label_to_semantic": {"person": "obstacle"},
                            "segmentation_aliases": {"obstacle": ["person"]},
                        },
                    }
                ),
                encoding="utf-8",
            )
            frames: list[dict[str, object]] = []
            components: list[dict[str, object]] = []
            traces: list[dict[str, object]] = []
            depths: list[dict[str, object]] = []
            for frame_id, offset in enumerate((0, 1)):
                image_sha = f"image-{frame_id}"
                mask = np.zeros(shape, dtype=bool)
                mask[4 + offset : 8 + offset, 4 + offset : 8 + offset] = True
                frames.append(
                    {
                        "source_id": "source",
                        "session_id": "session",
                        "frame_id": frame_id,
                        "source_capture_timestamp_ns": frame_id,
                        "image_sha256": image_sha,
                        "view_row_id": f"view-{frame_id}",
                        "packed_masks": {
                            "shape": list(shape),
                            "candidate_obstacle": _pack(mask),
                            "candidate_boundary_step_curb": _pack(np.zeros(shape, dtype=bool)),
                        },
                    }
                )
                components.append(
                    {
                        "component_id": f"source:{frame_id}:obstacle:0",
                        "source_id": "source",
                        "frame_id": frame_id,
                        "class_name": "obstacle",
                        "component_index": 0,
                        "area_pixels": 16,
                        "bbox_xyxy": [4 + offset, 4 + offset, 8 + offset, 8 + offset],
                    }
                )
                traces.append(
                    {
                        "schema_version": "fixture",
                        "source_id": "source",
                        "frame_id": frame_id,
                        "image_sha256": image_sha,
                        "source_capture_timestamp_ns": frame_id,
                        "detections": [
                            {
                                "label": "person",
                                "confidence": 0.9,
                                "left": 3 + offset,
                                "top": 3 + offset,
                                "right": 9 + offset,
                                "bottom": 9 + offset,
                                "frame_width": 16,
                                "frame_height": 16,
                            }
                        ],
                    }
                )
                depths.append(
                    {
                        "source_id": "source",
                        "frame_id": frame_id,
                        "cluster_id": "cluster-0",
                        "bbox_xyxy": [3 + offset, 3 + offset, 9 + offset, 9 + offset],
                        "median_depth": 2.0,
                    }
                )
            flow = root / "flow.jsonl"
            _jsonl(
                flow,
                [
                    {
                        "source_id": "source",
                        "frame_id": 1,
                        "previous_source_id": "source",
                        "previous_frame_id": 0,
                        "matrix_previous_to_current": [[1, 0, 1], [0, 1, 1]],
                    }
                ],
            )
            config_path = config
            frames_path = root / "frames.jsonl"
            components_path = root / "components.jsonl"
            trace_path = root / "trace.jsonl"
            depth_path = root / "depth.jsonl"
            _jsonl(frames_path, frames)
            _jsonl(components_path, components)
            _jsonl(trace_path, traces)
            _jsonl(depth_path, depths)
            output = root / "artifacts.local" / "correspondence"
            summary = run_batch(
                repo_root=root,
                config_path=config_path,
                frames_paths=[frames_path],
                components_paths=[components_path],
                yolo_trace_paths=[trace_path],
                output_root=output,
                depth_clusters_path=depth_path,
                optical_flow_path=flow,
            )
            self.assertEqual(summary["input_counts"]["frames"], 2)
            self.assertTrue(summary["evidence_availability"]["depth_clusters_input_supplied"])
            self.assertTrue(summary["evidence_availability"]["optical_flow_input_supplied"])
            self.assertGreaterEqual(summary["state_counts"]["component"].get("MATCH", 0), 1)
            rows = [json.loads(line) for line in (output / "pair_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["state"] == "MATCH" for row in rows))
            self.assertTrue(any(row["depth_consistency"]["state"] == "CONSISTENT" for row in rows))
            self.assertTrue(any(row["optical_flow"] is not None for row in rows))


if __name__ == "__main__":
    unittest.main()

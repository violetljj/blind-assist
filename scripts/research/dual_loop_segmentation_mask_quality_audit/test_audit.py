from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.dual_loop_segmentation_mask_quality_audit.audit import (
    CLASS_ORDER,
    DEFAULT_CONFIG,
    STATUS_INVALID,
    STATUS_PASS,
    STATUS_REVIEW,
    MaskQualityAuditError,
    audit_manifest,
    audit_rows,
    sha256_file,
    write_report,
)


class MaskQualityAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        self.rgb[..., 0] = np.arange(8, dtype=np.uint8)[None, :]
        self.rgb[..., 1] = np.arange(8, dtype=np.uint8)[:, None]
        self.mask = np.zeros((8, 8), dtype=np.uint8)
        self.mask[:2, :] = 3
        self.mask[2:4, 2:4] = 2
        self.mask[4:6, 4:7] = 1
        self.write("rgb-0.png", self.rgb, "RGB")
        self.write("mask-0.png", self.mask, "L")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, value: np.ndarray, mode: str) -> Path:
        path = self.root / name
        Image.fromarray(value, mode=mode).save(path)
        return path

    def row(self, *, frame_id: int = 0, mask_path: str = "mask-0.png", **extra: object) -> dict[str, object]:
        row: dict[str, object] = {
            "id": f"s-{frame_id}",
            "session_id": "session-1",
            "sequence_id": "sequence-1",
            "frame_id": frame_id,
            "rgb_path": "rgb-0.png",
            "mask_path": mask_path,
            "label_space": "blindassist_riskseg_v1",
            "class_order": list(CLASS_ORDER),
            "source_to_expected_mapping": {"0": 0, "1": 1, "2": 2, "3": 3},
            "rgb_frame_key": f"frame-{frame_id}",
            "mask_frame_key": f"frame-{frame_id}",
            "mask_decoder": "canonical",
            "rgb_sha256": sha256_file(self.root / "rgb-0.png"),
            "mask_sha256": sha256_file(self.root / mask_path),
        }
        row.update(extra)
        return row

    @staticmethod
    def review(row_id: str, status: str = STATUS_PASS, reason_codes: list[str] | None = None) -> dict[str, object]:
        return {
            "id": row_id,
            "status": status,
            "reason_codes": [] if reason_codes is None else reason_codes,
            "reviewer_id": "test-reviewer",
        }

    def config(self, **updates: object) -> dict[str, object]:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(updates)
        return config

    def audit(self, rows: list[dict[str, object]], reviews: list[dict[str, object]] | None = None, config: dict[str, object] | None = None) -> dict[str, object]:
        annotations = {str(item["id"]): item for item in (reviews or [])}
        return audit_rows(rows, base_root=self.root, config=config or self.config(), review_annotations=annotations)

    def test_clean_frame_requires_and_accepts_visual_pass(self) -> None:
        report = self.audit([self.row()], [self.review("s-0")])
        self.assertEqual(STATUS_PASS, report["status"])
        frame = report["frames"][0]
        self.assertEqual(STATUS_PASS, frame["status"])
        self.assertEqual([], frame["reason_codes"])
        self.assertTrue(frame["provenance"]["original_label_immutable"])
        self.assertFalse(frame["provenance"]["replacement_applied"])

    def test_no_visual_review_never_becomes_pass(self) -> None:
        report = self.audit([self.row()])
        self.assertEqual(STATUS_REVIEW, report["status"])
        self.assertIn("MANUAL_REVIEW_REQUIRED", report["frames"][0]["reason_codes"])

    def test_class_order_and_mapping_are_fail_closed(self) -> None:
        wrong_order = self.row(class_order=["walkable", "boundary_level_change", "blocking_obstacle", "unknown_nonwalkable"])
        missing_mapping = self.row(id="s-1")
        missing_mapping.pop("source_to_expected_mapping")
        swapped_mapping = self.row(id="s-2", source_to_expected_mapping={"0": 0, "1": 2, "2": 1, "3": 3})
        report = self.audit(
            [wrong_order, missing_mapping, swapped_mapping],
            [self.review("s-0"), self.review("s-1"), self.review("s-2")],
        )
        self.assertEqual(STATUS_INVALID, report["status"])
        self.assertIn("CLASS_ID_ORDER_MISMATCH", report["frames"][0]["reason_codes"])
        self.assertIn("CLASS_ID_MAPPING_UNVERIFIED", report["frames"][1]["reason_codes"])
        self.assertIn("CLASS_ID_MAPPING_MISMATCH", report["frames"][2]["reason_codes"])

    def test_out_of_range_id_and_dimension_mismatch_are_invalid(self) -> None:
        bad_mask = np.full((8, 8), 7, dtype=np.uint8)
        self.write("bad-id.png", bad_mask, "L")
        short_mask = np.zeros((7, 8), dtype=np.uint8)
        self.write("short.png", short_mask, "L")
        rows = [self.row(mask_path="bad-id.png"), self.row(frame_id=1, mask_path="short.png")]
        rows[1]["rgb_frame_key"] = "frame-1"
        rows[1]["mask_frame_key"] = "frame-1"
        report = self.audit(rows, [self.review("s-0"), self.review("s-1")])
        self.assertEqual(STATUS_INVALID, report["status"])
        self.assertIn("MASK_CLASS_ID_OUT_OF_RANGE", report["frames"][0]["reason_codes"])
        self.assertIn("RGB_MASK_DIMENSION_MISMATCH", report["frames"][1]["reason_codes"])

    def test_original_hash_binding_is_required(self) -> None:
        row = self.row()
        row.pop("rgb_sha256")
        row.pop("mask_sha256")
        report = self.audit([row], [self.review("s-0")])
        self.assertEqual(STATUS_INVALID, report["status"])
        self.assertIn("RGB_HASH_UNVERIFIED", report["frames"][0]["reason_codes"])
        self.assertIn("MASK_HASH_UNVERIFIED", report["frames"][0]["reason_codes"])

    def test_frame_key_and_non_nearest_resize_are_invalid(self) -> None:
        row = self.row(
            mask_resize_applied=True,
            mask_resize_interpolation="BILINEAR",
            rgb_frame_key="rgb-frame-0",
            mask_frame_key="mask-frame-0",
        )
        report = self.audit([row], [self.review("s-0")])
        codes = set(report["frames"][0]["reason_codes"])
        self.assertEqual(STATUS_INVALID, report["frames"][0]["status"])
        self.assertIn("RGB_MASK_FRAME_KEY_MISMATCH", codes)
        self.assertIn("MASK_NON_NEAREST_RESIZE", codes)
        self.assertIn("MASK_RESIZE_INTERPOLATION_CONTAMINATION", codes)

    def test_proposal_is_sidecar_and_swap_is_review_only(self) -> None:
        proposal = self.mask.copy()
        proposal[self.mask == 1] = 2
        proposal[self.mask == 2] = 1
        proposal_path = self.write("proposal.png", proposal, "L")
        original_bytes = (self.root / "mask-0.png").read_bytes()
        row = self.row(proposal_mask_path=proposal_path.name)
        config = self.config(
            heuristics={
                **DEFAULT_CONFIG["heuristics"],
                "proposal_class_min_pixels": 4,
            }
        )
        report = self.audit([row], [self.review("s-0")], config=config)
        frame = report["frames"][0]
        self.assertEqual(STATUS_REVIEW, frame["status"])
        self.assertIn("OBSTACLE_BOUNDARY_SWAP_SUSPECTED", frame["reason_codes"])
        self.assertEqual("PROPOSAL_ONLY", frame["proposal"]["authority"])
        self.assertFalse(frame["proposal"]["replacement_applied"])
        self.assertEqual(original_bytes, (self.root / "mask-0.png").read_bytes())

    def test_unknown_to_walkable_and_thin_object_are_review_candidates(self) -> None:
        proposal = self.mask.copy()
        proposal[self.mask == 3] = 0
        proposal[7, 0] = 1
        proposal_path = self.write("proposal-unknown-thin.png", proposal, "L")
        row = self.row(proposal_mask_path=proposal_path.name)
        report = self.audit([row], [self.review("s-0")])
        codes = set(report["frames"][0]["reason_codes"])
        self.assertIn("UNKNOWN_AS_WALKABLE_SUSPECTED", codes)
        self.assertIn("THIN_OBJECT_OR_BRANCH_MISSED_SUSPECTED", codes)
        self.assertEqual(STATUS_REVIEW, report["frames"][0]["status"])

    def test_temporal_flicker_marks_both_adjacent_frames(self) -> None:
        changed = np.full((8, 8), 1, dtype=np.uint8)
        self.write("mask-1.png", changed, "L")
        rows = [self.row(), self.row(frame_id=1, mask_path="mask-1.png")]
        report = self.audit(rows, [self.review("s-0"), self.review("s-1")])
        self.assertEqual(STATUS_REVIEW, report["status"])
        for frame in report["frames"]:
            self.assertIn("ADJACENT_LABEL_FLICKER_SUSPECTED", frame["reason_codes"])
        self.assertTrue(report["temporal_pairs"][0]["checked"])

    def test_review_invalid_without_reason_is_invalid(self) -> None:
        report = self.audit([self.row()], [self.review("s-0", status=STATUS_INVALID)])
        self.assertEqual(STATUS_INVALID, report["status"])
        self.assertIn("MANUAL_REVIEW_MISSING_REASON", report["frames"][0]["reason_codes"])

    def test_manifest_and_sidecar_outputs_are_append_only(self) -> None:
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(self.row(), ensure_ascii=False) + "\n", encoding="utf-8")
        review = self.root / "review.jsonl"
        review.write_text(json.dumps(self.review("s-0"), ensure_ascii=False) + "\n", encoding="utf-8")
        report = audit_manifest(manifest, base_root=self.root, config=self.config(), review_path=review)
        output_root = self.root / "artifacts.local" / "mask-qa"
        write_report(report, output_root)
        self.assertEqual(STATUS_PASS, json.loads((output_root / "summary.json").read_text(encoding="utf-8"))["status"])
        self.assertTrue((output_root / "frame_results.jsonl").is_file())
        self.assertTrue((output_root / "review_queue.json").is_file())
        with self.assertRaises(MaskQualityAuditError):
            write_report(report, output_root)


if __name__ == "__main__":
    unittest.main()

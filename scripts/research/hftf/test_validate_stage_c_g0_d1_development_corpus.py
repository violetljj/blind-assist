"""Tests for the independent HFTF G0-D1 corpus validator."""

from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_stage_c_g0_d1_development_corpus as target


def _labels() -> dict[str, object]:
    known = []
    risk = []
    clearance = []
    values = [
        -0.05,
        -0.04,
        -0.03,
        -0.02,
        -0.01,
        0.0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.20,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26,
        0.27,
        0.28,
        0.29,
        0.30,
    ]
    for _height in range(2):
        known.append([[1 for _column in range(6)] for _row in range(6)])
        risk.append(
            [
                [
                    int(values[row * 6 + column] < 0.0)
                    for column in range(6)
                ]
                for row in range(6)
            ]
        )
        clearance.append(
            [
                [values[row * 6 + column] for column in range(6)]
                for row in range(6)
            ]
        )
    return {
        "known_target": known,
        "risk_target_nullable": risk,
        "clearance_target_m_nullable": clearance,
    }


def _corpus(
    image_path: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, str],
]:
    roles = {
        **{f"train-{index}": "train" for index in range(6)},
        **{
            f"selection-{index}": "model_selection"
            for index in range(3)
        },
    }
    students = []
    receipts = []
    for session_id, role in roles.items():
        for frame_index in range(25):
            current_path = image_path.with_name(
                f"current-{session_id}-{frame_index}.png"
            )
            current_path.write_bytes(
                f"{session_id}:{frame_index}".encode("utf-8")
            )
            image_sha = hashlib.sha256(
                current_path.read_bytes()
            ).hexdigest()
            sample_id = f"{role}-{session_id}-{frame_index}"
            labels = _labels()
            student = {
                "sample_id": sample_id,
                "session_id": session_id,
                "role": role,
                "source_frame_index": frame_index,
                "manifest_id": f"manifest-{session_id}-{frame_index}",
                "current_rgb": {
                    "path": str(current_path.resolve()),
                    "sha256": image_sha,
                },
                "labels": labels,
            }
            receipt = {
                "schema": target.RECEIPT_SCHEMA,
                "sample_id": sample_id,
                "session_id": session_id,
                "role": role,
                "source_frame_index": frame_index,
                "manifest_id": student["manifest_id"],
                "teacher_view": {
                    "name": "reference",
                    "point_sample_stride_xy": 4,
                    "point_sample_offset_xy": 2,
                    "timeline": "current_only",
                },
                "teacher_inputs": {"opaque": "not student authorized"},
                "labels_sha256": target._sha256_bytes(
                    target._canonical_bytes(labels)
                ),
                "student_loader_authorized": False,
            }
            students.append(student)
            receipts.append(receipt)
    return students, receipts, roles


class LabelValidationTest(unittest.TestCase):
    def test_accepts_exact_known_risk_clearance_semantics(self) -> None:
        known, risk, clearance = target._validate_labels(_labels())
        self.assertEqual(len(known), 72)
        self.assertEqual(sum(risk), 10)
        self.assertTrue(all(value == value for value in clearance))

    def test_rejects_unknown_to_safe_and_sign_mismatch(self) -> None:
        unknown = _labels()
        unknown["known_target"][0][0][0] = 0
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            target._validate_labels(unknown)

        mismatch = _labels()
        mismatch["risk_target_nullable"][0][0][0] = 0
        with self.assertRaisesRegex(ValueError, "sign"):
            target._validate_labels(mismatch)

    def test_validation_output_is_atomic_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "validation.json"
            target._atomic_write_new(output, b'{"valid":true}\n')
            self.assertEqual(output.read_bytes(), b'{"valid":true}\n')
            self.assertEqual(
                [
                    path.name
                    for path in output.parent.iterdir()
                    if ".partial-" in path.name
                ],
                [],
            )
            with self.assertRaises(FileExistsError):
                target._atomic_write_new(output, b"replacement")


class RecordValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary.name) / "current.png"
        self.image_path.write_bytes(b"current-rgb")
        self.students, self.receipts, self.roles = _corpus(
            self.image_path
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def authority(self) -> dict[tuple[str, int], dict[str, object]]:
        return {
            (
                str(student["session_id"]),
                int(student["source_frame_index"]),
            ): {
                "sample_id": student["sample_id"],
                "role": student["role"],
                "manifest_id": student["manifest_id"],
                "current_rgb": student["current_rgb"],
                "teacher_inputs": receipt["teacher_inputs"],
                "labels": student["labels"],
            }
            for student, receipt in zip(
                self.students, self.receipts, strict=True
            )
        }

    def test_accepts_exact_six_three_disjoint_current_only_corpus(
        self,
    ) -> None:
        result = target._validate_records(
            self.students,
            self.receipts,
            self.roles,
            {"fresh-0"},
            authoritative=self.authority(),
        )
        self.assertEqual(
            result["record_counts"],
            {"train": 150, "model_selection": 75},
        )
        self.assertEqual(len(result["source_height_diagnostics"]), 18)
        self.assertEqual(result["unknown_to_safe_violation_count"], 0)

    def test_rejects_teacher_field_and_fresh_source(self) -> None:
        leaked = copy.deepcopy(self.students)
        leaked[0]["teacher_depth"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "top-level"):
            target._validate_records(
                leaked, self.receipts, self.roles, set()
            )

        fresh_students = copy.deepcopy(self.students)
        fresh_receipts = copy.deepcopy(self.receipts)
        for records in (fresh_students, fresh_receipts):
            records[0]["session_id"] = "fresh-0"
        with self.assertRaisesRegex(ValueError, "source"):
            target._validate_records(
                fresh_students,
                fresh_receipts,
                self.roles,
                {"fresh-0"},
            )

    def test_rejects_role_overlap_missing_frame_and_rgb_hash_drift(
        self,
    ) -> None:
        overlap = copy.deepcopy(self.students)
        overlap_receipts = copy.deepcopy(self.receipts)
        overlap[150]["session_id"] = "train-0"
        overlap_receipts[150]["session_id"] = "train-0"
        with self.assertRaisesRegex(ValueError, "source"):
            target._validate_records(
                overlap, overlap_receipts, self.roles, set()
            )

        bad_frame = copy.deepcopy(self.students)
        bad_receipts = copy.deepcopy(self.receipts)
        bad_frame[0]["source_frame_index"] = 1
        bad_receipts[0]["source_frame_index"] = 1
        with self.assertRaisesRegex(ValueError, "25 frames"):
            target._validate_records(
                bad_frame, bad_receipts, self.roles, set()
            )

        bad_hash = copy.deepcopy(self.students)
        bad_hash[0]["current_rgb"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "RGB byte"):
            target._validate_records(
                bad_hash, self.receipts, self.roles, set()
            )

    def test_rejects_manifest_teacher_and_self_consistent_label_forgery(
        self,
    ) -> None:
        authority = self.authority()
        manifest = copy.deepcopy(self.students)
        manifest_receipts = copy.deepcopy(self.receipts)
        manifest[0]["manifest_id"] = "forged-manifest"
        manifest_receipts[0]["manifest_id"] = "forged-manifest"
        with self.assertRaisesRegex(ValueError, "authoritative"):
            target._validate_records(
                manifest,
                manifest_receipts,
                self.roles,
                set(),
                authoritative=authority,
            )

        teacher = copy.deepcopy(self.receipts)
        teacher[0]["teacher_inputs"] = {"forged": True}
        with self.assertRaisesRegex(ValueError, "authoritative"):
            target._validate_records(
                self.students,
                teacher,
                self.roles,
                set(),
                authoritative=authority,
            )

        labels = copy.deepcopy(self.students)
        forged = _labels()
        forged["clearance_target_m_nullable"][0][0][0] = -0.06
        labels[0]["labels"] = forged
        forged_receipts = copy.deepcopy(self.receipts)
        forged_receipts[0]["labels_sha256"] = target._sha256_bytes(
            target._canonical_bytes(forged)
        )
        with self.assertRaisesRegex(ValueError, "authoritative"):
            target._validate_records(
                labels,
                forged_receipts,
                self.roles,
                set(),
                authoritative=authority,
            )


if __name__ == "__main__":
    unittest.main()

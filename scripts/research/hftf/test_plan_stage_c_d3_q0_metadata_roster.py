from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d2_official_train_metadata import MetadataApi
from plan_stage_c_d3_q0_metadata_roster import (
    ATTEMPT_SCHEMA,
    ATTEMPT_STATUS,
    CANONICAL_RELATIVE_OUTPUT,
    ROSTER_INSUFFICIENT,
    ROSTER_READY,
    CandidateIneligible,
    derive_exclusions,
    open_attempt,
    plan,
    qualify_candidate_exact13,
    repo_root,
    require_canonical_output,
    scan_metadata_roster,
    sha256,
    validate_contract,
    write_json_exclusive_fsync,
)


CONTRACT = (
    repo_root()
    / "docs/research/hftf/"
    "HFTF_STAGE_C_D3_Q0_METADATA_ROSTER_EXECUTION_CONTRACT_"
    "2026-08-02.json"
)
D2_QUALIFICATION = (
    repo_root()
    / "artifacts.local/evidence/hftf/"
    "stage-c-d2-official-train-metadata-qualification-20260802/"
    "qualification.json"
)


def session_id(index: int) -> str:
    return f"{index:064x}"


def object_receipt(name: str, *, generation: str = "11") -> dict[str, object]:
    return {
        "name": name,
        "generation": generation,
        "metageneration": "1",
        "size": "7",
        "md5Hash": "YWJj",
        "crc32c": "ZGVm",
    }


def description(fps: float = 20.0) -> dict[str, object]:
    return {
        "session_type": "synthetic",
        "session_camera_location": ["camera_chest"],
        "session_camera_details": [
            {
                "fps": fps,
                "left_camera_params": {
                    "fx": 10.0,
                    "fy": 11.0,
                    "cx": 5.0,
                    "cy": 6.0,
                    "image_width": 20,
                    "image_height": 12,
                },
            }
        ],
    }


def candidate(sid: str, rank: int) -> dict[str, object]:
    return {
        "session_id": sid,
        "official_split": "train",
        "role": "old",
        "metadata_eligible": True,
        "metadata_eligible_rank": rank,
        "source_fps": 20.0,
        "normalized_target_fps": 5.0,
        "selected_source_frames": list(range(0, 49, 4)),
        "description_object": {},
        "camera_pose_object_receipt": {},
        "camera_pose_content_read": False,
        "camera": {},
        "media_object_listing_receipts": {},
        "rgb_mask_depth_bytes_read": False,
        "geometry_teacher_outcome_read": False,
        "student_outcome_read": False,
    }


class FakeApi:
    def __init__(
        self,
        split_ids: list[str],
        *,
        split_generation: str,
        fail_ids: set[str] | None = None,
        selected_only: bool = False,
    ) -> None:
        self.split_ids = split_ids
        self.split_generation = split_generation
        self.fail_ids = fail_ids or set()
        self.selected_only = selected_only
        self.content_reads: list[str] = []

    def get_object(self, name: str, retries: int) -> dict[str, object]:
        generation = (
            self.split_generation
            if name.endswith("train_session_ids.txt")
            else "11"
        )
        if any(sid in name for sid in self.fail_ids):
            raise OSError("404 after internal retries")
        return object_receipt(name, generation=generation)

    def fetch_text(self, url: str, retries: int) -> str:
        self.content_reads.append(url)
        return "\n".join(self.split_ids) + "\n"

    def fetch_json(
        self,
        url: str,
        retries: int,
    ) -> dict[str, object]:
        self.content_reads.append(url)
        if not url.endswith("description.json?generation=11"):
            raise AssertionError(f"Unexpected JSON content read: {url}")
        return description()

    def list_objects(
        self,
        prefix: str,
        retries: int,
    ) -> list[dict[str, object]]:
        if prefix.endswith("/video_frames/"):
            suffix = ".png"
        elif prefix.endswith("/segmentation_masks/"):
            suffix = ".png"
        elif prefix.endswith("/depth_maps/"):
            suffix = ".float16.gz"
        else:
            raise AssertionError(f"Unexpected listing: {prefix}")
        indices = (
            list(range(0, 49, 4))
            if self.selected_only
            else list(range(50))
        )
        return [
            object_receipt(f"{prefix}{index:06d}{suffix}")
            for index in indices
        ]

    def value(self) -> MetadataApi:
        return MetadataApi(
            get_object=self.get_object,
            fetch_text=self.fetch_text,
            fetch_json=self.fetch_json,
            list_objects=self.list_objects,
        )


class D3Q0MetadataRosterTest(unittest.TestCase):
    def test_real_d2_exclusion_union_is_exact_84(self) -> None:
        categories, excluded = derive_exclusions(
            json.loads(D2_QUALIFICATION.read_text(encoding="utf-8"))
        )
        self.assertEqual(84, len(excluded))
        self.assertEqual(6, len(categories["d2_consumed_six_parent_cohort"]))
        self.assertEqual(sum(map(len, categories.values())), len(excluded))

    def test_overlapping_exclusion_categories_fail_closed(self) -> None:
        value = json.loads(D2_QUALIFICATION.read_text(encoding="utf-8"))
        duplicate = value["qualified_parents"][0]["session_id"]
        value["exclusions"]["category_session_ids"][
            "historical_burned_or_consumed"
        ].append(duplicate)
        with self.assertRaisesRegex(ValueError, "overlap"):
            derive_exclusions(value)

    def test_scan_is_ascending_and_stops_at_requested_count(self) -> None:
        ids = [session_id(index) for index in range(50)]
        locked, ledger = scan_metadata_roster(
            ids,
            {ids[0], ids[2]},
            lambda sid, rank: candidate(sid, rank),
            roster_slot_count=4,
        )
        self.assertEqual([ids[1], ids[3], ids[4], ids[5]], [
            row["session_id"] for row in locked
        ])
        self.assertEqual(6, len(ledger))

    def test_metadata_ineligible_candidate_does_not_consume_slot(self) -> None:
        ids = [session_id(index) for index in range(5)]

        def qualify(sid: str, rank: int) -> dict[str, object]:
            if sid == ids[1]:
                raise CandidateIneligible("metadata invalid")
            return candidate(sid, rank)

        locked, ledger = scan_metadata_roster(
            ids,
            set(),
            qualify,
            roster_slot_count=3,
        )
        self.assertEqual([ids[0], ids[2], ids[3]], [
            row["session_id"] for row in locked
        ])
        self.assertEqual(3, [row["d3_roster_slot_index"] for row in locked][-1])
        self.assertEqual("metadata invalid", ledger[1]["reason"])

    def test_locked_slot_firewall_is_explicit(self) -> None:
        sid = session_id(1)
        locked, _ = scan_metadata_roster(
            [sid],
            set(),
            lambda value, rank: candidate(value, rank),
            roster_slot_count=1,
        )
        row = locked[0]
        self.assertEqual("d3_q0_locked_truth_screening_slot", row["role"])
        self.assertFalse(row["media_pose_content_opened"])
        self.assertFalse(row["support_or_truth_computed"])
        self.assertFalse(row["effect_computed"])

    def test_only_selected_13_frames_is_metadata_eligible(self) -> None:
        sid = session_id(99)
        api = FakeApi(
            [sid],
            split_generation="unused",
            selected_only=True,
        )
        value = qualify_candidate_exact13(
            sid,
            1,
            3,
            api.value(),
        )
        self.assertEqual(list(range(0, 49, 4)), value["selected_source_frames"])
        for modality in ("rgb", "mask", "depth"):
            receipt = value["media_object_listing_receipts"][modality]
            self.assertEqual(13, receipt["required_frame_count"])
            self.assertEqual(
                list(range(0, 49, 4)),
                receipt["required_frame_indices"],
            )

    def test_scan_can_end_insufficient_without_replacement(self) -> None:
        ids = [session_id(index) for index in range(2)]
        locked, ledger = scan_metadata_roster(
            ids,
            set(),
            lambda sid, rank: candidate(sid, rank),
            roster_slot_count=3,
        )
        self.assertEqual(2, len(locked))
        self.assertEqual(2, len(ledger))

    def test_exclusive_writer_refuses_overwrite_and_fsyncs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.json"
            with mock.patch(
                "plan_stage_c_d3_q0_metadata_roster.os.fsync"
            ) as fsync:
                write_json_exclusive_fsync(path, {"value": 1})
            fsync.assert_called_once()
            with self.assertRaises(FileExistsError):
                write_json_exclusive_fsync(path, {"value": 2})

    def test_attempt_is_durable_and_closes_all_downstream_authority(self) -> None:
        context = {
            "contract_path": CONTRACT,
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evidence" / "roster.json"
            with mock.patch(
                "plan_stage_c_d3_q0_metadata_roster.os.fsync"
            ) as fsync:
                attempt = open_attempt(output, context, 3)
            value = json.loads(attempt.read_text(encoding="utf-8"))
        self.assertEqual(ATTEMPT_SCHEMA, value["schema"])
        self.assertEqual(ATTEMPT_STATUS, value["status"])
        self.assertFalse(value["first_network_request_started"])
        self.assertFalse(value["support_or_truth_qualification_authorized"])
        self.assertFalse(value["d3_effect_authorized"])
        fsync.assert_called_once()

    def test_noncanonical_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            require_canonical_output(Path("wrong.json"))
        expected = repo_root() / CANONICAL_RELATIVE_OUTPUT
        self.assertEqual(expected.resolve(), require_canonical_output(expected))

    def test_real_contract_validates_without_git_execution_gate(self) -> None:
        context = validate_contract(CONTRACT, verify_git=False)
        self.assertEqual(40, context["roster_slot_count"])
        self.assertEqual(84, len(context["excluded"]))
        self.assertEqual(3, context["retries"])

    def test_fake_plan_locks_exact_40_and_reads_no_media_or_pose_content(
        self,
    ) -> None:
        context = validate_contract(CONTRACT, verify_git=False)
        ids = sorted(
            set(context["excluded"])
            | {session_id(1000 + index) for index in range(40)}
        )
        api = FakeApi(
            ids,
            split_generation=context["split_generation"],
        )
        split_text = "\n".join(ids) + "\n"
        context = dict(context)
        context["split_sha256"] = hashlib.sha256(
            split_text.encode("utf-8")
        ).hexdigest()
        result = plan(
            CONTRACT,
            3,
            api=api.value(),
            context=context,
        )
        self.assertEqual(ROSTER_READY, result["terminal"])
        self.assertEqual(40, result["locked_roster_slot_count"])
        self.assertFalse(any(result["firewall"].values()))
        self.assertTrue(
            result["authorization"][
                "freeze_qualifier_effect_execution_contract"
            ]
        )
        self.assertTrue(all(
            "description.json" in url or "train_session_ids.txt" in url
            for url in api.content_reads
        ))

    def test_fake_plan_skips_metadata_failure_before_slot_lock(self) -> None:
        context = validate_contract(CONTRACT, verify_git=False)
        fresh = [session_id(2000 + index) for index in range(41)]
        ids = sorted(set(context["excluded"]) | set(fresh))
        failed = {fresh[0]}
        api = FakeApi(
            ids,
            split_generation=context["split_generation"],
            fail_ids=failed,
        )
        split_text = "\n".join(ids) + "\n"
        context = dict(context)
        context["split_sha256"] = hashlib.sha256(
            split_text.encode("utf-8")
        ).hexdigest()
        result = plan(
            CONTRACT,
            3,
            api=api.value(),
            context=context,
        )
        self.assertEqual(ROSTER_READY, result["terminal"])
        self.assertNotIn(
            fresh[0],
            [row["session_id"] for row in result["locked_slots"]],
        )
        self.assertEqual(40, result["locked_roster_slot_count"])

    def test_fake_plan_reports_insufficient_without_expanding_input(self) -> None:
        context = validate_contract(CONTRACT, verify_git=False)
        fresh = [session_id(3000 + index) for index in range(3)]
        ids = sorted(set(context["excluded"]) | set(fresh))
        api = FakeApi(
            ids,
            split_generation=context["split_generation"],
        )
        split_text = "\n".join(ids) + "\n"
        context = dict(context)
        context["split_sha256"] = hashlib.sha256(
            split_text.encode("utf-8")
        ).hexdigest()
        result = plan(
            CONTRACT,
            3,
            api=api.value(),
            context=context,
        )
        self.assertEqual(ROSTER_INSUFFICIENT, result["terminal"])
        self.assertEqual(3, result["locked_roster_slot_count"])
        self.assertFalse(
            result["authorization"][
                "freeze_qualifier_effect_execution_contract"
            ]
        )

    def test_wrong_retry_count_is_rejected_before_network(self) -> None:
        context = validate_contract(CONTRACT, verify_git=False)
        api = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Retries differ"):
            plan(CONTRACT, 2, api=api, context=context)
        api.get_object.assert_not_called()


if __name__ == "__main__":
    unittest.main()

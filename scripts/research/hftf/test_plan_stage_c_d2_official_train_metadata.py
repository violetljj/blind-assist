from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d2_official_train_metadata import (
    CANONICAL_RELATIVE_OUTPUT,
    CandidateIneligible,
    MetadataApi,
    _modality_receipt,
    _open_attempt,
    _parse_split_ids,
    _qualify_candidate,
    _require_output,
    _require_tracked_clean_path,
    _scan_candidates,
    _sha256,
    _timeline,
    _validate_frozen_inputs,
    _validate_intrinsics,
    _validate_parent_chain,
    _write_json_exclusive,
    _execution_failure_report,
    plan,
)


def _session_id(index: int) -> str:
    return f"{index:064x}"


def _object(name: str, size: int = 7) -> dict[str, object]:
    return {
        "name": name,
        "generation": "11",
        "metageneration": "1",
        "size": str(size),
        "md5Hash": "YWJj",
        "crc32c": "ZGVm",
    }


def _description(fps: float) -> dict[str, object]:
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


def _paths(repo: Path) -> tuple[Path, Path, Path]:
    design = (
        repo
        / "docs/research/hftf/"
        "HFTF_STAGE_C_CAUSAL_SIGNED_CLEARANCE_TRANSPORT_D2_"
        "2026-08-01.json"
    )
    t0_result = (
        repo
        / "docs/research/hftf/"
        "HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_"
        "TRANSPORT_RESULT_2026-08-01.json"
    )
    planner = (
        repo
        / "scripts/research/hftf/"
        "plan_stage_c_d2_official_train_metadata.py"
    )
    return design, t0_result, planner


def _write_execution_contract(
    root: Path,
    *,
    planner_sha256: str | None = None,
    scan_authorized: bool = True,
) -> Path:
    repo = Path(__file__).resolve().parents[3]
    design, t0_result, planner = _paths(repo)
    frozen = _validate_parent_chain(
        design,
        t0_result,
        _sha256(planner),
    )
    bindings = frozen["bindings"]

    def receipt(
        key: str,
        required_terminal: str | None = None,
    ) -> dict[str, object]:
        value: dict[str, object] = {
            "path": bindings[key]["path"],
            "sha256": bindings[key]["sha256"],
        }
        if required_terminal is not None:
            value["required_terminal"] = required_terminal
        return value

    contract = {
        "schema": (
            "blindassist_hftf_stage_c_d2_official_train_metadata_"
            "qualification_execution_contract"
        ),
        "status": "FROZEN_AFTER_T0_BEFORE_D2_METADATA_SCAN_OR_NEW_MEDIA",
        "workflow_profile": "THESIS_DEVELOPMENT",
        "parents": {
            "d2_design": receipt("d2_design"),
            "t0_result": receipt(
                "t0_result",
                "T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT",
            ),
            "g0_source_plan": receipt(
                "g0_source_plan",
                "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY",
            ),
            "f0_burn_ledger": receipt("f0_burn_ledger"),
            "r4_burn_ledger": receipt("r4_burn_ledger"),
            "r3_1_burn_ledger": receipt("r3_1_burn_ledger"),
            "f0_1_consumed_official_test_result": receipt(
                "f0_1_consumed_official_test_result",
                (
                    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_"
                    "STUDENT_SIGNAL_NOT_SUPPORTED_STOP"
                ),
            ),
        },
        "official_train_split": {
            "object_generation": frozen["split_generation"],
            "text_sha256": frozen["split_sha256"],
        },
        "source_selection": {
            "official_split": "train",
            "order": "ascending_session_id",
            "parent_count": 6,
            "source_replacement_after_scan": False,
            "insufficient_terminal": (
                "STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT"
            ),
        },
        "implementations": {
            "metadata_planner": {
                "path": (
                    "scripts/research/hftf/"
                    "plan_stage_c_d2_official_train_metadata.py"
                ),
                "sha256": planner_sha256 or _sha256(planner),
                "metadata_network_execution_authorized": True,
            }
        },
        "implementation_tests": {
            "metadata_planner_test": {
                "path": (
                    "scripts/research/hftf/"
                    "test_plan_stage_c_d2_official_train_metadata.py"
                ),
                "sha256": _sha256(
                    repo
                    / "scripts/research/hftf/"
                    "test_plan_stage_c_d2_official_train_metadata.py"
                ),
            },
            "test_count": 14,
            "tests_passed": 14,
        },
        "canonical_artifacts": {
            "metadata_qualification": str(
                CANONICAL_RELATIVE_OUTPUT
            ).replace("\\", "/")
        },
        "authorization": {
            "metadata_scan_execution_authorized": scan_authorized,
            "new_media_acquisition_authorized": False,
            "camera_pose_content_read_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "student_execution_authorized": False,
            "d2_mechanics_execution_authorized": False,
            "reserved_official_test_open_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "android_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
        "failure_policy": {
            "internal_retries_per_metadata_request": 3,
            "do_not_rerun_same_scan": True,
            "do_not_replace_or_append_sources_after_scan": True,
            "do_not_open_media_or_pose_content": True,
            "execution_failure_terminal": (
                "D2_METADATA_QUALIFICATION_NOT_EVALUABLE_NO_RETRY"
            ),
        },
    }
    path = root / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


class FakeMetadataApi:
    def __init__(self, session_id: str, fps: float = 20.0) -> None:
        self.session_id = session_id
        self.description = _description(fps)
        self.fetch_json_urls: list[str] = []
        self.fetch_text_urls: list[str] = []
        self.list_prefixes: list[str] = []

    def get_object(self, name: str, retries: int) -> dict[str, object]:
        return _object(name)

    def fetch_text(self, url: str, retries: int) -> str:
        self.fetch_text_urls.append(url)
        raise AssertionError("Candidate qualification must not fetch text")

    def fetch_json(
        self,
        url: str,
        retries: int,
    ) -> dict[str, object]:
        self.fetch_json_urls.append(url)
        if not url.endswith("description.json?generation=11"):
            raise AssertionError(f"Unexpected content read: {url}")
        return self.description

    def list_objects(
        self,
        prefix: str,
        retries: int,
    ) -> list[dict[str, object]]:
        self.list_prefixes.append(prefix)
        if prefix.endswith("/video_frames/"):
            suffix = ".png"
        elif prefix.endswith("/segmentation_masks/"):
            suffix = ".png"
        elif prefix.endswith("/depth_maps/"):
            suffix = ".float16.gz"
        else:
            raise AssertionError(f"Unexpected listing: {prefix}")
        return [
            _object(f"{prefix}{index:06d}{suffix}")
            for index in range(50)
        ]

    def value(self) -> MetadataApi:
        return MetadataApi(
            get_object=self.get_object,
            fetch_text=self.fetch_text,
            fetch_json=self.fetch_json,
            list_objects=self.list_objects,
        )


class FailOneMetadataApi(FakeMetadataApi):
    def __init__(self, failed_id: str) -> None:
        super().__init__(failed_id)
        self.failed_id = failed_id

    def get_object(self, name: str, retries: int) -> dict[str, object]:
        if self.failed_id in name:
            raise OSError("404 after internal retries")
        return super().get_object(name, retries)


class StageCD2OfficialTrainMetadataTest(unittest.TestCase):
    def test_real_frozen_binding_chain_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract = _write_execution_contract(Path(temp))
            frozen = _validate_frozen_inputs(
                contract,
                verify_tracked_contract=False,
            )
        self.assertEqual(78, len(frozen["excluded"]))
        self.assertEqual(
            {
                "historical_burned_or_consumed": 60,
                "g0_d1_development": 9,
                "closed_g0_d1_fresh_cohort": 3,
                "consumed_f0_1_official_test": 3,
                "g0_reserved_official_test": 3,
            },
            {
                key: len(value)
                for key, value in frozen[
                    "exclusion_categories"
                ].items()
            },
        )
        self.assertEqual(
            _sha256(_paths(Path(__file__).resolve().parents[3])[2]),
            frozen["bindings"]["metadata_planner"]["sha256"],
        )

    def test_candidate_reads_only_description_and_object_metadata(
        self,
    ) -> None:
        session_id = _session_id(101)
        fake = FakeMetadataApi(session_id)
        candidate = _qualify_candidate(
            session_id,
            1,
            3,
            fake.value(),
        )
        self.assertEqual(
            list(range(0, 49, 4)),
            candidate["selected_source_frames"],
        )
        self.assertEqual(1, len(fake.fetch_json_urls))
        self.assertEqual([], fake.fetch_text_urls)
        self.assertEqual(3, len(fake.list_prefixes))
        self.assertFalse(candidate["camera_pose_content_read"])
        self.assertFalse(candidate["rgb_mask_depth_bytes_read"])
        for value in candidate[
            "media_object_listing_receipts"
        ].values():
            self.assertEqual(50, value["required_frame_count"])
            self.assertEqual(
                50,
                len(value["required_frame_receipts"]),
            )

    def test_timeline_is_exact_for_5_and_20_fps(self) -> None:
        self.assertEqual(list(range(13)), _timeline(5.0))
        self.assertEqual(list(range(0, 49, 4)), _timeline(20.0))
        with self.assertRaisesRegex(
            CandidateIneligible,
            "exactly 5 or 20",
        ):
            _timeline(10.0)

    def test_receipts_intrinsics_and_frames_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            CandidateIneligible,
            "finite and positive",
        ):
            _validate_intrinsics(
                {
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 1.0,
                    "image_width": 2,
                    "image_height": 2,
                }
            )
        objects = [
            _object(f"x/{index:06d}.png")
            for index in range(49)
        ]
        with self.assertRaisesRegex(
            CandidateIneligible,
            "0..49",
        ):
            _modality_receipt(objects, ".png", "rgb")
        missing_md5 = [
            _object(f"x/{index:06d}.png")
            for index in range(50)
        ]
        missing_md5[7]["md5Hash"] = None
        with self.assertRaisesRegex(
            CandidateIneligible,
            "generation, positive size, and MD5",
        ):
            _modality_receipt(missing_md5, ".png", "rgb")

    def test_split_is_sorted_and_duplicates_fail(self) -> None:
        values = [_session_id(3), _session_id(1), _session_id(2)]
        self.assertEqual(
            sorted(values),
            _parse_split_ids("\n".join(values) + "\n"),
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            _parse_split_ids(
                f"{_session_id(1)}\n{_session_id(1)}\n"
            )

    def test_scan_excludes_then_selects_exactly_six(self) -> None:
        ordered = [_session_id(index) for index in range(10)]
        excluded = {ordered[0], ordered[1]}
        rejected = ordered[3]

        def qualify(
            session_id: str,
            rank: int,
        ) -> dict[str, object]:
            if session_id == rejected:
                raise CandidateIneligible("not eligible")
            return {
                "session_id": session_id,
                "metadata_eligible": True,
                "metadata_eligible_rank": rank,
            }

        selected, ledger = _scan_candidates(
            ordered,
            excluded,
            6,
            qualify,
        )
        self.assertEqual(6, len(selected))
        self.assertEqual(
            [
                ordered[2],
                ordered[4],
                ordered[5],
                ordered[6],
                ordered[7],
                ordered[8],
            ],
            [item["session_id"] for item in selected],
        )
        self.assertEqual(9, len(ledger))

    def test_insufficient_scan_is_exact_and_does_not_replace(self) -> None:
        ordered = [_session_id(index) for index in range(5)]
        selected, _ = _scan_candidates(
            ordered,
            set(),
            6,
            lambda session_id, rank: {
                "session_id": session_id,
                "metadata_eligible": True,
                "metadata_eligible_rank": rank,
            },
        )
        self.assertEqual(5, len(selected))

    def test_candidate_404_is_recorded_and_scan_continues(self) -> None:
        failed = _session_id(40)
        passing = _session_id(41)
        fake = FailOneMetadataApi(failed)
        selected, ledger = _scan_candidates(
            [failed, passing],
            set(),
            1,
            lambda session_id, rank: _qualify_candidate(
                session_id,
                rank,
                3,
                fake.value(),
            ),
        )
        self.assertEqual([passing], [item["session_id"] for item in selected])
        self.assertEqual(2, len(ledger))
        self.assertIn(
            "after 3 retries",
            str(ledger[0]["reason"]),
        )

    def test_output_is_canonical_under_artifacts(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        expected = (repo / CANONICAL_RELATIVE_OUTPUT).resolve()
        self.assertEqual(expected, _require_output(expected))
        with self.assertRaisesRegex(ValueError, "not canonical"):
            _require_output(repo / "qualification.json")

    def test_implementation_hash_drift_fails_closed(self) -> None:
        digest = hashlib.sha256(b"wrong").hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            contract = _write_execution_contract(
                Path(temp),
                planner_sha256=digest,
            )
            with self.assertRaisesRegex(
                ValueError,
                "implementation hash mismatch",
            ):
                _validate_frozen_inputs(
                    contract,
                    verify_tracked_contract=False,
                )

    def test_execution_contract_must_authorize_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract = _write_execution_contract(
                Path(temp),
                scan_authorized=False,
            )
            with self.assertRaisesRegex(
                ValueError,
                "authorization firewall",
            ):
                _validate_frozen_inputs(
                    contract,
                    verify_tracked_contract=False,
                )

    def test_execution_implementation_must_be_tracked_and_clean(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp).resolve()

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", *args],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            git("init")
            git("config", "user.email", "hftf-test@example.invalid")
            git("config", "user.name", "HFTF Test")
            planner = repo / "planner.py"
            planner.write_text("print('frozen')\n", encoding="utf-8")
            git("add", "planner.py")
            git("commit", "-m", "freeze planner")
            _require_tracked_clean_path(
                planner,
                repo,
                "metadata planner implementation",
            )
            planner.write_text("print('drifted')\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "not committed and clean",
            ):
                _require_tracked_clean_path(
                    planner,
                    repo,
                    "metadata planner implementation",
                )

    def test_retry_count_must_equal_frozen_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract = _write_execution_contract(Path(temp))
            frozen = _validate_frozen_inputs(
                contract,
                verify_tracked_contract=False,
            )
            with self.assertRaisesRegex(
                ValueError,
                "frozen contract value 3",
            ):
                plan(contract, 2, frozen=frozen)

    def test_attempt_marker_makes_failure_non_rerunnable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "attempt-root" / "qualification.json"
            frozen = {
                "bindings": {
                    "execution_contract": {
                        "path": "contract.json",
                        "sha256": "a" * 64,
                    }
                }
            }
            attempt = _open_attempt(output, frozen, 3)
            self.assertTrue(attempt.is_file())
            failure = _execution_failure_report(
                frozen,
                OSError("split unavailable"),
            )
            _write_json_exclusive(output, failure)
            self.assertEqual(
                "D2_METADATA_QUALIFICATION_NOT_EVALUABLE_NO_RETRY",
                json.loads(
                    output.read_text(encoding="utf-8")
                )["terminal"],
            )
            with self.assertRaises(FileExistsError):
                _open_attempt(output, frozen, 3)


if __name__ == "__main__":
    unittest.main()

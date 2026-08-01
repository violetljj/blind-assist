from __future__ import annotations

import base64
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acquire_stage_c_d2_six_source_media as subject


def _md5(payload: bytes) -> str:
    return base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(name: str, payload: bytes = b"payload") -> dict[str, object]:
    return {
        "name": name,
        "generation": "7",
        "metageneration": "1",
        "size": len(payload),
        "md5_base64": _md5(payload),
        "crc32c_base64": "fixture",
    }


def _source(index: int, fps: float) -> dict[str, object]:
    session_id = f"{index + 100:064x}"
    prefix = f"{subject.GCS_PREFIX}/sanpo-synthetic/{session_id}"
    selected = subject._expected_frames(fps)
    modalities = {}
    for key, (folder, suffix) in {
        "rgb": ("video_frames", ".png"),
        "mask": ("segmentation_masks", ".png"),
        "depth": ("depth_maps", ".float16.gz"),
    }.items():
        rows = []
        for frame in range(50):
            item = _receipt(
                f"{prefix}/camera_chest/left/{folder}/{frame:06d}{suffix}"
            )
            rows.append({"frame_index": frame, **item})
        canonical = [
            {
                "frame_index": row["frame_index"],
                "name": row["name"],
                "generation": row["generation"],
                "size": row["size"],
                "md5_base64": row["md5_base64"],
            }
            for row in rows
        ]
        modalities[key] = {
            "required_frame_count": 50,
            "required_frame_indices": [0, 49],
            "required_frame_receipts_sha256": (
                subject._canonical_json_sha256(canonical)
            ),
            "required_frame_receipts": rows,
        }
    return {
        "session_id": session_id,
        "official_split": "train",
        "role": "one_shot_thesis_development_mechanics_evaluation",
        "metadata_eligible": True,
        "metadata_eligible_rank": index + 1,
        "source_fps": fps,
        "normalized_target_fps": 5.0,
        "selected_source_frames": selected,
        "description_object": _receipt(f"{prefix}/description.json"),
        "camera_pose_object_receipt": _receipt(
            f"{prefix}/camera_chest/camera_poses.csv"
        ),
        "camera_pose_content_read": False,
        "camera": {
            "fx": 10.0,
            "fy": 11.0,
            "cx": 5.0,
            "cy": 6.0,
            "image_width": 20,
            "image_height": 12,
        },
        "media_object_listing_receipts": modalities,
        "rgb_mask_depth_bytes_read": False,
        "geometry_teacher_outcome_read": False,
        "student_outcome_read": False,
    }


def _metadata() -> dict[str, object]:
    sources = [_source(index, 5.0 if index % 2 else 20.0) for index in range(6)]
    return {
        "schema": subject.METADATA_SCHEMA,
        "terminal": subject.METADATA_READY,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "official_train_split": {
            "object_receipt": _receipt(
                f"{subject.GCS_PREFIX}/sanpo-synthetic/splits/train_session_ids.txt"
            ),
            "text_sha256": "a" * 64,
            "selection_order": "ascending_session_id",
            "input_split_order_used_for_selection": False,
        },
        "exclusions": {"excluded_parent_count": 78},
        "requested_parent_count": 6,
        "qualified_parent_count": 6,
        "qualified_parents": sources,
        "firewall": {
            "rgb_bytes_read": False,
            "panoptic_mask_bytes_read": False,
            "metric_depth_bytes_read": False,
            "camera_pose_content_read": False,
            "geometry_teacher_outcome_read": False,
            "student_outcome_read": False,
            "reserved_official_test_opened": False,
        },
    }


def _write_contract(root: Path) -> tuple[Path, Path, dict[str, object]]:
    metadata = _metadata()
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    acquirer = Path(subject.__file__).resolve()
    test_path = Path(__file__).resolve()
    sources = metadata["qualified_parents"]
    parent_values = {
        "d2_design": {
            "schema": subject.D2_DESIGN_SCHEMA,
            "status": subject.D2_DESIGN_STATUS,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "authorization": {"open_any_new_d2_media_now": False},
        },
        "d2_1_clarification": {
            "schema": subject.D2_1_SCHEMA,
            "status": subject.D2_1_STATUS,
            "outcome_firewall_at_freeze": {"rgb_bytes_open": False},
            "authorization": {
                "execute_d2_media_acquisition_now": False
            },
        },
        "metadata_qualification_result": {
            "schema": subject.TRACKED_METADATA_RESULT_SCHEMA,
            "terminal": subject.METADATA_READY,
            "durable_evidence": {
                "qualification": {
                    "path": str(metadata_path),
                    "sha256": _sha(metadata_path),
                    "required_terminal": subject.METADATA_READY,
                }
            },
            "authorization": {
                "freeze_d2_media_and_mechanics_implementation_contract": True,
                "execute_d2_media_acquisition_now": False,
            },
        },
        "t0_result": {
            "schema": subject.T0_RESULT_SCHEMA,
            "terminal": subject.T0_RESULT_TERMINAL,
        },
    }
    parent_paths = {}
    for key, value in parent_values.items():
        path = root / f"{key}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        parent_paths[key] = path
    contract = {
        "schema": subject.CONTRACT_SCHEMA,
        "status": subject.CONTRACT_STATUS,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "parents": {
            "d2_design": {
                "path": str(parent_paths["d2_design"]),
                "sha256": _sha(parent_paths["d2_design"]),
                "required_status": subject.D2_DESIGN_STATUS,
            },
            "d2_1_clarification": {
                "path": str(parent_paths["d2_1_clarification"]),
                "sha256": _sha(parent_paths["d2_1_clarification"]),
                "required_status": subject.D2_1_STATUS,
            },
            "metadata_qualification_result": {
                "path": str(parent_paths["metadata_qualification_result"]),
                "sha256": _sha(
                    parent_paths["metadata_qualification_result"]
                ),
                "required_terminal": subject.METADATA_READY,
            },
            "t0_result": {
                "path": str(parent_paths["t0_result"]),
                "sha256": _sha(parent_paths["t0_result"]),
                "required_terminal": subject.T0_RESULT_TERMINAL,
            },
            "metadata_qualification": {
                "path": str(metadata_path),
                "sha256": _sha(metadata_path),
                "required_terminal": subject.METADATA_READY,
            }
        },
        "source_cohort": {
            "parent_count": 6,
            "order": "metadata_eligible_rank",
            "sources": [
                {
                    "session_id": source["session_id"],
                    "metadata_eligible_rank": source["metadata_eligible_rank"],
                    "source_fps": float(source["source_fps"]),
                    "selected_source_frames": source["selected_source_frames"],
                    "metadata_qualified_parent_sha256": (
                        subject._canonical_json_sha256(source)
                    ),
                }
                for source in sources
            ],
        },
        "implementations": {
            "media_acquirer": {
                "path": subject.ACQUIRER_RELATIVE_PATH,
                "sha256": _sha(acquirer),
                "network_execution_authorized": True,
            },
            "sanpo_network_transport_dependency": {
                "path": subject.TRANSPORT_DEPENDENCY_RELATIVE_PATH,
                "sha256": _sha(
                    subject._repo_root()
                    / subject.TRANSPORT_DEPENDENCY_RELATIVE_PATH
                ),
                "network_transport_authorized": True,
            }
        },
        "implementation_tests": {
            "media_acquirer_test": {
                "path": subject.TEST_RELATIVE_PATH,
                "sha256": _sha(test_path),
            },
            "test_count": subject._test_definition_count(test_path),
            "tests_passed": subject._test_definition_count(test_path),
        },
        "canonical_artifacts": {
            "media_acquisition_root": str(
                (
                    subject._repo_root() / subject.CANONICAL_RELATIVE_ROOT
                ).resolve()
            )
        },
        "failure_policy": {
            "internal_retries_per_request": 3,
            "durable_attempt_file_flush_and_fsync_before_network": True,
            "do_not_rerun_same_acquisition": True,
            "do_not_replace_or_append_sources": True,
            "preserve_global_staging_on_failure": True,
            "failure_terminal": subject.NOT_EVALUABLE,
        },
        "authorization": {
            "six_source_media_acquisition_authorized": True,
            "pose_slice_materialization_authorized": True,
            "freeze_future_blind_preprocessor_contract_on_success": True,
            "future_blind_preprocessor_execution_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "effect_evaluation_authorized": False,
            "student_execution_authorized": False,
            "reserved_official_test_open_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "android_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, metadata_path, contract


class D2SixSourceMediaAcquirerTest(unittest.TestCase):
    def test_metadata_result_and_contract_bind_exact_six_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract_path, _, _ = _write_contract(Path(temp))
            frozen = subject.validate_execution_contract(
                contract_path, verify_git=False
            )
        self.assertEqual(6, len(frozen["sources"]))
        self.assertEqual(
            list(range(1, 7)),
            [source["metadata_eligible_rank"] for source in frozen["sources"]],
        )

    def test_contract_source_binding_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract_path, _, contract = _write_contract(Path(temp))
            contract["source_cohort"]["sources"][0]["source_fps"] = 5.0
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                subject.AcquisitionError, "exact source bindings"
            ):
                subject.validate_execution_contract(
                    contract_path, verify_git=False
                )

    def test_network_transport_dependency_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract_path, _, contract = _write_contract(Path(temp))
            contract["implementations"][
                "sanpo_network_transport_dependency"
            ]["sha256"] = "0" * 64
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                subject.AcquisitionError, "implementation receipt"
            ):
                subject.validate_execution_contract(
                    contract_path, verify_git=False
                )

    def test_source_blind_layout_covers_final_staging_and_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract_path, _, _ = _write_contract(Path(temp))
            frozen = subject.validate_execution_contract(
                contract_path, verify_git=False
            )
            layout = subject.plan_layout(
                Path(temp) / "short", frozen["sources"]
            )
            report = subject.path_preflight(layout)
        self.assertTrue(report["all_content_paths_under_limit"])
        self.assertFalse(report["session_id_present_in_any_content_path"])
        self.assertEqual(6, report["source_count"])
        self.assertGreater(report["content_path_count_including_tmp"], 500)

    def test_path_preflight_rejects_long_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sources = _metadata()["qualified_parents"]
            layout = subject.plan_layout(
                Path(temp) / ("x" * 170), sources
            )
            with self.assertRaisesRegex(
                subject.AcquisitionError, "path budget exceeded"
            ):
                subject.path_preflight(layout)

    def test_frozen_receipt_generation_size_and_md5_are_required(self) -> None:
        source = _source(0, 20.0)
        source["description_object"]["md5_base64"] = None
        with self.assertRaisesRegex(
            subject.AcquisitionError, "positive size, and MD5"
        ):
            subject._validate_metadata_source(source)

    def test_download_is_generation_bound_and_md5_verified(self) -> None:
        payload = b"verified"
        receipt = _receipt("bucket/object", payload)
        observed = {}
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "object.bin"

            def fake(url: str, path: Path, retries: int) -> None:
                observed["url"] = url
                temporary = subject._tmp_path(path)
                temporary.write_bytes(payload)
                temporary.replace(path)

            subject._download_verified(
                receipt, target, 3, downloader=fake
            )
            self.assertIn("?generation=7", observed["url"])
            self.assertEqual(payload, target.read_bytes())
        corrupt = dict(receipt)
        corrupt["md5_base64"] = _md5(b"wrong")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                subject.AcquisitionError, "MD5 mismatch"
            ):
                subject._download_verified(
                    corrupt,
                    Path(temp) / "object.bin",
                    3,
                    downloader=lambda _url, path, _r: path.write_bytes(payload),
                )

    def test_attempt_marker_blocks_rerun_and_freezes_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sources = _metadata()["qualified_parents"]
            layout = subject.plan_layout(Path(temp) / "root", sources)
            frozen = {
                "bindings": {"contract": {"sha256": "a" * 64}},
                "sources": sources,
                "acquisition_root": layout.acquisition_root,
            }
            preflight = subject.path_preflight(layout)
            subject.open_attempt(layout, frozen, preflight)
            self.assertTrue(layout.attempt_path.is_file())
            attempt = json.loads(layout.attempt_path.read_text(encoding="utf-8"))
            self.assertFalse(attempt["rerun_authorized"])
            self.assertFalse(
                attempt["future_blind_preprocessor_execution_authorized"]
            )
            with self.assertRaises(FileExistsError):
                subject.open_attempt(layout, frozen, preflight)

    def test_attempt_is_fsynced_before_acquisition_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sources = _metadata()["qualified_parents"]
            layout = subject.plan_layout(Path(temp) / "root", sources)
            frozen = {
                "bindings": {"contract": {"sha256": "a" * 64}},
                "sources": sources,
                "acquisition_root": layout.acquisition_root,
            }
            preflight = subject.path_preflight(layout)
            events: list[str] = []

            def fake_fsync(_file_descriptor: int) -> None:
                events.append("attempt_fsynced")

            def fake_acquire(
                _frozen: dict[str, object],
                observed_layout: subject.CohortLayout,
            ) -> dict[str, object]:
                self.assertTrue(observed_layout.attempt_path.is_file())
                self.assertEqual(["attempt_fsynced"], events)
                events.append("first_network_callback")
                return {"terminal": subject.ACQUISITION_READY}

            argv = [
                "acquire_stage_c_d2_six_source_media.py",
                "--execution-contract",
                "fixture.json",
                "--retries",
                "3",
            ]
            with (
                mock.patch.object(
                    subject,
                    "validate_execution_contract",
                    return_value=frozen,
                ),
                mock.patch.object(
                    subject, "plan_layout", return_value=layout
                ),
                mock.patch.object(
                    subject, "path_preflight", return_value=preflight
                ),
                mock.patch.object(subject, "acquire", side_effect=fake_acquire),
                mock.patch.object(subject.os, "fsync", side_effect=fake_fsync),
                mock.patch.object(subject.sys, "argv", argv),
            ):
                self.assertEqual(0, subject.main())
            self.assertEqual(
                ["attempt_fsynced", "first_network_callback"], events
            )

    def test_failure_report_preserves_staging_and_no_retry(self) -> None:
        report = subject.failure_report(
            {"bindings": {}}, OSError("download failed")
        )
        self.assertEqual(subject.NOT_EVALUABLE, report["terminal"])
        self.assertTrue(report["global_staging_preserved"])
        self.assertFalse(report["rerun_authorized"])
        self.assertFalse(report["partial_completion_authorized"])

    def test_retries_are_exactly_three(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            contract_path, _, contract = _write_contract(Path(temp))
            contract["failure_policy"]["internal_retries_per_request"] = 2
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                subject.AcquisitionError, "failure policy"
            ):
                subject.validate_execution_contract(
                    contract_path, verify_git=False
                )

    def test_manifest_keeps_teacher_and_effect_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = _source(0, 20.0)
            root = Path(temp)
            receipts = {
                key: [
                    _receipt(f"{key}/{index}", b"x")
                    for index in range(13)
                ]
                for key in ("rgb", "mask", "depth")
            }
            for timeline in range(13):
                alias = f"{timeline:02x}"
                for folder, suffix in (
                    ("i/train", ".png"),
                    ("m/train", ".png"),
                    ("d/train", ".f16.gz"),
                ):
                    path = root / folder / f"{alias}{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x")
            metadata = root / "source_metadata"
            metadata.mkdir()
            pose_lines = [
                (
                    "tracking_state,pos_x,pos_y,pos_z,"
                    "q_x,q_y,q_z,q_w"
                )
            ]
            pose_lines.extend(
                f"TrackingState.READY,{index},0,0,0,0,0,1"
                for index in range(50)
            )
            (metadata / "camera_poses.csv").write_text(
                "\n".join(pose_lines) + "\n",
                encoding="utf-8",
            )
            result = subject._write_source_package(
                source,
                root,
                description_receipt=_receipt("description"),
                labelmap_receipt=_receipt("labelmap"),
                annotation_receipt=_receipt("annotation"),
                pose_receipt=_receipt("pose"),
                split_receipt=_receipt("split"),
                media_receipts=receipts,
            )
            manifest = [
                json.loads(line)
                for line in (root / "manifest.replay.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            pose_slices_exist = all(
                (root / "pose" / f"{index:02x}.json").is_file()
                for index in range(13)
            )
        self.assertEqual(13, len(manifest))
        self.assertEqual(13, len(result["frames"]))
        self.assertTrue(pose_slices_exist)
        self.assertTrue(result["future_blind_preprocessor_candidate"])
        self.assertTrue(
            all(
                row["authorization"]["geometry_teacher_execution"] is False
                and row["authorization"]["effect_evaluation"] is False
                for row in manifest
            )
        )


if __name__ == "__main__":
    unittest.main()

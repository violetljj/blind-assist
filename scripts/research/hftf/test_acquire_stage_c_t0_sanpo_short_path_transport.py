from __future__ import annotations

import base64
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquire_stage_c_t0_sanpo_short_path_transport import (
    AcquisitionConfig,
    EXECUTION_CONTRACT_SCHEMA,
    EXECUTION_CONTRACT_STATUS,
    EXECUTION_ROLE,
    MAX_CONTENT_PATH_EXCLUSIVE,
    NOT_EVALUABLE,
    REPLAY_SCHEMA,
    TransportError,
    _require_gcs_receipt,
    _tmp_path,
    download_verified,
    enumerate_content_paths,
    plan_layout,
    preflight,
    preflight_layout,
    require_fresh_layout,
    validate_execution_contract,
)
from run_stage_c_t0_sanpo_short_path_canary import (
    CANARY_READY,
    run_canary,
)
from verify_sanpo_pose_geometry_authority import _source_pose_contract


def _md5(payload: bytes) -> str:
    return base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")


class StageCT0SanpoShortPathTransportTest(unittest.TestCase):
    def config(self, *, session_id: str = "s" * 64) -> AcquisitionConfig:
        return AcquisitionConfig(
            session_id=session_id,
            camera="camera_chest",
            lens="left",
            official_split="test",
            start_frame=0,
            target_fps=10.0,
            frame_count=25,
        )

    def test_governance_failure_terminal_is_no_source_replacement(self) -> None:
        self.assertEqual(
            (
                "T0_SANPO_SHORT_PATH_TRANSPORT_NOT_EVALUABLE_"
                "NO_SOURCE_REPLACEMENT"
            ),
            NOT_EVALUABLE,
        )

    def test_long_identity_never_enters_content_paths_including_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_id = "long-session-" + ("x" * 512)
            layout = plan_layout(Path(directory) / "t", self.config(session_id=session_id))
            report = preflight_layout(layout)
            rendered = [str(path) for path in enumerate_content_paths(layout)]
        self.assertTrue(report["all_content_paths_under_limit"])
        self.assertLess(
            report["maximum_content_path_length"],
            MAX_CONTENT_PATH_EXCLUSIVE,
        )
        self.assertFalse(any(session_id in path for path in rendered))
        self.assertTrue(any(path.endswith(".tmp") for path in rendered))

    def test_path_budget_rejects_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ("x" * 170)
            layout = plan_layout(root, self.config())
            with self.assertRaisesRegex(TransportError, "path budget exceeded"):
                preflight_layout(layout)
            self.assertFalse(layout.output_root.exists())
            self.assertFalse(layout.staging_root.exists())

    def test_fresh_output_and_partial_stage_both_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = plan_layout(Path(directory) / "t", self.config())
            layout.output_root.mkdir(parents=True)
            with self.assertRaisesRegex(TransportError, "overwrite"):
                require_fresh_layout(layout)
        with tempfile.TemporaryDirectory() as directory:
            layout = plan_layout(Path(directory) / "t", self.config())
            layout.staging_root.mkdir(parents=True)
            with self.assertRaisesRegex(TransportError, "partial"):
                require_fresh_layout(layout)

    def test_preflight_is_read_only_and_source_blind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "transport"
            result = preflight(root, self.config())
            self.assertEqual(
                "T0_SANPO_SHORT_PATH_PREFLIGHT_READY", result["terminal"]
            )
            self.assertFalse(result["network_opened"])
            self.assertFalse(result["source_opened"])
            self.assertFalse(root.exists())

    def test_network_contract_only_allows_exact_consumed_development_source(
        self,
    ) -> None:
        script_path = Path(
            sys.modules[
                "acquire_stage_c_t0_sanpo_short_path_transport"
            ].__file__
        ).resolve()
        repo_root = script_path.parents[3]
        docs_root = repo_root / "docs/research/hftf"
        transport_root = Path(tempfile.gettempdir()) / "t0-contract-root"
        config = self.config(
            session_id=(
                "12b65d2c76d7ad0c17d7ac791089b8ca"
                "e0bb059c9b02a6f23129044192bc93bb"
            )
        )
        canonical_root = (
            repo_root
            / (
                "artifacts.local/evidence/datasets/"
                "hftf-f0-1-train-train-12b65d2c-25frames-20260801"
            )
        )
        source_plan = (
            repo_root
            / (
                "artifacts.local/evidence/hftf/"
                "stage-c-g0-signed-clearance-source-plan-20260801/"
                "source_plan.json"
            )
        )
        contract = {
            "schema": EXECUTION_CONTRACT_SCHEMA,
            "status": EXECUTION_CONTRACT_STATUS,
            "implementations": {
                "acquirer": {
                    "path": (
                        "scripts/research/hftf/"
                        "acquire_stage_c_t0_sanpo_short_path_transport.py"
                    ),
                    "sha256": hashlib.sha256(
                        script_path.read_bytes()
                    ).hexdigest(),
                    "network_execution_authorized": True,
                }
            },
            "source": {
                "session_id": config.session_id,
                "official_split": "train",
                "role": EXECUTION_ROLE,
                "outcome_open_before_t0": True,
                "fresh_evidence_credit": False,
                "reserved_source": False,
                "canonical_consumed_package": {
                    "root": str(canonical_root),
                    "manifest_sha256": hashlib.sha256(
                        (
                            canonical_root / "manifest.replay.jsonl"
                        ).read_bytes()
                    ).hexdigest(),
                    "dataset_spec_sha256": hashlib.sha256(
                        (
                            canonical_root / "dataset_spec.json"
                        ).read_bytes()
                    ).hexdigest(),
                },
            },
            "acquisition_config": {
                "session_id": config.session_id,
                "camera": config.camera,
                "lens": config.lens,
                "official_split": "train",
                "start_frame": config.start_frame,
                "target_fps": config.target_fps,
                "frame_count": config.frame_count,
            },
            "transport_root": str(transport_root),
            "parents": {
                "g0_source_plan": {
                    "path": str(source_plan),
                    "sha256": hashlib.sha256(
                        source_plan.read_bytes()
                    ).hexdigest(),
                }
            },
            "authorization": {
                "consumed_development_transport_execution": True,
                "fresh_or_reserved_source_open": False,
                "teacher_or_student_execution": False,
            },
        }
        train_config = AcquisitionConfig(
            session_id=config.session_id,
            camera=config.camera,
            lens=config.lens,
            official_split="train",
            start_frame=config.start_frame,
            target_fps=config.target_fps,
            frame_count=config.frame_count,
        )
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=".t0-contract-test-",
            dir=docs_root,
            delete=False,
        )
        contract_path = Path(handle.name)
        try:
            with handle:
                json.dump(contract, handle)
            loaded = validate_execution_contract(
                contract_path, transport_root, train_config
            )
            self.assertEqual(EXECUTION_ROLE, loaded["source"]["role"])
            test_config = AcquisitionConfig(
                session_id=config.session_id,
                camera=config.camera,
                lens=config.lens,
                official_split="test",
                start_frame=config.start_frame,
                target_fps=config.target_fps,
                frame_count=config.frame_count,
            )
            with self.assertRaisesRegex(
                TransportError, "not the frozen source"
            ):
                validate_execution_contract(
                    contract_path, transport_root, test_config
                )
            contract["acquisition_config"]["official_split"] = "test"
            contract_path.write_text(
                json.dumps(contract), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                TransportError, "not the frozen source"
            ):
                validate_execution_contract(
                    contract_path, transport_root, test_config
                )
            contract["acquisition_config"]["official_split"] = "train"
            contract["source"]["reserved_source"] = True
            contract_path.write_text(
                json.dumps(contract), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                TransportError, "outcome-open Development"
            ):
                validate_execution_contract(
                    contract_path, transport_root, train_config
                )
        finally:
            contract_path.unlink(missing_ok=True)

    def test_generation_and_md5_are_mandatory_and_verified(self) -> None:
        payload = b"official-object-bytes"
        receipt = {
            "name": "sanpo_dataset/v0/object",
            "generation": "42",
            "size": len(payload),
            "md5Hash": _md5(payload),
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "short.bin"
            observed: dict[str, str] = {}

            def fake(url: str, path: Path, retries: int) -> None:
                observed["url"] = url
                temporary = _tmp_path(path)
                temporary.write_bytes(payload)
                temporary.replace(path)

            download_verified(receipt, target, 1, downloader=fake)
            self.assertIn("?generation=42", observed["url"])
            self.assertEqual(payload, target.read_bytes())
            self.assertFalse(_tmp_path(target).exists())
        for missing in ("generation", "md5Hash"):
            invalid = dict(receipt)
            invalid.pop(missing)
            with self.assertRaisesRegex(TransportError, "requires"):
                _require_gcs_receipt(invalid)
        corrupt = dict(receipt)
        corrupt["md5Hash"] = _md5(b"different")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TransportError, "MD5 mismatch"):
                download_verified(
                    corrupt,
                    Path(directory) / "short.bin",
                    1,
                    downloader=lambda _url, path, _retries: path.write_bytes(payload),
                )

    def test_replay_schema_and_pose_binding_remain_verifier_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "source_metadata"
            metadata.mkdir()
            description_path = metadata / "source_session_description.json"
            description_path.write_text(
                json.dumps(
                    {
                        "session_camera_details": [
                            {"camera_name": "camera_chest", "fps": 20.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pose_path = metadata / "camera_poses.csv"
            with pose_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "tracking_state",
                        "pos_x",
                        "pos_y",
                        "pos_z",
                        "q_x",
                        "q_y",
                        "q_z",
                        "q_w",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "tracking_state": "TrackingState.READY",
                        "pos_x": 0,
                        "pos_y": 0,
                        "pos_z": 1.7,
                        "q_x": 0,
                        "q_y": 0,
                        "q_z": 0,
                        "q_w": 1,
                    }
                )

            def inventory(path: Path, name: str) -> dict[str, object]:
                payload = path.read_bytes()
                return {
                    "name": name,
                    "generation": "7",
                    "size": len(payload),
                    "md5_base64": _md5(payload),
                    "crc32c_base64": "fixture",
                }

            description = inventory(description_path, "fixture/description.json")
            poses = inventory(pose_path, "fixture/camera_poses.csv")
            spec = {
                "schema": REPLAY_SCHEMA,
                "sampling": {
                    "source_fps": 20.0,
                    "target_fps": 10.0,
                    "selected_source_frames": [0],
                },
                "source_inventory": {
                    "description": description,
                    "camera_poses": poses,
                },
            }
            rows = [
                {
                    "id": "identity-remains-in-manifest",
                    "session_id": "s" * 512,
                    "source_frame_index": 0,
                    "source_timestamp_ms": 0,
                    "image_path": "i/test/00.png",
                    "source_mask_path": "m/test/00.png",
                    "source_depth_path": "d/test/00.f16.gz",
                    "modalities": {
                        "rgb": {"name": "fixture/video_frames/000000.png"},
                        "panoptic_mask": {
                            "name": "fixture/segmentation_masks/000000.png"
                        },
                        "metric_depth": {
                            "name": "fixture/depth_maps/000000.float16.gz"
                        },
                    },
                }
            ]
            live_description = {
                "name": description["name"],
                "generation": description["generation"],
                "size": description["size"],
                "md5Hash": description["md5_base64"],
                "crc32c": description["crc32c_base64"],
            }
            live_poses = {
                "name": poses["name"],
                "generation": poses["generation"],
                "size": poses["size"],
                "md5Hash": poses["md5_base64"],
                "crc32c": poses["crc32c_base64"],
            }
            result = _source_pose_contract(
                root, spec, rows, live_description, live_poses
            )
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(1, result["binding_count"])

    def test_offline_filesystem_canary(self) -> None:
        result = run_canary()
        self.assertEqual(CANARY_READY, result["terminal"])
        self.assertGreater(result["long_session_id_length"], 500)
        self.assertTrue(result["generation_bound_url_observed"])
        self.assertTrue(result["size_and_md5_verified"])
        self.assertFalse(result["network_opened"])


if __name__ == "__main__":
    unittest.main()

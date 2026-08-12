from __future__ import annotations

import copy
import unittest
from unittest import mock

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_inventory as runner


def _synthetic_inventory() -> dict:
    parents = []
    for visit, video, rank in fresh_pool.EXPECTED_POOL:
        parents.append(
            {
                "visit_id": visit,
                "video_id": video,
                "official_fold": "Training",
                "pool_rank_sha256": rank,
                "container_bindings": {
                    "upsampling": {
                        "path": f"{runner.SOURCE_ROOT}/upsampling/Training/{video}.zip",
                        "bytes": 1,
                        "sha256": "A" * 64,
                        "declared_uncompressed_bytes": 1,
                        "recognized_member_index_sha256": "D" * 64,
                    },
                    "intrinsics": {
                        "path": f"{runner.SOURCE_ROOT}/raw/Training/{video}/lowres_wide_intrinsics.zip",
                        "bytes": 1,
                        "sha256": "B" * 64,
                        "declared_uncompressed_bytes": 1,
                        "recognized_member_index_sha256": "E" * 64,
                    },
                    "trajectory": {
                        "path": f"{runner.SOURCE_ROOT}/raw/Training/{video}/lowres_wide.traj",
                        "bytes": 1,
                        "sha256": "C" * 64,
                        "row_count": 2,
                    },
                },
                "modality_member_counts": {"color": 1, "highres_depth": 1, "lowres_depth": 1, "confidence": 1},
                "intrinsics_member_count": 1,
                "frame_plan": {
                    "video_id": video,
                    "source_common_frame_count": 1,
                    "exact_intrinsics_common_frame_count": 1,
                    "exact_pose_bounded_frame_count": 1,
                    "exact_timestamp_tokens": ["1.0"],
                    "pose_rejected": [],
                    "selection_rule": "ALL_SAME_STEM_MODALITIES_EXACT_INTRINSICS_WITH_BOUNDED_POSE_SORTED_BY_EXACT_NS_THEN_TOKEN",
                    "truth_or_model_fields_read": False,
                },
                "declared_materialized_bytes": 3,
            }
        )
    value = {
        "schema": "blindassist.taro.o1r.r11_fresh_pool_inventory.v1",
        "protocol_content_sha256": runner.PROTOCOL_CONTENT_SHA256,
        "authorization_receipt_content_sha256": runner.AUTHORIZATION_CONTENT_SHA256,
        "download_formal_result_content_sha256": runner.DOWNLOAD_FORMAL_CONTENT_SHA256,
        "download_manifest_content_sha256": runner.DOWNLOAD_MANIFEST_CONTENT_SHA256,
        "pool_content_sha256": runner.POOL_CONTENT_SHA256,
        "request_plan_sha256": runner.REQUEST_PLAN_SHA256,
        "parent_count": 48,
        "asset_count": 144,
        "compressed_source_bytes": 2960390828,
        "exact_pose_bounded_frame_count": 48,
        "declared_materialized_bytes": 144,
        "parents": parents,
        "inventory_policy": dict(runner.EXPECTED_INVENTORY_POLICY),
        "read_accounting": {
            "zip_central_directory_metadata_read_operations": 48 * 2,
            "trajectory_payload_reads": 48,
            "zip_member_payload_reads": 0,
            "highres_depth_member_payload_reads": 0,
            "pixel_arrays_decoded": 0,
            "model_executions": 0,
            "network_requests": 0,
        },
        "zip_declared_crc_indexed": True,
        "zip_member_payload_crc_validated": False,
        "pixel_arrays_decoded": False,
        "source_frames_materialized": False,
        "faro_values_interpreted": False,
        "truth_values_interpreted": False,
        "model_outputs_read": False,
        "training": False,
    }
    value["content_sha256"] = adapter.canonical_sha256(value)
    return value


def _reseal(value: dict) -> dict:
    clone = copy.deepcopy(value)
    clone.pop("content_sha256", None)
    clone["content_sha256"] = adapter.canonical_sha256(clone)
    return clone


class PoolInventoryTests(unittest.TestCase):
    def test_exact_roster_and_scope_validate(self) -> None:
        value = _synthetic_inventory()
        result = runner.validate_inventory(value)
        self.assertEqual(result["parent_count"], 48)
        self.assertFalse(result["pixel_arrays_decoded"])
        self.assertFalse(result["faro_values_interpreted"])

    def test_roster_frame_and_scope_tamper_fail_closed(self) -> None:
        for mutate in (
            lambda value: value["parents"][0].__setitem__("visit_id", "tampered"),
            lambda value: value["parents"][0]["frame_plan"]["exact_timestamp_tokens"].append("2.0"),
            lambda value: value.__setitem__("faro_values_interpreted", True),
        ):
            with self.subTest(mutate=mutate):
                value = _synthetic_inventory()
                mutate(value)
                with self.assertRaises(runner.PoolInventoryError):
                    runner.validate_inventory(_reseal(value))

    def test_container_byte_and_frame_order_tamper_fail_closed(self) -> None:
        for mutate in (
            lambda value: value["parents"][0]["container_bindings"]["upsampling"].__setitem__("path", "wrong.zip"),
            lambda value: value["parents"][0]["container_bindings"]["trajectory"].__setitem__("sha256", "lowercase"),
            lambda value: value["parents"][0].__setitem__("declared_materialized_bytes", 4),
            lambda value: value.__setitem__("declared_materialized_bytes", "144"),
            lambda value: value["parents"][0]["frame_plan"].update(
                {
                    "source_common_frame_count": 2,
                    "exact_intrinsics_common_frame_count": 2,
                    "exact_pose_bounded_frame_count": 2,
                    "exact_timestamp_tokens": ["2.0", "1.0"],
                }
            ),
        ):
            with self.subTest(mutate=mutate):
                value = _synthetic_inventory()
                mutate(value)
                with self.assertRaises(runner.PoolInventoryError):
                    runner.validate_inventory(_reseal(value))

    def test_zero_pose_bounded_parent_is_valid_inventory_but_not_ready(self) -> None:
        value = _synthetic_inventory()
        frame_plan = value["parents"][0]["frame_plan"]
        frame_plan["exact_pose_bounded_frame_count"] = 0
        frame_plan["exact_timestamp_tokens"] = []
        frame_plan["pose_rejected"] = [
            {"timestamp_token": "1.0", "reason_code": "FRAME_OUTSIDE_TRAJECTORY"}
        ]
        value["exact_pose_bounded_frame_count"] = 47
        result = runner.validate_inventory(_reseal(value))
        counts = [row["frame_plan"]["exact_pose_bounded_frame_count"] for row in result["parents"]]
        self.assertFalse(all(count > 0 for count in counts))

    def test_inventory_containers_bind_exactly_to_download_receipts(self) -> None:
        inventory = _synthetic_inventory()
        receipt_by_path = {}
        for parent in inventory["parents"]:
            for binding in parent["container_bindings"].values():
                relative = binding["path"].removeprefix(f"{runner.SOURCE_ROOT}/")
                receipt_by_path[relative] = {
                    "bytes": binding["bytes"],
                    "sha256": binding["sha256"],
                }
        runner._validate_inventory_download_binding(
            inventory,
            {
                "source_files_verified": True,
                "receipt_by_path": receipt_by_path,
            },
        )
        first = next(iter(receipt_by_path.values()))
        first["sha256"] = "F" * 64
        with self.assertRaises(runner.PoolInventoryError):
            runner._validate_inventory_download_binding(
                inventory,
                {
                    "source_files_verified": True,
                    "receipt_by_path": receipt_by_path,
                },
            )

    def test_formal_entrypoint_and_authority_are_narrow(self) -> None:
        self.assertEqual(runner.EXPECTED_ARGV[:2], ["-m", "scripts.research.taro_o1r_r11_abstention_runtime.run_pool_inventory"])
        for key in ("pixel_array_decode", "source_frame_materialization", "model_execution", "faro_read", "truth_scoring", "training", "device", "deployment", "product", "safety"):
            self.assertFalse(runner.EXPECTED_AUTHORITY[key])
        self.assertFalse(runner.EXPECTED_AUTHORITY["zip_member_payload_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["zip_member_payload_crc_validation"])

    def test_download_evidence_records_are_exact_without_source_preflight(self) -> None:
        evidence = runner.verify_download_evidence(verify_source_files=False)
        self.assertEqual(evidence["source_bytes"], 2960390828)
        self.assertEqual(len(evidence["rows"]), 144)
        self.assertFalse(evidence["source_files_verified"])

    def test_metadata_index_never_opens_or_reads_zip_members(self) -> None:
        video = fresh_pool.EXPECTED_POOL[0][1]
        source = runner._repo_path(runner.SOURCE_ROOT)
        upsampling_path = source / f"upsampling/Training/{video}.zip"
        intrinsics_path = source / f"raw/Training/{video}/lowres_wide_intrinsics.zip"
        with (
            mock.patch.object(materializer.zipfile.ZipFile, "testzip", side_effect=AssertionError("payload CRC read")),
            mock.patch.object(materializer.zipfile.ZipFile, "open", side_effect=AssertionError("member opened")),
            mock.patch.object(materializer.zipfile.ZipFile, "read", side_effect=AssertionError("member read")),
        ):
            upsampling, _ = runner.index_upsampling_archive_metadata_only(
                upsampling_path,
                video,
                maximum_declared_uncompressed_bytes=runner.MAXIMUM_MATERIALIZED_BYTES,
            )
            intrinsics, _ = runner.index_intrinsics_archive_metadata_only(
                intrinsics_path,
                video,
                maximum_declared_uncompressed_bytes=runner.MAXIMUM_MATERIALIZED_BYTES,
            )
        self.assertEqual(set(upsampling), {"color", "highres_depth", "lowres_depth", "confidence"})
        self.assertGreater(len(upsampling["highres_depth"]), 0)
        self.assertGreater(len(intrinsics), 0)


if __name__ == "__main__":
    unittest.main()

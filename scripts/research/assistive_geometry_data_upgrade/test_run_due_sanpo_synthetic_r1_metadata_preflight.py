from __future__ import annotations

import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.research.assistive_geometry_data_upgrade import (
    run_due_sanpo_synthetic_r1_metadata_preflight as runner,
)


def _body_receipt(name: str, body: bytes) -> dict[str, object]:
    return {
        "name": name,
        "generation": "101",
        "metageneration": "1",
        "size": str(len(body)),
        "md5Hash": base64.b64encode(hashlib.md5(body, usedforsecurity=False).digest()).decode("ascii"),
        "crc32c": "AAAAAA==",
        "contentType": "application/octet-stream",
    }


def _frame_receipt(name: str, size: int = 17) -> dict[str, object]:
    return {
        "name": name,
        "generation": "201",
        "metageneration": "1",
        "size": str(size),
        "md5Hash": "AAAAAAAAAAAAAAAAAAAAAA==",
        "crc32c": "AAAAAA==",
        "contentType": "application/octet-stream",
    }


class FakeProvider:
    def __init__(self, lock: dict[str, object], count: int = 30, pose_header: str | None = None) -> None:
        self.lock = lock
        self.request_count = 0
        self.network_response_bytes = 0
        self.list_pages: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []
        self.metadata_names = {item["role"]: item["path"] for item in lock["object_contract"]["metadata_objects"]}
        description = {
            "session_type": "synthetic",
            "session_camera_location": [runner.CAMERA],
            "session_camera_details": [
                {
                    "fps": 20.0,
                    "left_camera_params": {
                        "image_width": 1280,
                        "image_height": 720,
                        "fx": 900.0,
                        "fy": 901.0,
                        "cx": 640.0,
                        "cy": 360.0,
                    },
                }
            ],
        }
        pose_header = pose_header or "x,y,z,qx,qy,qz,qw"
        pose_rows = "\n".join([pose_header, *("0,0,0,0,0,0,1" for _ in range(35))]) + "\n"
        self.bodies = {
            self.metadata_names["session_description"]: json.dumps(description).encode(),
            self.metadata_names["global_labelmap"]: json.dumps({"ground": 1, "wall": 2}).encode(),
            self.metadata_names["frame_annotation_type"]: json.dumps({str(i): "panoptic" for i in range(count)}).encode(),
            self.metadata_names["camera_pose_table"]: pose_rows.encode(),
        }
        self.objects = {name: _body_receipt(name, body) for name, body in self.bodies.items()}
        self.listings: dict[str, list[dict[str, object]]] = {}
        for spec in lock["object_contract"]["frame_prefixes"]:
            self.listings[spec["prefix"]] = [
                _frame_receipt(f"{spec['prefix']}{index:06d}{spec['suffix']}", size=100 + index)
                for index in range(count)
            ]

    def get_object_metadata(self, object_name: str) -> dict[str, object]:
        self.calls.append(("HEAD", object_name))
        self.request_count += 1
        return copy.deepcopy(self.objects[object_name])

    def read_metadata_object(self, object_name: str, generation: str) -> bytes:
        self.calls.append(("BODY", object_name))
        self.request_count += 1
        if object_name not in self.bodies:
            raise AssertionError("frame body attempted")
        self.network_response_bytes += len(self.bodies[object_name])
        return self.bodies[object_name]

    def list_frame_objects(self, prefix: str) -> list[dict[str, object]]:
        self.calls.append(("LIST", prefix))
        self.request_count += 1
        self.list_pages[prefix] = 1
        return copy.deepcopy(self.listings[prefix])


class SanpoSyntheticR1MetadataPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = runner.load_json(runner.LOCK_PATH)

    def evaluate(self, provider: FakeProvider | None = None):
        provider = provider or FakeProvider(self.lock)
        return (*runner.evaluate_preflight(provider, self.lock), provider)

    def test_execution_lock_validates(self) -> None:
        runner.validate_execution_lock(self.lock)

    def test_pose_unbound_inventory_is_partial(self) -> None:
        inventory, schema, result, provider = self.evaluate()
        self.assertEqual("PARTIAL_POSE_UNBOUND", result["decision"])
        self.assertEqual(30, inventory["numeric_index_intersection_count"])
        self.assertFalse(schema["pose_table"]["pose_binding_admitted"])
        self.assertEqual(runner.BODY_CANARY_SUCCESSOR, result["unique_successor"])
        self.assertFalse(result["authority"]["source_data_support_established"])
        self.assertFalse(result["execution_disclosure"]["frame_body_requested_or_read"])

    def test_only_metadata_bodies_are_read(self) -> None:
        _, _, _, provider = self.evaluate()
        body_names = [name for method, name in provider.calls if method == "BODY"]
        self.assertEqual(
            [item["path"] for item in self.lock["object_contract"]["metadata_objects"]],
            body_names,
        )
        frame_prefixes = [item["prefix"] for item in self.lock["object_contract"]["frame_prefixes"]]
        self.assertFalse(any(any(name.startswith(prefix) for prefix in frame_prefixes) for name in body_names))
        self.assertFalse(any("splits/" in name for _, name in provider.calls))

    def test_lowest_25_is_deterministic(self) -> None:
        provider = FakeProvider(self.lock, count=30)
        rgb_prefix = self.lock["object_contract"]["frame_prefixes"][0]["prefix"]
        provider.listings[rgb_prefix].reverse()
        inventory, _, result = runner.evaluate_preflight(provider, self.lock)
        self.assertEqual(list(range(25)), inventory["selected_lowest_25"])
        self.assertEqual(list(range(25)), result["selected_lowest_25"])

    def test_fewer_than_25_is_not_evaluable(self) -> None:
        provider = FakeProvider(self.lock, count=24)
        _, _, result = runner.evaluate_preflight(provider, self.lock)
        self.assertEqual("NOT_EVALUABLE", result["decision"])
        self.assertEqual(runner.STOP_SUCCESSOR, result["unique_successor"])

    def test_float16_suffix_is_removed_as_a_whole(self) -> None:
        spec = self.lock["object_contract"]["frame_prefixes"][2]
        indexed, receipts = runner._frame_inventory(
            [_frame_receipt(f"{spec['prefix']}000008.float16.gz")], spec, 10
        )
        self.assertEqual([8], list(indexed))
        self.assertEqual(8, receipts[0]["numeric_index"])

    def test_noncanonical_numeric_alias_is_rejected(self) -> None:
        spec = self.lock["object_contract"]["frame_prefixes"][0]
        with self.assertRaisesRegex(runner.PreflightError, "canonical six-digit"):
            runner._frame_inventory([_frame_receipt(f"{spec['prefix']}1.png")], spec, 10)

    def test_duplicate_numeric_index_is_rejected(self) -> None:
        spec = self.lock["object_contract"]["frame_prefixes"][0]
        item = _frame_receipt(f"{spec['prefix']}000001.png")
        with self.assertRaisesRegex(runner.PreflightError, "duplicate numeric index"):
            runner._frame_inventory([item, copy.deepcopy(item)], spec, 10)

    def test_frame_receipt_cannot_claim_body_read(self) -> None:
        spec = self.lock["object_contract"]["frame_prefixes"][0]
        _, receipts = runner._frame_inventory(
            [_frame_receipt(f"{spec['prefix']}000001.png")], spec, 10
        )
        self.assertFalse(receipts[0]["body_read"])
        self.assertIsNone(receipts[0]["sha256_after_read"])
        self.assertTrue(receipts[0]["provider_receipt_only"])

    def test_missing_provider_hash_is_rejected(self) -> None:
        spec = self.lock["object_contract"]["frame_prefixes"][0]
        item = _frame_receipt(f"{spec['prefix']}000001.png")
        item.pop("crc32c")
        with self.assertRaisesRegex(runner.PreflightError, "provider hash"):
            runner._frame_inventory([item], spec, 10)

    def test_metadata_md5_mismatch_is_rejected(self) -> None:
        provider = FakeProvider(self.lock)
        name = provider.metadata_names["session_description"]
        provider.objects[name]["md5Hash"] = "wrong"
        with self.assertRaisesRegex(runner.PreflightError, "MD5 mismatch"):
            runner.evaluate_preflight(provider, self.lock)

    def test_missing_camera_is_rejected(self) -> None:
        provider = FakeProvider(self.lock)
        name = provider.metadata_names["session_description"]
        value = json.loads(provider.bodies[name])
        value["session_camera_location"] = ["camera_head"]
        provider.bodies[name] = json.dumps(value).encode()
        provider.objects[name] = _body_receipt(name, provider.bodies[name])
        with self.assertRaisesRegex(runner.PreflightError, "camera missing"):
            runner.evaluate_preflight(provider, self.lock)

    def test_nonfinite_fps_is_rejected(self) -> None:
        provider = FakeProvider(self.lock)
        name = provider.metadata_names["session_description"]
        value = json.loads(provider.bodies[name])
        value["session_camera_details"][0]["fps"] = "NaN"
        provider.bodies[name] = json.dumps(value).encode()
        provider.objects[name] = _body_receipt(name, provider.bodies[name])
        with self.assertRaisesRegex(runner.PreflightError, "fps invalid"):
            runner.evaluate_preflight(provider, self.lock)

    def test_pose_rows_are_not_parsed_or_admitted(self) -> None:
        _, schema, result, _ = self.evaluate()
        pose = schema["pose_table"]
        self.assertEqual(35, pose["data_row_count"])
        self.assertFalse(pose["numeric_pose_values_parsed"])
        self.assertFalse(pose["row_count_coverage_is_frame_binding"])
        self.assertFalse(result["pose_and_time"]["pose_binding_admitted"])

    def test_explicit_header_does_not_grant_pose_or_timestamp_authority(self) -> None:
        provider = FakeProvider(
            self.lock,
            pose_header="frame_index,timestamp_ms,x,y,z,qx,qy,qz,qw",
        )
        pose_name = provider.metadata_names["camera_pose_table"]
        rows = "\n".join(
            ["frame_index,timestamp_ms,x,y,z,qx,qy,qz,qw", *(f"{i},{i * 50},0,0,0,0,0,0,1" for i in range(35))]
        ) + "\n"
        provider.bodies[pose_name] = rows.encode()
        provider.objects[pose_name] = _body_receipt(pose_name, provider.bodies[pose_name])
        _, _, result = runner.evaluate_preflight(provider, self.lock)
        self.assertEqual("PASS", result["decision"])
        self.assertFalse(result["pose_and_time"]["pose_binding_admitted"])
        self.assertFalse(result["pose_and_time"]["explicit_timestamp_materialized"])
        self.assertFalse(result["authority"]["pose_or_timestamp_admitted"])

    def test_inventory_counts_never_fill_capability_counts(self) -> None:
        _, _, result, _ = self.evaluate()
        self.assertGreater(result["inventory_counts"]["metric_depth_object_inventory_count"], 0)
        self.assertTrue(all(value == 0 for value in result["capability_counts"].values()))
        self.assertFalse(result["authority"]["support_truth_established"])
        self.assertFalse(result["authority"]["boundary_truth_materialized"])

    def test_existing_output_root_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "preflight"
            root.mkdir()
            with mock.patch.object(runner, "OUTPUT_ROOT", root):
                with self.assertRaisesRegex(runner.PreflightError, "failed-attempt file set drift"):
                    runner.execute(self.lock, root)

    def test_output_root_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(runner.PreflightError, "output root escape"):
                runner.execute(self.lock, Path(directory) / "elsewhere")

    def test_lock_cannot_authorize_frame_body(self) -> None:
        mutated = copy.deepcopy(self.lock)
        mutated["execution_authority"]["frame_body_request_or_read"] = True
        with self.assertRaisesRegex(runner.PreflightError, "execution authority drift"):
            runner.validate_execution_lock(mutated)

    def test_observed_404_terminal_is_fail_closed(self) -> None:
        inventory, schema, result = runner.build_observed_failure_terminal(
            self.lock,
            attempt_sha256="A" * 64,
            retry_attempt_sha256="B" * 64,
        )
        self.assertEqual("NOT_EVALUABLE", result["decision"])
        self.assertEqual(0, inventory["numeric_index_intersection_count"])
        self.assertFalse(schema["pose_header_or_row_count_inspected"])
        self.assertFalse(result["execution_disclosure"]["frame_prefix_listing_performed"])
        self.assertFalse(result["execution_disclosure"]["frame_body_requested_or_read"])
        self.assertTrue(all(value == 0 for value in result["capability_counts"].values()))
        self.assertEqual(runner.STOP_SUCCESSOR, result["unique_successor"])


if __name__ == "__main__":
    unittest.main()

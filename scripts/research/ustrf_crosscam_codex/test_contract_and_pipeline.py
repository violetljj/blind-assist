from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent.parent.parent))

from scripts.research.ustrf_crosscam_codex import evaluate_proxy_benchmark
from scripts.research.ustrf_crosscam_codex import prepare_review_bundle
from scripts.research.ustrf_crosscam_codex import validate_teacher_reviews
from scripts.research.ustrf_crosscam_codex.contract import (
    CANDIDATE_SCHEMA,
    CONTRACT_ID,
    ContractError,
    load_json,
    sha256_file,
    validate_review,
    write_json,
)
from scripts.research.ustrf_crosscam_codex.projected_corridor_geometry import (
    INSIDE,
    OUTSIDE,
    UNCERTAIN,
    classify_bottom_center,
    robust_relation,
    validate_polygon,
)


FALSE_FLAGS = {
    "human_event_truth_present": False,
    "metric_geometry_present": False,
    "training_authorized": False,
    "u0_authority_granted": False,
    "android_runtime_change_authorized": False,
    "production_model_replacement_authorized": False,
}


class CrossCameraCodexPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "fixture.mp4"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
        )
        self.assertTrue(writer.isOpened())
        for index in range(40):
            frame = np.full((240, 320, 3), 230, dtype=np.uint8)
            cv2.rectangle(frame, (130, 100), (190, 220), (20, 20, 20), -1)
            cv2.putText(frame, str(index), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            writer.write(frame)
        writer.release()
        self.source_receipt = self.root / "source_receipt.json"
        write_json(
            self.source_receipt,
            {
                "schema": "blindassist_ustrf_crosscam_source_receipt_v1",
                "contract_id": CONTRACT_ID,
                "source_id": "synthetic-contract-fixture",
                "dataset_name": "Synthetic contract fixture",
                "dataset_page": "https://example.invalid/fixture",
                "citation": "Synthetic unit-test fixture; not external evidence.",
                "license_name": "test-only",
                "license_url": "https://example.invalid/license",
                "camera_domain": "synthetic_head_mounted",
                "lawfully_available": True,
                "public_data": True,
                "privacy_review_status": "not_required_public_release",
                "video_sha256": sha256_file(self.video),
                **FALSE_FLAGS,
            },
        )
        self.config = self.root / "config.json"
        write_json(
            self.config,
            {
                "schema": "blindassist_ustrf_crosscam_codex_config_v1",
                "contract_id": CONTRACT_ID,
                "teacher_interval_ms": 250,
                "causal_interval_ms": 500,
                "max_abs_seek_error_ms": 125,
                "contact_sheet_columns": 4,
                "contact_sheet_page_size": 16,
                "assumed_geometry": {
                    "authority": "assumed_geometry_v1",
                    "pseudo_metric": True,
                    "camera_height_m": 1.6,
                    "horizontal_fov_deg": 70.0,
                    "vertical_fov_deg": 55.0,
                    "walking_speed_mps": 1.2,
                    "route_width_m": 0.9,
                    "risk_horizon_s": 3.0,
                    "route_polygon_xy_norm": [[0.26, 0.98], [0.74, 0.98], [0.56, 0.44], [0.44, 0.44]],
                },
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def prepare_and_review(self) -> tuple[Path, Path]:
        bundle_dir = self.root / "bundle"
        prepare_review_bundle.run(
            Namespace(
                video=self.video,
                source_receipt=self.source_receipt,
                config=self.config,
                output_dir=bundle_dir,
                start_ms=0,
                duration_ms=2000,
            )
        )
        bundle_path = bundle_dir / "bundle_manifest.json"
        reviews_dir = bundle_dir / "reviews"
        reviews_dir.mkdir()
        for template_path in sorted((bundle_dir / "review_templates").glob("*.json")):
            review = load_json(template_path)
            review["review_id"] = template_path.stem
            review["route_valid"] = "yes"
            review["overall_risk"] = "critical"
            review["abstain_reasons"] = []
            frames = load_json(bundle_path)["review_artifacts"][review["role"]]["frames"]
            review["events"] = [
                {
                    "category": "static_obstacle",
                    "route_relation": "inside",
                    "distance_band": "0-2m",
                    "ttc_band": "0-1.5s",
                    "required_action": "stop",
                    "confidence": "high",
                    "start_frame": frames[0]["frame_id"],
                    "end_frame": frames[-1]["frame_id"],
                    "peak_frame": frames[len(frames) // 2]["frame_id"],
                    "evidence_frames": [frames[0]["frame_id"], frames[-1]["frame_id"]],
                }
            ]
            write_json(reviews_dir / template_path.name, review)
        consensus_path = self.root / "consensus.json"
        validate_teacher_reviews.run(
            Namespace(bundle_manifest=bundle_path, reviews_dir=reviews_dir, output=consensus_path)
        )
        return bundle_path, consensus_path

    def test_end_to_end_proxy_comparison(self) -> None:
        bundle_path, consensus_path = self.prepare_and_review()
        consensus = load_json(consensus_path)
        candidate_path = self.root / "candidate.json"
        write_json(
            candidate_path,
            {
                "schema": CANDIDATE_SCHEMA,
                "contract_id": CONTRACT_ID,
                "candidate_id": "fixture-external-candidate",
                "candidate_kind": "unit_test",
                "bundle_manifest_sha256": sha256_file(bundle_path),
                "events": deepcopy(consensus["roles"]["full_context_teacher"]["events"]),
                **FALSE_FLAGS,
            },
        )
        output = self.root / "report.json"
        report = evaluate_proxy_benchmark.run(
            Namespace(consensus=consensus_path, candidate=[candidate_path], tolerance_ms=1000, output=output)
        )
        self.assertEqual(report["decision"], "PROXY_COMPARISON_AVAILABLE")
        self.assertEqual(report["candidates"][1]["metrics"]["event_recall"], 1.0)
        self.assertEqual(report["candidates"][1]["metrics"]["critical_miss_count"], 0)

    def test_hash_binding_and_unknown_fail_closed(self) -> None:
        bundle_path, _ = self.prepare_and_review()
        bundle = load_json(bundle_path)
        template = load_json(next((bundle_path.parent / "review_templates").glob("*.json")))
        template["bundle_manifest_sha256"] = "0" * 64
        with self.assertRaises(ContractError):
            validate_review(template, bundle=bundle, bundle_sha256=sha256_file(bundle_path))
        template["bundle_manifest_sha256"] = sha256_file(bundle_path)
        template["abstain_reasons"] = []
        with self.assertRaises(ContractError):
            validate_review(template, bundle=bundle, bundle_sha256=sha256_file(bundle_path))

    def test_projected_corridor_uses_ground_contact_and_uncertainty(self) -> None:
        polygon = [[0.56, 0.98], [0.92, 0.98], [0.66, 0.44], [0.55, 0.44]]
        pexels_car = [325.7878, 203.0786, 367.3694, 243.1192]
        relations = [
            classify_bottom_center(
                pexels_car,
                frame_width=640,
                frame_height=360,
                polygon_xy_norm=polygon,
                uncertainty_frame_ratio=ratio,
            ).relation
            for ratio in (0.01, 0.02, 0.03)
        ]
        self.assertEqual(relations, [OUTSIDE, UNCERTAIN, UNCERTAIN])
        self.assertEqual(robust_relation(relations), UNCERTAIN)
        central = classify_bottom_center(
            [420.0, 220.0, 450.0, 260.0],
            frame_width=640,
            frame_height=360,
            polygon_xy_norm=polygon,
            uncertainty_frame_ratio=0.03,
        )
        self.assertEqual(central.relation, INSIDE)

    def test_projected_corridor_rejects_invalid_polygon(self) -> None:
        with self.assertRaises(ValueError):
            validate_polygon([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])


if __name__ == "__main__":
    unittest.main()

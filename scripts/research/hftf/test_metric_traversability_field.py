#!/usr/bin/env python3

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from metric_traversability_field import (
    AlertMapper,
    TraversabilityPolicy,
    build_metric_traversability_field,
)


def synthetic_depth(with_center_wall: bool = True) -> tuple[np.ndarray, np.ndarray]:
    height, width = 120, 160
    fx = fy = 120.0
    cx, cy = width / 2.0, height / 2.0
    depth = np.full((height, width), np.nan, dtype=np.float64)
    camera_height = 1.20
    for row in range(round(cy) + 4, height):
        depth[row, :] = fy * camera_height / (row - cy)
    if with_center_wall:
        depth[28:86, 70:90] = 1.0
    intrinsics = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return depth, intrinsics


class MetricTraversabilityFieldTest(unittest.TestCase):
    def test_rejects_a_downward_ground_normal_before_building_lateral_basis(self) -> None:
        depth, intrinsics = synthetic_depth()
        with patch(
            "metric_traversability_field.fit_ground_plane",
            return_value=(np.asarray([0.45, 0.66, -0.60]), 0.68, 0.02),
        ):
            field = build_metric_traversability_field(
                depth,
                intrinsics,
                metric_scale={
                    "status": "VALID",
                    "scale": 1.0,
                    "anchor_age_ns": 0,
                    "anchor_source": "orientation-test",
                },
                source_model="orientation-test",
            )

        self.assertEqual("UNKNOWN", field["status"])
        self.assertIn(
            "UNKNOWN_IMPLAUSIBLE_GROUND_ORIENTATION",
            field["unknown_reasons"],
        )

    def test_image_left_and_right_preserve_signed_birdseye_direction(self) -> None:
        scale = {
            "status": "VALID",
            "scale": 1.0,
            "anchor_age_ns": 0,
            "anchor_source": "orientation-test",
        }
        nearest_thetas = []
        for columns in (slice(34, 54), slice(106, 126)):
            depth, intrinsics = synthetic_depth(False)
            depth[28:86, columns] = 1.0
            field = build_metric_traversability_field(
                depth,
                intrinsics,
                metric_scale=scale,
                source_model="orientation-test",
            )
            occupied = [
                item
                for item in field["clearance_profile"]
                if item["nearest_intrusion_m"] is not None
            ]
            nearest_thetas.append(
                min(occupied, key=lambda item: item["nearest_intrusion_m"])[
                    "theta_deg"
                ]
            )

        self.assertLess(nearest_thetas[0], 0)
        self.assertGreater(nearest_thetas[1], 0)

    def test_builds_continuous_profile_envelopes_and_intrusion_regions(self) -> None:
        depth, intrinsics = synthetic_depth()
        field = build_metric_traversability_field(
            depth,
            intrinsics,
            metric_scale={
                "status": "VALID",
                "scale": 1.0,
                "anchor_age_ns": 10,
                "anchor_source": "synthetic-test-anchor",
            },
            source_model="synthetic-test-depth",
            timestamp_ns=100,
        )

        self.assertEqual(field["status"], "VALID")
        self.assertEqual(len(field["clearance_profile"]), 17)
        self.assertEqual(
            [row["horizon_m"] for row in field["sweep_envelopes"]],
            [1.0, 1.5, 2.0],
        )
        center = next(
            row for row in field["clearance_profile"] if row["theta_deg"] == 0
        )
        self.assertIsNotNone(center["nearest_intrusion_m"])
        self.assertIsNotNone(center["risk_score"])
        self.assertEqual(center["provenance"]["metric_scale_source"], "synthetic-test-anchor")
        self.assertTrue(field["intrusion_regions"])
        self.assertEqual(field["authority"], "DEVELOPMENT_ONLY_SHADOW_DEMO")

    def test_missing_scale_fails_closed_without_metric_outputs(self) -> None:
        depth, intrinsics = synthetic_depth()
        field = build_metric_traversability_field(
            depth,
            intrinsics,
            metric_scale={"status": "UNKNOWN_STALE_METRIC_SCALE_ANCHOR"},
            source_model="synthetic-test-depth",
        )

        self.assertEqual(field["status"], "UNKNOWN")
        self.assertEqual(field["clearance_profile"], [])
        self.assertIn(
            "UNKNOWN_STALE_METRIC_SCALE_ANCHOR", field["unknown_reasons"]
        )
        self.assertEqual(AlertMapper().map(field)["status"], "SILENT_UNKNOWN")

    def test_alert_is_lossy_shadow_projection_and_temporal_trend_is_observed_only(self) -> None:
        depth, intrinsics = synthetic_depth()
        scale = {
            "status": "VALID",
            "scale": 1.0,
            "anchor_age_ns": 10,
            "anchor_source": "synthetic-test-anchor",
        }
        previous = build_metric_traversability_field(
            depth * 1.15,
            intrinsics,
            metric_scale=scale,
            source_model="synthetic-test-depth",
            timestamp_ns=100,
        )
        current = build_metric_traversability_field(
            depth,
            intrinsics,
            metric_scale=scale,
            source_model="synthetic-test-depth",
            timestamp_ns=200,
            previous_field=previous,
            previous_timestamp_ns=100,
        )

        decision = AlertMapper(alert_horizon_m=1.5).map(current)
        self.assertEqual(decision["authority"], "SHADOW_DEMO_ONLY")
        self.assertEqual(decision["status"], "CENTER_RISK")
        self.assertIn("不代表安全方向", decision["observed_opening_hint"]["text_zh"])
        self.assertEqual(current["temporal_trend"]["status"], "VALID_OBSERVED_DELTA")
        self.assertIn("not motion", current["temporal_trend"]["claim_ceiling"])

    def test_policy_rejects_duplicate_direction_bins(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            TraversabilityPolicy(direction_degrees=(0, 0)).validate()

    def test_failed_image_quality_suppresses_metric_geometry(self) -> None:
        depth, intrinsics = synthetic_depth()
        field = build_metric_traversability_field(
            depth,
            intrinsics,
            metric_scale={
                "status": "VALID",
                "scale": 1.0,
                "anchor_age_ns": 0,
                "anchor_source": "synthetic-test-anchor",
            },
            source_model="synthetic-test-depth",
            image_quality={"status": "FAIL", "pass": False},
        )
        self.assertEqual(field["status"], "UNKNOWN")
        self.assertIn("UNKNOWN_IMAGE_QUALITY", field["unknown_reasons"])
        self.assertEqual(field["clearance_profile"], [])

    def test_repository_schema_and_protocol_bind_frozen_authority(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        schema = json.loads(
            (repository / "schemas" / "metric_traversability_field_r0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        protocol = json.loads(
            (
                repository
                / "docs"
                / "research"
                / "hftf"
                / "METRIC_TRAVERSABILITY_FIELD_SHADOW_DEMO_R0_PROTOCOL_2026-08-04.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            schema["properties"]["authority"]["const"],
            "DEVELOPMENT_ONLY_SHADOW_DEMO",
        )
        self.assertEqual(protocol["direction_degrees"], {"minimum": -40, "maximum": 40, "step": 5})
        self.assertFalse(protocol["safety_authority"])


if __name__ == "__main__":
    unittest.main()

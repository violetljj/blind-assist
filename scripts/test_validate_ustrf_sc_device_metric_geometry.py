from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("device_gate", SCRIPTS / "validate_ustrf_sc_device_metric_geometry.py")
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)
REPORT_SPEC = importlib.util.spec_from_file_location("research_report", SCRIPTS / "report_ustrf_sc_research_benchmark.py")
assert REPORT_SPEC and REPORT_SPEC.loader
research_report = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(research_report)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeviceMetricGeometryValidatorTest(unittest.TestCase):
    def test_complete_bound_bundle_authorizes_shadow_but_never_production(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            value = self.bundle(root)
            report = subject.validate(value, root=root, require_complete=True)
            self.assertTrue(report["device_metric_geometry_admitted"])
            self.assertTrue(report["geometry_shadow_authorized"])
            self.assertFalse(report["production_authority"])

    def test_incomplete_template_reports_blocked_without_fabricating_evidence(self) -> None:
        report = subject.validate({
            "schema": subject.SCHEMA,
            "status": "not_collected",
            "production_authority": False,
        }, root=Path("."), require_complete=False)
        self.assertFalse(report["device_metric_geometry_admitted"])
        self.assertIn("MISSING_DEVICE_IDENTITY", report["blockers"])
        self.assertIn("MISSING_EVIDENCE_ARTIFACTS", report["blockers"])

    def test_unstable_pose_or_wrong_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            value = self.bundle(root)
            value["frame_clock"]["pose_reference_mode"] = "EPHEMERAL_PER_FRAME"
            with self.assertRaisesRegex(subject.ContractError, "INTER_FRAME_STABLE"):
                subject.validate(value, root=root, require_complete=True)

    def test_hollow_artifact_or_summary_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            value = self.bundle(root)
            artifact_path = root / "calibration.json"
            artifact_path.write_text(json.dumps({"name": "calibration"}), encoding="utf-8")
            value["evidence_artifacts"]["calibration"]["sha256"] = sha(artifact_path)
            with self.assertRaisesRegex(subject.ContractError, "source_evidence"):
                subject.validate(value, root=root, require_complete=True)

            value = self.bundle(root)
            value["calibration"]["sample_count"] = 31
            with self.assertRaisesRegex(subject.ContractError, "metrics do not exactly match"):
                subject.validate(value, root=root, require_complete=True)

    def test_blocked_partial_bundle_verifies_blocker_receipt_without_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blocker = root / "freshness-summary.json"
            blocker.write_text(json.dumps({"source_aligned": 0, "candidates": 843}), encoding="utf-8")
            report = subject.validate({
                "schema": subject.SCHEMA,
                "status": "blocked",
                "production_authority": False,
                "blocker_code": "BLOCKED_ON_SOURCE_ALIGNED_METRIC_DEPTH_AND_INTER_FRAME_STABLE_POSE",
                "blocker_evidence": [{"path": blocker.name, "sha256": sha(blocker)}],
                "device_identity": {"device_id": "device-1"},
                "evidence_artifacts": {},
                "calibration": {},
                "frame_clock": {},
                "body_local_ground_truth": {},
                "route_event_truth": {},
                "target_device_benchmark": {},
            }, root=root, require_complete=False)
            self.assertFalse(report["device_metric_geometry_admitted"])
            self.assertEqual(
                "BLOCKED_ON_SOURCE_ALIGNED_METRIC_DEPTH_AND_INTER_FRAME_STABLE_POSE",
                report["declared_blocker_code"],
            )
            self.assertIn("MISSING_CALIBRATION", report["blockers"])

    def test_research_report_device_gate_consumes_raw_bundle_instead_of_hardcoded_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "device-evidence.json"
            manifest.write_text(json.dumps(self.bundle(root)), encoding="utf-8")
            passed, detail = research_report._device_metric_geometry_gate(manifest)
            self.assertTrue(passed)
            self.assertIn("geometry shadow only", detail)

            value = self.bundle(root)
            value["evidence_artifacts"]["calibration"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(subject.ContractError, "mismatched SHA256"):
                subject.validate(value, root=root, require_complete=True)

    @staticmethod
    def ai_review_receipt(root: Path, name: str, subject_id: str, input_sha: str) -> tuple[str, str]:
        path = root / f"{name}-ai-review.json"
        path.write_text(json.dumps({
            "schema": "blindassist_ai_review_consensus_v1",
            "subject_id": subject_id,
            "input_sha256": input_sha,
            "reviews": [
                {
                    "reviewer_id": f"gpt-{name}", "reviewer_type": "ai_model",
                    "reviewer_role": "gpt_multimodal_reviewer", "provider": "openai",
                    "model": "gpt-multimodal", "model_version": "2026-07-21",
                    "review_run_id": f"gpt-{name}-run", "workflow_id": "metric_geometry_review_v1",
                    "prompt_sha256": "1" * 64, "input_sha256": input_sha,
                    "isolated_context": True, "other_review_visible_before_submission": False,
                    "confidence": 0.91, "abstained": False, "abstain_reasons": [], "verdict": "accept",
                },
                {
                    "reviewer_id": f"codex-{name}", "reviewer_type": "ai_model",
                    "reviewer_role": "codex_evidence_reviewer", "provider": "openai",
                    "model": "codex", "model_version": "2026-07-21",
                    "review_run_id": f"codex-{name}-run", "workflow_id": "metric_geometry_review_v1",
                    "prompt_sha256": "2" * 64, "input_sha256": input_sha,
                    "isolated_context": True, "other_review_visible_before_submission": False,
                    "confidence": 0.89, "abstained": False, "abstain_reasons": [], "verdict": "accept",
                },
            ],
            "consensus": {"method": "model_consensus", "disposition": "accept"},
        }), encoding="utf-8")
        return path.name, sha(path)

    @staticmethod
    def bundle(root: Path) -> dict:
        calibration_review_path, calibration_review_sha = DeviceMetricGeometryValidatorTest.ai_review_receipt(
            root, "calibration", "cal-1", "a" * 64,
        )
        ground_review_path, ground_review_sha = DeviceMetricGeometryValidatorTest.ai_review_receipt(
            root, "body-ground", "body-local-ground:device-1:cal-1", "b" * 64,
        )
        value = {
            "schema": subject.SCHEMA,
            "status": "complete",
            "production_authority": False,
            "device_identity": {
                "device_id": "device-1",
                "hardware_revision": "hw-1",
                "mount_revision": "mount-1",
                "camera_frame": "camera-1",
                "body_frame": "body-1",
                "device_stage": "fixed_body_mount",
                "reused_phone_evidence": False,
            },
            "calibration": {
                "calibration_id": "cal-1",
                "camera_calibration_version": "camera-cal-1",
                "camera_frame": "camera-1",
                "body_frame": "body-1",
                "collector_id": "collector",
                "ai_review_receipt_path": calibration_review_path,
                "ai_review_receipt_sha256": calibration_review_sha,
                "sample_count": 30,
                "pose_coverage_bins": 5,
                "intrinsics_p95_reprojection_px": 1.5,
                "depth_registration_p95_error_m": 0.03,
                "mount_translation_repeatability_m": 0.01,
                "mount_rotation_repeatability_deg": 1.0,
            },
            "frame_clock": {
                "camera_frame": "camera-1",
                "capture_timestamps_strictly_monotonic": True,
                "pose_reference_mode": "INTER_FRAME_STABLE",
                "source_aligned_metric_depth_pair_count": 100,
                "source_aligned_metric_depth_fraction": 0.95,
                "maximum_cross_sensor_sync_error_ms": 20.0,
            },
            "body_local_ground_truth": {
                "body_frame": "body-1",
                "collector_id": "collector",
                "ai_review_receipt_path": ground_review_path,
                "ai_review_receipt_sha256": ground_review_sha,
                "sample_count": 30,
                "p95_plane_distance_error_m": 0.03,
                "clear_obstacle_head_drop_and_missing_depth_covered": True,
            },
            "route_event_truth": {
                "route_conditioned_truth_eligible": True,
                "episode_count": 120,
                "matched_pair_count": 60,
            },
            "target_device_benchmark": {
                "device_id": "device-1",
                "mount_revision": "mount-1",
                "calibration_id": "cal-1",
                "failure_count": 0,
                "pipeline_p95_ms": 70.0,
                "stale_output_count": 0,
                "latest_only_queue_verified": True,
                "thermal_throttle_count": 0,
            },
        }
        binding = subject.artifact_binding(value["device_identity"], value["calibration"])
        artifacts = {}
        for name in subject.REQUIRED_ARTIFACTS:
            source_path = root / f"{name}-source.json"
            source_path.write_text(json.dumps({"raw_or_gate_evidence": name}), encoding="utf-8")
            path = root / f"{name}.json"
            path.write_text(json.dumps({
                "schema": subject.ARTIFACT_SCHEMAS[name],
                "artifact_kind": name,
                "bundle_binding": binding,
                "metrics": value[name],
                "source_evidence": [{"path": source_path.name, "sha256": sha(source_path)}],
            }), encoding="utf-8")
            artifacts[name] = {"path": path.name, "sha256": sha(path)}
        value["evidence_artifacts"] = artifacts
        return value


if __name__ == "__main__":
    unittest.main()

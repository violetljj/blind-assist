from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = load("pilot_validator", "validate_ustrf_sc_route_conditioned_event_pilot.py")
fixture = load("truth_fixture", "test_validate_route_conditioned_event_truth.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RouteConditionedEventPilotTest(unittest.TestCase):
    def test_valid_pilot_only_passes_collection_pipeline_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, manifest = self.package(root)
            report = subject.validate_pilot(config, manifest, root=root)
            self.assertTrue(report["pilot_collection_pipeline_audit_passed"])
            self.assertEqual(2, report["episode_count"])
            self.assertEqual(42, report["capture_frame_count"])
            for key in (
                "route_conditioned_truth_eligible", "u0_evaluation_eligible", "s0_probe_eligible",
                "training_eligible", "android_runtime_change_authorized", "production_model_replacement_authorized",
            ):
                self.assertFalse(report[key])

    def test_frame_ledger_tamper_and_timestamp_order_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, manifest = self.package(root)
            ledger_path = root / manifest["episodes"][0]["capture_frame_ledger_path"]
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["frames"][1]["frame_payload_sha256"] = "f" * 64
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            manifest["episodes"][0]["capture_frame_ledger_sha256"] = sha(ledger_path)
            with self.assertRaisesRegex(subject.ContractError, "capture_frame_ledger_sha256"):
                subject.validate_pilot(config, manifest, root=root)

            config, manifest = self.package(root)
            ledger_path = root / manifest["episodes"][0]["capture_frame_ledger_path"]
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["frames"][1]["capture_timestamp_ns"] = ledger["frames"][0]["capture_timestamp_ns"]
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            row = manifest["episodes"][0]
            ledger_sha = sha(ledger_path)
            row["capture_frame_ledger_sha256"] = ledger_sha
            for stem in ("capture_clock_receipt", "route_intent"):
                path = root / row[f"{stem}_path"]
                value = json.loads(path.read_text(encoding="utf-8"))
                value["capture_frame_ledger_sha256"] = ledger_sha
                path.write_text(json.dumps(value), encoding="utf-8")
                row[f"{stem}_sha256"] = sha(path)
            with self.assertRaisesRegex(subject.ContractError, "strictly increasing"):
                subject.validate_pilot(config, manifest, root=root)

            config, manifest = self.package(root)
            row = manifest["episodes"][0]
            ledger_path = root / row["capture_frame_ledger_path"]
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["frames"][1]["capture_timestamp_ns"] += 50_000_000
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            ledger_sha = sha(ledger_path)
            row["capture_frame_ledger_sha256"] = ledger_sha
            for stem in ("capture_clock_receipt", "route_intent"):
                path = root / row[f"{stem}_path"]
                value = json.loads(path.read_text(encoding="utf-8"))
                value["capture_frame_ledger_sha256"] = ledger_sha
                path.write_text(json.dumps(value), encoding="utf-8")
                row[f"{stem}_sha256"] = sha(path)
            with self.assertRaisesRegex(subject.ContractError, "clock and video PTS drift"):
                subject.validate_pilot(config, manifest, root=root)

    def test_wrong_scope_matrix_or_truth_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, manifest = self.package(root)
            manifest["pilot_authority"] = "truth"
            with self.assertRaisesRegex(subject.ContractError, "authority mismatch"):
                subject.validate_pilot(config, manifest, root=root)

            config, manifest = self.package(root)
            manifest["route_conditioned_truth_eligible"] = True
            with self.assertRaisesRegex(subject.ContractError, "route_conditioned_truth_eligible=false"):
                subject.validate_pilot(config, manifest, root=root)

            config, manifest = self.package(root)
            manifest["episodes"][1]["scene_id"] = "outside-pilot"
            with self.assertRaises(subject.ContractError):
                subject.validate_pilot(config, manifest, root=root)

    def test_review_disagreement_requires_third_independent_adjudicator(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, manifest = self.package(root)
            row = manifest["episodes"][0]
            review_path = root / row["independent_review_paths"][1]
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["alertable_start_ms"] += 1000
            review_path.write_text(json.dumps(review), encoding="utf-8")
            row["independent_review_sha256s"][1] = sha(review_path)
            adjudication_path = root / row["adjudication_evidence_path"]
            adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
            adjudication["input_review_sha256s"] = row["independent_review_sha256s"]
            adjudication_path.write_text(json.dumps(adjudication), encoding="utf-8")
            row["adjudication_evidence_sha256"] = sha(adjudication_path)
            with self.assertRaisesRegex(subject.ContractError, "requires an independent human adjudicator"):
                subject.validate_pilot(config, manifest, root=root)

    def test_pilot_origin_cannot_be_relabelled_as_official_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, manifest = self.package(root)
            manifest.update({
                "schema": "blindassist_sanpo_counterfactual_episode_manifest_v1",
                "contract_id": config["contract_id"],
                "collection_scope": "official_full_matrix",
                "collection_status": "complete",
                "pilot": False,
            })
            with self.assertRaisesRegex(ValueError, "origin_scope"):
                subject._BASE.validate(config, manifest, root=root, require_complete=True)

    @classmethod
    def package(cls, root: Path) -> tuple[dict, dict]:
        config = fixture.RouteConditionedEventTruthTest.config()
        config["collection_scope"] = "official_full_matrix"
        config["design"]["pilot_before_full_matrix"] = {
            "contract_id": "test-route-conditioned-pilot-v1",
            "origin_scope": "pipeline_audit_pilot_capture",
            "session_count": 1,
            "matched_pairs_per_scene": 1,
            "episode_count": 2,
            "authority": "collection-pipeline-audit-only",
        }
        config["episode_record_schema"]["required_fields"].extend([
            "origin_scope",
            "capture_frame_ledger_path", "capture_frame_ledger_sha256",
            "independent_review_paths", "independent_review_sha256s",
            "adjudication_evidence_path", "adjudication_evidence_sha256",
        ])
        config["capture_frame_ledger_policy"] = {
            "ledger_schema": "blindassist_capture_frame_ledger_v1",
            "clock_receipt_schema": "blindassist_capture_clock_receipt_v1",
            "maximum_capture_gap_ns": 600000000,
            "maximum_clock_pts_alignment_error_ns": 20000000,
        }
        config["independent_human_review_policy"] = {
            "review_schema": "blindassist_independent_human_event_review_v1",
            "adjudication_schema": "blindassist_independent_human_event_adjudication_v1",
            "required_independent_review_count": 2,
            "anchor_agreement_tolerance_ms": 500,
        }
        manifest = fixture.RouteConditionedEventTruthTest.manifest(root)
        manifest.update({
            "schema": subject.PILOT_MANIFEST_SCHEMA,
            "contract_id": "test-route-conditioned-pilot-v1",
            "source_truth_contract_id": "test-route-conditioned-truth-v1",
            "collection_scope": subject.PILOT_SCOPE,
            "collection_status": "pilot_complete",
            "pilot_authority": subject.PILOT_AUTHORITY,
            "route_conditioned_truth_eligible": False,
            "u0_evaluation_eligible": False,
            "s0_probe_eligible": False,
            "training_eligible": False,
            "android_runtime_change_authorized": False,
            "production_model_replacement_authorized": False,
        })
        config["source_receipt_schema"]["required_origin_scope"] = "official_full_matrix_capture"
        for receipt in manifest["source_receipts"]:
            receipt["origin_scope"] = "pipeline_audit_pilot_capture"
        for index, row in enumerate(manifest["episodes"]):
            row["origin_scope"] = "pipeline_audit_pilot_capture"
            row["capture_context"].update({
                "calibration_id": "calibration-v1",
                "route_choice": "straight-through",
            })
            ledger_path = root / f"{row['episode_id']}-frames.json"
            clock_path = root / row["capture_clock_receipt_path"]
            route_path = root / row["route_intent_path"]
            clock = json.loads(clock_path.read_text(encoding="utf-8"))
            route = json.loads(route_path.read_text(encoding="utf-8"))
            receipt_id = f"clock-{row['episode_id']}"
            frames = [
                {
                    "frame_id": f"{row['episode_id']}-f{frame_index:02d}",
                    "frame_index": frame_index,
                    "capture_timestamp_ns": 1_000_000_000 + frame_index * 500_000_000 + index,
                    "video_pts_ms": frame_index * 500,
                    "episode_time_ms": frame_index * 500,
                    "camera_frame": "camera-v1",
                    "dropped": False,
                    "duplicate": False,
                    "frame_payload_sha256": f"{frame_index % 16:x}" * 64,
                }
                for frame_index in range(21)
            ]
            ledger = {
                "schema": "blindassist_capture_frame_ledger_v1",
                "episode_id": row["episode_id"],
                "source_video_sha256": row["video_sha256"],
                "clock_receipt_id": receipt_id,
                "camera_frame": "camera-v1",
                "frames": frames,
            }
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            ledger_sha = sha(ledger_path)
            clock.update({
                "receipt_id": receipt_id,
                "video_sha256": row["video_sha256"],
                "capture_frame_ledger_sha256": ledger_sha,
                "clock_domain": "camera-monotonic",
                "clock_source": "camera2-sensor-timestamp",
                "frame_count": len(frames),
                "decoded_frame_count": len(frames),
                "first_capture_timestamp_ns": frames[0]["capture_timestamp_ns"],
                "last_capture_timestamp_ns": frames[-1]["capture_timestamp_ns"],
                "first_video_pts_ms": 0,
                "last_video_pts_ms": 10000,
                "maximum_capture_gap_ns": 500_000_000,
            })
            projection_path = root / f"{row['episode_id']}-projection.json"
            projection_path.write_text(json.dumps({
                "schema": "blindassist_route_projection_receipt_v1",
                "episode_id": row["episode_id"],
                "input_video_sha256": row["video_sha256"],
                "capture_frame_ledger_sha256": ledger_sha,
                "camera_frame": "camera-v1",
                "calibration_id": "calibration-v1",
            }), encoding="utf-8")
            route.update({
                "route_plan_id": "shared-straight-route-plan",
                "capture_frame_ledger_sha256": ledger_sha,
                "input_video_sha256": row["video_sha256"],
                "camera_frame": "camera-v1",
                "calibration_id": "calibration-v1",
            })
            route["provider"].update({
                "implementation_sha256": "d" * 64,
                "config_sha256": "e" * 64,
            })
            route["coordinate_contract"].update({
                "projection_receipt_path": projection_path.name,
                "projection_receipt_sha256": sha(projection_path),
            })
            for frame, sample in zip(frames, route["samples"]):
                sample.update({
                    "source_frame_id": frame["frame_id"],
                    "consuming_frame_id": frame["frame_id"],
                    "source_capture_timestamp_ns": frame["capture_timestamp_ns"],
                    "generated_at_timestamp_ns": frame["capture_timestamp_ns"],
                })
            clock_path.write_text(json.dumps(clock), encoding="utf-8")
            route_path.write_text(json.dumps(route), encoding="utf-8")
            row.update({
                "capture_frame_ledger_path": ledger_path.name,
                "capture_frame_ledger_sha256": ledger_sha,
                "capture_clock_receipt_sha256": sha(clock_path),
                "route_intent_sha256": sha(route_path),
            })
            review_paths = []
            review_hashes = []
            for reviewer_id in row["annotation_reviewer_ids"]:
                review_path = root / f"{row['episode_id']}-{reviewer_id}.json"
                review_path.write_text(json.dumps({
                    "schema": "blindassist_independent_human_event_review_v1",
                    "episode_id": row["episode_id"],
                    "reviewer_id": reviewer_id,
                    "reviewer_type": "human",
                    "model_assistance_used": False,
                    "other_review_visible_before_submission": False,
                    "should_alert": row["expected_should_alert"],
                    "critical": row["expected_critical"],
                    "criticality_reason": row["criticality_reason"],
                    "route_relation": row["risk_profile"]["corridor_relation"],
                    "first_visible_ms": row["first_visible_ms"],
                    "alertable_start_ms": row["alertable_start_ms"],
                    "passed_or_cleared_ms": row["passed_or_cleared_ms"],
                    "negative_reason": row.get("negative_reason"),
                }), encoding="utf-8")
                review_paths.append(review_path.name)
                review_hashes.append(sha(review_path))
            adjudication_path = root / f"{row['episode_id']}-independent-adjudication.json"
            adjudication_path.write_text(json.dumps({
                "schema": "blindassist_independent_human_event_adjudication_v1",
                "episode_id": row["episode_id"],
                "input_review_sha256s": review_hashes,
                "method": "reviewer_consensus",
                "should_alert": row["expected_should_alert"],
                "critical": row["expected_critical"],
                "criticality_reason": row["criticality_reason"],
                "first_visible_ms": row["first_visible_ms"],
                "alertable_start_ms": row["alertable_start_ms"],
                "passed_or_cleared_ms": row["passed_or_cleared_ms"],
                "negative_reason": row.get("negative_reason"),
            }), encoding="utf-8")
            row.update({
                "independent_review_paths": review_paths,
                "independent_review_sha256s": review_hashes,
                "adjudication_evidence_path": adjudication_path.name,
                "adjudication_evidence_sha256": sha(adjudication_path),
            })
        return config, manifest


if __name__ == "__main__":
    unittest.main()

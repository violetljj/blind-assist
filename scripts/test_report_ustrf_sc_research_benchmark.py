import importlib.util
import hashlib
import json
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("report_ustrf_sc_research_benchmark.py")
SPEC = importlib.util.spec_from_file_location("ustrf_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ResearchBenchmarkReportTest(unittest.TestCase):
    def test_passing_research_receipts_remain_device_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal])
            MODULE.write(report, root / "output")
            self.assertEqual("CONDITIONAL_RESEARCH_GO", report["decision"])
            self.assertEqual([True, True, True, False], [gate["passed"] for gate in report["gates"]])
            self.assertTrue((root / "output" / "research_benchmark_report.json").is_file())
            self.assertTrue((root / "output" / "research_benchmark_report.html").is_file())

    def test_source_native_dynamic_tracks_are_recorded_without_authorizing_seconds_ttc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; tracks = root / "tracks.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            tracks.write_text(json.dumps({"consecutive_track_pair_count": 10, "source_moving_pair_count": 3, "classification": {"precision": 1.0, "recall": 1.0}, "physical_ttc_seconds_admitted": False, "ustrf_motion_input_admitted": False}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], tracks)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_timestamped_source_ttc_gate_requires_time_and_remains_non_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; source_ttc = root / "source_ttc.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            source_ttc.write_text(json.dumps({"source_ttc_seconds_available": True, "front_facing_track_pair_count": 10, "timestamp": {"median_period_seconds": .1}, "kinematics": {"approaching_track_pair_count": 4, "ttc_collision_candidate_count_within_horizon": 1}, "ustrf_motion_input_admitted": False}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], None, source_ttc)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_timestamped_rgbd_pose_sequence_remains_source_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; rgbd = root / "rgbd.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            rgbd.write_text(json.dumps({"ok": True, "source_rgbd_pose_sequence_admitted": True, "frame_count": 2, "timestamp": {"median_interval_seconds": .02}, "depth": {"minimum_m": 1.0, "maximum_m": 20.0}, "camera_coordinate_handedness": "left_handed_or_image_reflected", "ustrf_metric_geometry_input_admitted": False}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], None, None, rgbd)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_body_capsule_corridor_gate_requires_cuda_truth_and_kotlin_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; corridor = root / "corridor.json"; replay = root / "replay.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            corridor.write_text(json.dumps({"scene_count": 256, "body_frame_ground_truth": True, "local_ground_truth": True, "dynamic_event_truth": True, "expected_clear_stop_count": 0}), encoding="utf-8")
            replay.write_text(json.dumps({"scene_count": 256, "expected_stop_count": 59, "actual_stop_count": 59, "clear_stop_count": 0, "action_match_count": 256, "eligible_corridor_selection_count": 240, "matching_corridor_selection_count": 240, "fault_scene_count": 16, "fault_stop_count": 16}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], None, None, None, corridor, replay)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_dynamic_rgbd_pose_source_gate_remains_non_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; rgbd = root / "rgbd.json"; reprojection = root / "reprojection.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            rgbd.write_text(json.dumps({"audit_passed": True, "frame_count": 100, "rgb_depth_association": {"within_20ms_fraction": 1.0}, "rgb_pose_association": {"within_20ms_fraction": .998}, "depth": {"valid_fraction": .86}, "ustrf_metric_geometry_input_admitted": False}), encoding="utf-8")
            reprojection.write_text(json.dumps({"aggregate": {"median_valid_projection_fraction": .84, "median_pair_median_abs_depth_residual_m": .013}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], None, None, None, None, None, rgbd, reprojection)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_dynamic_2d_labels_remain_source_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; labels = root / "labels.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            labels.write_text(json.dumps({"pairing": {"paired_frames": 1200, "image_without_label": 0, "label_without_image": 0}, "labels": {"annotated_boxes": 2000, "valid_normalized_box_fraction": 1.0}, "temporal": {"median_frame_rate_hz": 23.0}, "admission": {"external_2d_dynamic_object_truth_admitted": True, "not_admitted_for": ["physical TTC"]}}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], revel_rgb_labels_audit=labels)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_metric_person_sensor_trajectories_remain_source_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; vicon = root / "vicon.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            person = {"valid_nonorigin_pose_count": 20000, "relative_to_sensor": {"synchronized_valid_pose_fraction": .95, "continuity_filtered_relative_pair_count": 20000, "sensor_local_range_m": {"min": 1.0, "median": 3.0}}}
            vicon.write_text(json.dumps({"source": {"world_frame": ["/vicon/world"]}, "helmet_people": {"green": person, "yellow": person}, "admission": {"external_metric_person_sensor_trajectory_truth_admitted": True, "not_admitted_for": ["physical assistive TTC"]}}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], revel_vicon_trajectory_audit=vicon)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_rgb_vicon_cross_modal_alignment_remains_source_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; reprojection = root / "reprojection.json"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            item = {"labelled_frame_count": 6000, "valid_sensor_and_person_sync_count": 4000, "inside_any_matching_box_fraction_of_usable": .90, "closest_matching_box_outside_distance_px": {"p95": 2.0}, "ambiguous_same_class_frame_count": 1}
            reprojection.write_text(json.dumps({"frame_alignment": {"archive_image_count": 8580, "bag_image_count": 8580}, "reprojection": {"green": item, "yellow": item}, "admission": {"source_cross_modal_2d_3d_alignment_admitted": True, "not_admitted_for": ["physical assistive TTC"]}}), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], revel_rgb_vicon_reprojection_audit=reprojection)
            self.assertEqual(5, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertFalse(report["gates"][4]["passed"])

    def test_bounded_detector_measurement_requires_guard_and_remains_non_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.json"; dynamic = root / "dynamic.json"; temporal = root / "temporal.json"; detector = root / "detector.json"; guard = root / "guard.json"; alignment = root / "alignment.json"
            details = root / "details.jsonl"; motion_details = root / "motion-details.jsonl"
            geometry.write_text(json.dumps({"ok": True, "static_reprojection_rmse_meters": 0.0, "visibility_gap_false_drop_count": 0, "drop_expected_sample_count": 2, "drop_detected_sample_count": 2}), encoding="utf-8")
            dynamic.write_text(json.dumps({"max_velocity_error_mps": 0.0, "max_ttc_error_ms": 1.0, "collision_label_accuracy": 1.0, "rejected_sequence_count": 2}), encoding="utf-8")
            temporal.write_text(json.dumps({"trajectory": "fixture", "aggregate": {"median_valid_projection_fraction": .97, "median_pair_median_abs_depth_residual_m": .004}, "ustrf_geometry_input_admitted": False}), encoding="utf-8")
            details.write_text("".join(json.dumps({"selected_index": index}) + "\n" for index in range(512)), encoding="utf-8")
            details_sha256 = hashlib.sha256(details.read_bytes()).hexdigest()
            motion_details.write_text("".join(json.dumps({"box": index}) + "\n" for index in range(770)), encoding="utf-8")
            motion_details_sha256 = hashlib.sha256(motion_details.read_bytes()).hexdigest()
            detector.write_text(json.dumps({
                "format": "blindassist_revel_yolo11n_person_benchmark_v2",
                "dataset": {"total_frames": 8580, "evaluated_frames": 512, "selection": "uniform", "selected_first_index": 0, "selected_last_index": 8579, "person_ground_truth_boxes": 770},
                "model": {"batch": 1, "half": False},
                "fixed_score_metrics": {"precision": .83, "recall": .89, "f1": .86},
                "ap50_over_score_floor": .93,
                "recall_by_normalized_box_area": {"small": {"ground_truth": 37, "recall": .24}, "medium": {"ground_truth": 354, "recall": .88}, "large": {"ground_truth": 379, "recall": .96}},
                "details_receipt": {"path": str(details), "frame_records": 512, "sha256": details_sha256},
                "admission": {"offline_rgb_person_detection_baseline_admitted": True},
                "production_authority": False,
            }), encoding="utf-8")
            guard.write_text(json.dumps({
                "format": "blindassist_guarded_gpu_run_v1", "exit_code": 0, "stop_reason": None, "monitor_samples": 100,
                "limits": {"max_temperature_c": 72, "max_frames": 512, "batch": 1},
                "observed": {"max_temperature_c": 49, "max_power_draw_w": 36, "relevant_system_events": 0},
            }), encoding="utf-8")
            alignment.write_text(json.dumps({
                "format": "blindassist_revel_detector_vicon_failure_alignment_v2",
                "source": {
                    "details_sha256": details_sha256,
                    "bag_sha256_from_audit": "bag-sha",
                    "bag_sha256_verified": "bag-sha",
                    "box_records_receipt": {"path": str(motion_details), "records": 770, "sha256": motion_details_sha256},
                    "motion_contract": {"timestamp_basis": "rosbag record time", "minimum_continuous_interval_s": .005, "maximum_continuous_interval_s": .05, "maximum_single_track_world_speed_mps": 5.0, "approach_recede_deadband_mps": .10, "offline_noncausal": True},
                },
                "summary": {
                    "box_count": 770,
                    "vicon_aligned_box_count": 502,
                    "document_range_summary": {"within_0_5m": {"ground_truth": 448, "recall": .94, "recall_wilson95": [.91, .96]}, "beyond_5m": {"ground_truth": 54, "recall": .72, "recall_wilson95": [.59, .82]}},
                    "source_motion_aligned_box_count": 500,
                    "recall_by_source_radial_motion": {"approaching": {"ground_truth": 200, "recall": .80}, "quasi_static": {"ground_truth": 100, "recall": .90}, "receding": {"ground_truth": 200, "recall": .95}},
                    "recall_by_source_ttc_proxy": {"0-1s": {"ground_truth": 20}, "1-2s": {"ground_truth": 30}, "2-3s": {"ground_truth": 50}, "3s+": {"ground_truth": 100}},
                    "document_motion_summary": {"ttc_proxy_within_3s": {"ground_truth": 100, "recall": .70}},
                },
                "admission": {"source_detector_range_stratification_admitted": True, "source_detector_radial_motion_stratification_admitted": True, "not_admitted_for": ["physical assistive TTC"]},
                "production_authority": False,
            }), encoding="utf-8")
            report = MODULE.build(geometry, dynamic, [temporal], revel_detector_benchmark=detector, revel_detector_guard=guard, revel_detector_vicon_alignment=alignment)
            self.assertEqual(7, len(report["gates"]))
            self.assertTrue(report["gates"][3]["passed"])
            self.assertIn("small/medium/large recall=0.24/0.88/0.96", report["gates"][3]["detail"])
            self.assertTrue(report["gates"][4]["passed"])
            self.assertIn("0-5m recall=0.94", report["gates"][4]["detail"])
            self.assertTrue(report["gates"][5]["passed"])
            self.assertIn("motion-aligned=500/770", report["gates"][5]["detail"])
            self.assertFalse(report["gates"][6]["passed"])


if __name__ == "__main__":
    unittest.main()

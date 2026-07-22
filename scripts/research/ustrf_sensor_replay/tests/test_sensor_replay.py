from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODULE))
from contract import nearest_pose, quaternion_matrix, safe_file, validate_pose
from finalize_reviews import _rows
from finalize_r3_reviews import build_consensus_events, validate_review
from lilocbench_calibration import (
    compose_transform_chain,
    parse_depth_to_color_yaml,
    parse_intrinsics_yaml,
    parse_transformations_yaml,
    register_depth_to_color,
    transform_matrix,
    validate_front_color_optical,
    world_from_color_optical,
)
from normalize_sources import _associate_nearest, _lilocbench_package, _openloris_camera_pose
from prepare_lilocbench_rgbd import _associate_sorted, sanitize_raw_depth
from prepare_r3_review_bundle import PROMPT as R3_REVIEW_PROMPT
from prescreen_lilocbench_sources import build_report as build_lilocbench_report
from prescreen_openloris_sources import build_report, trajectory_stats
from run_replay import _rotation_error_deg


class ContractTest(unittest.TestCase):
    def test_quaternion_pose_is_valid(self):
        pose=quaternion_matrix([1,2,3,0,0,0,1]); validate_pose(pose); self.assertEqual(pose[0][3],1)

    def test_nearest_pose(self):
        stamp,values=nearest_pose([(0.0,[0]*7),(1.0,[1]*7)],.6); self.assertEqual(stamp,1.0); self.assertEqual(values[0],1)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"ok").write_text("x",encoding="utf-8")
            self.assertEqual(safe_file(root,"ok"),root.resolve()/"ok")
            with self.assertRaises(ValueError): safe_file(root,"../outside")

    def test_left_handed_pose_rejected(self):
        pose=np.eye(4); pose[0,0]=-1
        with self.assertRaises(ValueError): validate_pose(pose.tolist())

    def test_review_row_shapes(self):
        self.assertEqual(_rows({"sources":[{"source_id":"s"}]})[0]["source_id"],"s")
        self.assertEqual(_rows({"reviews":[{"source_id":"s"}]})[0]["source_id"],"s")

    def test_rotation_error(self):
        self.assertAlmostEqual(_rotation_error_deg(np.eye(3),np.eye(3)),0.0)

    def test_tum_association_is_one_to_one(self):
        first=[(0.0,"a"),(.03,"b")]; second=[(.01,"x"),(.04,"y")]
        self.assertEqual(_associate_nearest(first,second,.02),[(0.0,"a",.01,"x"),(.03,"b",.04,"y")])

    def test_r3_review_rejects_candidate_visibility(self):
        value={"schema":"blindassist_ustrf_sensor_replay_r3_review_v1","reviewer_role":"a","independent_review":True,"other_reviewer_outputs_viewed":False,"candidate_alerts_viewed":True,"sources":[]}
        with self.assertRaisesRegex(ValueError,"not isolated"):
            validate_review(value,"a",{})

    def test_r3_review_requires_manifest_binding(self):
        value={"schema":"blindassist_ustrf_sensor_replay_r3_review_v1","reviewer_role":"a","independent_review":True,"other_reviewer_outputs_viewed":False,"candidate_alerts_viewed":False,"sources":[{"source_id":"s","manifest_sha256":"bad","route_valid":False,"events":[]}]}
        with self.assertRaisesRegex(ValueError,"manifest binding"):
            validate_review(value,"a",{"s":"good"})

    def test_r3_review_prompt_requires_governance_bindings(self):
        for field in ("reviewer_type", "workflow_id", "prompt_sha256", "input_sha256", "isolated_context", "candidate_output_visible"):
            self.assertIn(field, R3_REVIEW_PROMPT)

    def test_r3_review_event_consensus_keeps_tolerance_and_flags_count_disagreement(self):
        first={"events":[{"onset_frame":10,"alertable_frame":12,"passed_or_cleared_frame":20,"end_frame":22,"critical":True}]}
        second={"events":[{"onset_frame":12,"alertable_frame":14,"passed_or_cleared_frame":21,"end_frame":24,"critical":True}]}
        events,disagreement=build_consensus_events(first,second,15,100)
        self.assertIsNone(disagreement)
        self.assertEqual(events[0]["onset_frame"],11)
        second["events"].append(dict(second["events"][0]))
        events,disagreement=build_consensus_events(first,second,15,100)
        self.assertEqual(events,[])
        self.assertEqual(disagreement,"event count disagreement")

    def test_openloris_ground_truth_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"groundtruth.txt"
            path.write_text("#Time px py pz qx qy qz qw\n0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n",encoding="utf-8")
            stats=trajectory_stats(path)
            self.assertEqual(stats["ground_truth_sample_count"],2)
            self.assertAlmostEqual(stats["path_length_m"],1.0)

    def test_openloris_marker_pose_is_transformed_to_color_camera(self):
        world_to_marker=np.eye(4); world_to_marker[0,3]=10.0
        base_to_marker=np.eye(4); base_to_marker[0,3]=2.0
        base_to_color=np.eye(4); base_to_color[0,3]=1.0
        camera=_openloris_camera_pose(world_to_marker,base_to_color,base_to_marker,"marker")
        self.assertAlmostEqual(camera[0,3],9.0)

    def test_openloris_base_pose_is_transformed_to_color_camera(self):
        world_to_base=np.eye(4); world_to_base[0,3]=10.0
        base_to_color=np.eye(4); base_to_color[0,3]=1.0
        camera=_openloris_camera_pose(world_to_base,base_to_color,None,"base_link")
        self.assertAlmostEqual(camera[0,3],11.0)

    def test_openloris_prescreen_never_grants_source_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); repo=root/"repo"; repo.mkdir()
            prereg=repo/"prereg.json"; prereg.write_text("{}\n",encoding="utf-8")
            import hashlib, json
            prereg_sha=hashlib.sha256(prereg.read_bytes()).hexdigest()
            archive=root/"groundtruth.zip"; archive.write_bytes(b"groundtruth")
            archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
            truth=root/"truth"
            candidates=[]
            for index in range(3):
                trajectory_id=f"office1-{index+1}"
                target=truth/trajectory_id; target.mkdir(parents=True)
                (target/"groundtruth.txt").write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n",encoding="utf-8")
                candidates.append({"trajectory_id":trajectory_id,"scene":"office","priority":1})
            config={"frozen_r3_prereg":{"path":"prereg.json","sha256":prereg_sha},"complete_sequence_two_model_review":{"anchor_tolerance_frames":15,"minimum_admitted_trajectories_before_source_count_credit":3},"dataset":{"groundtruth_archive":{"lfs_sha256":archive_sha}},"archives":[{"scene":"office","path":"office.tar","lfs_sha256":"a","trajectory_authority":"mocap"}],"trajectory_candidates":candidates}
            config_path=root/"config.json"; config_path.write_text(json.dumps(config),encoding="utf-8")
            report=build_report(repo,config_path,truth,archive)
            self.assertEqual(report["candidate_trajectory_count"],3)
            self.assertEqual(report["admitted_trajectory_count"],0)
            self.assertFalse(report["three_source_count_credit"])
            self.assertTrue(all(row["source_count_credit"] == 0 for row in report["candidates"]))

    def test_lilocbench_prescreen_preserves_rgb_frame_time_semantics_and_allows_public_download(self):
        with tempfile.TemporaryDirectory() as directory:
            import hashlib, json
            root=Path(directory); repo=root/"repo"; repo.mkdir()
            prereg_value={
                "synchronization":{"maximum_rgb_pose_delta_ms":40.0,"minimum_source_aligned_fraction":0.95},
                "route":{"truth_horizon_frames":24,"causal_history_frames":12,"minimum_forward_displacement_m":0.03,"maximum_unknown_rate":0.50},
            }
            prereg=repo/"prereg.json"; prereg.write_text(json.dumps(prereg_value)+"\n",encoding="utf-8")
            prereg_sha=hashlib.sha256(prereg.read_bytes()).hexdigest()
            truth=root/"gt_poses.txt"
            truth.write_text("\n".join(
                f"{index/20:.2f} {index/20:.4f} 0 0 0 0 0 1"
                for index in range(81)
            )+"\n",encoding="utf-8")
            truth_sha=hashlib.sha256(truth.read_bytes()).hexdigest()
            config={
                "frozen_r3_prereg":{"path":"prereg.json","sha256":prereg_sha},
                "complete_sequence_two_model_review":{"anchor_tolerance_frames":15,"minimum_admitted_trajectories_before_source_count_credit":3},
                "dataset":{"official_page_url":"https://example.test/dataset","ground_truth_frame":"base_link","data_rights":{"status":"unverified","official_direct_download":True,"source_admission_rights_gate_passed":False,"full_rgbd_download_authorized":False}},
                "sequence":{"sequence_id":"dynamics_0","ground_truth":{"url":"https://example.test/gt","size_bytes":truth.stat().st_size,"sha256":truth_sha,"http_etag":"test","http_last_modified":"test"},"individual_files_rgbd_archive":{"downloaded":False}},
                "gt_only_route_prescreen":{"rgb_frame_rate_hz":15.0},
            }
            config_path=root/"config.json"; config_path.write_text(json.dumps(config),encoding="utf-8")
            report=build_lilocbench_report(repo,config_path,truth)
            self.assertEqual(report["time_semantics"]["truth_horizon_seconds"],1.6)
            self.assertEqual(report["time_semantics"]["causal_history_seconds"],0.8)
            self.assertAlmostEqual(report["ground_truth"]["maximum_gap_s"],0.05)
            self.assertTrue(report["gt_route_prescreen_passed"])
            self.assertTrue(report["ordinary_public_download"])
            self.assertTrue(report["full_rgbd_download_authorized"])
            self.assertEqual(report["source_count_credit"],0)
            self.assertFalse(report["evaluator_ran"])
            config["dataset"]["data_rights"]["official_direct_download"]=False
            config_path.write_text(json.dumps(config),encoding="utf-8")
            private_report=build_lilocbench_report(repo,config_path,truth)
            self.assertFalse(private_report["ordinary_public_download"])
            self.assertFalse(private_report["full_rgbd_download_authorized"])
            self.assertFalse(private_report["source_admission_rights_gate_passed"])

    def test_lilocbench_front_color_optical_chain_has_forward_axis_and_camera_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"transformations.yaml"
            path.write_text("""
- parent: base_link
  child: camera_front_link
  translation:
    x: 0.0945807064
    y: 0.0370990463
    z: 0.6601355405
  rotation:
    x: 0.0041801186
    y: 0.0201761720
    z: -0.0004593315
    w: 0.9997875963
- parent: camera_front_link
  child: camera_front_color_frame
  translation:
    x: -0.0001586774
    y: -0.0590095557
    z: 0.0000811873
  rotation:
    x: -0.0001419995
    y: 0.0003050308
    z: 0.0024367822
    w: 0.9999969602
- parent: camera_front_color_frame
  child: camera_front_color_optical_frame
  translation:
    x: 0
    y: 0
    z: 0
  rotation:
    x: -0.5
    y: 0.5
    z: -0.5
    w: 0.5
""".strip()+"\n",encoding="utf-8")
            entries=parse_transformations_yaml(path)
            base_from_color=compose_transform_chain(entries,["base_link","camera_front_link","camera_front_color_frame","camera_front_color_optical_frame"])
            axis=validate_front_color_optical(base_from_color)
            np.testing.assert_allclose(axis,[0.999154,0.004129,-0.040917],atol=2e-6)
            world_from_base=np.eye(4); world_from_base[0,3]=10.0
            world_from_camera=world_from_color_optical(world_from_base,base_from_color)
            np.testing.assert_allclose(world_from_camera,world_from_base@base_from_color,atol=1e-12)

    def test_lilocbench_intrinsics_and_depth_to_color_identity_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"intrinsics.yaml"
            path.write_text("""
height: 3
width: 3
distortion_model: plumb_bob
distortion_coefficients:
- 0
- 0
- 0
- 0
- 0
K:
- 1
- 0
- 0
- 0
- 1
- 0
- 0
- 0
- 1
R:
- 1
- 0
- 0
- 0
- 1
- 0
- 0
- 0
- 1
P:
- 1
- 0
- 0
- 0
- 0
- 1
- 0
- 0
- 0
- 0
- 1
- 0
binning_x: 0
binning_y: 0
""".strip()+"\n",encoding="utf-8")
            calibration=parse_intrinsics_yaml(path)
            raw=np.asarray([[1000,0,0],[0,2000,0],[0,0,3000]],dtype=np.uint16)
            aligned=register_depth_to_color(raw,1000.0,calibration,calibration,np.eye(4))
            np.testing.assert_allclose(aligned,raw.astype(np.float32)/1000.0,atol=1e-6)

    def test_lilocbench_depth_to_color_direction_and_nearest_z(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            intrinsics=root/"intrinsics.yaml"
            intrinsics.write_text("""
height: 1
width: 3
distortion_model: plumb_bob
distortion_coefficients: [0, 0, 0, 0, 0]
K: [10, 0, 1, 0, 10, 0, 0, 0, 1]
R: [1, 0, 0, 0, 1, 0, 0, 0, 1]
P: [10, 0, 1, 0, 0, 10, 0, 0, 0, 0, 1, 0]
binning_x: 0
binning_y: 0
""".strip()+"\n",encoding="utf-8")
            calibration=parse_intrinsics_yaml(intrinsics)
            extrinsics=root/"extrinsics_depth_to_color.yaml"
            extrinsics.write_text("""
parent: color
child: depth
translation:
  x: -0.1
  y: 0
  z: 0
rotation:
  x: 0
  y: 0
  z: 0
  w: 1
""".strip()+"\n",encoding="utf-8")
            color_from_depth=transform_matrix(parse_depth_to_color_yaml(extrinsics))
            raw=np.asarray([[0,1000,0]],dtype=np.uint16)
            aligned=register_depth_to_color(raw,1000.0,calibration,calibration,color_from_depth)
            np.testing.assert_allclose(aligned,[[1.0,0.0,0.0]],atol=1e-6)

            collision=dict(calibration)
            collision["K"]=[0.1,0,0,0,1,0,0,0,1]
            collision["P"]=[0.1,0,0,0,0,1,0,0,0,0,1,0]
            raw_collision=np.asarray([[1000,2000,0]],dtype=np.uint16)
            collided=register_depth_to_color(raw_collision,1000.0,calibration,collision,np.eye(4))
            np.testing.assert_allclose(collided,[[1.0,0.0,0.0]],atol=1e-6)

    def test_lilocbench_calibration_rejects_wrong_direction_duplicates_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            wrong=root/"wrong.yaml"
            wrong.write_text("parent: depth\nchild: color\ntranslation:\n  x: 0\n  y: 0\n  z: 0\nrotation:\n  x: 0\n  y: 0\n  z: 0\n  w: 1\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"direction"):
                parse_depth_to_color_yaml(wrong)

            entry={"parent":"base","child":"camera","translation":{"x":0.0,"y":0.0,"z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0}}
            with self.assertRaisesRegex(ValueError,"duplicate"):
                compose_transform_chain([entry,dict(entry)],["base","camera"])
            nonfinite={**entry,"translation":{"x":float("nan"),"y":0.0,"z":0.0}}
            with self.assertRaisesRegex(ValueError,"non-finite"):
                transform_matrix(nonfinite)

            calibration={"height":1,"width":1,"distortion_model":"plumb_bob","distortion_coefficients":[0,0,0,0,0],"K":[1,0,0,0,1,0,0,0,1],"R":[1,0,0,0,1,0,0,0,1],"P":[1,0,0,0,0,1,0,0,0,0,1,0],"binning_x":0,"binning_y":0}
            with self.assertRaisesRegex(ValueError,"raster or scale"):
                register_depth_to_color(np.asarray([[1000]],dtype=np.uint16),float("nan"),calibration,calibration,np.eye(4))
            bad_transform=np.eye(4); bad_transform[0,3]=float("nan")
            with self.assertRaisesRegex(ValueError,"non-finite"):
                register_depth_to_color(np.asarray([[1000]],dtype=np.uint16),1000.0,calibration,calibration,bad_transform)

    def test_lilocbench_sorted_association_skips_unpaired_first_color(self):
        color=[(0.0,Path("c0")),(.1,Path("c1")),(.2,Path("c2"))]
        depth=[(.1001,Path("d1")),(.2001,Path("d2")),(.3001,Path("d3"))]
        pairs=_associate_sorted(color,depth,.02)
        self.assertEqual([(row[0],row[2]) for row in pairs],[(.1,.1001),(.2,.2001)])

    def test_lilocbench_saturated_depth_is_unknown_not_clipped(self):
        raw=np.asarray([[0,1000,65535]],dtype=np.uint16)
        sanitized,count=sanitize_raw_depth(raw)
        self.assertEqual(count,1)
        np.testing.assert_array_equal(sanitized,[[0,1000,0]])
        self.assertEqual(int(raw[0,2]),65535)

    def test_lilocbench_package_composes_base_ground_truth_into_color_optical_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            import json
            root=Path(directory)
            (root/"color/images").mkdir(parents=True)
            (root/"aligned_depth/images").mkdir(parents=True)
            (root/"calibration").mkdir()
            (root/"color/images/1.png").write_bytes(b"rgb")
            (root/"aligned_depth/images/1.png").write_bytes(b"depth")
            (root/"color.txt").write_text("1.000 color/images/1.png\n",encoding="utf-8")
            (root/"aligned_depth.txt").write_text("1.001 aligned_depth/images/1.png\n",encoding="utf-8")
            (root/"groundtruth.txt").write_text("1.000 2 0 0 0 0 0 1\n",encoding="utf-8")
            chain=["base_link","camera_front_color_optical_frame"]
            receipt_path=root/"preparation_receipt.json"
            receipt_path.write_text(json.dumps({"schema":"blindassist_ustrf_lilocbench_rgbd_preparation_v1","selected_camera":"camera_front","depth_registered_to_color":True,"frame_count":1,"depth_scale_units_per_meter":1000.0,"transform_chain":chain}),encoding="utf-8")
            (root/"calibration/color_intrinsics.yaml").write_text("""
height: 1
width: 1
distortion_model: plumb_bob
distortion_coefficients: [0, 0, 0, 0, 0]
K: [1, 0, 0, 0, 1, 0, 0, 0, 1]
R: [1, 0, 0, 0, 1, 0, 0, 0, 1]
P: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
binning_x: 0
binning_y: 0
""".strip()+"\n",encoding="utf-8")
            half=2 ** -0.5
            (root/"transformations.yaml").write_text(f"""
- parent: base_link
  child: camera_front_color_optical_frame
  translation:
    x: 1
    y: 0
    z: 0
  rotation:
    x: 0
    y: {half}
    z: 0
    w: {half}
""".strip()+"\n",encoding="utf-8")
            import hashlib
            source={"maximum_rgb_depth_delta_s":.02,"maximum_rgb_pose_delta_s":.04,"camera_transform_chain":chain,"depth_scale":1000.0,"pose_stability":"INTER_FRAME_STABLE","preparation_receipt_sha256":hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
            frames=_lilocbench_package(root,source,0)
            self.assertEqual(len(frames),1)
            self.assertAlmostEqual(frames[0]["camera_to_world"][0][3],3.0)
            self.assertEqual(frames[0]["intrinsics_fx_fy_cx_cy"],[1.0,1.0,0.0,0.0])


if __name__=="__main__": unittest.main()

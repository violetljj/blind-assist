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
from finalize_r3_reviews import validate_review
from normalize_sources import _associate_nearest, _openloris_camera_pose
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


if __name__=="__main__": unittest.main()

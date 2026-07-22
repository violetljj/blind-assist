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
from normalize_sources import _associate_nearest
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


if __name__=="__main__": unittest.main()

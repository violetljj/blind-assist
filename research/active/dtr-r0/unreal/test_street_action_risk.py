"""Analytic geometry and shared-action counterexamples; no simulator truth inputs."""
import copy
import math
import unittest

from street_action_risk import sweep_intervals, evaluate_candidates, Candidate, supported_tracks
from street_live_policy import MotionPolicy


def square(x, y, half=0.10):
    return [[x-half, y-half], [x+half, y-half], [x+half, y+half], [x-half, y+half]]


def footprint(x, y, *, vx=0.0, vy=0.0, age=0.0, disposition="MEASURED"):
    return {"track_id": "observed-track", "footprint_xy": square(x,y),
            "velocity_forward_mps": vx, "velocity_right_mps": vy,
            "evidence_age_s": age, "disposition": disposition}


def corridors(front=None, valid=1.0):
    return {"clearance_m": {str(k):12.0 for k in (0.0,-.72,.72,-.52,.52)},
            "front_obstacle_m": front, "valid_fraction": valid}


class SweptGeometryTest(unittest.TestCase):
    def test_face_entry_exit_has_closed_form(self):
        intervals=sweep_intervals((0,0),(1,0),[[1,-.2],[2,-.2],[2,.2],[1,.2]],.3,3)
        self.assertEqual(len(intervals),1)
        self.assertAlmostEqual(intervals[0][0],.7)
        self.assertAlmostEqual(intervals[0][1],2.3)

    def test_rounded_corner_is_not_square_expansion(self):
        polygon=[[1,1],[2,1],[2,2],[1,2]]
        self.assertEqual(sweep_intervals((.72,.72),(0,0),polygon,.3,1),[])
        self.assertEqual(sweep_intervals((.8,.8),(0,0),polygon,.3,1),[[0.0,1]])

    def test_short_contact_between_time_grid_samples_is_retained(self):
        intervals=sweep_intervals((0,0),(1,0),square(.05,0,.005),.001,.2)
        self.assertAlmostEqual(intervals[0][0],.044)
        self.assertAlmostEqual(intervals[0][1],.056)

    def test_vertex_tangent_is_included(self):
        polygon=[[1,1],[2,1],[2,2],[1,2]]
        hit=sweep_intervals((0,.7),(1,0),polygon,.3,3)
        self.assertAlmostEqual(hit[0][0],1)
        self.assertAlmostEqual(hit[0][1],2)

    def test_translation_orientation_and_rotation_invariance(self):
        polygon=square(1.5,0,.2)
        original=sweep_intervals((0,0),(1,0),polygon,.3,3)
        transformed=[[4-p[1],2+p[0]] for p in reversed(polygon)]
        result=sweep_intervals((4,2),(0,1),transformed,.3,3)
        for actual,expected in zip(result[0],original[0]):
            self.assertAlmostEqual(actual,expected)

    def test_relative_motion_changes_collision_time(self):
        stationary={"time_s":1,"tracks":[footprint(1.5,0)]}
        approaching={"time_s":1,"tracks":[footprint(1.5,0,vx=-1)]}
        candidate=[Candidate("straight",1,0,0)]
        a=evaluate_candidates(candidate,stationary,t=1,x=0,y=0)
        b=evaluate_candidates(candidate,approaching,t=1,x=0,y=0)
        self.assertAlmostEqual(a['candidates'][0]['first_entry_s'],1.1)
        self.assertAlmostEqual(b['candidates'][0]['first_entry_s'],.55)


class CandidatePolicyTest(unittest.TestCase):
    def call(self, mode, *, tracks=(), frame_time=1, front=None, valid=1, raw=False):
        frame={'time_s':frame_time,'tracks':list(tracks)}
        before=copy.deepcopy(frame)
        result=MotionPolicy(mode).command(t=1,x=0,y=0,goal_x=8,dtr_risk=raw,
                                         corridors=corridors(front,valid),motion_frame=frame)
        self.assertEqual(frame,before)
        return result

    def test_same_action_set_but_dynamic_risk_changes_selected_motion(self):
        tracks=[footprint(1.5,0)]
        depth=self.call('CANDIDATE_DEPTH',tracks=tracks)
        dynamic=self.call('CANDIDATE_DTR',tracks=tracks)
        self.assertEqual(depth['candidate_evaluation'],dynamic['candidate_evaluation'])
        self.assertEqual(depth['action'],'WALK')
        self.assertFalse(depth['candidate_intervention'])
        self.assertTrue(dynamic['candidate_intervention'])
        chosen=next(c for c in dynamic['candidate_evaluation']['candidates']
                    if c['candidate']==dynamic['selected_candidate'])
        self.assertFalse(chosen['conflicts'])
        self.assertNotEqual(dynamic['vy_mps'],0)

    def test_scalar_alert_does_not_transfer_plan_credentials(self):
        result=self.call('CANDIDATE_DTR',tracks=[footprint(1.5,3)],raw=True)
        self.assertTrue(result['raw_dtr_route_risk'])
        self.assertFalse(result['dtr_route_risk'])
        self.assertFalse(result['candidate_intervention'])
        self.assertEqual(result['action'],'WALK')
        self.assertEqual(result['candidate_evaluation']['authority'],'UNISSUED_ACTION_HYPOTHESES')

    def test_missing_stale_and_malformed_evidence_remain_unknown(self):
        for tracks,stamp in (([],1),([footprint(1,0,age=.7)],1),([footprint(1,0)],2)):
            result=self.call('CANDIDATE_DTR',tracks=tracks,frame_time=stamp,raw=True)
            self.assertFalse(result['candidate_intervention'])
            self.assertTrue(all(c['state']=='UNKNOWN' for c in result['candidate_evaluation']['candidates']))
            self.assertEqual(result['candidate_evaluation']['global_observability'],'UNKNOWN')
        self.assertEqual(supported_tracks({'time_s':'invalid'},1)[0],[])

    def test_depth_imminent_stop_is_not_overridden(self):
        result=self.call('CANDIDATE_DTR',tracks=[footprint(0,0,vy=1)],front=.2)
        self.assertEqual(result['action'],'BRAKE_IMMINENT')
        self.assertEqual((result['vx_mps'],result['vy_mps']),(0,0))
        self.assertEqual(len(result['candidate_evaluation']['candidates']),1)

    def test_unknown_depth_retains_stop(self):
        result=self.call('CANDIDATE_DTR',tracks=[footprint(1.5,0)],valid=0)
        self.assertEqual(result['action'],'WAIT_UNKNOWN_DEPTH')

    def test_candidate_depth_matches_existing_depth_motion_over_sequence(self):
        old,new=MotionPolicy('DEPTH_ONLY'),MotionPolicy('CANDIDATE_DEPTH')
        for t,x,y,front in ((0,0,0,None),(.2,.2,0,2),(.4,.27,-.13,1.9),
                            (.6,.34,-.26,1.8),(4,3,-.71,None),(6,5,-.71,None)):
            args=dict(t=t,x=x,y=y,goal_x=8,dtr_risk=True,corridors=corridors(front))
            a,b=old.command(**args),new.command(**args)
            for key in ('action','vx_mps','vy_mps','target_y_m','risk'):
                self.assertEqual(a[key],b[key])


if __name__=='__main__':
    unittest.main()

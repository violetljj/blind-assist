import unittest

from cnh_route_source_capture import prevent_reentry, probe_radius_m


class ProbeCoverageTests(unittest.TestCase):
    def test_endpoint_neighborhood_enclosed_without_changing_gate(self):
        pose=lambda x,y=0: dict(x=x,y=y,z=1.)
        layout=dict(camera=pose(0), clips=[dict(poses=[pose(0),pose(3.9)])])
        self.assertAlmostEqual(probe_radius_m(layout),11.9)
        layout['clips'][0]['poses']=[pose(0),pose(.5)]
        self.assertAlmostEqual(probe_radius_m(layout),8.5)

    def test_candidates_and_all_clip_poses_contribute(self):
        pose=lambda x,y=0: dict(x=x,y=y,z=1.)
        layout=dict(camera=pose(0),clips=[dict(poses=[pose(0)])],
            candidates=[dict(camera=pose(3,4),clips=[dict(poses=[pose(0,6)])])])
        self.assertEqual(probe_radius_m(layout),14.)


class SlateReentryTests(unittest.TestCase):
    def test_nested_slate_pump_does_not_repeat_transition(self):
        transitions=[]
        @prevent_reentry
        def tick(delta):
            transitions.append('begin')
            tick(delta)
            transitions.append('end')
        tick(0)
        tick(0)
        self.assertEqual(transitions,['begin','end','begin','end'])

    def test_exception_releases_callback_guard(self):
        calls=[]
        @prevent_reentry
        def tick():
            calls.append(1)
            if len(calls)==1:
                raise ValueError('compile failed')
            return 'recovered'
        with self.assertRaises(ValueError):
            tick()
        self.assertEqual(tick(),'recovered')


if __name__=='__main__':
    unittest.main()

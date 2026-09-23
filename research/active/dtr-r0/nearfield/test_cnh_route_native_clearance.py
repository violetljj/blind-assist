import copy
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from cnh_route_native_clearance import evaluate


def manifest():
    p = dict(x=0., y=0., z=0., pitch=0., yaw=0., roll=0.)
    return dict(native_coverage_complete=True,
        native_entities=[dict(id='wall', bounds_min_m=[10,10,10], bounds_max_m=[11,11,11], bounds_conservative=True, deformation_bounded=True)],
        clips=[dict(id=kind, trajectory_model='piecewise_linear_fixed_orientation', poses=[copy.deepcopy(p),copy.deepcopy(p)]) for kind in ('centre','boundary','outside','removed')])


def box(d, lo, hi):
    d['native_entities'][0].update(bounds_min_m=lo,bounds_max_m=hi)


class ClearanceTest(unittest.TestCase):
    def test_clear_not_admission(self):
        r=evaluate(manifest()); self.assertEqual(r['status'],'PASS'); self.assertFalse(r['runtime_admission'])

    def test_large_origin_surface_bounds(self):
        d=manifest(); d['native_entities'][0]['actor_origin_m']=[10000]*3
        box(d,[1,-.1,-.1],[2,.1,.1]); self.assertEqual(evaluate(d)['status'],'FAIL')

    def test_head_and_sides(self):
        for y in (-.59,.59):
            d=manifest(); box(d,[1,y,.1],[1.1,y+.001,.15]); r=evaluate(d)
            self.assertEqual(r['status'],'FAIL'); self.assertTrue(any(c['region']=='HEAD' for c in r['conflicts']))

    def test_body(self):
        d=manifest(); box(d,[1,0,-.85],[1.1,.01,-.8])
        self.assertTrue(any(c['region']=='BODY' for c in evaluate(d)['conflicts']))

    def test_fourth_clip_trajectory_interior(self):
        d=manifest(); d['clips'][3]['poses'][0]['y']=-5; d['clips'][3]['poses'][1]['y']=5
        box(d,[1,2,0],[1.1,2.1,.1]); self.assertEqual({c['clip_id'] for c in evaluate(d)['conflicts']},{'removed'})

    def test_unknown_and_empty(self):
        d=manifest(); d['native_coverage_complete']=False
        r=evaluate(d); self.assertEqual(r['status'],'NOT_RUN'); self.assertTrue(r['broadphase_clear']); self.assertFalse(r['selected'])
        d=manifest(); d['native_entities']=[]; self.assertEqual(evaluate(d)['status'],'NOT_RUN')

    def test_unbounded_deformation(self):
        d=manifest(); d['native_entities'][0]['deformation_bounded']=False; self.assertFalse(evaluate(d)['selected'])

    def test_exact_margin(self):
        d=manifest(); box(d,[1,.75,0],[1.1,.85,.1]); self.assertEqual(evaluate(d)['status'],'FAIL')
        box(d,[1,.750001,0],[1.1,.85,.1]); self.assertEqual(evaluate(d)['status'],'PASS')

    def test_rotating_segment(self):
        d=manifest(); d['clips'][0]['poses'][1]['yaw']=90; self.assertEqual(evaluate(d)['status'],'NOT_RUN')

    def test_rotated_camera_basis(self):
        d=manifest()
        for clip in d['clips']:
            for pose in clip['poses']: pose['yaw']=90
        box(d,[-.1,1,-.1],[.1,2,.1]); self.assertEqual(evaluate(d)['status'],'FAIL')

    def test_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.json'; output=Path(folder)/'result.json'
            source.write_text(json.dumps(manifest()),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('cnh_route_native_clearance.py')),str(source),'--output',str(output)]
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
            original=output.read_bytes()
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(output.read_bytes(),original)

    def test_missing_clip(self):
        d=manifest(); d['clips'].pop(); self.assertEqual(evaluate(d)['status'],'NOT_RUN')

    def test_four_arbitrary_clip_names_do_not_replace_required_kinds(self):
        d=manifest(); d['clips'][-1]['id']='other'
        self.assertFalse(evaluate(d)['selected'])

    def test_bad_bounds_and_inserted_entity(self):
        d=manifest(); d['native_entities'][0]['bounds_min_m'][0]=float('nan')
        with self.assertRaises(ValueError): evaluate(d)
        d=manifest(); d['native_entities'][0]['kind']='inserted'
        with self.assertRaises(ValueError): evaluate(d)


if __name__=='__main__': unittest.main()

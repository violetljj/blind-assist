"""No-UE integration checks for regional route/fixture capture specifications.

RGB/depth reads are mocked: these tests prove index/path/route joins, not rendered
pixels, native pavement admission or successful sensor capture.
"""
import copy
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from city_field_fixtures import fixture_objects, validate_fixture
from run_city_field_collection import compile_region
import city_field_contract as contract


def mock_plan():
    regions, routes = [], []
    for n, split in enumerate(('train', 'dev', 'test')):
        region = f'region_{n}'
        regions.append(dict(region_id=region, split=split, bounds_xy_m=[n*100, 0, n*100+40, 40]))
        for j, scene in enumerate(('sidewalk', 'intersection', 'narrow_passage')):
            rid = f'{region}_route_{j}'
            yaw, floor = 37.+j*53, 2.3+n
            co,si = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            waypoints = [dict(camera=dict(x=n*100+k*co, y=j*10+k*si, z=floor+1.65,
                yaw=yaw, pitch=-3., roll=0.), floor_z_m=floor, route_distance_m=float(k)) for k in range(3)]
            routes.append(dict(route_id=rid, region_id=region, split=split, scene_type=scene,
                status='ADMITTED', waypoints=waypoints, pair_groups=[rid+'_fixture'],
                instance_ids=[rid+'__'+part for part in ('thin_pole','body_arm','head_crossbar','sign_panel')]))
    return dict(regions=regions, routes=routes)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.plan = mock_plan()
        self.template = dict(cases=[{'name': 'stale'}], native_targets=[{'target_id': 'stale'}],
                             suite_contract={'stale': True}, map_asset='/Game/TestCity')

    def test_three_regions_nine_routes_exact_case_coverage_and_no_mutation(self):
        original_plan, original_template = copy.deepcopy(self.plan), copy.deepcopy(self.template)
        names = []
        for region in self.plan['regions']:
            spec = compile_region(self.plan, self.template, region['region_id'])
            self.assertEqual(len(spec['cases']), 3*(3+5))
            self.assertEqual(len(spec['native_targets']), 3*4)
            self.assertEqual(spec['inventory_indices'], [0,8,16])
            self.assertTrue(spec['export_controlled_targets'])
            for route in [r for r in self.plan['routes'] if r['region_id']==region['region_id']]:
                cases = [c for c in spec['cases'] if c['route_id']==route['route_id']]
                self.assertEqual([c['variant'] for c in cases], ['route_baseline']*3+
                    ['clear','thin_pole','body_protrusion','head_bar','suspended_sign'])
                self.assertEqual(len({c['pair_id'] for c in cases[3:]}), 1)
                self.assertTrue(all(c['pair_id'] is None for c in cases[:3]))
            names.extend(c['name'] for c in spec['cases'])
        self.assertEqual(len(names), 72)
        self.assertEqual(len(set(names)), 72)
        self.assertEqual(self.plan, original_plan)
        self.assertEqual(self.template, original_template)

    def test_route_target_ids_presence_and_transformed_supports(self):
        all_ids = set()
        for region in self.plan['regions']:
            spec = compile_region(self.plan, self.template, region['region_id'])
            for target in spec['native_targets']:
                self.assertRegex(target['target_id'], r'^[A-Za-z0-9_-]+$')
                self.assertEqual(target['target_id'], target['instance_id'])
                self.assertTrue(target['target_id'].startswith(target['route_id']+'__'))
                self.assertNotIn(target['target_id'], all_ids)
                all_ids.add(target['target_id'])
            for case in spec['cases']:
                route_ids = {t['target_id'] for t in spec['native_targets'] if t['route_id']==case['route_id']}
                active = {o['instance_id'] for o in case['objects'] if o.get('target_part')}
                self.assertEqual(set(case['active_target_ids']), active)
                self.assertEqual(set(case['absent_target_ids']), route_ids-active)
                self.assertEqual(set(case['controlled_target_presence']), route_ids)
                for tid in route_ids:
                    self.assertEqual(case['controlled_target_presence'][tid], 'PRESENT' if tid in active else 'NOT_PRESENT')
                if case['variant'] in ('clear','route_baseline'):
                    self.assertEqual(active,set())
                    self.assertEqual(len(case['absent_target_ids']),4)
                else:
                    self.assertEqual(len(active),1)
                if case['variant']!='route_baseline':
                    camera = case['camera']; yaw = camera['yaw']; angle = math.radians(yaw)
                    anchor = [camera['x']+2*math.cos(angle), camera['y']+2*math.sin(angle),case['floor_z_m']]
                    contacts = validate_fixture(case['objects'], anchor, yaw)
                    if case['variant']=='clear':
                        self.assertEqual(contacts, {'BODY': [], 'HEAD': []})
        self.assertEqual(len(all_ids),36)

    def test_unsafe_instance_prefix_rejected_before_native_export(self):
        for prefix in ('route/escape', 'route name', '区域', '../route'):
            with self.assertRaises(ValueError):
                fixture_objects('thin_pole', (0.,0.,0.), 0., prefix)

    def test_region_ingestion_keeps_exact_rgb_depth_sample_and_route_identity(self):
        spec = compile_region(self.plan, self.template, 'region_1')
        frames = [dict(sample_index=i,rgb_path=f'rgb/{i:04d}.png') for i in range(len(spec['cases']))]
        dataset = dict(frames=frames, calibration=dict(width=640,height=360,depth_max_m=100))
        receipt = dict(status='PASS',source_unchanged=True,spec_sha256='mock_spec_hash',frame_count=len(frames),
            readiness={'status':'PASS'},view_readiness=[dict(index=i,status='PASS') for i in range(len(frames))],
            pair_exports=dict(profile=dict(completed=len(frames),failed=0,pending=0),
                rows=[dict(sample_index=i,mode='gpu_async') for i in range(len(frames))]))
        capture = Path(__file__).resolve().parents[1]
        def read(path):
            return {'dataset.json':dataset,'spec.json':spec,'receipt.json':receipt}[Path(path).name]
        def asset(path, kind, cal):
            return dict(path=str(path),status='PRESENT',valid=True,sha256=f'mock_{kind}_{Path(path).stem}')
        with patch.object(contract,'read',side_effect=read), patch.object(contract,'sha',return_value='mock_spec_hash'), \
             patch.object(contract,'_asset',side_effect=asset), patch.dict(sys.modules,{'numpy':types.ModuleType('numpy')}):
            bundles = [contract.ingest_native_bundle(self.plan,capture,r['route_id']) for r in self.plan['routes'] if r['region_id']=='region_1']
        seen = []
        for bundle in bundles:
            self.assertEqual(len(bundle['frames']),8)
            for frame in bundle['frames']:
                sid = frame['sample_index']; case = spec['cases'][sid]
                seen.append(sid)
                self.assertEqual(frame['assignment']['route_id'],case['route_id'])
                self.assertEqual(frame['assignment']['route_id'],bundle['route_id'])
                self.assertEqual(frame['camera'],case['camera'])
                self.assertEqual(frame['variant'],case['variant'])
                self.assertEqual(frame['pair_id'],case['pair_id'])
                self.assertEqual(Path(frame['rgb']['path']).stem,f'{sid:04d}')
                self.assertEqual(Path(frame['depth']['path']).stem,f'{sid:04d}')
                self.assertEqual(frame['synchronization']['status'],'PAIRED')
                self.assertEqual(frame['labels']['status'],'UNKNOWN')
                self.assertEqual(len(frame['targets']),4)
                present = [t for t in frame['targets'] if t['label_status']!='NOT_PRESENT']
                self.assertEqual({t['target_id'] for t in present},set(case['active_target_ids']))
                absent = [t for t in frame['targets'] if t['label_status']=='NOT_PRESENT']
                self.assertEqual({t['target_id'] for t in absent},set(case['absent_target_ids']))
                self.assertTrue(all(t['label_status']=='UNKNOWN' for t in present))
        self.assertEqual(sorted(seen), list(range(24)))
        report = contract.validate_bundles(self.plan,bundles)
        self.assertFalse([i for i in report['issues'] if i['code'] in ('PAIR_VARIANTS','UNDECLARED_PAIR')])


if __name__ == '__main__':
    unittest.main()

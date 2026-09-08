"""Synthetic CPU fixtures for collection/split integrity; no model execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import city_field_contract as c


def plan():
    regions=[];routes=[]
    for j,split in enumerate(c.SPLITS):
        x=j*100
        regions.append(dict(region_id=split,split=split,bounds_xy_m=[x,-10,x+30,40]))
        for i,scene in enumerate(c.SCENE_TYPES):
            rid=split+'_'+scene
            routes.append(dict(route_id=rid,region_id=split,split=split,scene_type=scene,
                waypoints=[dict(camera=dict(x=x+5,y=i*10,z=2,yaw=0,pitch=-5,roll=0))],
                instance_ids=[rid+'_target'],pair_groups=[]))
    return dict(schema='city-collection-field-v1',required_scene_types_per_region=list(c.SCENE_TYPES),
        separation=dict(min_region_gap_m=20,min_route_gap_cross_split_m=20,adjacent_frame_gap_m=10),
        regions=regions,routes=routes)


def bundle(p,index=0):
    r=p['routes'][index]
    frame=dict(sample_index=0,camera=r['waypoints'][0]['camera'],
        rgb=dict(status='PRESENT',sha256=r['route_id'],valid=True),
        depth=dict(status='PRESENT',sha256='depth',valid=True),
        synchronization=dict(status='PAIRED'),source_load_status='READY',
        calibration=dict(width=640,height=360,horizontal_fov_degrees=100.,depth_max_m=100.),
        labels=dict(status='KNOWN',near=[0,0],support_unknown_cells=[0,0]),targets=[])
    return dict(schema='city-field-bundle-v1',bundle_id=r['route_id'],route_id=r['route_id'],
        region_id=r['region_id'],split=r['split'],scene_type=r['scene_type'],expected_frame_count=1,
        source=dict(status='PASS',source_unchanged=True,load_status='READY'),frames=[frame])


class FieldContractTest(unittest.TestCase):
    def codes(self,result): return {i['code'] for i in result['issues']}

    def test_floor_probe_thresholds_and_unknown_nominal(self):
        camera=dict(x=1,y=2,z=2.43,yaw=0,pitch=-5,roll=0)
        def check(z,nominal=.73):
            return c.floor_evidence(camera,nominal,dict(hit=True,point_m=[1,2,z]))
        self.assertEqual(check(.75)['status'],'CONSISTENT')
        self.assertEqual(check(.751)['status'],'REVIEW')
        self.assertEqual(check(1.24)['status'],'ANOMALY')
        self.assertEqual(check(2.43)['status'],'ANOMALY')
        missing=check(.73,None)
        self.assertEqual(missing['status'],'UNKNOWN')
        self.assertEqual(missing['measured_floor_z_m'],.73)
        self.assertIsNone(missing['deviation_m'])
        self.assertEqual(c.floor_evidence(camera,.73,dict(hit=False))['status'],'UNKNOWN')

    def test_engine_pose_pair_tolerances_and_nominal_provenance(self):
        command=dict(x=1,y=2,z=2.43,yaw=0,pitch=-5,roll=0)
        pose,report=c.capture_pose_evidence(command,None,True)
        self.assertEqual(pose,command)
        self.assertEqual(report['status'],'NOMINAL_ONLY')
        actual=dict(command,x=1.004)
        pose,report=c.capture_pose_evidence(command,dict(rgb=actual,depth=actual,authority='ENGINE_ACTOR_TRANSFORMS_AT_PAIRED_STATIC_CAPTURE'),True)
        self.assertEqual(pose,actual)
        self.assertEqual(report['status'],'MEASURED')
        self.assertTrue(report['command_mismatch'])
        _,report=c.capture_pose_evidence(command,dict(rgb=actual,depth=dict(actual,x=1.006)),True)
        self.assertEqual(report['status'],'ANOMALY')
        _,report=c.capture_pose_evidence(command,dict(rgb=command,depth=dict(command,yaw=.011)),True)
        self.assertEqual(report['status'],'ANOMALY')

    def test_floor_and_pose_validation_statuses(self):
        p=plan();b=bundle(p);frame=b['frames'][0]
        frame['floor_probe']=dict(status='REVIEW',reason='Deviation exceeds .02m')
        frame['pose_evidence']=dict(status='NOMINAL_ONLY',required=True)
        codes=self.codes(c.validate_bundle(p,b))
        self.assertTrue({'FLOOR_REVIEW','POSE_NOMINAL_ONLY'}<=codes)
        frame['floor_probe']['status']='ANOMALY'
        frame['pose_evidence'].update(status='ANOMALY',reason='Different RGB/depth poses')
        codes=self.codes(c.validate_bundle(p,b))
        self.assertTrue({'FLOOR_ANOMALY','CAPTURE_POSE_ANOMALY'}<=codes)

    def test_ready_requires_three_scene_types_in_every_split_region(self):
        p=plan()
        self.assertEqual(c.validate_plan(p)['readiness'],'READY')
        p['routes']=p['routes'][:3]
        result=c.validate_plan(p)
        self.assertEqual(result['status'],'REVIEW')
        self.assertIn('SPLIT_SCENE_COVERAGE',self.codes(result))

    def test_region_names_do_not_establish_distance(self):
        p=plan();p['regions'][1]['bounds_xy_m']=[29,-10,59,40]
        self.assertIn('REGION_GAP',self.codes(c.validate_plan(p)))

    def test_shared_declared_instance_or_pair_fails(self):
        p=plan();p['routes'][3]['instance_ids']=p['routes'][0]['instance_ids']
        p['routes'][0]['pair_groups']=['same'];p['routes'][3]['pair_groups']=['same']
        codes=self.codes(c.validate_plan(p))
        self.assertTrue({'CROSS_SPLIT_INSTANCE','CROSS_SPLIT_PAIR'}<=codes)

    def test_complete_collection_accepts_scene_mix(self):
        p=plan();self.assertEqual(c.validate_bundles(p,[bundle(p,i) for i in range(9)],True)['status'],'PASS')

    def test_duplicate_rgb_and_adjacent_actual_frames_fail(self):
        p=plan();a=bundle(p);b=bundle(p,3)
        b['frames'][0]['rgb']=copy.deepcopy(a['frames'][0]['rgb'])
        b['frames'][0]['camera']=dict(a['frames'][0]['camera'],x=6)
        codes=self.codes(c.validate_bundles(p,[a,b]))
        self.assertTrue({'CROSS_SPLIT_DUPLICATE_RGB','CROSS_SPLIT_ADJACENT_FRAMES','FRAME_OUTSIDE_REGION'}<=codes)

    def test_clear_hazards_stay_together(self):
        p=plan();p['routes'][0]['pair_groups']=['paired']
        b=bundle(p);b['frames'][0].update(pair_id='paired',variant='clear')
        self.assertIn('PAIR_VARIANTS',self.codes(c.validate_bundle(p,b)))
        other=copy.deepcopy(b['frames'][0]);other.update(sample_index=1,variant='thin_pole')
        b['frames'].append(other);b['expected_frame_count']=2
        self.assertEqual(c.validate_bundle(p,b)['status'],'PASS')

    def test_missing_frame_sync_and_load_have_distinct_codes(self):
        p=plan();b=bundle(p);b['expected_frame_count']=2
        b['frames'][0]['depth']=dict(status='MISSING',valid=False)
        b['frames'][0]['synchronization']['status']='UNKNOWN'
        b['source']['load_status']='UNKNOWN'
        result=c.validate_bundle(p,b)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertTrue({'MISSING_FRAME','MISSING_DEPTH','UNSYNCHRONIZED_RGB_DEPTH','SOURCE_LOAD_UNVERIFIED'}<=self.codes(result))

    def test_unknown_is_retained_and_known_erasure_fails(self):
        p=plan();b=bundle(p);b['frames'][0]['labels'].update(status='UNKNOWN',near=[-1,0])
        before=copy.deepcopy(b)
        self.assertEqual(c.validate_bundle(p,b)['status'],'REVIEW')
        self.assertEqual(before,b)
        b['frames'][0]['labels']['status']='KNOWN'
        self.assertIn('UNKNOWN_ERASURE',self.codes(c.validate_bundle(p,b)))

    def test_per_case_assignment_cannot_change_split(self):
        p=plan();b=bundle(p);b['frames'][0]['assignment']=dict(split='test')
        self.assertIn('FRAME_ASSIGNMENT',self.codes(c.validate_bundle(p,b)))

    def test_explicit_fixture_absence_is_not_unknown_or_scene_clear(self):
        p=plan();b=bundle(p)
        b['frames'][0]['targets']=[dict(identity=p['routes'][0]['instance_ids'][0],
            source_kind='CONTROLLED_SYNTHETIC_FIXTURE',label_status='NOT_PRESENT',relative_geometry={})]
        self.assertEqual(c.validate_bundle(p,b)['status'],'PASS')
        self.assertEqual(b['frames'][0]['labels']['near'],[0,0])

    def test_multiroute_capture_ingestion_keeps_indices_and_synthetic_identity(self):
        import numpy as np
        from PIL import Image
        p=plan();parent=Path(__file__).resolve().parents[1]/'artifacts.local/tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent,prefix='city-field-test-') as directory:
            root=Path(directory)
            for folder in ('source','model','evaluator/native','labels'):(root/folder).mkdir(parents=True,exist_ok=True)
            cases=[];bindings=[]
            for i in range(2):
                r=p['routes'][i];stable=r['instance_ids'][0]
                cases.append(dict(name=f'case{i}',camera=r['waypoints'][0]['camera'],route_id=r['route_id'],region_id=r['region_id'],
                    split=r['split'],scene_type=r['scene_type'],objects=[dict(instance_id=stable,target_part=True,center_m=[6,i*10,1])]))
                bindings.append(dict(sample_index=i,targets=[dict(instance_id=stable,target_id=f't{i}',component_path=f'recreated{i}',position_m=[6,i*10,1])]))
                Image.new('RGB',(640,360),(i,0,0)).save(root/f'model/{i}.png')
                np.save(root/f'evaluator/native/{i:04d}.npy',np.ones((360,640),np.float32))
            def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
            write(root/'source/spec.json',dict(cases=cases,export_controlled_targets=True,
                native_targets=[dict(target_id=f't{i}',instance_id=p['routes'][i]['instance_ids'][0]) for i in range(2)]))
            receipt=dict(status='PASS',source_unchanged=True,spec_sha256=c.sha(root/'source/spec.json'),frame_count=2,
                readiness=dict(status='READY'),view_readiness=[dict(index=i,status='READY') for i in range(2)],
                pair_exports=dict(rows=[dict(sample_index=i,mode='gpu_async') for i in range(2)],profile=dict(completed=2,failed=0,pending=0)),
                controlled_target_bindings=bindings)
            write(root/'receipt.json',receipt)
            write(root/'model/dataset.json',dict(frames=[dict(sample_index=i,rgb_path=f'{i}.png') for i in range(2)],
                calibration=dict(width=640,height=360,horizontal_fov_degrees=100,depth_max_m=100)))
            near=np.array([[-1,-1],[-1,-1]],np.int8);support=np.full((2,2,360,640),-1,np.int8)
            np.save(root/'labels/near.npy',near);np.save(root/'labels/support.npy',support)
            write(root/'labels/native-route-labels.json',dict(schema='city-native-route-labels-v1',sample_indices=[0,1],near='near.npy',support='support.npy',targets=[],
                provenance=dict(spec_sha256=c.sha(root/'source/spec.json'),receipt_sha256=c.sha(root/'receipt.json'))))
            label_hash=c.sha(root/'labels/near.npy')
            b=c.ingest_native_bundle(p,root,p['routes'][1]['route_id'],root/'labels')
            self.assertEqual(b['expected_frame_count'],1)
            self.assertEqual([f['sample_index'] for f in b['frames']],[1])
            self.assertEqual(b['frames'][0]['targets'][0]['identity'],p['routes'][1]['instance_ids'][0])
            self.assertEqual(b['frames'][0]['targets'][0]['source_kind'],'CONTROLLED_SYNTHETIC_FIXTURE')
            self.assertEqual(b['frames'][0]['targets'][0]['relative_geometry']['horizontal_distance_m'],1)
            self.assertEqual(c.sha(root/'labels/near.npy'),label_hash)
            self.assertEqual(c.validate_bundle(p,b)['status'],'REVIEW')
            (root/'model/1.png').write_bytes(b'changed-after-ingestion')
            self.assertIn('CHANGED_RGB',self.codes(c.validate_bundle(p,b)))


if __name__=='__main__': unittest.main()

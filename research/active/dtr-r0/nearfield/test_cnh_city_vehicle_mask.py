"""Pure-Python lifecycle falsifiers; not native UE/render acceptance."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cnh_city_vehicle_mask as module


class Vector:
    def __init__(self,x=0,y=0,z=0):self.x,self.y,self.z=x,y,z


class Rotation(Vector):
    w=1.
    def rotator(self):return deepcopy(self)


class Transform:
    def __init__(self,location=None,rotation=None,scale=None):
        self.translation=deepcopy(location or Vector());self.rotation=deepcopy(rotation or Rotation());self.scale3d=deepcopy(scale or Vector(1,1,1))


class Static:
    pass


class Instanced(Static):
    def __init__(self):
        self.transforms=[Transform(Vector(x,0,0)) for x in (1,30,7)]
        self.static_mesh=SimpleNamespace(get_path_name=lambda:'/Game/Vehicle/Test')
        self.properties={name:0 for name in ('forced_lod_model','min_draw_distance','ld_max_draw_distance',
            'instance_start_cull_distance','instance_end_cull_distance')}
        self.properties.update(override_materials=['material-a'],num_custom_data_floats=2,
            per_instance_sm_custom_data=[1.,2.,3.,4.,5.,6.],instancing_random_seed=71,
            additional_random_seeds=[dict(start_instance_index=1,random_seed=13)])
        self.calls=[];self.corrupt_metadata=False
    def get_path_name(self):return '/Actor/ISM'
    def get_editor_property(self,name):return deepcopy(self.properties[name])
    def get_instance_count(self):return len(self.transforms)
    def get_instance_transform(self,index,world):return deepcopy(self.transforms[index])
    def update_instance_transform(self,index,transform,*flags):
        self.calls.append((index,flags));self.transforms[index]=deepcopy(transform)
        if self.corrupt_metadata and transform.scale3d.x==0:self.properties['additional_random_seeds'][0]['random_seed']+=1
        return True


def context(component):
    unreal=SimpleNamespace(StaticMeshComponent=Static,InstancedStaticMeshComponent=Instanced,Transform=Transform,Vector=Vector)
    unreal.BlindAssistCaptureLibrary=SimpleNamespace(get_ism_preservation_state=native_reader)
    actor=SimpleNamespace(get_class=lambda:SimpleNamespace(get_name=lambda:'Actor'),
        get_components_by_class=lambda cls:[component])
    api=SimpleNamespace(get_all_level_actors=lambda:[actor])
    spec=dict(map_asset='/Game/Map/Small_City_LVL',native_geometry_policy=module.POLICY,
        city_derived_control=dict(authority='CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC',benchmark_eligible=False))
    return unreal,api,spec


def native_reader(component):
    return json.dumps(dict(schema='cnh_ism_preservation_state_v1',data_status='AVAILABLE',
        instance_count=component.get_instance_count(),**{k:component.properties[k] for k in (
            'num_custom_data_floats','per_instance_sm_custom_data','instancing_random_seed','additional_random_seeds')}))


def bounds(unreal,mesh,transform):
    x=transform.translation.x
    return [x-.1,-.1,-.1],[x+.1,.1,.1]


class VehicleMaskTests(unittest.TestCase):
    def test_near_far_union_and_closed_eight_metre_boundary(self):
        bounds_list=[([7.9,0,0],[8.1,1,1]),([8.01,0,0],[9,1,1]),([30,0,0],[31,1,1]),([-.1,-.1,-.1],[.1,.1,.1])]
        self.assertEqual(module.selected_indices(bounds_list,[[0,0,0]]),[0,3])
        self.assertEqual(module.selected_indices(bounds_list,[[0,0,0],[30,0,0]]),[0,2,3])

    def test_apply_restore_keeps_indices_custom_data_and_exact_far_transform(self):
        component=Instanced();before=deepcopy(component.transforms);metadata=module.metadata(component,True,native_reader)
        with patch.object(module,'camera_centres',return_value=[[0,0,0]]),patch.object(module,'transformed_bounds',side_effect=bounds):
            session=module.apply(*context(component))
        self.assertEqual([c[0] for c in component.calls],[0,2])
        self.assertEqual(module.transform_state(component.transforms[1]),module.transform_state(before[1]))
        self.assertEqual(module.metadata(component,True,native_reader),metadata)
        self.assertEqual([r['current_index'] for r in session.receipt['components'][0]['mapping']],[0,2])
        session.restore()
        self.assertEqual([module.transform_state(t) for t in component.transforms],[module.transform_state(t) for t in before])
        self.assertEqual(module.metadata(component,True,native_reader),metadata)
        self.assertTrue(session.receipt['restoration']['restored']);self.assertEqual(session.entries,[])

    def test_metadata_mutation_is_rejected_in_apply_and_restoration(self):
        component=Instanced();component.corrupt_metadata=True
        with patch.object(module,'camera_centres',return_value=[[0,0,0]]),patch.object(module,'transformed_bounds',side_effect=bounds):
            with self.assertRaisesRegex(RuntimeError,'metadata'):
                module.apply(*context(component))
        self.assertEqual([t.scale3d.x for t in component.transforms],[1,1,1])

    def test_restore_detects_untouched_far_instance_corruption(self):
        component=Instanced();unreal,api,_=context(component);receipt={}
        session=module.Session(unreal,api,receipt)
        originals=deepcopy(component.transforms)
        session.entries.append((component,True,originals,[0],module.metadata(component,True,native_reader)))
        component.transforms[1].translation.x+=.001
        with self.assertRaisesRegex(RuntimeError,'transform exceeds'):
            session.restore()
        self.assertFalse(receipt['restoration']['restored'])
        self.assertTrue(receipt['restoration']['errors'])


if __name__=='__main__':unittest.main()

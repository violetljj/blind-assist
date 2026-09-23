import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from cnh_route_source_clearance import material_padding, normalized_guid, audit


class MaterialBoundTests(unittest.TestCase):
    def test_compiled_no_deformation_or_positive_world_clamp(self):
        report=dict(data_status='AVAILABLE', capability='NO_COMPILED_MATERIAL_DEFORMATION', uses_pdo=False, uses_displacement=False, uses_wpo=False)
        self.assertEqual(material_padding(report),0.)
        clamped=dict(report,capability='WPO_CLAMP_CONFIGURED',uses_wpo=True,configured_max_wpo_per_axis_cm=12)
        self.assertIsNone(material_padding(clamped))
        self.assertEqual(material_padding(clamped,dict(status='VERIFIED',all_component_materials=True,max_wpo_extent_cm=12)),.12)

    def test_unknown_unbounded_and_other_displacement_fail_closed(self):
        base=dict(data_status='AVAILABLE', capability='WPO_CLAMP_CONFIGURED', uses_pdo=False, uses_displacement=False, uses_wpo=True)
        for value in (None,0,-1,float('nan'),float('inf'),True):
            self.assertIsNone(material_padding(base,dict(status='VERIFIED',all_component_materials=True,max_wpo_extent_cm=value)))
        for changes in (dict(data_status='UNKNOWN'),dict(uses_pdo=True),dict(uses_displacement=True),dict(uses_wpo=None)):
            self.assertIsNone(material_padding(dict(base,**changes),dict(status='VERIFIED',all_component_materials=True,max_wpo_extent_cm=10)))

    def test_guid_representation_normalized(self):
        self.assertEqual(normalized_guid('{ABC-123}'),normalized_guid('abc123'))
        reflected = SimpleNamespace(get_editor_property=lambda k: dict(a=-1,b=0,c=1,d=2)[k])
        self.assertEqual(normalized_guid(reflected),'ffffffff000000000000000100000002')

    def test_bounded_loaded_world_pass_and_missing_descriptor_unknown(self):
        class Component:
            static_mesh=object()
            class_name='StaticMeshComponent'
            bounds_scale=1.
            def get_path_name(self): return 'component'
            def get_class(self): return SimpleNamespace(get_name=lambda:self.class_name)
            def get_editor_property(self,key): return {'mobility':'STATIC','visible':True,'hidden_in_game':False,'bounds_scale':self.bounds_scale}.get(key)
            def get_num_materials(self): return 1
            def get_material(self,index): return SimpleNamespace(get_path_name=lambda:'material')
            def get_world_transform(self): return object()
        component=Component()
        class Actor:
            def get_path_name(self): return 'actor'
            def get_editor_property(self,key): return 'ABC' if key=='actor_guid' else None
            def get_components_by_class(self,cls): return [component]
        api=SimpleNamespace(get_all_level_actors=lambda:[Actor()])
        material=dict(data_status='AVAILABLE',capability='NO_COMPILED_MATERIAL_DEFORMATION',
                      uses_wpo=False,uses_pdo=False,uses_displacement=False)
        u=SimpleNamespace(PrimitiveComponent=object,StaticMeshComponent=Component,
            InstancedStaticMeshComponent=type('Instance',(object,),{}),ComponentMobility=SimpleNamespace(STATIC='STATIC'),
            BlindAssistCaptureLibrary=SimpleNamespace(get_material_geometry_capability=lambda m:json.dumps(material)))
        pose=dict(x=0,y=0,z=1.6,pitch=0,yaw=0,roll=0)
        clips=[dict(id=name,trajectory_model='piecewise_linear_fixed_orientation',poses=[pose,pose])
               for name in ('centre','boundary','outside','removed')]
        layouts=[dict(layout_id=str(i),clips=clips) for i in range(2)]
        source=dict(map_asset='/Game/Map/Small_City_LVL',native_region=dict(requested_guids=['ABC']))
        with patch('cnh_route_source_clearance.transformed_bounds',return_value=([20,20,20],[21,21,21])):
            result=audit(u,api,layouts,{},source)
            self.assertEqual(result['candidates'][0]['status'],'PASS_LOADED_WORLD_ONLY')
            self.assertFalse(result['native_coverage_complete'])
            source['native_region']['requested_guids'].append('MISSING')
            result=audit(u,api,layouts,{},source)
            self.assertEqual(result['candidates'][0]['status'],'UNKNOWN')
            source['native_region']['requested_guids']=['ABC']
            u.ComponentMobility.STATIC='DIFFERENT_FROM_MOVABLE_COMPONENT'
            source.update(motion_scope='TICK_DISABLED_SETTLED_EDITOR_NO_SIMULATION',
                          native_tick_freeze=dict(actors_disabled=1,components_disabled=1,actor_paths=['actor'],errors=[]))
            self.assertEqual(audit(u,api,layouts,{},source)['candidates'][0]['status'],'PASS_LOADED_WORLD_ONLY')
            source['native_tick_freeze']['actor_paths']=['another_actor']
            self.assertEqual(audit(u,api,layouts,{},source)['candidates'][0]['status'],'UNKNOWN')
            source['native_tick_freeze']['actor_paths']=['actor']
            source['native_tick_freeze']['errors']=['component freeze failed']
            self.assertEqual(audit(u,api,layouts,{},source)['candidates'][0]['status'],'UNKNOWN')
            source['native_tick_freeze']['errors']=[]
            component.class_name='SplineMeshComponent'
            result=audit(u,api,layouts,{},source)
            self.assertEqual(result['unsupported_primitives'][0]['reason'],'STATIC_MESH_SUBCLASS_DEFORMATION_NOT_AUDITED')
            self.assertEqual(result['candidates'][0]['status'],'UNKNOWN')
            for debug_class in ('ZoneGraphRenderingComponent','ZoneGraphCrowdLaneAnnotations','ZoneGraphDisturbanceAnnotation'):
                component.class_name=debug_class
                source['capture_debug_flags']={'Navigation':False,'ZoneGraph':False}
                result=audit(u,api,layouts,{},source)
                self.assertEqual(result['excluded_primitives'][0]['reason'],'DEBUG_VISUALIZATION_EXCLUDED_BY_CAPTURE_FLAGS')
                for flags in ({'Navigation':False},{'Navigation':False,'ZoneGraph':True},{'Navigation':True,'ZoneGraph':False}):
                    source['capture_debug_flags']=flags
                    self.assertFalse(audit(u,api,layouts,{},source)['excluded_primitives'])
            component.class_name='ZoneGraphRenderingComponentSubclass'
            source['capture_debug_flags']={'Navigation':False,'ZoneGraph':False}
            self.assertFalse(audit(u,api,layouts,{},source)['excluded_primitives'])
            component.class_name='DynamicMeshComponent'
            u.StaticMeshComponent=type('OtherStaticMesh',(object,),{})
            for invalid_scale in (.5,0.,float('nan'),None,True):
                component.bounds_scale=invalid_scale
                result=audit(u,api,layouts,{},source)
                self.assertEqual(result['unsupported_primitives'][0]['reason'],'BOUND_EXTRACTION_FAILED')
                self.assertEqual(result['candidates'][0]['status'],'UNKNOWN')


if __name__=='__main__':unittest.main()

"""CPU contract checks; no UE or source-map modification."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cnh_route_source_compare_adapter as adapter


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(scope=adapter.SCOPE, benchmark_eligible=False,
            map_asset='/Game/BAResearchSlice/Street200V7', map_file='unused',
            map_sha256=adapter.hashlib.sha256(b'map').hexdigest(),
            layouts=[dict(layout_id=str(i), physical_site_id=str(i),
                camera=dict(x=20*i, y=0, z=1.6, pitch=0, yaw=90, roll=0)) for i in range(2)])

    def test_fixed_source_contract(self):
        with patch.object(adapter.Path, 'read_bytes', return_value=b'map'):
            adapter.validate_spec(self.spec)
            for change in (dict(layouts=self.spec['layouts'][:1]), dict(map_sha256='wrong'),
                           dict(benchmark_eligible=True), dict(background_hlod_min_distance_m=50)):
                with self.assertRaises(ValueError):
                    adapter.validate_spec(dict(self.spec, **change))
            repeated=deepcopy(self.spec)
            repeated['layouts'][1]['physical_site_id']='0'
            with self.assertRaises(ValueError):
                adapter.validate_spec(repeated)

    def test_city_requires_region_enclosing_probe_spheres(self):
        spec=dict(self.spec, map_asset='/Game/Map/Small_City_LVL')
        with patch.object(adapter.Path, 'read_bytes', return_value=b'map'):
            with self.assertRaises(ValueError):
                adapter.validate_spec(spec)
            spec['world_partition_region_m']=dict(min=[-8,-8,-7], max=[28,8,10])
            adapter.validate_spec(spec)
            spec['world_partition_region_m']['max'][0]=27.9
            with self.assertRaises(ValueError):
                adapter.validate_spec(spec)

    def test_candidate_count_and_region_validate_before_loading(self):
        spec=deepcopy(self.spec)
        spec['world_partition_region_m']=dict(min=[-8,-8,-7],max=[28,8,10])
        camera=dict(spec['layouts'][0]['camera'])
        with patch.object(adapter.Path,'read_bytes',return_value=b'map'):
            spec['layouts'][0]['candidates']=[dict(camera=camera)]
            adapter.validate_spec(spec)
            spec['layouts'][0]['candidates']=[dict(camera=camera)]*17
            with self.assertRaises(ValueError): adapter.validate_spec(spec)
            spec['layouts'][0]['candidates']=[]
            with self.assertRaises(ValueError): adapter.validate_spec(spec)
            spec['layouts'][0]['candidates']=[dict(camera=dict(camera,x=100))]
            with self.assertRaises(ValueError): adapter.validate_spec(spec)

    def test_stereo_uses_camera_right_and_preserves_left_attributes(self):
        class Actor:
            def set_actor_location(self, value, *args): self.position=value
            def set_actor_rotation(self, value, *args): self.rotation=value
        editor=SimpleNamespace(set_level_viewport_camera_info=lambda *args:None)
        u=SimpleNamespace(Vector=lambda *v:v, Rotator=lambda **v:v,
            UnrealEditorSubsystem=object, get_editor_subsystem=lambda _:editor)
        captures={name:Actor() for name in ('rgb_left','rgb_right','depth_right','normal_left')}
        adapter.place_captures(u,captures,self.spec['layouts'][0]['camera'])
        self.assertAlmostEqual(captures['rgb_right'].position[0],-6.)
        self.assertAlmostEqual(captures['rgb_right'].position[1],0.)
        self.assertEqual(captures['rgb_right'].position,captures['depth_right'].position)
        self.assertEqual(captures['rgb_left'].position,captures['normal_left'].position)

    def test_native_tick_freeze_records_failures(self):
        component=SimpleNamespace(set_component_tick_enabled=lambda value:None,get_path_name=lambda:'component')
        actor=SimpleNamespace(set_actor_tick_enabled=lambda value:None,
            get_components_by_class=lambda cls:[component],get_path_name=lambda:'actor')
        api=SimpleNamespace(get_all_level_actors=lambda:[actor])
        u=SimpleNamespace(ActorComponent=object)
        receipt=adapter.freeze_native_ticks(u,api)
        self.assertEqual(receipt['native_tick_freeze'],dict(actors_disabled=1,components_disabled=1,actor_paths=['actor'],errors=[]))
        def fail(value): raise RuntimeError('refused')
        component.set_component_tick_enabled=fail
        receipt=adapter.freeze_native_ticks(u,api)
        self.assertEqual(len(receipt['native_tick_freeze']['errors']),1)


if __name__=='__main__':
    unittest.main()

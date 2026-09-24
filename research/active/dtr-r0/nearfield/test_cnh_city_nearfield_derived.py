import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from cnh_city_nearfield_derived import camera_centres, selection, Session, apply, POLICY, NearFarPreflightError


class SelectionTests(unittest.TestCase):
    def test_bounds_intersection_not_centroid(self):
        self.assertTrue(selection([([7., -1., -1.], [30., 1., 1.])], [(0., 0., 0.)]))
        self.assertFalse(selection([([8.01, 0., 0.], [9., 1., 1.])], [(0., 0., 0.)]))

    def test_union_selects_second_camera_and_rejects_far_instance(self):
        centres = [(0., 0., 0.), (100., 0., 0.)]
        near = ([99., 0., 0.], [101., 1., 1.])
        far = ([49., 0., 0.], [51., 1., 1.])
        self.assertTrue(selection([near], centres))
        with self.assertRaisesRegex(ValueError, 'Mixed near/far'):
            selection([near, far], centres)
        self.assertFalse(selection([], centres))

    def test_all_clip_endpoints_included_without_candidate_expansion(self):
        pose = lambda x: dict(x=x, y=0., z=1.)
        spec = dict(layouts=[dict(layout_id=str(i), camera=pose(i),
            clips=[dict(poses=[pose(i), pose(i+10)])], candidates=[dict(camera=pose(999))]) for i in (0, 1)])
        self.assertEqual(camera_centres(spec), [(0.,0.,1.), (10.,0.,1.), (1.,0.,1.), (11.,0.,1.)])
        spec['layouts'] *= 2
        with self.assertRaises(ValueError):
            camera_centres(spec)


class RestoreTests(unittest.TestCase):
    def test_mixed_component_rejects_before_any_asset_mutation(self):
        class Component:
            static_mesh='source'
            def get_instance_count(self): return 2
            def get_instance_transform(self,index,world): return index
            def get_path_name(self): return 'mixed-component'
        assets=Mock()
        u=SimpleNamespace(StaticMeshEditorSubsystem=object,StaticMeshComponent=Component,
            InstancedStaticMeshComponent=Component,get_editor_subsystem=Mock(),
            AssetToolsHelpers=SimpleNamespace(get_asset_tools=lambda:assets))
        api=SimpleNamespace(get_all_level_actors=lambda:[SimpleNamespace(get_components_by_class=lambda cls:[Component()])])
        pose=dict(x=0,y=0,z=0)
        spec=dict(map_asset='/Game/Map/Small_City_LVL',native_geometry_policy=POLICY,
            city_derived_control=dict(authority='CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC',benchmark_eligible=False),
            layouts=[dict(layout_id=str(i),camera=pose,clips=[dict(poses=[pose])]) for i in range(2)])
        with patch('cnh_city_nearfield_derived.transformed_bounds',side_effect=[([-1,-1,-1],[1,1,1]),([20,20,20],[21,21,21])]):
            with self.assertRaises(NearFarPreflightError) as caught:
                apply(u,api,spec)
        assets.duplicate_asset.assert_not_called()
        self.assertFalse(caught.exception.receipt['mutations_started'])
        self.assertEqual(caught.exception.receipt['status'],'REJECTED_MIXED_NEAR_FAR_INSTANCES')

    def test_restore_runs_all_assignments_after_one_failure(self):
        calls = []
        class Component:
            def __init__(self, name, fail=False):
                self.name, self.fail = name, fail
            def set_static_mesh(self, mesh):
                calls.append(self.name)
                if self.fail:
                    raise RuntimeError('injected restoration failure')
                self.static_mesh = mesh
            def set_forced_lod_model(self, lod): self.lod = lod
            def get_editor_property(self, key):
                return self.lod if key == 'forced_lod_model' else []
            def get_world_transform(self): return 'T'
            def get_num_materials(self): return 0
        state = dict(transforms=['T'], overrides='[]', materials=[])
        session = Session(SimpleNamespace(), {})
        session.assignments = [(Component('good'), 'original', 0, state, False),
                               (Component('bad', True), 'original', 0, state, False)]
        with self.assertRaisesRegex(RuntimeError, 'injected restoration failure'):
            session.restore()
        self.assertEqual(calls, ['bad', 'good'])
        self.assertFalse(session.receipt['restoration']['restored'])


if __name__ == '__main__':
    unittest.main()

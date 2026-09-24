"""Lifecycle checks for the consumed City per-instance substitution control."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import cnh_city_instance_substitution as module
from test_cnh_city_vehicle_mask import Instanced, Transform, Vector, native_reader


class Mesh:
    def __init__(self, path, enabled=True):
        self.path, self.enabled = path, enabled
        self.material = NS(get_path_name=lambda: '/Material/A')
        self.slots = [NS(get_editor_property=lambda key: self.material)]
    def get_path_name(self): return self.path
    def get_name(self): return self.path.rsplit('/', 1)[-1]
    def get_editor_property(self, key):
        if key == 'static_materials': return self.slots
        raise KeyError(key)


class Component(Instanced):
    def __init__(self, name, mesh):
        super().__init__()
        self.name, self.static_mesh = name, mesh
        self.transforms.append(Transform(Vector(6, 0, 0)))
    def get_path_name(self): return self.name
    def get_num_materials(self): return 1
    def get_material(self, slot): return self.static_mesh.material


class OverlayComponent:
    def __init__(self):
        self.static_mesh = None
        self.props = {'forced_lod_model': 0, 'disallow_nanite': False}
        self.materials = {}
    def set_static_mesh(self, mesh): self.static_mesh = mesh
    def set_editor_property(self, key, value, **kwargs): self.props[key] = value
    def get_editor_property(self, key): return self.props[key]
    def set_forced_lod_model(self, value): self.props['forced_lod_model'] = value
    def set_material(self, slot, material): self.materials[slot] = material
    def get_material(self, slot): return self.materials[slot]


class OverlayActor:
    def __init__(self, index):
        self.index = index
        self.static_mesh_component = OverlayComponent()
        self.transform = None
    def set_actor_transform(self, transform, *args): self.transform = transform
    def get_actor_transform(self): return self.transform
    def get_path_name(self): return '/Derived/' + str(self.index)


class Editor:
    class Settings:
        def __init__(self, mesh): self.mesh = mesh
        def get_editor_property(self, key): return self.mesh.enabled
        def set_editor_property(self, key, value): self.mesh.enabled = value
        def __str__(self): return 'NaniteEnabled=' + str(self.mesh.enabled)
    def get_nanite_settings(self, mesh): return self.Settings(mesh)
    def set_nanite_settings(self, mesh, settings, apply): pass


class SubstitutionTests(unittest.TestCase):
    def test_selection_is_exact_for_curb_and_near_only_for_vehicles(self):
        bounds = [([.9, 0, 0], [1.1, 1, 1]), ([30, 0, 0], [31, 1, 1]),
                  ([7.9, 0, 0], [8.1, 1, 1]), ([6, 0, 0], [7, 1, 1])]
        self.assertEqual(module.chosen_instances(module.CURB_COMPONENT, module.CURB_MESH,
                         bounds, [[0, 0, 0]], module.MODES[0]), [3])
        self.assertEqual(module.chosen_instances('/Vehicle/ISM', '/Game/Vehicle/Car',
                         bounds, [[0, 0, 0]], module.MODES[1]), [0, 2, 3])
        self.assertEqual(module.chosen_instances('/Vehicle/ISM', '/Game/Vehicle/Car',
                         bounds, [[0, 0, 0]], module.MODES[0]), [])
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            module.chosen_instances(module.CURB_COMPONENT, '/Game/Other', bounds,
                                    [[0, 0, 0]], module.MODES[0])

    def test_full_substitution_preserves_far_indices_and_restores(self):
        curb = Component(module.CURB_COMPONENT, Mesh(module.CURB_MESH))
        vehicle = Component('/Vehicle/ISM', Mesh('/Game/Vehicle/Car.Car'))
        originals = {c.get_path_name(): [module.transform_state(t) for t in c.transforms]
                     for c in (curb, vehicle)}
        editor = Editor()
        actors = []
        for component in (curb, vehicle):
            actors.append(NS(get_class=lambda: NS(get_name=lambda: 'TemplateActor'),
                             get_components_by_class=lambda cls, c=component: [c]))
        overlays = []
        def spawn(cls, location):
            result = OverlayActor(len(overlays))
            overlays.append(result)
            return result
        api = NS(get_all_level_actors=lambda: actors, spawn_actor_from_class=spawn,
                 destroy_actor=lambda actor: overlays.remove(actor) is None)
        def duplicate(name, package, mesh):
            clone = Mesh(package+'/'+name, False)
            clone.material, clone.slots = mesh.material, mesh.slots
            return clone
        assets = NS(duplicate_asset=duplicate)
        u = NS(StaticMeshComponent=object, InstancedStaticMeshComponent=Component,
               StaticMeshActor=OverlayActor, StaticMeshEditorSubsystem=object,
               AssetToolsHelpers=NS(get_asset_tools=lambda: assets),
               BlindAssistCaptureLibrary=NS(get_ism_preservation_state=native_reader),
               PropertyAccessChangeNotifyMode=NS(NEVER=0), Transform=Transform, Vector=Vector,
               get_editor_subsystem=lambda cls: editor)
        spec = dict(map_asset='/Game/Map/Small_City_LVL', native_geometry_policy=module.POLICY,
                    city_derived_control=dict(authority='CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC',
                        benchmark_eligible=False, substitution_mode='NEAR_VEHICLES_AND_CURB'))
        def bounds(u, mesh, transform):
            x = transform.translation.x
            return [x-.1, 0, 0], [x+.1, 1, 1]
        with patch.object(module, 'camera_centres', return_value=[[0, 0, 0]]), \
             patch.object(module, 'transformed_bounds', side_effect=bounds):
            session = module.apply(u, api, spec)
        self.assertEqual(session.receipt['substituted_instances'], 4)
        self.assertEqual(session.receipt['substituted_vehicle_instances'], 3)
        self.assertEqual([module.transform_state(t) for t in vehicle.transforms][1],
                         originals['/Vehicle/ISM'][1])
        self.assertTrue(all(not actor.static_mesh_component.static_mesh.enabled for actor in overlays))
        self.assertTrue(all(actor.static_mesh_component.props ==
                            {'forced_lod_model': 1, 'disallow_nanite': True} for actor in overlays))
        session.restore()
        self.assertEqual(overlays, [])
        self.assertTrue(session.receipt['restoration']['restored'])
        for component in (curb, vehicle):
            self.assertEqual([module.transform_state(t) for t in component.transforms],
                             originals[component.get_path_name()])

    def test_frozen_spec_rejects_changed_predicted_winner(self):
        frames = []
        for clip in ('centre', 'boundary', 'outside', 'removed'):
            frames.append(dict(layout_id='city-engineering-0', pose_index=1, clip_id=clip,
                worst_background_rays=[dict(x=223, y=277, surface_index=15, mesh=module.CURB_MESH)],
                predicted_surface_attribution=[dict(surface_index=15, mesh=module.CURB_MESH,
                    component=module.CURB_COMPONENT, instance=module.CURB_INDEX)]))
        with TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            path.write_text(json.dumps(dict(status='DIAGNOSTIC_ONLY', benchmark_eligible=False,
                                            frames=frames)), encoding='utf-8')
            with patch('cnh_city_nearfield_derived.frozen_spec', return_value=dict(city_derived_control={})):
                spec = module.frozen_spec(temp, path, module.MODES[0])
                self.assertEqual(spec['native_geometry_policy'], module.POLICY)
                frames[0]['predicted_surface_attribution'][0]['instance'] = 4
                path.write_text(json.dumps(dict(status='DIAGNOSTIC_ONLY', benchmark_eligible=False,
                                                frames=frames)), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'attribution changed'):
                    module.frozen_spec(temp, path, module.MODES[0])


if __name__ == '__main__': unittest.main()

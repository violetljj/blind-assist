"""CPU adapter regressions; do not claim Unreal render acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

from cnh_route_scene_probe import intersects_sphere, probe, transformed_bounds


def vector(x=0, y=0, z=0):
    return NS(x=x, y=y, z=z)


def transform(x):
    return NS(translation=vector(x), scale3d=vector(1, 1, 1), rotation=NS(x=0, y=0, z=0, w=1))


class Mesh:
    def get_path_name(self): return '/Game/Test/LargeMesh'
    def get_bounding_box(self): return NS(min=vector(-10000, -100, -100), max=vector(0, 100, 100))
    def get_editor_property(self, name): raise AttributeError(name)
    def get_num_sections(self, lod): return 1


class Static:
    pass


class Instanced(Static):
    static_mesh = Mesh()
    def get_path_name(self): return '/Map/Actor/Instances'
    def get_class(self): return NS(get_name=lambda: 'HierarchicalInstancedStaticMeshComponent')
    def get_instance_count(self): return 2
    def get_instance_transform(self, index, world):
        assert world is True
        return transform([10000, 30000][index])
    def get_num_materials(self): return 1
    def get_material(self, slot): return None


class ProbeTest(unittest.TestCase):
    def unreal(self):
        return NS(Vector=vector, StaticMeshComponent=Static, InstancedStaticMeshComponent=Instanced,
            PrimitiveComponent=object, MathLibrary=NS(transform_location=lambda t, p:
                vector(t.translation.x+p.x, t.translation.y+p.y, t.translation.z+p.z)),
            ProceduralMeshLibrary=NS(get_section_from_static_mesh=lambda m, lod, section:
                ([vector(0), vector(100), vector(0, 100)], [0, 1, 2],
                 [vector(0, 0, 1)]*3, [vector()]*3, [])))

    def test_large_origin_does_not_drop_near_surface(self):
        low, high = transformed_bounds(self.unreal(), Mesh(), transform(10000))
        self.assertEqual(low, [0, -1, -1])
        self.assertTrue(intersects_sphere(low, high, [0, 0, 0], 5))
        self.assertFalse(intersects_sphere([6, -1, -1], [7, 1, 1], [0, 0, 0], 5))

    def test_instance_transform_selection_export_and_fail_closed(self):
        actor = NS(get_path_name=lambda: '/Map/Actor', get_components_by_class=lambda cls: [Instanced()])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'probe'
            result = probe(self.unreal(), NS(get_all_level_actors=lambda: [actor]),
                           dict(x=0, y=0, z=0), 5, out)
            self.assertEqual(result['status'], 'NOT_ADMITTED')
            self.assertEqual(result['instance_count'], 1)
            self.assertEqual(result['instances'][0]['instance_index'], 0)
            self.assertEqual(result['instances'][0]['actual_translation_m'], [100, 0, 0])
            self.assertEqual(result['meshes'][0]['nanite'], 'UNKNOWN')
            self.assertEqual(result['meshes'][0]['status'], 'EXPORTED_LOD0_NOT_RENDER_VERIFIED')
            self.assertEqual(result['instances'][0]['materials'][0]['world_position_offset'], 'UNKNOWN')
            self.assertEqual(json.loads((out/'scene-probe.json').read_text())['instance_raster'], 'NOT_IMPLEMENTED')
            with self.assertRaises(FileExistsError):
                probe(self.unreal(), NS(get_all_level_actors=lambda: [actor]), dict(x=0,y=0,z=0), 5, out)

    def test_missing_mesh_api_retains_failure_receipt(self):
        u = self.unreal(); u.ProceduralMeshLibrary = None
        actor = NS(get_path_name=lambda: '/Map/Actor', get_components_by_class=lambda cls: [Instanced()])
        with tempfile.TemporaryDirectory() as directory:
            result = probe(u, NS(get_all_level_actors=lambda: [actor]), dict(x=0,y=0,z=0), 5, Path(directory)/'p')
            self.assertEqual(result['meshes'][0]['status'], 'NOT_EXPORTED')
            self.assertIn('no proxy fallback', result['meshes'][0]['error'])

    def test_hidden_editor_visualization_is_not_native_geometry(self):
        component = Instanced()
        component.get_editor_property = lambda name: True if name in ('is_editor_only', 'hidden_in_game') else None
        actor = NS(get_path_name=lambda: '/Map/Actor', get_components_by_class=lambda cls: [component])
        with tempfile.TemporaryDirectory() as directory:
            result = probe(self.unreal(), NS(get_all_level_actors=lambda: [actor]),
                           dict(x=0,y=0,z=0), 5, Path(directory)/'p')
            self.assertEqual(result['instance_count'], 0)
            self.assertEqual(len(result['excluded_editor_primitives']), 1)

    def test_invalid_bounds_and_nonfinite_radius(self):
        for low, high, radius in [([2,0,0], [1,1,1], 5), ([0,0,0], [1,1,1], float('nan'))]:
            with self.assertRaises(ValueError):
                intersects_sphere(low, high, [0,0,0], radius)


if __name__ == '__main__':
    unittest.main()

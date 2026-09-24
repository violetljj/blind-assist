import json
import unittest
from types import SimpleNamespace
from cnh_route_street_static_background import apply, verify


class StreetStaticTests(unittest.TestCase):
    def test_null_material_remains_unresolved(self):
        component = SimpleNamespace(get_editor_property=lambda k: k=='visible',
            get_num_materials=lambda: 1, get_material=lambda i: None,
            get_path_name=lambda: '/component')
        api = SimpleNamespace(get_all_level_actors=lambda: [SimpleNamespace(
            get_components_by_class=lambda _: [component])])
        u = SimpleNamespace(PrimitiveComponent=object, BlindAssistCaptureLibrary=SimpleNamespace(
            get_material_geometry_capability=lambda _: self.fail('No material queried')))
        receipt = apply(u, api)
        self.assertEqual(receipt['slots'], [])
        self.assertEqual(receipt['unresolved'][0]['reason'], 'NULL_MATERIAL')

    def test_pending_or_deforming_shader_never_passes(self):
        for capability in ({'data_status':'UNAVAILABLE'},
                           {'data_status':'AVAILABLE','capability':'WPO_CLAMP_CONFIGURED'}):
            u = SimpleNamespace(load_asset=lambda _: object(), BlindAssistCaptureLibrary=SimpleNamespace(
                get_material_geometry_capability=lambda _: json.dumps(capability)))
            receipt = {'materials':[{'derivation':{'derived_material':'/derived'}}]}
            with self.assertRaises(RuntimeError):
                verify(u, receipt)
            self.assertNotIn('compiled_verification', receipt)


if __name__ == '__main__':
    unittest.main()

import unittest
from cnh_route_depth_surface import DepthSurfaceFilter


class DepthSurfaceTests(unittest.TestCase):
    def instance(self,mode,masked=False,name='/Material'):
        return dict(component_path='/Component',mesh=dict(asset_path='/Mesh'),materials=[dict(
            asset_path=name,effective_render_material=dict(data_status='AVAILABLE',blend_mode=mode,is_masked=masked))])

    def test_shadow_mask_does_not_make_translucent_surface_write_depth(self):
        f=DepthSurfaceFilter()
        self.assertFalse(f(self.instance(2,True),dict(section=0,material_slot=0)))
        self.assertFalse(f(self.instance(7),dict(section=1,material_slot=0)))
        self.assertEqual(len(f.excluded),2)

    def test_force_mask_name_does_not_override_effective_opaque_mode(self):
        f=DepthSurfaceFilter()
        self.assertTrue(f(self.instance(0,name='/ForceMask'),dict(section=0,material_slot=0)))
        self.assertTrue(f(self.instance(1,True),dict(section=0,material_slot=0)))
        self.assertEqual(len(f.masked),1)

    def test_unknown_mapping_fails_closed(self):
        with self.assertRaises(ValueError):DepthSurfaceFilter()(self.instance(0),dict(section=0))
        with self.assertRaises(ValueError):DepthSurfaceFilter()(self.instance(99),dict(section=0,material_slot=0))


if __name__=='__main__':unittest.main()

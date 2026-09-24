import unittest
from cnh_route_derived_assets import _disable_mfpd_on_clone, _stable_state


class DerivedStateTests(unittest.TestCase):
    def test_wrapper_address_is_not_configuration(self):
        a = "<Struct 'Settings' (0x1234) {enabled: True, precision: 2}>"
        b = "<Struct 'Settings' (0000ABCD) {enabled: True, precision: 2}>"
        self.assertEqual(_stable_state(a), _stable_state(b))
        self.assertNotEqual(_stable_state(a), _stable_state(b.replace('True','False')))

    def test_values_outside_struct_wrapper_remain_exact(self):
        value = "texture=/Game/Asset, parameter=(0x1234), enabled=False"
        self.assertEqual(_stable_state(value), value)

    def test_mfpd_switch_changes_clone_only_and_receipts_actual_used_textures(self):
        class Texture:
            def __init__(self, name): self.name = name
            def get_path_name(self): return self.name

        class Material:
            def __init__(self, name): self.name, self.mfpd = name, True
            def get_path_name(self): return self.name

        class Library:
            updates = 0
            def get_static_switch_parameter_names(self, material): return ['Enable MFPD']
            def get_material_instance_static_switch_parameter_value(self, material, name):
                return material.mfpd
            def get_material_used_textures(self, material):
                names = ['/Game/UniqueAlbedo', '/Game/UniqueNormal', '/Game/UniqueRoughness']
                if material.mfpd: names += ['/Game/SharedMFPD']
                return [Texture(name) for name in names]
            def set_material_instance_static_switch_parameter_value(self, material, name, value):
                material.mfpd = value
            def update_material_instance(self, material): self.updates += 1

        class Unreal:
            MaterialEditingLibrary = Library()

        source, clone = Material('/Game/Source'), Material('/Game/UnsavedClone')
        receipt = _disable_mfpd_on_clone(Unreal, source, clone)
        self.assertTrue(source.mfpd)
        self.assertFalse(clone.mfpd)
        self.assertEqual(Unreal.MaterialEditingLibrary.updates, 1)
        self.assertEqual(receipt['removed_editor_used_textures'], ['/Game/SharedMFPD'])
        self.assertEqual(receipt['derived_editor_used_textures_after'],
                         ['/Game/UniqueAlbedo', '/Game/UniqueNormal', '/Game/UniqueRoughness'])
        self.assertEqual(receipt['source_editor_used_textures_before'],
                         receipt['source_editor_used_textures_after'])
        self.assertTrue(receipt['source_used_textures_unchanged'])


if __name__ == '__main__':
    unittest.main()

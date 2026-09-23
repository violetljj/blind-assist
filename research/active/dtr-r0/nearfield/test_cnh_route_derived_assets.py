import unittest
from cnh_route_derived_assets import _stable_state


class DerivedStateTests(unittest.TestCase):
    def test_wrapper_address_is_not_configuration(self):
        a = "<Struct 'Settings' (0x1234) {enabled: True, precision: 2}>"
        b = "<Struct 'Settings' (0000ABCD) {enabled: True, precision: 2}>"
        self.assertEqual(_stable_state(a), _stable_state(b))
        self.assertNotEqual(_stable_state(a), _stable_state(b.replace('True','False')))

    def test_values_outside_struct_wrapper_remain_exact(self):
        value = "texture=/Game/Asset, parameter=(0x1234), enabled=False"
        self.assertEqual(_stable_state(value), value)


if __name__ == '__main__':
    unittest.main()

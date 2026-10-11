"""Raw-to-model zone orientation contract; no accuracy assertions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

import cnh_zone_orientation as O

NEAR = Path(__file__).resolve().parents[2] / 'dtr-r0/nearfield'


class OrientationTests(unittest.TestCase):
    def test_display_top_left_is_raw_zone_63(self):
        # The accepted display grid lists zone 63 top-left and zone 0 bottom-right.
        self.assertEqual(O.model_zone_for_raw(63), 0)
        self.assertEqual(O.model_zone_for_raw(0), 63)
        self.assertEqual(O.model_zone_for_raw(7), 56)   # raw bottom-left of display row 7

    def test_layouts_agree_and_involution(self):
        rng = np.random.default_rng(3)
        flat = rng.normal(size=(5, 64, 16))
        grid = flat.reshape(5, 8, 8, 16)
        a = O.to_model(flat).reshape(5, 8, 8, 16)
        b = O.to_model(grid)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(O.to_model(O.to_model(flat)), flat)
        amb = rng.normal(size=(5, 64))
        np.testing.assert_array_equal(O.to_model(amb).reshape(5, 8, 8), O.to_model(amb.reshape(5, 8, 8)))

    def test_raw_bottom_row_lands_in_model_lower_half(self):
        # A return in raw row 0 (scene bottom) must reach model row 7, i.e. y>0 (below).
        h = np.zeros((64, 16)); h[0:8, 5] = 1
        m = O.to_model(h).reshape(8, 8, 16)
        self.assertTrue(m[7, :, 5].all() and not m[:7, :, 5].any())

    def test_projector_convention_row0_is_up(self):
        sys.path.insert(0, str(NEAR))
        import cnh_cvr_projection as C
        # A point above the optical axis (y<0, Y down) falls in projector row 0..3.
        xy = np.array([0.0, -0.3])
        ij = np.floor((xy + C.EDGE) / (2 * C.EDGE) * 8).astype(int)
        self.assertLess(ij[1], 4)

    def test_mounting_guard(self):
        O.check_mounting()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'view.json'
            p.write_text(json.dumps({'raw_to_current_display': 'row=raw_row,col=raw_col'}), encoding='utf8')
            with self.assertRaises(ValueError):
                O.check_mounting(p)


if __name__ == '__main__':
    unittest.main()

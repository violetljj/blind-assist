import math
import unittest
from body_query_5000_sites import candidates, select


class SiteTests(unittest.TestCase):
    def grid(self):
        return [dict(x=x*3., y=y*3., z=.7, hit=True, component_path='sidewalk', instance_index=0)
                for x in range(12) for y in range(12)]

    def test_no_wall_or_step_or_missing_forward(self):
        rows = self.grid()
        for r in rows:
            if (r['x'], r['y']) == (6, 3): r['z'] = 1.0
            if (r['x'], r['y']) == (3, 6): r['component_path'] = 'sidewalk_tree_grill'
        pool = candidates(rows, {})
        self.assertFalse(any(p['camera_xy_m'] == [3, 3] for p in pool))
        self.assertFalse(any(p['camera_xy_m'] == [0, 0] for p in pool))

    def test_separation_variety_and_reproducibility(self):
        pool = candidates(self.grid(), {})
        selected = select(pool, 25)
        self.assertEqual(len(selected), 25)
        self.assertEqual(selected, select(list(reversed(pool)), 25))
        self.assertEqual(len({p['yaw_deg'] for p in selected}), 4)
        self.assertTrue(all(math.dist(a['camera_xy_m'], b['camera_xy_m']) >= 6
                            for i, a in enumerate(selected) for b in selected[i+1:]))


if __name__ == '__main__':
    unittest.main()

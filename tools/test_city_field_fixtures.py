"""Attachment and sweep regressions for the reusable controlled City fixtures."""
import math
import unittest

from city_field_fixtures import KINDS, PROVENANCE, corridor_objects, fixture_objects, validate_fixture


class FixtureTests(unittest.TestCase):
    def test_clear_and_target_bands_after_arbitrary_placement(self):
        expected = {'clear': (False, False), 'thin_pole': (True, True),
                    'body_protrusion': (True, False), 'head_bar': (False, True),
                    'suspended_sign': (False, True)}
        for yaw in (0., 37., 90., 180., -123.):
            for kind in KINDS:
                with self.subTest(kind=kind, yaw=yaw):
                    anchor = (93.2, -52.1, 7.6)
                    objects = fixture_objects(kind, anchor, yaw, 'site_1')
                    hits = validate_fixture(objects, anchor, yaw)
                    self.assertEqual(tuple(bool(hits[b]) for b in ('BODY', 'HEAD')), expected[kind])
                    targets = {o['name'] for o in objects if o['target_part']}
                    self.assertTrue(set(hits['BODY']+hits['HEAD']) <= targets)
                    self.assertTrue(all(o['geometry_provenance'] == PROVENANCE for o in objects))

    def test_matched_portal_and_ids(self):
        clear = fixture_objects('clear', (0., 0., 0.), 0., 'bay')
        for kind in KINDS:
            objects = fixture_objects(kind, (0., 0., 0.), 0., 'bay')
            self.assertEqual(objects[:len(clear)], clear)
            moved = fixture_objects(kind, (10., 20., 3.), 37., 'bay')
            self.assertEqual([o['instance_id'] for o in objects], [o['instance_id'] for o in moved])
            other = fixture_objects(kind, (0., 0., 0.), 0., 'other')
            self.assertFalse({o['instance_id'] for o in objects} & {o['instance_id'] for o in other})

    def test_rotation_known_quarter_turn(self):
        zero = fixture_objects('body_protrusion', (0., 0., 0.), 0., 'bay')
        quarter = fixture_objects('body_protrusion', (10., 20., 3.), 90., 'bay')
        for first, second in zip(zero, quarter):
            x,y,z = first['center_m']
            for actual, expected in zip(second['center_m'], (10-y, 20+x, 3+z)):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(first['size_m'], second['size_m'])
            self.assertEqual(second['rotation_deg']['yaw'], 90.)

    def test_each_declared_support_really_contacts_child(self):
        # Independent Euclidean interval gap check in unrotated engineering space.
        # Zero distance to another component/floor is required for every part.
        for kind in KINDS:
            objects = fixture_objects(kind, (0., 0., 0.), 0., 'bay')
            named = {o['name']: o for o in objects}
            for child in objects:
                if child['support_parent'] == 'ground':
                    self.assertAlmostEqual(child['center_m'][2]-child['size_m'][2]/2, 0.)
                else:
                    parent = named[child['support_parent']]
                    gap = [max(0., abs(c-p)-(cs+ps)/2) for c,p,cs,ps in
                           zip(child['center_m'], parent['center_m'], child['size_m'], parent['size_m'])]
                    self.assertLess(math.sqrt(sum(v*v for v in gap)), 1e-8)

    def test_detached_geometry_and_floating_root_are_rejected(self):
        for part in ('sign_panel', 'left_foot'):
            objects = fixture_objects('suspended_sign', (0., 0., 0.), 0., 'bay')
            bad = next(o for o in objects if o['name'] == 'bay__'+part)
            bad['center_m'][0 if part == 'sign_panel' else 2] += .8
            with self.assertRaises(ValueError):
                validate_fixture(objects, (0., 0., 0.), 0.)

    def test_support_cycles_duplicate_ids_and_nonfinite_rejected(self):
        objects = fixture_objects('clear', (0., 0., 0.), 0., 'bay')
        objects[0]['support_parent'] = objects[1]['name']
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            validate_fixture(objects, (0., 0., 0.), 0.)
        objects = fixture_objects('clear', (0., 0., 0.), 0., 'bay')
        objects[1]['instance_id'] = objects[0]['instance_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_fixture(objects, (0., 0., 0.), 0.)
        with self.assertRaises(ValueError):
            fixture_objects('clear', (0., 0., float('nan')), 0., 'bay')

    def test_corridor_is_supported_and_clear_for_whole_length(self):
        for yaw in (0., 37., -90.):
            anchor = (13., -80., 4.)
            corridor = corridor_objects(anchor, yaw, 'bay')
            for along in (-6., -3., 0., 3., 6.):
                co,si = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                sample_anchor = (anchor[0]+along*co, anchor[1]+along*si, anchor[2])
                self.assertEqual(validate_fixture(corridor, sample_anchor, yaw), {'BODY': [], 'HEAD': []})
            all_objects = corridor + fixture_objects('clear', anchor, yaw, 'bay')
            self.assertEqual(validate_fixture(all_objects, anchor, yaw), {'BODY': [], 'HEAD': []})
        corridor = corridor_objects((0.,0.,0.), 0., 'bay', width_m=1.)
        self.assertTrue(all(abs(o['center_m'][1])-o['size_m'][1]/2 >= .5-1e-8 for o in corridor))
        with self.assertRaises(ValueError):
            corridor_objects((0.,0.,0.), 0., 'bay', width_m=.55)

    def test_body_arm_butt_joint_and_corridor_rails_do_not_overlap(self):
        # compile_region anchors the fixture two metres ahead of the mid-route
        # camera; the admitted corridor is centred at that camera position.
        fixture = fixture_objects('body_protrusion', (2.,0.,0.), 0., 'bay')
        arm = next(o for o in fixture if o['target_part'])
        clamp = next(o for o in fixture if o['name'] == arm['support_parent'])
        self.assertAlmostEqual(arm['center_m'][1]-arm['size_m'][1]/2,
                               clamp['center_m'][1]+clamp['size_m'][1]/2)
        corridor = corridor_objects((0.,0.,0.), 0., 'bay')
        for support in corridor:
            overlap = [min(a+sa/2,b+sb/2)-max(a-sa/2,b-sb/2) for a,b,sa,sb in
                       zip(arm['center_m'],support['center_m'],arm['size_m'],support['size_m'])]
            self.assertTrue(any(v <= 0. for v in overlap), support['name'])
        self.assertEqual(validate_fixture(fixture+corridor,(2.,0.,0.),0.),
                         {'BODY':['bay__body_arm'],'HEAD':[]})


if __name__ == '__main__':
    unittest.main()

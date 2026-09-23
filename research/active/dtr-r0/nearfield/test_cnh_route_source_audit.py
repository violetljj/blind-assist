"""Exercise cross-role leakage, derived assets and physical site overlap."""
import copy
import unittest

from cnh_route_source_audit import POLICY, audit


def fixture():
    assets = [dict(id=k, family_ids=[k], visual_dependency_ids=['mesh:' + k, 'texture:' + k],
                   dependency_closure_complete=True) for k in ('building', 'train-pole', 'test-pole')]
    layouts = [dict(id=s, split=s, site_ids=['site:' + s], visible_physical_instance_ids=['actor:' + s],
                    physical_visibility_closure_complete=True,
                    uses=[dict(asset_id='building', role='background', intersects_query=False),
                          dict(asset_id=s + '-pole', role='target', intersects_query=True)]) for s in ('train', 'test')]
    return dict(isolation_policy=POLICY, assets=assets, layouts=layouts)


class SourceAuditTests(unittest.TestCase):
    def test_shared_background_is_disclosed_without_benchmark_admission(self):
        result = audit(fixture())
        self.assertEqual(result['violations'], [])
        self.assertEqual(len(result['shared_background_identities']), 3)
        self.assertFalse(result['benchmark_eligible'])

    def test_intruding_building_is_not_exempt(self):
        source = fixture()
        source['layouts'][0]['uses'][0]['intersects_query'] = True
        self.assertTrue(audit(source)['violations'])

    def test_distractor_and_background_cannot_launder_target_family(self):
        for role in ('distractor', 'background'):
            with self.subTest(role=role):
                source = fixture()
                source['layouts'][1]['uses'][1].update(asset_id='train-pole', role=role, intersects_query=False)
                self.assertTrue(audit(source)['violations'])

    def test_renamed_family_or_shared_texture_is_rejected(self):
        for field in ('family_ids', 'visual_dependency_ids'):
            source = fixture()
            source['assets'][2][field] = copy.deepcopy(source['assets'][1][field])
            self.assertTrue(audit(source)['violations'])

    def test_site_or_distant_physical_instance_overlap(self):
        for field in ('site_ids', 'visible_physical_instance_ids'):
            source = fixture()
            source['layouts'][1][field] = source['layouts'][0][field]
            self.assertEqual(audit(source)['violations'][0]['code'], 'PHYSICAL_SOURCE_OVERLAP')

    def test_unknown_closures_and_intersections_are_not_passes(self):
        for owner, field in (('assets', 'dependency_closure_complete'), ('layouts', 'physical_visibility_closure_complete')):
            source = fixture()
            source[owner][0][field] = False
            with self.assertRaises(ValueError):
                audit(source)
        source = fixture()
        source['layouts'][0]['uses'][0]['intersects_query'] = None
        with self.assertRaises(ValueError):
            audit(source)


if __name__ == '__main__':
    unittest.main()

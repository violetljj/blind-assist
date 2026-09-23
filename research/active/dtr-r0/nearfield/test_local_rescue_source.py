"""Fresh geometry and frozen public-inference contract checks; no capture/fit."""
import copy
import unittest

import numpy as np

from local_rescue_source import specification, check_spec, planned_bounds
from local_rescue_inference import synthetic_checks, validate_identities
from local_stability_spec import specification as stability_spec
from local_transfer_spec import specification as transfer_spec
from query_occupancy_spec import specification as query_spec
from data_coverage_spec import specification as coverage_spec
from query_occupancy_data import geometric_labels
from launch_local_rescue_fresh import validate_cases, MANDATORY_INPUTS


class FreshRescueSourceTests(unittest.TestCase):
    def test_new_relative_geometry_against_all_four_sources(self):
        spec = specification()
        result = check_spec(spec, [query_spec(), coverage_spec(), transfer_spec(), stability_spec()])
        self.assertEqual(result['old_specs_compared'], 4)
        self.assertEqual((result['positive_frames'], result['negative_frames'], result['positive_events']),
                         (224, 352, 32))
        old = stability_spec()
        self.assertTrue(all(g['size_m'] != h['size_m'] for g, h in zip(spec['groups'], old['groups'])))
        self.assertTrue(all(c['nominal_front_m'] != o['nominal_front_m']
                            for c, o in zip(spec['cases'], old['cases'])))
        validate_cases(spec)

    def test_modified_source_rejected(self):
        spec = copy.deepcopy(specification())
        spec['cases'][0]['objects'][0]['size_m'][0] += .01
        with self.assertRaises(AssertionError):
            check_spec(spec)

    def test_old_group_rejected_and_public_extract_contract(self):
        spec = specification()
        ids = [dict(**c, id=c['name'], index=i) for i, c in enumerate(spec['cases'])]
        self.assertEqual(validate_identities(ids, set())['frames'], 576)
        with self.assertRaises(ValueError):
            validate_identities(ids, {ids[0]['base_group_id']})
        self.assertEqual(synthetic_checks()['cases'], 10)

    def test_union_preserves_composite_hole(self):
        bounds = [(np.array([.4, 0., 2.]), np.array([.6, 1., 2.2])),
                  (np.array([-.6, 0., 2.]), np.array([.6, .1, 2.2]))]
        self.assertEqual(geometric_labels(bounds)[0][1], 6)
        self.assertLess(geometric_labels([(np.array([-.6, 0., 2.]),
                                          np.array([.6, 1., 2.2]))])[0][1], 6)

    def test_selected_gate_is_mandatory_before_capture(self):
        self.assertIn('artifacts.local/evidence/ba-local-rescue-gate-20260923-run/gate.json', MANDATORY_INPUTS)
        self.assertIn('research/active/dtr-r0/nearfield/LOCAL_RESCUE_GATE_PROTOCOL_20260923.md', MANDATORY_INPUTS)


if __name__ == '__main__':
    unittest.main()

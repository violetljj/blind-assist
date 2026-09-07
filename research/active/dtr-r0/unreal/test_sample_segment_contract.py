"""Synthetic contract checks; these do not certify the live UE sample map."""
import json
import unittest

from sample_segment_contract import convert_components, evaluate_route


def component(identity, center, extent, **extra):
    return dict(id=identity, asset="synthetic/test-mesh", center_m=center, extent_m=extent, **extra)


class SampleSegmentContractTests(unittest.TestCase):
    def fixture(self):
        return convert_components([
            component("floor", [34, 0, .07], [12, 6, .05]),
            component("planter/instance/0", [30, 2, .52], [.6, .5, .4], collision_enabled=False),
            component("planter/instance/1", [38, -2, .52], [.6, .5, .4]),
            component("roof", [30, 0, 3], [4, 3, .4]),
        ])

    def test_center_witness_and_seeded_visible_planter_contact(self):
        contract = self.fixture()
        clear = evaluate_route([[26, 0], [34, 0]], contract)
        contact = evaluate_route([[26, 2], [34, 2]], contract)
        self.assertEqual("CLEAR_WITHIN_SUPPLIED_ROSTER", clear["status"])
        self.assertEqual("PROXY_CONTACT", contact["status"])
        self.assertEqual(["planter/instance/0"], [v["id"] for v in contact["contacts"]])
        self.assertAlmostEqual(1.22, clear["min_clearance_m"])
        json.dumps([contract, clear, contact], allow_nan=False)

    def test_clipping_preserves_native_roster_and_explicit_exclusions(self):
        contract = convert_components([
            component("cross-edge", [22, 0, 1], [1, 1, 1]),
            component("floor", [30, 0, .12], [1, 1, .025]),
            component("overhead", [30, 0, 2.42], [1, 1, .5]),
            component("outside", [50, 0, 1], [1, 1, 1]),
        ])
        self.assertEqual(21, contract["native_components"][0]["min_m"][0])
        self.assertEqual(22, contract["obstacles"][0]["min_m"][0])
        self.assertEqual(3, len(contract["exclusions"]))
        self.assertEqual("UNKNOWN", evaluate_route([[45.9, 0]], contract)["status"])

    def test_exact_circular_corner_and_tangent(self):
        contract = convert_components([component("box", [30, 0, 1], [.5, .5, .5])])
        self.assertFalse(evaluate_route([[29.3, .7]], contract)["contact"])
        self.assertTrue(evaluate_route([[26, .78], [34, .78]], contract)["contact"])
        self.assertTrue(evaluate_route([[26, 2], [30, 0], [34, 2]], contract)["contact"])

    def test_ground_threshold_configurable_and_instances_distinct(self):
        row = component("low", [30, 0, .1], [.5, .5, .05])
        self.assertEqual(1, len(convert_components([row])["obstacles"]))
        self.assertFalse(convert_components([row], ground_upper_m=.16)["obstacles"])
        self.assertEqual(2, len(self.fixture()["obstacles"]))

    def test_invalid_input_fails_instead_of_silently_reducing_roster(self):
        row = component("one", [30, 0, 1], [.5, .5, .5])
        with self.assertRaises(ValueError):
            convert_components([row, row])
        with self.assertRaises(ValueError):
            convert_components([component("bad", [30, 0, float("nan")], [1, 1, 1])])
        with self.assertRaises(ValueError):
            convert_components([component("bad", [30, 0, 1], [-1, 1, 1])])
        with self.assertRaises(ValueError):
            evaluate_route([], self.fixture())


if __name__ == "__main__":
    unittest.main()

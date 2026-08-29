from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import named_poi_portal_binding as binding
import named_poi_portal_binding_adjudicate as adjudication


class PortalBindingTest(unittest.TestCase):
    def test_head_topology_self_test(self) -> None:
        self.assertEqual(binding.self_test()["status"], "PASS")

    def test_all_none_cannot_promote(self) -> None:
        self.assertEqual(adjudication.self_test()["status"], "PASS")

    def test_source_buildings_are_split_disjoint(self) -> None:
        source = json.loads(
            (HERE / "named_poi_portal_binding_source_v1.json").read_text(encoding="utf-8")
        )
        train = {
            row["id"]
            for row in source["splits"]["train"]["prior_portal_source"]["entities"]
        } | {row["id"] for row in source["splits"]["train"]["added_entities"]}
        test = {
            row["id"]
            for row in source["splits"]["test"]["prior_portal_source"]["entities"]
        }
        development_protocol = json.loads(
            (HERE / "named_poi_multifacet_reference_ray_protocol_v1.json").read_text(
                encoding="utf-8"
            )
        )
        development = {
            target
            for target, roles in development_protocol["targets"].items()
            if roles["evaluation_indices"]
        }
        self.assertEqual((len(train), len(development), len(test)), (10, 4, 6))
        self.assertFalse(train & development)
        self.assertFalse(train & test)
        self.assertFalse(development & test)

    def test_added_training_truth_is_complete(self) -> None:
        source = json.loads(
            (HERE / "named_poi_portal_binding_source_v1.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (HERE / "named_poi_portal_binding_source_audit_v1.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {row["id"] for row in source["splits"]["train"]["added_entities"]}
        observed = set(audit["portal_set_truth"]["entities"])
        self.assertEqual(observed, expected)
        self.assertEqual(audit["algorithm_calls_before_freeze"], 0)
        self.assertEqual(audit["model_calls_before_freeze"], 0)

    def test_protocol_features_are_unique(self) -> None:
        protocol = json.loads(
            (HERE / "named_poi_portal_binding_protocol_v1.json").read_text(encoding="utf-8")
        )
        names = protocol["feature_names"]
        self.assertEqual(len(names), 31)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()

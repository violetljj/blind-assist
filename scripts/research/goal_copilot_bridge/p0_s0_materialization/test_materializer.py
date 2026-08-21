from __future__ import annotations

import copy
import unittest

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


def source(name: str) -> dict:
    return {
        "source_name": name,
        "snapshot_or_release": "fixture-v1",
        "record_ids": [f"{name}-1"],
        "license": "fixture-license",
        "attribution": f"fixture {name}",
        "retrieved_at": "2026-08-21T00:00:00Z",
        "content_sha256": "a" * 64,
    }


def admitted_record() -> dict:
    frames = [
        {"frame_id": "f1", "sequence_id": "s1", "camera_position": {"lon": 3.72, "lat": 51.05}, "is_panorama_slice": False},
        {"frame_id": "f2", "sequence_id": "s2", "camera_position": {"lon": 3.7201, "lat": 51.05}, "is_panorama_slice": False},
    ]
    base = {
        "building_id": "b1", "anchor_id": "osm-node-1", "geometry_verified": True, "map_anchored": True,
        "ray_range_m": 20.0, "candidate_anchor_distance_m": 1.0, "second_anchor_margin_m": 5.0,
        "bbox_xyxy": [0.1, 0.1, 0.4, 0.8],
        "generator_provenance": {
            "provider_id": "fixture-provider", "model_id": "fixture-model", "model_version": "1",
            "config_sha256": "b" * 64, "input_sha256": "c" * 64,
            "candidate_source": "synthetic-mechanics-fixture", "lineage_group": "fixture-lineage",
        },
    }
    candidates = [
        dict(base, candidate_id="c1", frame_id="f1", predicted_entrance_geo={"lon": 3.72005, "lat": 51.0501}, ray_heading_deg=20.0),
        dict(base, candidate_id="c2", frame_id="f2", predicted_entrance_geo={"lon": 3.72006, "lat": 51.0501}, ray_heading_deg=50.0),
    ]
    return {
        "record_id": "episode-1",
        "sources": {"mapillary": source("mapillary"), "overture": source("overture"), "osm": source("osm")},
        "crosswalk": {"status": "ADMITTED", "unique": True, "building_id": "b1", "method": "CONTAINED_UNIQUE"},
        "anchor": {"status": "ADMITTED", "unique": True, "anchor_id": "osm-node-1", "method": "TOPOLOGICAL"},
        "frames": frames,
        "candidates": candidates,
        "conflicts": [],
        "evaluated_system_overlap": "NO_KNOWN_OVERLAP",
        "target_visibility_identifiable": True,
        "entrance_semantics_match_goal": True,
        "ancestry_deduplicated": True,
    }


def bundle(record: dict | None = None) -> dict:
    return {
        "protocol_id": materializer.PROTOCOL_ID,
        "upstream_commit": materializer.UPSTREAM_COMMIT,
        "provider_input": {"goal": {"poi_name": "Fixture Library"}, "coarse_area": "Ghent"},
        "records": [record or admitted_record()],
    }


class MaterializerTest(unittest.TestCase):
    def test_primary_requires_true_independent_multiview(self) -> None:
        result = materializer.materialize_bundle(bundle())
        self.assertEqual("P0_S0_MATERIALIZATION_CANARY_PASS", result["verdict"])
        self.assertEqual(1, result["primary_admitted_count"])
        self.assertEqual(["s1", "s2"], result["results"][0]["multiview_pairs"][0]["sequence_ids"])

    def test_five_meter_cluster_does_not_imply_multiview(self) -> None:
        record = admitted_record()
        record["frames"][1]["camera_position"] = copy.deepcopy(record["frames"][0]["camera_position"])
        record["candidates"][1]["frame_id"] = "f1"
        result = materializer.materialize_bundle(bundle(record))
        self.assertEqual(0, result["primary_admitted_count"])
        self.assertEqual(1, result["secondary_admitted_count"])
        self.assertEqual(0, result["rejected_count"])
        self.assertEqual("SILVER_B_MAP_GEOMETRY", result["results"][0]["quality_class"])
        self.assertIn("INSUFFICIENT_MULTIVIEW_SUPPORT", result["results"][0]["failure_codes"])

    def test_every_hard_gate_fails_closed(self) -> None:
        mutations = [
            (lambda r: r["crosswalk"].update(unique=False), "TARGET_BUILDING_CROSSWALK_AMBIGUOUS"),
            (lambda r: r["anchor"].update(status="REJECTED"), "NO_VALID_MAP_ANCHOR"),
            (lambda r: r["candidates"][0].update(geometry_verified=False) or r["candidates"][1].update(geometry_verified=False), "INSUFFICIENT_GEOMETRY_SUPPORT"),
            (lambda r: r["conflicts"].append("OSM_BUILDING_CONFLICT"), "EVIDENCE_CONFLICT"),
            (lambda r: r.update(evaluated_system_overlap="TEACHER_LINEAGE_OVERLAP"), "EVALUATOR_LINEAGE_OVERLAP"),
            (lambda r: r.update(target_visibility_identifiable=False), "TARGET_VISIBILITY_NOT_IDENTIFIABLE"),
            (lambda r: r.update(entrance_semantics_match_goal=False), "ENTRANCE_RELATION_NOT_IDENTIFIABLE"),
            (lambda r: r.update(ancestry_deduplicated=False), "ANCESTRY_OVERLAP"),
        ]
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                record = admitted_record()
                mutate(record)
                rejected = materializer.materialize_bundle(bundle(record))["results"][0]
                self.assertEqual("REJECTED", rejected["admission_status"])
                self.assertIn(expected, rejected["failure_codes"])

    def test_provenance_and_license_are_separate_failures(self) -> None:
        record = admitted_record()
        record["sources"]["osm"].pop("record_ids")
        record["sources"]["overture"]["license"] = ""
        failures = materializer.materialize_bundle(bundle(record))["results"][0]["failure_codes"]
        self.assertIn("PROVENANCE_INCOMPLETE", failures)
        self.assertIn("LICENSE_METADATA_INCOMPLETE", failures)

    def test_provider_input_leak_fails_entire_canary(self) -> None:
        value = bundle()
        value["provider_input"]["target_building_id"] = "b1"
        result = materializer.materialize_bundle(value)
        self.assertEqual("P0_S0_PROTOCOL_OR_SCHEMA_INADEQUACY", result["verdict"])
        self.assertEqual(["target_building_id"], result["provider_input_leaks"])

    def test_malformed_candidate_is_schema_inadequacy_not_crash(self) -> None:
        record = admitted_record()
        record["candidates"][0].pop("ray_range_m")
        result = materializer.materialize_bundle(bundle(record))
        self.assertEqual("P0_S0_PROTOCOL_OR_SCHEMA_INADEQUACY", result["verdict"])
        self.assertEqual(["SCHEMA_INADEQUACY"], result["results"][0]["failure_codes"])

    def test_replay_is_deterministic(self) -> None:
        first = materializer.materialize_bundle(bundle())
        second = materializer.materialize_bundle(copy.deepcopy(bundle()))
        self.assertEqual(materializer.canonical_bytes(first), materializer.canonical_bytes(second))

    def test_real_preflight_reports_both_current_blockers(self) -> None:
        report = materializer.prerequisite_report(token_present=False, candidate_generator_authorized=False)
        self.assertEqual(
            ["MAPILLARY_ACCESS_TOKEN_MISSING", "MANDATORY_CANDIDATE_GENERATOR_NOT_AUTHORIZED"],
            report["failure_codes"],
        )
        self.assertFalse(report["ready_for_real_episode_materialization"])

    def test_blocked_closeout_never_claims_an_episode(self) -> None:
        source_report = {
            "report_sha256": "b" * 64,
            "source_files": {
                "overture_buildings": {"feature_count": 3},
                "overture_places": {"feature_count": 4},
                "osm": {"entrance_count": 2},
            },
        }
        report = materializer.blocked_closeout(source_report, token_present=False)
        self.assertEqual("P0_S0_SOURCE_OR_LICENSE_BLOCKED", report["verdict"])
        self.assertEqual(0, report["materialized_episode_count"])
        self.assertEqual(0, report["evaluator_dry_run_episode_count"])


if __name__ == "__main__":
    unittest.main()

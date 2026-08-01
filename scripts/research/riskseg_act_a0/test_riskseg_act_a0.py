from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from PIL import Image

from .build_review_bundle import (
    _opaque_id,
    _validate_manifest,
    build_bundle,
)
from .common import (
    CELL_IDS,
    PASS_CONDITION,
    PASS_IDS,
    A0Error,
    anchor_indices,
    assert_no_forbidden_public_fields,
    condition_offsets,
    derive_actionability,
    load_json,
    round_half_up,
    sha256_file,
    validate_review_row,
    write_json,
    write_jsonl,
)
from .score_reviews import EXPECTED_DECLARATION, score
from .validate_result import validate


class AnchorAndFirewallTest(unittest.TestCase):
    def test_round_half_up_and_frozen_anchor_formula(self) -> None:
        self.assertEqual(round_half_up(Decimal("10.5")), 11)
        self.assertEqual(anchor_indices(30), (10, 13, 16, 19))
        self.assertEqual(anchor_indices(31), (10, 13, 17, 20))
        for frame_count in (30, 31, 64, 120):
            for anchor in anchor_indices(frame_count):
                self.assertGreaterEqual(anchor - 10, 0)
                self.assertLess(anchor + 10, frame_count)

    def test_future_firewall_offsets(self) -> None:
        self.assertEqual(condition_offsets("CURRENT_ONLY"), (0,))
        self.assertEqual(condition_offsets("CAUSAL_HISTORY"), tuple(range(-10, 1)))
        self.assertEqual(
            condition_offsets("HINDSIGHT_REFERENCE"), tuple(range(-10, 11))
        )
        self.assertFalse(any(offset > 0 for offset in condition_offsets("CAUSAL_HISTORY")))

    def test_opaque_ids_are_distinct_across_all_passes(self) -> None:
        seed = bytes(range(32))
        identifiers = {
            _opaque_id(seed, pass_id, event, anchor)
            for pass_id in PASS_IDS
            for event in range(30)
            for anchor in range(4)
        }
        self.assertEqual(len(identifiers), 720)
        self.assertTrue(all(value.startswith("a0_") and len(value) == 27 for value in identifiers))


class DerivationTest(unittest.TestCase):
    def _row(self) -> dict:
        return {
            "review_item_id": "a0_fixture",
            "hazard_aux": "UNRESOLVED",
            "intrusion_cells": {cell: "UNKNOWN" for cell in CELL_IDS},
            "boundary_relation": "AMBIGUOUS",
            "alertable": "UNKNOWN",
            "passed": "UNKNOWN",
            "knownness": "UNKNOWN",
            "non_actionable_reason": None,
            "quality_state": "OTHER_NOT_EVALUABLE",
            "rationale_code": "INSUFFICIENT_RGB_EVIDENCE",
        }

    def test_unknown_derives_only_to_abstain(self) -> None:
        row = self._row()
        self.assertEqual(derive_actionability(row), "ABSTAIN_NOT_EVALUABLE")
        normalized = validate_review_row(row, where="fixture")
        self.assertEqual(normalized["derived_actionability"], "ABSTAIN_NOT_EVALUABLE")
        row["derived_actionability"] = "NON_ACTIONABLE_NOW"
        with self.assertRaisesRegex(A0Error, "contradicts"):
            validate_review_row(row, where="fixture")

    def test_public_absence_declaration_is_allowed_but_oracle_path_is_not(self) -> None:
        assert_no_forbidden_public_fields(
            {"truth_oracle_model_output_absent": True}
        )
        with self.assertRaisesRegex(A0Error, "forbidden public key"):
            assert_no_forbidden_public_fields({"oracle_mask_path": "secret.png"})


class BundleIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _manifest(
        self,
        *,
        size: tuple[int, int] = (512, 288),
        bad_timestamp: bool = False,
    ) -> Path:
        image_root = self.root / f"images-{size[0]}-{bad_timestamp}"
        image_root.mkdir()
        frames = []
        for index in range(30):
            path = image_root / f"{index:03d}.png"
            Image.new("RGB", size, (index, index, index)).save(path)
            timestamp = index * 100
            if bad_timestamp and index >= 20:
                timestamp += 1
            frames.append(
                {
                    "frame_index": index,
                    "image_path": str(path.relative_to(self.root)).replace("\\", "/"),
                    "image_sha256": sha256_file(path),
                    "timestamp_ms": timestamp,
                }
            )
        manifest = {
            "event_count": 1,
            "events": [
                {
                    "source_session_id": "fixture-session",
                    "bucket": "parallel_curb_negative",
                    "frames": frames,
                }
            ],
        }
        path = self.root / f"manifest-{size[0]}-{bad_timestamp}.json"
        write_json(path, manifest)
        return path

    def test_manifest_rejects_wrong_rgb_geometry_and_timestamp_span(self) -> None:
        wrong_size = self._manifest(size=(64, 48))
        with self.assertRaisesRegex(A0Error, "512x288"):
            _validate_manifest(wrong_size, sha256_file(wrong_size), expected_event_count=1)
        bad_time = self._manifest(bad_timestamp=True)
        with self.assertRaisesRegex(A0Error, "1000 ms"):
            _validate_manifest(bad_time, sha256_file(bad_time), expected_event_count=1)

    def test_bundle_has_six_isolated_firewalled_passes_and_private_time_evidence(self) -> None:
        manifest = self._manifest()
        output = self.root / "bundle"
        build_bundle(
            manifest_path=manifest,
            expected_manifest_sha256=sha256_file(manifest),
            output_root=output,
            master_seed=bytes(range(32)),
            expected_event_count=1,
        )
        all_ids: set[str] = set()
        orders: list[list[tuple[str, int]]] = []
        key = load_json(output / "_private" / "scoring_key.json")
        for pass_id in PASS_IDS:
            public = load_json(output / "passes" / pass_id / "pass_manifest.json")
            self.assertEqual(public["item_count"], 4)
            ids = {item["review_item_id"] for item in public["items"]}
            self.assertFalse(all_ids & ids)
            all_ids |= ids
            expected_offsets = list(condition_offsets(PASS_CONDITION[pass_id]))
            for item in public["items"]:
                self.assertEqual(
                    [media["relative_offset"] for media in item["media"]],
                    expected_offsets,
                )
                self.assertTrue(
                    (output / "passes" / pass_id / item["contact_sheet"]["path"]).is_file()
                )
                self.assertNotIn("event", json.dumps(item).lower())
                self.assertNotIn("session", json.dumps(item).lower())
            pass_units = [unit for unit in key["units"] if unit["pass_id"] == pass_id]
            orders.append(
                [(unit["parent_id"], unit["anchor_slot"]) for unit in pass_units]
            )
            self.assertTrue(
                all(
                    unit["history_span_ms"] == unit["future_span_ms"] == 1000
                    for unit in pass_units
                )
            )
        self.assertEqual(len(all_ids), 24)
        self.assertGreater(len({tuple(order) for order in orders}), 1)


class ScoringAndIndependentValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle_root = self.root / "bundle"
        self.key_path = self.bundle_root / "_private" / "scoring_key.json"
        self.key_path.parent.mkdir(parents=True)
        units = []
        self.item_by_pass_slot: dict[tuple[str, int], str] = {}
        for pass_id in PASS_IDS:
            for slot in range(4):
                item_id = f"a0_{pass_id.lower()}_{slot}"
                self.item_by_pass_slot[(pass_id, slot)] = item_id
                units.append(
                    {
                        "review_item_id": item_id,
                        "pass_id": pass_id,
                        "condition": PASS_CONDITION[pass_id],
                        "parent_id": "parent_fixture",
                        "hidden_source_stratum": "fixture_stratum",
                        "anchor_slot": slot,
                        "anchor_frame_index": 10 + slot * 3,
                    }
                )
        write_json(
            self.key_path,
            {
                "schema_version": "blindassist.riskseg_act_a0.scoring_key.v1",
                "protocol_id": "RISKSEG_ACT_A0",
                "parent_event_count": 1,
                "anchor_count": 4,
                "passes": list(PASS_IDS),
                "units": units,
            },
        )
        pass_rows = []
        self.pass_manifest_hashes: dict[str, str] = {}
        for pass_id in PASS_IDS:
            pass_root = self.bundle_root / "passes" / pass_id
            pass_root.mkdir(parents=True)
            write_json(
                pass_root / "pass_manifest.json",
                {
                    "pass_id": pass_id,
                    "information_condition": PASS_CONDITION[pass_id],
                    "item_count": 4,
                },
            )
            write_jsonl(pass_root / "review_template.jsonl", [])
            manifest_sha = sha256_file(pass_root / "pass_manifest.json")
            self.pass_manifest_hashes[pass_id] = manifest_sha
            pass_rows.append(
                {
                    "pass_id": pass_id,
                    "information_condition": PASS_CONDITION[pass_id],
                    "path": f"passes/{pass_id}",
                    "item_count": 4,
                    "manifest_sha256": manifest_sha,
                    "review_template_sha256": sha256_file(
                        pass_root / "review_template.jsonl"
                    ),
                }
            )
        write_json(
            self.bundle_root / "bundle_receipt.json",
            {
                "schema_version": "blindassist.riskseg_act_a0.review_bundle.v1",
                "protocol_id": "RISKSEG_ACT_A0",
                "status": "READY_FOR_SIX_ISOLATED_REVIEWS",
                "anchor_count_per_pass": 4,
                "passes": pass_rows,
                "scoring_key": {
                    "path": "_private/scoring_key.json",
                    "sha256": sha256_file(self.key_path),
                    "must_not_be_shared_with_reviewers": True,
                },
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _row(self, pass_id: str, slot: int) -> dict:
        base = {
            "review_item_id": self.item_by_pass_slot[(pass_id, slot)],
            "hazard_aux": "NONE_IN_SCOPE",
            "intrusion_cells": {cell: "NON_INTRUDING" for cell in CELL_IDS},
            "boundary_relation": "NOT_APPLICABLE",
            "alertable": "NO",
            "passed": "NO",
            "knownness": "KNOWN",
            "non_actionable_reason": "CLEAR_FORWARD",
            "quality_state": "STABLE",
            "rationale_code": "FIXTURE",
        }
        if slot == 0:
            base.update(
                {
                    "hazard_aux": "BLOCKING_OBSTACLE",
                    "alertable": "YES",
                    "non_actionable_reason": None,
                }
            )
            base["intrusion_cells"]["NEAR_CENTER"] = "INTRUDING"
        elif slot == 2:
            base.update(
                {
                    "hazard_aux": "UNRESOLVED",
                    "boundary_relation": "AMBIGUOUS",
                    "alertable": "UNKNOWN",
                    "passed": "UNKNOWN",
                    "knownness": "UNKNOWN",
                    "non_actionable_reason": None,
                    "quality_state": "OTHER_NOT_EVALUABLE",
                }
            )
            base["intrusion_cells"] = {cell: "UNKNOWN" for cell in CELL_IDS}
        elif slot == 3:
            base.update(
                {
                    "hazard_aux": "BOUNDARY_LEVEL_CHANGE",
                    "boundary_relation": "PARALLEL_BOUNDARY",
                    "non_actionable_reason": "LATERAL_OR_PARALLEL",
                }
            )
        return base

    def _submissions(self, *, disagreement: bool = False) -> dict[str, Path]:
        submissions = {}
        for pass_index, pass_id in enumerate(PASS_IDS):
            root = self.root / f"submission-{pass_id}"
            root.mkdir()
            rows = [self._row(pass_id, slot) for slot in range(4)]
            if disagreement and pass_id == "CURRENT_B":
                rows[0]["alertable"] = "NO"
                rows[0]["non_actionable_reason"] = "TOO_EARLY_OR_NOT_CURRENTLY_ALERTABLE"
            review_path = root / "review.jsonl"
            write_jsonl(review_path, rows)
            write_json(
                root / "submission_receipt.json",
                {
                    "schema_version": "blindassist.riskseg_act_a0.submission_receipt.v1",
                    "protocol_id": "RISKSEG_ACT_A0",
                    "pass_id": pass_id,
                    "assigned_pass_manifest_sha256": self.pass_manifest_hashes[
                        pass_id
                    ],
                    "reviewer_identity": f"fresh-reviewer-{pass_index}",
                    "fresh_agent_for_this_pass": True,
                    "isolation_declaration": EXPECTED_DECLARATION,
                    "review_jsonl_path": "review.jsonl",
                    "review_jsonl_sha256": sha256_file(review_path),
                },
            )
            submissions[pass_id] = root
        return submissions

    def test_exact_agreement_scoring_and_independent_recomputation(self) -> None:
        submissions = self._submissions()
        result = score(
            bundle_root=self.bundle_root,
            scoring_key_path=self.key_path,
            submissions=submissions,
        )
        for condition in ("CURRENT_ONLY", "CAUSAL_HISTORY", "HINDSIGHT_REFERENCE"):
            metrics = result["condition_metrics"][condition]
            self.assertEqual(metrics["alertable_exact_agreement"]["value"], 1.0)
            self.assertEqual(metrics["passed_exact_agreement"]["value"], 1.0)
            self.assertEqual(metrics["intrusion_spatial_f1"]["value"], 1.0)
            self.assertGreater(metrics["intrusion_spatial_f1"]["excluded_unknown_cell_pairs"], 0)
            self.assertEqual(
                metrics["unknown_to_non_actionable_violation_count"], 0
            )
            self.assertEqual(metrics["parent_event_sequence_match"]["value"], 1.0)
        result_path = self.root / "result.json"
        write_json(result_path, result)
        receipt = validate(
            bundle_root=self.bundle_root,
            result_path=result_path,
            scoring_key_path=self.key_path,
            submissions=submissions,
        )
        self.assertEqual(receipt["status"], "VALID")

    def test_disagreement_is_counted_not_repaired(self) -> None:
        result = score(
            bundle_root=self.bundle_root,
            scoring_key_path=self.key_path,
            submissions=self._submissions(disagreement=True),
        )
        current = result["condition_metrics"]["CURRENT_ONLY"]
        self.assertEqual(current["alertable_exact_agreement"]["matches"], 3)
        self.assertEqual(current["derived"]["exact_agreement"]["matches"], 3)
        self.assertEqual(current["parent_event_sequence_match"]["matches"], 0)


if __name__ == "__main__":
    unittest.main()

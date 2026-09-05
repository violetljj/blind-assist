"""Synthetic mechanics only; no natural-camera or physical-ownership claims."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from l10_candidate_carrier_association import AssociationState, BoundedCarrierContinuity, associate_frame
from l10_candidate_carrier_replay import adapt_observation

_NAMED_POI_PATH = Path(__file__).resolve().parents[1] / "named-poi-v1"
HAS_NAMED_POI_WIP = (_NAMED_POI_PATH / "named_poi_v1.py").is_file()
if HAS_NAMED_POI_WIP:
    sys.path.insert(0, str(_NAMED_POI_PATH))
    from named_poi_v1 import Decision, FusionConfig, NamedPoiFusion, NamedPoiSequence, PoiTarget, SequencePhase


TARGET = PoiTarget("hospital", ("Central Hospital",), 22.28, 114.18) if HAS_NAMED_POI_WIP else None


def candidate(candidate_id="right", box=(0.6, 0.2, 0.9, 0.9), score=0.8, track="right-track"):
    return {
        "candidate_id": candidate_id, "bbox_xyxy_norm": list(box), "poi_id": "hospital", "score": score,
        "track_id": track, "track_evidence_id": f"synthetic-tracker:{candidate_id}",
    }


def text(text_id="name", value="Central Hospital", box=(0.65, 0.3, 0.85, 0.4), role="IDENTITY_SIGN"):
    return {
        "text_id": text_id, "text": value, "bbox_xyxy_norm": list(box),
        "role": role, "role_evidence_id": "synthetic-role-producer",
    }


def frame(frame_id="1", *, candidates=None, texts=None, relations=()):
    return {
        "source_id": "synthetic-source", "frame_id": frame_id, "roi_id": "full-frame",
        "candidate_roster_complete": True,
        "location": {"latitude": 22.28, "longitude": 114.18},
        "appearance_evidence": [candidate()] if candidates is None else candidates,
        "text_evidence": [text()] if texts is None else texts, "carrier_relations": list(relations),
    }


def relation(predicate="IDENTITY_TEXT_OF", status="SUPPORTED", **overrides):
    return {
        "text_id": "name", "candidate_id": "right", "source_id": "synthetic-source",
        "frame_id": "1", "roi_id": "full-frame", "evidence_id": "synthetic-relation-producer",
        "predicate": predicate, "status": status, **overrides,
    }


class NativeCarrierContractTest(unittest.TestCase):
    def test_geometric_attachment_and_identity_attachment_are_distinct(self):
        local = associate_frame(frame(texts=[text(role="UNKNOWN")])).candidates[0]
        named = associate_frame(frame()).candidates[0]
        self.assertEqual(AssociationState.LOCAL_SUPPORT, local.state)
        self.assertEqual(AssociationState.IDENTITY_SUPPORT, named.state)

    def test_text_on_right_does_not_attach_to_left(self):
        observed = frame(candidates=[candidate("left", (0.1, 0.2, 0.4, 0.9), 0.99), candidate()])
        left, right = associate_frame(observed).candidates
        self.assertFalse(left.local_text_ids)
        self.assertEqual(("name",), right.identity_text_ids)

    def test_same_text_with_two_carriers_stays_unresolved(self):
        result = associate_frame(frame(candidates=[candidate("outer", (0, 0, 1, 1)), candidate()]))
        self.assertTrue(all(item.state == AssociationState.UNRESOLVED for item in result.candidates))

    def test_local_crop_edge_retains_support_outside_endpoint_box(self):
        observed = frame(texts=[text(box=(0.6, 0.05, 0.8, 0.15), role="UNKNOWN")], relations=[relation("LOCAL_OBSERVATION_OF")])
        item = associate_frame(observed).candidates[0]
        self.assertEqual(AssociationState.LOCAL_SUPPORT, item.state)
        self.assertFalse(item.identity_texts)

    def test_private_crop_witness_keeps_its_own_roi_without_inventing_a_transform(self):
        observed = frame(texts=[{
            "text_id": "name", "text": "Central Hospital", "roi_id": "private-crop:right",
            "bbox_xyxy_px": [449, 438, 567, 533], "role": "UNKNOWN",
        }], relations=[relation(
            "LOCAL_OBSERVATION_OF", text_roi_id="private-crop:right", candidate_roi_id="full-frame",
        )])
        item = associate_frame(observed).candidates[0]
        self.assertEqual(AssociationState.LOCAL_SUPPORT, item.state)
        self.assertFalse(item.identity_texts)
        observed["carrier_relations"] = []
        self.assertEqual(AssociationState.UNRESOLVED, associate_frame(observed).candidates[0].state)

    def test_missing_provenance_does_not_support_identity(self):
        observed = frame()
        observed.pop("source_id")
        self.assertEqual(AssociationState.UNRESOLVED, associate_frame(observed).candidates[0].state)

    def test_advertisement_role_and_explicit_unknown_relation_do_not_support_identity(self):
        for observed in (frame(texts=[text(role="ADVERTISEMENT")]), frame(relations=[relation(status="UNKNOWN")])):
            self.assertFalse(associate_frame(observed).candidates[0].identity_texts)

    def test_complete_name_is_preserved_for_target_conditioned_adapters(self):
        observed = frame(texts=[{**text(), "name_completeness": "COMPLETE"}])
        self.assertEqual(("Central Hospital",), associate_frame(observed).candidates[0].complete_identity_names)

    def test_continuity_is_same_track_and_bounded(self):
        lease = BoundedCarrierContinuity(2)
        lease.confirm(associate_frame(frame()).candidates[0])
        self.assertTrue(lease.advance(associate_frame(frame("2", texts=[])).candidates[0]))
        self.assertTrue(lease.advance(associate_frame(frame("3", texts=[])).candidates[0]))
        self.assertFalse(lease.advance(associate_frame(frame("4", texts=[])).candidates[0]))

    def test_repeated_frame_or_new_track_cannot_refresh_continuity(self):
        for observed in (frame(texts=[]), frame("2", texts=[], candidates=[candidate(track="other")])):
            lease = BoundedCarrierContinuity()
            lease.confirm(associate_frame(frame()).candidates[0])
            self.assertFalse(lease.advance(associate_frame(observed).candidates[0]))

    def test_a_new_roi_on_the_same_frame_does_not_refresh_continuity(self):
        lease = BoundedCarrierContinuity()
        lease.confirm(associate_frame(frame()).candidates[0])
        observed = frame(texts=[])
        observed["roi_id"] = "another-crop"
        self.assertFalse(lease.advance(associate_frame(observed).candidates[0]))

    def test_explicit_contradiction_revokes_a_track(self):
        lease = BoundedCarrierContinuity()
        lease.confirm(associate_frame(frame()).candidates[0])
        observed = frame("2", relations=[relation(status="CONTRADICTED", frame_id="2")])
        self.assertFalse(lease.advance(associate_frame(observed).candidates[0]))
        self.assertIsNone(lease.track_key)


class LegacyAdapterContractTest(unittest.TestCase):
    def observation(self, private=False):
        binding = {
            "portal_id": "P1", "portal_box_xyxy": [0, 470, 133, 854],
            "portal_confidence": 0.8, "text": "7010", "text_score": 0.9,
            "authority_edge": None if not private else {
                "type": "PORTAL_TO_PRIVATE_MEDIUM_OBSERVATION_TO_EXACT_TOKEN",
                "source_portal_id": "P1", "unique_portal_witness_count": 1,
            },
        }
        return {
            "render_receipt": {"viewport_shape_hwc": [1280, 1440, 3], "viewport_pixel_sha256": "synthetic-frame"},
            "runtime_output": {
                "selected_binding": binding, "portal_candidate_count": 2,
                "state": "PIXEL_BOUND_PPOCRV6_MEDIUM_UNIQUE_PORTAL_WITNESS" if private else "PIXEL_BOUND_MASK_PORTAL_PROPOSAL",
                "observation_branch": "PPOCRV6_MEDIUM_UNIQUE_PORTAL_WITNESS" if private else "V2_RESULT_RETAINED",
                "exact_target_text_candidates": [{"canonical": "7010", "box_xyxy": [449, 438, 567, 533]}],
            },
        }

    def test_crop_local_coordinates_are_never_treated_as_full_frame(self):
        public = adapt_observation(self.observation(private=True), "synthetic")
        self.assertNotIn("bbox_xyxy_norm", public["text_evidence"][0])
        self.assertEqual([449, 438, 567, 533], public["text_evidence"][0]["bbox_xyxy_px"])
        result = associate_frame(public).candidates[0]
        self.assertEqual(AssociationState.LOCAL_SUPPORT, result.state)
        self.assertFalse(result.identity_texts)

    def test_existing_topology_halo_witness_retains_only_its_local_scope(self):
        observation = self.observation()
        public = adapt_observation(observation, "synthetic")
        result = associate_frame(public).candidates[0]
        self.assertEqual(AssociationState.LOCAL_SUPPORT, result.state)
        self.assertEqual(("selected-producer-text",), result.local_text_ids)
        self.assertFalse(result.identity_texts)
        observation["evaluator_truth"] = {"outcome": "WRONG_BINDING", "target_visible": False}
        self.assertEqual(public, adapt_observation(observation, "synthetic"))


@unittest.skipUnless(HAS_NAMED_POI_WIP, "Optional preexisting Named-POI WIP is absent")
class CarrierAssociationTest(unittest.TestCase):
    def test_goal_text_on_right_cannot_be_borrowed_by_higher_scoring_left(self):
        observed = frame(candidates=[candidate("left", (0.1, 0.2, 0.4, 0.9), 0.99, "left-track"), candidate()])
        result = NamedPoiFusion().decide(TARGET, observed)
        self.assertEqual(Decision.CONFIRM_FULL_TEXT, result.decision)
        self.assertEqual("right", result.carrier.candidate_id)
        self.assertAlmostEqual(0.75, result.bbox_x_center)
        observed["appearance_evidence"] = observed["appearance_evidence"][:1]
        borrowed = NamedPoiFusion().decide(TARGET, observed)
        self.assertIsNone(borrowed.confirmed_poi_id)
        self.assertIsNone(borrowed.bbox_x_center)

    def test_matched_carrier_does_not_require_complete_endpoint_extent(self):
        observed = frame(texts=[text(box=(0.6, 0.05, 0.8, 0.15), role="UNKNOWN")], relations=[relation()])
        self.assertEqual(Decision.CONFIRM_FULL_TEXT, NamedPoiFusion().decide(TARGET, observed).decision)

    def test_geometric_overlap_alone_is_local_support(self):
        observed = frame(texts=[text(role="UNKNOWN")])
        result = associate_frame(observed)
        self.assertEqual(AssociationState.LOCAL_SUPPORT, result.candidates[0].state)
        self.assertIn("NO_OWNERSHIP", result.claim_scope)
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_advertisement_and_direction_sign_do_not_name_the_carrier(self):
        for role in ("ADVERTISEMENT", "DIRECTION_SIGN", "SIBLING_SIGN"):
            with self.subTest(role=role):
                observed = frame(texts=[text(role=role)], relations=[relation()])
                self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_same_name_on_adjacent_doors_remains_ambiguous(self):
        observed = frame(
            candidates=[candidate("left", (0.1, 0.2, 0.4, 0.9), 0.99), candidate()],
            texts=[text("left-name", box=(0.15, 0.3, 0.35, 0.4)), text()],
        )
        result = NamedPoiFusion().decide(TARGET, observed)
        self.assertEqual(Decision.SCAN, result.decision)
        self.assertIn("ambiguous", result.reason)

    def test_competing_overlapping_carriers_do_not_pick_by_appearance(self):
        observed = frame(candidates=[candidate("outer", (0.5, 0.1, 1, 1), 0.99), candidate()])
        self.assertEqual(("name",), associate_frame(observed).unresolved_text_ids)
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_explicit_unique_relation_resolves_spatial_ambiguity(self):
        observed = frame(candidates=[candidate("outer", (0.5, 0.1, 1, 1), 0.99), candidate()], relations=[relation()])
        self.assertEqual("right", NamedPoiFusion().decide(TARGET, observed).carrier.candidate_id)

    def test_private_crop_support_is_preserved_without_identity_promotion(self):
        observed = frame(texts=[text(box=(0.6, 0.05, 0.8, 0.15), role="UNKNOWN")], relations=[relation("LOCAL_OBSERVATION_OF")])
        item = associate_frame(observed).candidates[0]
        self.assertEqual(AssociationState.LOCAL_SUPPORT, item.state)
        self.assertEqual(("name",), item.local_text_ids)
        self.assertFalse(item.identity_texts)

    def test_missing_spatial_input_is_unresolved_not_absent(self):
        observed = frame(texts=[{"text": "Central Hospital"}])
        self.assertEqual(Decision.SCAN, NamedPoiFusion().decide(TARGET, observed).decision)
        self.assertEqual(AssociationState.UNRESOLVED, associate_frame(observed).candidates[0].state)

    def test_source_frame_roi_and_valid_box_are_required(self):
        for key, value in (("source_id", "other"), ("frame_id", "other"), ("roi_id", "other"), ("bbox_xyxy_norm", [0, 0, float("nan"), 1])):
            with self.subTest(key=key):
                observed = frame()
                observed["text_evidence"][0][key] = value
                self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_explicit_relation_cannot_bypass_frame_provenance(self):
        observed = frame(texts=[text(role="UNKNOWN")], relations=[relation(frame_id="old")])
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_unknown_or_sibling_relation_is_not_overridden_by_overlap(self):
        for edge in (relation(status="UNKNOWN"), relation(predicate="SIBLING_OF")):
            self.assertIsNone(NamedPoiFusion().decide(TARGET, frame(relations=[edge])).confirmed_poi_id)

    def test_two_stale_rows_cannot_claim_the_current_frame(self):
        observed = frame()
        observed["appearance_evidence"][0]["frame_id"] = "old"
        observed["text_evidence"][0]["frame_id"] = "old"
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_missing_target_and_nonmatching_text_provide_no_identity(self):
        for observed in (frame(candidates=[], texts=[text()]), frame(texts=[text(value="East Pharmacy")]), frame(texts=[])):
            self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_explicit_complete_other_name_cannot_fall_through_to_partial_match(self):
        observed = frame(texts=[{**text(value="Another Hospital"), "name_completeness": "COMPLETE"}])
        result = NamedPoiFusion().decide(TARGET, observed)
        self.assertIsNone(result.confirmed_poi_id)
        self.assertEqual(AssociationState.CONTRADICTED, result.carrier.state)
        observed["text_evidence"][0]["text"] = "Central Hospital"
        self.assertEqual(Decision.CONFIRM_FULL_TEXT, NamedPoiFusion().decide(TARGET, observed).decision)

    def test_partial_candidate_roster_cannot_establish_unique_identity_by_geometry(self):
        observed = frame()
        observed["candidate_roster_complete"] = False
        self.assertEqual(AssociationState.LOCAL_SUPPORT, associate_frame(observed).candidates[0].state)
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)

    def test_two_text_rows_can_support_one_carrier_without_first_row_poisoning_it(self):
        observed = frame(texts=[text("advert", "Sale", role="ADVERTISEMENT"), text()])
        self.assertEqual(Decision.CONFIRM_FULL_TEXT, NamedPoiFusion().decide(TARGET, observed).decision)

    def test_duplicate_candidate_and_text_ids_are_unresolved(self):
        observed = frame(candidates=[candidate(), candidate()])
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)
        observed = frame(texts=[text(), text()])
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)


@unittest.skipUnless(HAS_NAMED_POI_WIP, "Optional preexisting Named-POI WIP is absent")
class CarrierContinuityTest(unittest.TestCase):
    def test_same_track_continues_without_borrowing_higher_scoring_sibling(self):
        sequence = NamedPoiSequence(TARGET)
        self.assertEqual(SequencePhase.CONFIRMED, sequence.observe(frame()).phase)
        current = frame("2", candidates=[candidate("left", (0.1, 0.2, 0.4, 0.9), 0.99, "left-track"), candidate(score=0.65)], texts=[])
        result = sequence.observe(current)
        self.assertEqual(SequencePhase.TRACKING, result.phase)
        self.assertEqual("right", result.identity.carrier.candidate_id)

    def test_new_source_or_sibling_cannot_inherit_a_lock(self):
        for kind in ("source", "track"):
            sequence = NamedPoiSequence(TARGET, max_hold_frames=0)
            sequence.observe(frame())
            current = frame("2", texts=[])
            if kind == "source":
                current["source_id"] = "different-source"
            else:
                current["appearance_evidence"][0]["track_id"] = "sibling"
            self.assertEqual(SequencePhase.LOST_SCAN, sequence.observe(current).phase)

    def test_track_only_continuity_has_a_fixed_lifetime(self):
        sequence = NamedPoiSequence(TARGET, FusionConfig(max_track_only_frames=2), max_hold_frames=0)
        sequence.observe(frame())
        self.assertEqual(SequencePhase.TRACKING, sequence.observe(frame("2", texts=[])).phase)
        self.assertEqual(SequencePhase.TRACKING, sequence.observe(frame("3", texts=[])).phase)
        self.assertEqual(SequencePhase.LOST_SCAN, sequence.observe(frame("4", texts=[])).phase)

    def test_replayed_frame_does_not_refresh_continuity(self):
        sequence = NamedPoiSequence(TARGET, max_hold_frames=0)
        sequence.observe(frame())
        self.assertEqual(SequencePhase.LOST_SCAN, sequence.observe(frame(texts=[])).phase)

    def test_current_relation_contradiction_revokes_previous_identity(self):
        sequence = NamedPoiSequence(TARGET)
        sequence.observe(frame())
        current = frame("2", relations=[relation(status="CONTRADICTED", frame_id="2")])
        self.assertEqual(SequencePhase.LOST_SCAN, sequence.observe(current).phase)

    def test_complete_other_name_revokes_the_same_track_binding(self):
        sequence = NamedPoiSequence(TARGET)
        sequence.observe(frame())
        current = frame("2", texts=[{**text(value="Another Hospital"), "name_completeness": "COMPLETE"}])
        self.assertEqual(SequencePhase.LOST_SCAN, sequence.observe(current).phase)

    def test_missing_track_receipt_does_not_continue_identity(self):
        sequence = NamedPoiSequence(TARGET, max_hold_frames=0)
        sequence.observe(frame())
        current = frame("2", texts=[])
        current["appearance_evidence"][0].pop("track_evidence_id")
        self.assertEqual(SequencePhase.LOST_SCAN, sequence.observe(current).phase)

    def test_fusion_ignores_evaluator_only_target_labels(self):
        observed = frame(texts=[])
        observed["target_visible"] = True
        observed["evaluator_truth"] = {"target_present": True, "target_poi_id": "hospital", "candidate_id": "right"}
        self.assertIsNone(NamedPoiFusion().decide(TARGET, observed).confirmed_poi_id)


if __name__ == "__main__":
    unittest.main()

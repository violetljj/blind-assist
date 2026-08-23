from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0 import audit_sun3d_referent_identifiability as sut


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "artifacts.local" / "evidence" / "sun3d-native-door-approach-v0"
REVIEW = Path(__file__).with_name("sun3d_referent_identifiability_review_v0.json")


@unittest.skipUnless(EVIDENCE.exists(), "sealed local SUN3D evidence is unavailable")
class Sun3dReferentIdentifiabilityAuditTest(unittest.TestCase):
    def test_consumed_episode_is_not_publicly_bound_to_object_45(self) -> None:
        roster = json.loads((EVIDENCE / "frozen-roster-private-truth.json").read_text(encoding="utf-8"))
        response = sut.requests.get(roster["source"]["annotation_url"], timeout=90)
        response.raise_for_status()
        annotation_bytes = response.content
        audit = sut.build_audit(
            roster,
            json.loads((EVIDENCE / "pixels-manifest.json").read_text(encoding="utf-8")),
            json.loads((EVIDENCE / "formal" / "proposal-provider-output.json").read_text(encoding="utf-8")),
            json.loads((EVIDENCE / "formal" / "final-report.json").read_text(encoding="utf-8")),
            json.loads(annotation_bytes),
            sut.sun3d._sha256_bytes(annotation_bytes),
            json.loads(REVIEW.read_text(encoding="utf-8")),
        )
        self.assertEqual(sut.TERMINAL, audit["terminal"])
        self.assertEqual("AMBIGUOUS", audit["episode_referent_resolution"])
        self.assertEqual(3, audit["critical_counts"]["usable_proposal_frames_ambiguous_for_private_target"])
        self.assertEqual(
            "NOT_EVALUABLE_PUBLIC_REFERENT_NOT_IDENTIFIABLE",
            audit["claim_disposition"]["selection_given_usable_proposal_0_of_3"],
        )
        self.assertEqual([{"object_id": 57, "name": "door: bathroom"}], audit["other_native_door_family_objects"])


if __name__ == "__main__":
    unittest.main()

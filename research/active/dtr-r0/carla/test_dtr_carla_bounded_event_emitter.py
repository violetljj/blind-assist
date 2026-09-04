from __future__ import annotations

import unittest

import dtr_carla_bounded_event_emitter as emitter


def evidence(
    risk: bool,
    *,
    receipt: str = "r1",
    changed: bool = False,
    entry: float | None = 1.0,
) -> dict:
    return {
        "route_mode": "ISSUED_PLAN",
        "authority": "VALID",
        "plan_receipt_sha256": receipt,
        "route_mode_changed": changed,
        "route_risk": risk,
        "minimum_entry_s": entry if risk else None,
    }


class BoundedEventEmitterTest(unittest.TestCase):
    def test_hold_is_bounded_without_changing_collision_evidence(self) -> None:
        layer = emitter.BoundedEventEmitter()
        self.assertEqual(
            "ACTIVE_EVIDENCE",
            layer.update(time_s=1.0, evidence=evidence(True))["emission_state"],
        )
        held = layer.update(time_s=1.4, evidence=evidence(False))
        self.assertEqual("ACTIVE_BOUNDED_HOLD", held["emission_state"])
        self.assertTrue(held["route_risk"])
        self.assertFalse(held["raw_evidence_route_risk"])
        self.assertEqual("CLEAR", layer.update(time_s=1.61, evidence=evidence(False))["emission_state"])

    def test_route_change_releases_immediately(self) -> None:
        layer = emitter.BoundedEventEmitter()
        layer.update(time_s=2.0, evidence=evidence(True))
        released = layer.update(
            time_s=2.1,
            evidence=evidence(False, receipt="r2", changed=True),
        )
        self.assertEqual("CLEAR", released["emission_state"])
        self.assertFalse(released["route_risk"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from .freeze_fresh_holdout import HoldoutFreezeError, _protocol_session_ids


class FreshHoldoutFreezeTests(unittest.TestCase):
    def _protocol(self) -> dict[str, object]:
        return {
            "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
            "status": "DESIGN_FROZEN",
            "data_roles": {
                "fresh_source_native_pixel_truth": {
                    "status_before_acquisition": "FROZEN_IDENTITY_PIXEL_TRUTH_NOT_ACCESSED",
                    "official_split": "test",
                    "camera": "camera_chest",
                    "lens": "left",
                    "session_ids": ["a", "b"],
                }
            },
        }

    def test_protocol_identity_is_required(self) -> None:
        protocol = self._protocol()
        self.assertEqual(_protocol_session_ids(protocol), ["a", "b"])
        protocol["status"] = "DRAFT"
        with self.assertRaises(HoldoutFreezeError):
            _protocol_session_ids(protocol)

    def test_duplicate_identity_is_rejected(self) -> None:
        protocol = self._protocol()
        role = protocol["data_roles"]["fresh_source_native_pixel_truth"]
        role["session_ids"] = ["a", "a"]
        with self.assertRaises(HoldoutFreezeError):
            _protocol_session_ids(protocol)


if __name__ == "__main__":
    unittest.main()

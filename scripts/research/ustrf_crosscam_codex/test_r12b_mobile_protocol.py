from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs/ustrf_crosscam_mobile_r12b_prereg_v1.json"
TRANSPORT_V1 = ROOT / "configs/ustrf_crosscam_mobile_r12b_exact_frame_transport_v1.json"
TRANSPORT_V2 = ROOT / "configs/ustrf_crosscam_mobile_r12b_exact_frame_transport_v2.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R12bMobileProtocolTest(unittest.TestCase):
    def test_candidate_order_and_hard_stops_remain_frozen(self) -> None:
        protocol = load(PROTOCOL)
        self.assertEqual("seen_diagnostic_not_held_out", protocol["dataset_role"])
        self.assertFalse(protocol["authority"]["new_held_out_read"])
        self.assertTrue(protocol["parent_protocol"]["r13_inventory_must_remain_locked"])
        self.assertEqual(
            ["r12b_c1_same640_gpu_delegate", "r12b_c2_sameweights_fp16_320_cpu", "r12b_c3_same320_gpu_delegate"],
            [row["candidate_id"] for row in protocol["candidate_ladder"]],
        )
        self.assertEqual("first_candidate_passing_all_canary_gates", protocol["selection_and_stop"]["selection_rule"])
        self.assertFalse(protocol["selection_and_stop"]["vancouver_may_select_or_reject_candidate"])
        self.assertFalse(protocol["frozen_detector_semantics"]["prompt_or_class_change_allowed"])
        self.assertFalse(protocol["frozen_detector_semantics"]["threshold_change_allowed"])
        self.assertFalse(protocol["frozen_detector_semantics"]["bbox_or_polygon_change_allowed"])

    def test_decoder_only_transport_amendment_is_hash_bound(self) -> None:
        v1 = load(TRANSPORT_V1)
        v2 = load(TRANSPORT_V2)
        self.assertEqual(sha256(TRANSPORT_V1), v2["supersedes_transport_contract_sha256"])
        self.assertEqual("2/9", v2["parent"]["failed_ffmpeg_pixel_parity"])
        self.assertEqual("opencv VideoCapture CAP_PROP_POS_MSEC", v2["transport"]["decoder"])
        self.assertEqual("9/9", v2["admission"]["static_canary_host_pixel_parity_required"])
        self.assertEqual("9/9", v2["admission"]["static_canary_exact_frame_target_status_parity_required"])
        self.assertEqual(v1["selected_candidate"], v2["selected_candidate"])
        self.assertFalse(v2["hard_stops"]["c2_or_c3_may_run_after_c1_pass"])
        self.assertFalse(v2["hard_stops"]["vancouver_may_change_transport_or_candidate"])
        self.assertFalse(v2["hard_stops"]["r13_inventory_may_open"])
        sidecar_sha = TRANSPORT_V2.with_suffix(TRANSPORT_V2.suffix + ".sha256").read_text().split()[0]
        self.assertEqual(sha256(TRANSPORT_V2), sidecar_sha)


if __name__ == "__main__":
    unittest.main()

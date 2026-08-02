from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ingest_egowalk_metadata import _segments
from materialize_pending_package import _reason
from pipeline import dataset_id_for_ledger_row, role_for_ledger_row, stable_id


def test_dataset_mapping_and_roles() -> None:
    assert dataset_id_for_ledger_row({"dataset": "SANPO", "session_root": "sanpo-real/session"}) == "SANPO-Real"
    assert dataset_id_for_ledger_row({"dataset": "SANPO", "session_root": "sanpo-synthetic/session"}) == "SANPO-Synthetic"
    assert dataset_id_for_ledger_row({"dataset": "JRDB", "session_root": "x"}) == "JRDB"
    assert role_for_ledger_row({"is_consumed": "True", "is_burned": "False", "history_roles": "train"}) == "THESIS_DEVELOPMENT_CONSUMED"
    assert role_for_ledger_row({"is_consumed": "False", "is_burned": "False", "history_roles": "fresh"}) == "HOLD_ROLE_REVIEW"


def test_stable_id() -> None:
    assert stable_id("x", "a", 1) == stable_id("x", "a", 1)
    assert stable_id("x", "a", 1) != stable_id("x", "a", 2)


def test_contract_json_is_valid() -> None:
    contract_path = Path(__file__).with_name("contract.json")
    data = json.loads(contract_path.read_text(encoding="utf-8"))
    assert data["schema"] == "hftf_d7_public_real_contract_v1"
    assert data["review_firewall"]["confirmation_candidate_selection_see_model_output"] is False
    assert "NOT_EVALUABLE" in data["event_buckets"]


def test_egowalk_segments_fail_closed_on_gap_and_invalid_pose() -> None:
    def row(frame: int, timestamp: int, x: float = 0.0) -> dict[str, object]:
        return {
            "frame": frame,
            "timestamp": timestamp,
            "cart_x": x,
            "cart_y": 0.0,
            "cart_z": 0.0,
            "quat_x": 0.0,
            "quat_y": 0.0,
            "quat_z": 0.0,
            "quat_w": 1.0,
        }

    rows = [row(0, 1000), row(1, 1200), row(2, 2500), row(3, 2700), row(4, 2900), row(5, 3100)]
    rows[4]["cart_x"] = None
    segments = _segments(rows, max_gap_ms=1000, max_position_jump_m=5.0)
    assert [[item["frame"] for item in segment] for segment in segments] == [[0, 1], [2, 3], [5]]


def test_egowalk_review_reason_changes_only_after_complete_rgb_receipt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "receipts").mkdir()
        candidate = {"dataset_id": "EgoWalk"}
        assert _reason(candidate, output_root=root) == "RAW_RGB_OR_EXTRACTED_VIDEO_REVIEW_NOT_LAWFULLY_CLOSED"
        (root / "receipts" / "egowalk_rgb_receipt_test.json").write_text(
            json.dumps({"status": "PARTIAL_DOWNLOAD"}), encoding="utf-8"
        )
        assert _reason(candidate, output_root=root) == "RAW_RGB_OR_EXTRACTED_VIDEO_REVIEW_NOT_LAWFULLY_CLOSED"
        (root / "receipts" / "egowalk_rgb_receipt_test.json").write_text(
            json.dumps({"status": "PUBLIC_EXTRACTED_RGB_DOWNLOADED"}), encoding="utf-8"
        )
        assert _reason(candidate, output_root=root) == "INDEPENDENT_RGB_GEOMETRY_REVIEW_NOT_RUN"

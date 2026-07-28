from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "rgb_segment_confirmation_mvsec_r1"))
sys.path.insert(0, str(BASE / "rgb_segment_confirmation_mvsec_r2"))
parser_core = importlib.import_module("mvsec_target_chunks")
evidence = importlib.import_module("mvsec_identity_evidence_r2")


def image(timestamp_ns: int, sequence: int):
    payload = bytes([sequence]) * 12
    return parser_core.ImageMessage(
        bag_timestamp_ns=timestamp_ns,
        header_timestamp_ns=timestamp_ns,
        sequence=sequence,
        frame_id="davis_left",
        encoding="mono8",
        height=3,
        width=4,
        is_bigendian=0,
        step=4,
        payload=payload,
        serialized_sha256=f"{sequence:064x}",
        payload_sha256=f"{sequence + 100:064x}",
    )


class EvidenceR2Tests(unittest.TestCase):
    def test_exception_code_is_allowlisted_not_raw_arbitrary_text(self) -> None:
        try:
            raise ValueError("IMAGE_GEOMETRY_PAIRING_TIE")
        except ValueError as error:
            safe = evidence.classify_exception(error)
        self.assertEqual(safe["error_code"], "IMAGE_GEOMETRY_PAIRING_TIE")

        try:
            raise ValueError("secret-like arbitrary message / path")
        except ValueError as error:
            redacted = evidence.classify_exception(error)
        self.assertEqual(redacted["error_code"], "UNCLASSIFIED_VALUEERROR")
        self.assertNotIn("secret-like", json.dumps(redacted))

    def test_pairing_diagnostics_exposes_tie_reuse_and_guards(self) -> None:
        tie = evidence.pairing_diagnostics(
            [image(80, 1), image(90, 2), image(110, 3), image(120, 4)],
            geometry_timestamps_ns=[100],
            maximum_delta_ns=10,
        )
        self.assertEqual(tie["nearest_tie_count"], 1)
        self.assertFalse(tie["guard_before_available"])
        self.assertFalse(tie["guard_after_available"])

        reuse = evidence.pairing_diagnostics(
            [image(80, 1), image(100, 2), image(120, 3)],
            geometry_timestamps_ns=[99, 101],
            maximum_delta_ns=2,
        )
        self.assertEqual(reuse["nearest_reuse_count"], 1)

        valid = evidence.pairing_diagnostics(
            [image(80, 1), image(100, 2), image(120, 3)],
            geometry_timestamps_ns=[100],
            maximum_delta_ns=1,
        )
        self.assertTrue(valid["guard_before_available"])
        self.assertTrue(valid["guard_after_available"])

    def test_metadata_and_pairing_ledgers_are_hash_chained_no_pixels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            metadata_path = repo / "metadata.jsonl"
            metadata = evidence.ImageMetadataLedger(metadata_path)
            row = image(100, 1)
            metadata.append(
                row,
                chunk_index=7,
                chunk_pos=1000,
                record_ordinal=2,
                record_offset=40,
                serialized_bytes=80,
            )
            binding = metadata.binding(repo)
            self.assertFalse(binding["pixel_payload_materialized"])
            self.assertNotIn(row.payload, metadata_path.read_bytes())

            pairing_path = repo / "pairing.jsonl"
            evidence.write_pairing_diagnostic_ledger(
                pairing_path,
                [image(80, 1), image(100, 2), image(120, 3)],
                geometry_timestamps_ns=[100],
                maximum_delta_ns=1,
            )
            for path, sequence_field in (
                (metadata_path, "sequence_index"),
                (pairing_path, "sequence_index"),
            ):
                previous = None
                for index, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines()
                ):
                    payload = json.loads(line)
                    saved = payload.pop("row_sha256")
                    canonical = json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    actual = hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest()
                    self.assertEqual(payload[sequence_field], index)
                    self.assertEqual(payload["previous_row_sha256"], previous)
                    self.assertEqual(saved, actual)
                    previous = saved


if __name__ == "__main__":
    unittest.main()

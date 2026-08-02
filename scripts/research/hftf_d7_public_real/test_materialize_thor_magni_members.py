from __future__ import annotations

import struct
import unittest

from materialize_thor_magni_members import (
    LOCAL_HEADER_SIGNATURE,
    _decode_payload,
    _parse_local_header,
    _read_range,
    _safe_member_path,
    _validate_payload,
)
from pipeline import ContractError


class MaterializeThorMagniMembersTest(unittest.TestCase):
    def test_range_reader_is_chunked_at_request_cap(self) -> None:
        class FakeReader:
            max_request_bytes = 3

            def __init__(self) -> None:
                self.position = 0
                self.payload = b"0123456789"

            def seek(self, offset: int) -> None:
                self.position = offset

            def read(self, size: int) -> bytes:
                value = self.payload[self.position : self.position + size]
                self.position += len(value)
                return value

        self.assertEqual(_read_range(FakeReader(), 2, 7), b"2345678")

    def test_local_header_parser_reads_offset_fields(self) -> None:
        header = struct.pack(
            "<4s5H3I2H",
            LOCAL_HEADER_SIGNATURE,
            20,
            0,
            8,
            0,
            0,
            0,
            4,
            8,
            5,
            7,
        )
        self.assertEqual(_parse_local_header(header, member_name="x.txt"), (8, 5, 7))

    def test_path_traversal_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            _safe_member_path(r"F:\\tmp\\raw", "THOR_MAGNI/../escape.txt")

    def test_deflate_payload_is_verified(self) -> None:
        import zlib

        payload = b"d7-source-intake"
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(payload) + compressor.flush()
        decoded = _decode_payload(compressed, compression=8, member_name="x.txt")
        self.assertEqual(decoded, payload)
        row = {"file_size": len(payload), "crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"}
        self.assertEqual(len(_validate_payload(decoded, row, member_name="x.txt")), 64)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from io import BytesIO
import struct
import tarfile
import unittest
import zipfile

from scripts.research.svrf_o0.probe_materialization_capability import (
    parse_tar_header,
    parse_zip_central_directory,
    zip_directory_location,
)


class SvrfO0MaterializationCapabilityTest(unittest.TestCase):
    def test_tar_header_parser_preserves_member_identity_and_size(self) -> None:
        payload = BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            info = tarfile.TarInfo("locked/source/frame.png")
            info.size = 3
            archive.addfile(info, BytesIO(b"rgb"))
        header = parse_tar_header(payload.getvalue()[:512])
        self.assertEqual(header["name"], "locked/source/frame.png")
        self.assertEqual(header["size"], 3)

    def test_zip_directory_location_and_member_parser(self) -> None:
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("spring/train/0005/frame_left/frame.png", b"rgb")
            archive.writestr("spring/train/0014/frame_left/frame.png", b"more-rgb")
        data = payload.getvalue()
        offset, size, count = zip_directory_location(data, total_bytes=len(data), tail_start=0)
        members = parse_zip_central_directory(data[offset : offset + size], expected_entries=count)
        self.assertEqual(count, 2)
        self.assertEqual(
            [member["name"] for member in members],
            [
                "spring/train/0005/frame_left/frame.png",
                "spring/train/0014/frame_left/frame.png",
            ],
        )

    def test_zip64_location_uses_64_bit_directory_identity(self) -> None:
        directory_offset = 2**32 + 123
        directory_size = 630_200
        entry_count = 5_000
        zip64_offset = directory_offset + directory_size
        zip64 = struct.pack(
            "<4sQ2H2L4Q",
            bytes.fromhex("504b0606"),
            44,
            798,
            45,
            0,
            0,
            entry_count,
            entry_count,
            directory_size,
            directory_offset,
        )
        locator = struct.pack("<4sLQL", bytes.fromhex("504b0607"), 0, zip64_offset, 1)
        eocd = struct.pack(
            "<4s4H2LH",
            bytes.fromhex("504b0506"),
            0,
            0,
            0xFFFF,
            0xFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0,
        )
        tail = zip64 + locator + eocd
        self.assertEqual(
            zip_directory_location(tail, total_bytes=zip64_offset + len(tail), tail_start=zip64_offset),
            (directory_offset, directory_size, entry_count),
        )


if __name__ == "__main__":
    unittest.main()

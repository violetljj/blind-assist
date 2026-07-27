from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import fetch_remote_zip_members as fetch


class RangeBytes(BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.payload = payload
        self.requests = 0
        self.transferred = 0

    def _range(self, start: int, end: int):
        self.requests += 1
        body = self.payload[start : end + 1]
        self.transferred += len(body)
        return body, {}


class CoalescedExtractionTest(unittest.TestCase):
    def test_stored_and_deflated_members_are_crc_checked(self) -> None:
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "depth/1.png",
                b"first-depth",
                compress_type=zipfile.ZIP_STORED,
            )
            archive.writestr(
                "depth/2.png",
                b"second-depth" * 20,
                compress_type=zipfile.ZIP_DEFLATED,
            )
        reader = RangeBytes(payload.getvalue())
        with zipfile.ZipFile(reader) as archive:
            selected = archive.infolist()
            with tempfile.TemporaryDirectory() as directory:
                runs = fetch.extract_coalesced(
                    reader,
                    archive,
                    selected,
                    Path(directory),
                    0,
                )
                self.assertEqual(
                    (Path(directory) / "depth/1.png").read_bytes(),
                    b"first-depth",
                )
                self.assertEqual(
                    (Path(directory) / "depth/2.png").read_bytes(),
                    b"second-depth" * 20,
                )
                self.assertEqual(sum(run["member_count"] for run in runs), 2)


if __name__ == "__main__":
    unittest.main()

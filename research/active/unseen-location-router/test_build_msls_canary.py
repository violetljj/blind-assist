from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_msls_canary.py")
SPEC = importlib.util.spec_from_file_location("ulr_build_msls_canary", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MslsCanaryTest(unittest.TestCase):
    def _city(self, root: Path, city: str) -> None:
        for partition, offset in (("database", 0.0), ("query", 0.00001)):
            folder = root / "train_val" / city / partition
            raw, post, seq = [], [], []
            for index in range(20):
                key = f"{partition}-{index}"
                raw.append({"key": key, "lon": 10.0 + index * 0.001 + offset, "lat": 50.0, "ca": 0, "captured_at": index, "pano": False})
                post.append({"key": key, "easting": index * 100.0 + offset * 1000, "northing": 0.0, "night": False, "control_panel": False, "view_direction": "Forward", "unique_cluster": index})
                seq.append({"key": key, "sequence_id": f"{partition}-sequence", "frame_number": index})
            write_csv(folder / "raw.csv", raw)
            write_csv(folder / "postprocessed.csv", post)
            write_csv(folder / "seq_info.csv", seq)

    def test_canary_admits_aligned_frame_gps_and_keeps_test_unopened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._city(root, "train-city")
            self._city(root, "dev-city")
            result = MODULE.build_canary(root, train_cities=("train-city",), development_cities=("dev-city",))
        self.assertEqual("ADMITTED", result["status"])
        self.assertTrue(result["admission_checks"]["query_frame_gps_100_percent"])
        self.assertTrue(result["admission_checks"]["k8_coverage_at_least_90_percent"])
        self.assertTrue(result["admission_checks"]["k16_coverage_at_least_99_percent"])
        self.assertFalse(result["test_metadata_read"])
        self.assertEqual(0, result["test_images_read"])

    def test_canary_rejects_missing_query_gps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._city(root, "train-city")
            self._city(root, "dev-city")
            raw_path = root / "train_val" / "dev-city" / "query" / "raw.csv"
            rows = MODULE._read_csv(raw_path)
            rows[0]["lat"] = ""
            write_csv(raw_path, rows)
            result = MODULE.build_canary(root, train_cities=("train-city",), development_cities=("dev-city",))
        self.assertEqual("REJECTED", result["status"])
        self.assertFalse(result["admission_checks"]["query_frame_gps_100_percent"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from materialize_thor_magni_windows import (
    _body_centroid_columns,
    _parse_float,
    _parse_int,
    _read_synchronized_rows,
    _scene_column,
)
from pipeline import ContractError


class MaterializeThorMagniWindowsTest(unittest.TestCase):
    def test_scene_column_binds_body_and_device(self) -> None:
        index, column, body = _scene_column(
            ["Frame", "Time", "Helmet_4 PPL_SceneFNr", "Helmet_6 TB3_SceneFNr"],
            "PPL",
        )
        self.assertEqual((index, column, body), (2, "Helmet_4 PPL_SceneFNr", "Helmet_4"))

    def test_scene_column_requires_one_match(self) -> None:
        with self.assertRaises(ContractError):
            _scene_column(["Frame", "Time"], "PPL")

    def test_locale_numbers_and_centroid_columns(self) -> None:
        self.assertEqual(_parse_float("1,25"), 1.25)
        self.assertEqual(_parse_int("4.0"), 4)
        self.assertIsNone(_parse_int("N/A"))
        header = [
            "Frame",
            "Helmet_4 Centroid_X",
            "Helmet_4 Centroid_Y",
            "Helmet_4 Centroid_Z",
        ]
        self.assertEqual(_body_centroid_columns(header, ["Helmet_4"]), {"Helmet_4": (1, 2, 3)})

    def test_qtm_rows_are_preserved_when_scene_frame_repeats(self) -> None:
        content = "\n".join([
            "FILE_ID,unit",
            "BODY_NAMES,Helmet_4",
            "BODY_ROLES,Visitors-Alone",
            "Frame,Time,Helmet_4 Centroid_X,Helmet_4 Centroid_Y,Helmet_4 Centroid_Z,Helmet_4 PPL_SceneFNr",
            "1,0.01,1,2,3,0",
            "2,0.02,1,2,3,0",
            "3,0.03,1,2,3,1",
            "",
        ])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            path.write_text(content, encoding="utf-8")
            parsed = _read_synchronized_rows(path, device="PPL")
        self.assertEqual(parsed["source_row_count"], 3)
        self.assertEqual(parsed["unique_scene_frame_count"], 2)
        self.assertEqual(len(parsed["scene_groups"][0]), 2)


if __name__ == "__main__":
    unittest.main()

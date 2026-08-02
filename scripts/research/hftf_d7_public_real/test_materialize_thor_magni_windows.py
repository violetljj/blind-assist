from __future__ import annotations

import unittest

from materialize_thor_magni_windows import (
    _body_centroid_columns,
    _parse_float,
    _parse_int,
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


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_dav2_selective_w8a16_a5_r0 import converter_command


class SelectiveW8A16ConversionTest(unittest.TestCase):
    def test_command_freezes_selective_conversion_contract(self) -> None:
        command = converter_command(
            Path("python"),
            Path("qairt-converter"),
            Path("model.onnx"),
            Path("overrides.json"),
            Path("model.dlc"),
        )
        self.assertIn("--quantization_overrides", command)
        self.assertEqual(command[command.index("--float_bitwidth") + 1], "16")
        self.assertEqual(command[command.index("--target_soc_model") + 1], "SM8650")
        self.assertNotIn("--act_bitwidth", command)


if __name__ == "__main__":
    unittest.main()

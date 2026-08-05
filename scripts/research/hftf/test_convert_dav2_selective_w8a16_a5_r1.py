#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_dav2_selective_w8a16_a5_r1 import generic_htp_command


class SelectiveW8A16GenericHtpTest(unittest.TestCase):
    def test_command_omits_unsupported_host_soc_name(self) -> None:
        command = generic_htp_command(
            Path("python"),
            Path("converter"),
            Path("model.onnx"),
            Path("overrides.json"),
            Path("model.dlc"),
        )
        self.assertEqual(command[command.index("--target_backend") + 1], "HTP")
        self.assertNotIn("--target_soc_model", command)
        self.assertNotIn("--act_bitwidth", command)


if __name__ == "__main__":
    unittest.main()

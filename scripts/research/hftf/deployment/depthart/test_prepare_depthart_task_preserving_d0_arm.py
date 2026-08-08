from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.prepare_depthart_task_preserving_d0_arm import (
    build_command,
)


ROOT = Path(__file__).resolve().parents[5]


class PrepareDepthArtTaskPreservingD0ArmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads((ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.json").read_text(encoding="utf-8"))
        self.source = json.loads((ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json").read_text(encoding="utf-8"))

    def test_fp16_uses_converter_without_calibration(self) -> None:
        command = build_command(Path("python"), self.protocol, self.source, ROOT, "D0_FP16_R0", Path("fp16.dlc"), None)
        self.assertIn("--float_bitwidth", command)
        self.assertEqual(command[command.index("--float_bitwidth") + 1], "16")
        self.assertNotIn("--input_list", command)

    def test_w8a16_uses_quantizer_and_calibration(self) -> None:
        with tempfile.NamedTemporaryFile() as handle:
            calibration = Path(handle.name)
            command = build_command(Path("python"), self.protocol, self.source, ROOT, "D0_W8A16_R0", Path("w8a16.dlc"), calibration)
        self.assertIn("--input_list", command)
        self.assertEqual(command[command.index("--act_bitwidth") + 1], "16")
        self.assertEqual(command[command.index("--weights_bitwidth") + 1], "8")

    def test_int8_requires_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a frozen calibration list"):
            build_command(Path("python"), self.protocol, self.source, ROOT, "D0_INT8_R0", Path("int8.dlc"), None)


if __name__ == "__main__":
    unittest.main()

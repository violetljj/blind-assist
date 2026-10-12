"""Build/run actual shared firmware with host mocks; never contacts hardware."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


class SharedFirmwareTests(unittest.TestCase):
    def test_profile_migration_and_failover_state_machine(self):
        root = Path(__file__).resolve().parents[4]
        compiler = Path("E:/codex-tools/tools/llvm/bin/clang++.exe")
        if not compiler.is_file():
            self.skipTest("Host clang++ unavailable")
        environment = os.environ.copy()
        vcvars = Path("E:/codex-tools/msvc-buildtools-2022/VC/Auxiliary/Build/vcvars64.bat")
        if vcvars.is_file():
            prepared = subprocess.run(["cmd", "/d", "/c", f'{vcvars} >nul && set'],
                                      capture_output=True, text=True, timeout=20)
            self.assertEqual(0, prepared.returncode, "MSVC environment setup failed")
            for line in prepared.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    environment[key] = value
        scratch = root / "artifacts.local/hardware-bringup/test-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dual-wifi-", dir=scratch) as temporary:
            directory = Path(temporary)
            for header in ("Arduino.h", "WiFi.h", "WiFiUdp.h", "WebServer.h", "Preferences.h", "esp_timer.h", "esp_mac.h"):
                (directory / header).write_text("#pragma once\n", encoding="utf-8")
            binary = directory / "wifi-profile-test.exe"
            result = subprocess.run([str(compiler), "-std=c++17", "-I", str(directory),
                                     str(Path(__file__).with_suffix(".cpp")), "-o", str(binary)],
                                    capture_output=True, text=True, timeout=40, env=environment)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("PASS:", result.stdout)


if __name__ == "__main__":
    unittest.main()

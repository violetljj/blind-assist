from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class DiscoverSanpoSequenceCandidatesCliTest(unittest.TestCase):
    def test_negative_start_index_fails_before_network(self) -> None:
        script = Path(__file__).with_name("discover_sanpo_sequence_candidates.py")
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output",
                "unused.json",
                "--start-session-index",
                "-1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("start-session-index non-negative", result.stderr)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from materialize_crowdbot_holdout_sources import unlink_with_retry


class MaterializerCleanupTest(unittest.TestCase):
    @mock.patch("materialize_crowdbot_holdout_sources.time.sleep")
    @mock.patch.object(Path, "unlink")
    def test_transient_windows_lock_is_retried(
        self,
        unlink: mock.Mock,
        sleep: mock.Mock,
    ) -> None:
        unlink.side_effect = [
            PermissionError(32, "file is in use"),
            PermissionError(32, "file is in use"),
            None,
        ]

        unlink_with_retry(Path("raw.bag"))

        self.assertEqual(unlink.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.5, 1.0])

    @mock.patch.object(Path, "unlink", side_effect=FileNotFoundError)
    def test_already_released_file_is_idempotent(self, unlink: mock.Mock) -> None:
        unlink_with_retry(Path("raw.bag"))
        unlink.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

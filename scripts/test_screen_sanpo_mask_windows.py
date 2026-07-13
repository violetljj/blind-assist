from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import screen_sanpo_mask_windows as screen


class ScreenSanpoMaskWindowsRetryTest(unittest.TestCase):
    def test_retry_fetch_recovers_before_limit(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise OSError("transient")
            return "ok"

        with patch.object(screen.time, "sleep"):
            self.assertEqual(screen.retry_fetch(operation, 3, "fixture"), "ok")
        self.assertEqual(calls, 3)

    def test_mask_array_retries_transport_failure(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                from io import BytesIO
                from PIL import Image

                output = BytesIO()
                Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), mode="RGB").save(output, format="PNG")
                return output.getvalue()

        replies = [OSError("tls eof"), Response()]

        def urlopen(*_args, **_kwargs):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        with patch.object(screen, "urlopen", side_effect=urlopen), patch.object(screen.time, "sleep"):
            value = screen.mask_array("https://example.invalid/mask.png", 2)
        self.assertEqual(value.shape, (2, 3, 3))
        self.assertFalse(replies)

    def test_retry_fetch_fails_after_declared_limit(self) -> None:
        with patch.object(screen.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
                screen.retry_fetch(lambda: (_ for _ in ()).throw(OSError("tls eof")), 2, "fixture")


if __name__ == "__main__":
    unittest.main()

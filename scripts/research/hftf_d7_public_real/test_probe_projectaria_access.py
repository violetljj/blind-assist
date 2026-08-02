from __future__ import annotations

import io
import unittest
from email.message import Message

from probe_projectaria_access import build_probe


class _Response:
    def __init__(self, status: int, body: bytes = b"{}") -> None:
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        self._body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class ProjectAriaAccessProbeTest(unittest.TestCase):
    def test_all_api_routes_401_is_blocked(self) -> None:
        def opener(request: object, timeout: float) -> _Response:
            url = request.full_url  # type: ignore[attr-defined]
            return _Response(200 if "explorer.projectaria.com" in url else 401, b"shell-or-auth")

        payload = build_probe("Project-Aria-Everyday-Activities", opener=opener)

        self.assertEqual(payload["access_status"], "ACCESS_BLOCKED_API_KEY_REQUIRED")
        self.assertFalse(payload["credentials_submitted"])
        self.assertFalse(payload["media_download_attempted"])
        self.assertEqual(len(payload["probes"]), 5)

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_probe("unknown", opener=lambda request, timeout: _Response(401))


if __name__ == "__main__":
    unittest.main()

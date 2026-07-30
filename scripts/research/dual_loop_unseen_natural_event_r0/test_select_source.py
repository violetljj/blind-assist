import unittest

from scripts.research.dual_loop_unseen_natural_event_r0.select_source import (
    eligible_reason,
    select,
)


def row(title: str, **overrides):
    value = {
        "title": title,
        "mediatype": "VIDEO",
        "mime": "video/webm",
        "duration_s": 300.0,
        "width": 1920,
        "height": 1080,
        "size_bytes": 100_000_000,
        "license_short_name": "CC BY 4.0",
    }
    value.update(overrides)
    return value


class SelectSourceTest(unittest.TestCase):
    def test_rejects_output_unrelated_metadata_failures(self):
        admitted, failures = eligible_reason(
            row("File:Walking example.webm", duration_s=179.9)
        )
        self.assertFalse(admitted)
        self.assertEqual(["DURATION_OUTSIDE_180_TO_900_S"], failures)

    def test_unicode_title_order_selects_first_eligible(self):
        registry = [
            row("File:Z Walking.webm"),
            row("File:A City Tour.webm"),
        ]
        import scripts.research.dual_loop_unseen_natural_event_r0.select_source as module

        original = module.fetch_derivatives
        module.fetch_derivatives = lambda _: [
            {
                "transcode_key": "480p.vp9.webm",
                "width": 854,
                "height": 480,
                "url": "https://example.invalid/video.webm",
                "type": "video/webm",
            }
        ]
        try:
            _, selected = select(sorted(registry, key=lambda item: item["title"]))
        finally:
            module.fetch_derivatives = original
        self.assertEqual("File:A City Tour.webm", selected["title"])


if __name__ == "__main__":
    unittest.main()

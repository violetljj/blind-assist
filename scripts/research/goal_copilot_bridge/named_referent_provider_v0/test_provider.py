from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.research.goal_copilot_bridge.named_referent_provider_v0.provider import (
    MapBearingAdapter,
    NamedReferentProviderV0,
    text_match,
)
from scripts.research.goal_copilot_bridge.named_referent_provider_v0.schema import (
    CHANNELS,
    CurrentFrame,
    GoalReferencePack,
    provider_identity,
    validate_output,
)


class _Adapter:
    def __init__(self, channel: str, *, fail: bool = False):
        self.channel = channel
        self.fail = fail

    def identity(self):
        return provider_identity(provider=f"fake-{self.channel}", implementation_version="test")

    def collect(self, frame, goal):
        del frame, goal
        if self.fail:
            raise RuntimeError("deliberate provider failure")
        return {
            "channel": self.channel,
            "status": "AVAILABLE",
            "provider_identity": self.identity(),
            "latency_ms": 1.0,
            "items": [],
            "error": None,
        }


class ProviderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.image = Path(self.temporary.name) / "frame.jpg"
        Image.new("RGB", (32, 24), "white").save(self.image)
        self.frame = CurrentFrame.from_mapping({"frame_id": "frame-1", "image_path": self.image, "heading_degrees": 350})
        self.goal = GoalReferencePack.from_mapping({
            "name": "翠華餐廳",
            "aliases": ["翠華", "Tsui Wah"],
            "reference_images": [self.image],
            "logo": self.image,
            "map_bearing_degrees": 10,
        })

    def tearDown(self):
        self.temporary.cleanup()

    def test_text_match_keeps_exact_substring_and_fuzzy_distinct(self):
        self.assertEqual(text_match("STARBUCKS", "Starbucks")["match_class"], "EXACT")
        self.assertEqual(text_match("Welcome Starbucks Coffee", "Starbucks")["match_class"], "SUBSTRING")
        self.assertEqual(text_match("Starbuks", "Starbucks")["match_class"], "FUZZY")

    def test_provider_keeps_four_channels_and_fails_one_closed(self):
        adapters = {channel: _Adapter(channel, fail=channel == "proposal_evidence") for channel in CHANNELS}
        output = NamedReferentProviderV0(adapters).run(self.frame, self.goal)
        validate_output(output)
        self.assertEqual(tuple(output["evidence"]), CHANNELS)
        self.assertEqual(output["evidence"]["proposal_evidence"]["status"], "NOT_EVALUABLE")
        self.assertEqual(output["evidence"]["proposal_evidence"]["items"], [])
        self.assertNotIn("fused_score", output)
        self.assertNotIn("decision", output)

    def test_unconfigured_provider_is_not_evaluable(self):
        output = NamedReferentProviderV0({"bearing_evidence": MapBearingAdapter()}).run(self.frame, self.goal)
        self.assertEqual(output["evidence"]["text_evidence"]["error"]["code"], "PROVIDER_UNCONFIGURED")
        self.assertEqual(output["evidence"]["bearing_evidence"]["items"][0]["normalized_match"]["signed_clockwise_delta_degrees"], 20)


if __name__ == "__main__":
    unittest.main()

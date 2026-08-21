from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from scripts.research.goal_copilot_bridge.p0_s0_materialization import crossview_identity as identity


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "docs/research/goal-copilot/p0_s1_crossview_identity_config.json"


def make_image(path: Path, *, shifted: bool = False, different: bool = False) -> None:
    image = Image.new("RGB", (128, 96), "white")
    draw = ImageDraw.Draw(image)
    if different:
        draw.ellipse((45, 20, 85, 80), fill=(20, 30, 180))
    else:
        offset = 3 if shifted else 0
        draw.rectangle((40 + offset, 20, 80 + offset, 85), fill=(80, 40, 20))
        draw.rectangle((48 + offset, 35, 72 + offset, 82), fill=(180, 160, 120))
    image.save(path, format="PNG")


def candidate(candidate_id: str, frame_id: str, *, lon: float = 3.72, heading: float = 20.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "frame_id": frame_id,
        "building_id": "b1",
        "anchor_id": "a1",
        "bbox_xyxy": [38.0, 18.0, 84.0, 88.0],
        "predicted_entrance_geo": {"lon": lon, "lat": 51.05},
        "ray_heading_deg": heading,
        "ray_range_m": 12.0,
        "geometry_verified": True,
        "map_anchored": True,
    }


def metadata_item(frame_id: str, path: Path, *, sequence: str, lon: float, captured_at: int) -> dict:
    return {
        "id": frame_id,
        "path": str(path),
        "sequence_id": sequence,
        "coordinates": [lon, 51.05],
        "captured_at": captured_at,
        "camera_parameters": [0.5],
        "width": 128,
        "height": 96,
    }


class CrossviewIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = identity.load_config(CONFIG_PATH)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.left_path = self.root / "left.png"
        self.right_path = self.root / "right.png"
        make_image(self.left_path)
        make_image(self.right_path, shifted=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_same_sequence_pair_can_establish_identity(self) -> None:
        left = candidate("c1", "f1", heading=20.0)
        right = candidate("c2", "f2", lon=3.720005, heading=35.0)
        metadata = {
            "f1": metadata_item("f1", self.left_path, sequence="s1", lon=3.72, captured_at=1000),
            "f2": metadata_item("f2", self.right_path, sequence="s1", lon=3.72005, captured_at=2000),
        }
        result = identity.assess_pair(left, right, metadata, self.config)
        self.assertEqual("ENTRANCE_IDENTITY_ESTABLISHED", result["disposition"])
        self.assertTrue(all(result["gates"].values()))

    def test_cross_sequence_is_never_strong_even_when_other_gates_pass(self) -> None:
        left = candidate("c1", "f1", heading=20.0)
        right = candidate("c2", "f2", lon=3.720005, heading=35.0)
        metadata = {
            "f1": metadata_item("f1", self.left_path, sequence="s1", lon=3.72, captured_at=1000),
            "f2": metadata_item("f2", self.right_path, sequence="s2", lon=3.72005, captured_at=2000),
        }
        result = identity.assess_pair(left, right, metadata, self.config)
        self.assertTrue(result["disposition"].startswith("CROSS_SEQUENCE_"))
        self.assertFalse(result["gates"]["same_sequence"])

    def test_insufficient_parallax_fails_same_sequence(self) -> None:
        left = candidate("c1", "f1", heading=20.0)
        right = candidate("c2", "f2", lon=3.720001, heading=35.0)
        metadata = {
            "f1": metadata_item("f1", self.left_path, sequence="s1", lon=3.72, captured_at=1000),
            "f2": metadata_item("f2", self.right_path, sequence="s1", lon=3.720001, captured_at=2000),
        }
        result = identity.assess_pair(left, right, metadata, self.config)
        self.assertEqual("SAME_SEQUENCE_IDENTITY_NOT_ESTABLISHED", result["disposition"])
        self.assertFalse(result["gates"]["camera_parallax"])

    def test_different_appearance_fails_same_sequence(self) -> None:
        make_image(self.right_path, different=True)
        left = candidate("c1", "f1", heading=20.0)
        right = candidate("c2", "f2", lon=3.720005, heading=35.0)
        metadata = {
            "f1": metadata_item("f1", self.left_path, sequence="s1", lon=3.72, captured_at=1000),
            "f2": metadata_item("f2", self.right_path, sequence="s1", lon=3.72005, captured_at=2000),
        }
        result = identity.assess_pair(left, right, metadata, self.config)
        self.assertEqual("SAME_SEQUENCE_IDENTITY_NOT_ESTABLISHED", result["disposition"])
        self.assertFalse(result["gates"]["appearance_compatibility"])

    def test_forbidden_manual_truth_input_fails_schema(self) -> None:
        bundle = {"records": [], "manual_truth": True}
        report = identity.assess_identity(bundle, {"images": []}, {"results": []}, self.config)
        self.assertEqual("P0_S1_SCHEMA_INADEQUACY", report["verdict"])
        self.assertEqual(["bundle.manual_truth"], report["forbidden_input_leaks"])

    def test_replay_is_deterministic(self) -> None:
        bundle = {"records": []}
        first = identity.assess_identity(bundle, {"images": []}, {"results": []}, self.config)
        second = identity.assess_identity(copy.deepcopy(bundle), {"images": []}, {"results": []}, self.config)
        self.assertEqual(identity.canonical_bytes(first), identity.canonical_bytes(second))


if __name__ == "__main__":
    unittest.main()

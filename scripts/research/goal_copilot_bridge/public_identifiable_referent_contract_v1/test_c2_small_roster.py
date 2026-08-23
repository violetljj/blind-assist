from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import c2_small_roster as sut


def _polygon(object_id: int, x0: int, y0: int, x1: int, y1: int) -> dict:
    return {
        "object": object_id,
        "x": [x0, x1, x1, x0],
        "y": [y0, y0, y1, y1],
        "XYZ": [[0.0, 0.0, 3.0], [0.1, 0.0, 3.0], [0.0, 0.1, 3.0]],
    }


def _synthetic_metadata(*, moving: bool = True) -> tuple[dict, str]:
    source_ids = [1, 41, 81, 121]
    frames = []
    for index in range(4):
        polygons = [_polygon(0, 100, 100, 300, 340)]
        if index == 2:
            polygons.append(_polygon(1, 360, 120, 500, 330))
        frames.append({"polygon": polygons})
    annotation = {
        "objects": [{"name": "chair: 1"}, {"name": "chair: 2"}, {"name": "wall"}],
        "fileList": [f"{source_id:07d}-000000.jpg" for source_id in source_ids],
        "frames": frames,
    }
    matrices = []
    for source_frame_id in range(121):
        x = source_frame_id * 0.01 if moving else 0.0
        matrices.extend([1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return annotation, " ".join(str(value) for value in matrices)


def _jpeg(seed: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (sut.IMAGE_WIDTH, sut.IMAGE_HEIGHT), (seed * 11 % 255, seed * 17 % 255, seed * 23 % 255)).save(
        stream, format="JPEG"
    )
    return stream.getvalue()


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.content = payload

    def raise_for_status(self) -> None:
        return None


class C2SmallRosterTest(unittest.TestCase):
    def test_metadata_selection_requires_real_view_change_and_prefers_distractor_signal(self) -> None:
        annotation, extrinsics = _synthetic_metadata()
        episode = sut._episode_from_metadata("synthetic/source", annotation, extrinsics)
        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertEqual(0, episode["native_object_id"])
        self.assertEqual(3, len(episode["later_observations"]))
        self.assertEqual(1, episode["selected_later_distractor_frame_count"])
        self.assertTrue(all(item["translation_from_reference_m"] >= sut.MIN_TRANSLATION_M for item in episode["later_observations"]))

        static_annotation, static_extrinsics = _synthetic_metadata(moving=False)
        self.assertIsNone(sut._episode_from_metadata("synthetic/static", static_annotation, static_extrinsics))

    def test_materialization_locks_all_identities_before_any_later_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.md"
            protocol_path.write_text("frozen synthetic protocol\n", encoding="utf-8")
            payloads: dict[str, bytes] = {}
            episodes = []
            seed = 1
            for index in range(5):
                case_id = f"c2-ref-{index + 1:03d}"
                reference_url = f"https://example.invalid/{case_id}/reference.jpg"
                payloads[reference_url] = _jpeg(seed)
                seed += 1
                later_rows = []
                for later_index in range(3):
                    observation_id = f"{case_id}-later-{later_index + 1:02d}"
                    image_url = f"https://example.invalid/{case_id}/{observation_id}.jpg"
                    payloads[image_url] = _jpeg(seed)
                    seed += 1
                    later_rows.append(
                        {
                            "observation_id": observation_id,
                            "image_url": image_url,
                            "bbox_xyxy_normalized": [0.2, 0.2, 0.6, 0.8],
                        }
                    )
                episodes.append(
                    {
                        "case_id": case_id,
                        "sequence": f"synthetic/source-{index + 1}",
                        "private_physical_instance_id": f"synthetic-instance-{index + 1}",
                        "private_identity_source_sha256": f"{index + 1:064x}",
                        "target_world_xyz_map_derived": [float(index), 0.0, 3.0],
                        "selected_later_distractor_frame_count": index % 2,
                        "reference": {
                            "image_url": reference_url,
                            "bbox_xyxy_normalized": [0.1, 0.1, 0.7, 0.9],
                        },
                        "later_observations": later_rows,
                    }
                )
            roster = {
                "schema_version": sut.SCHEMA_VERSION,
                "protocol_id": sut.PROTOCOL_ID,
                "frozen_at_utc": "2020-01-01T00:00:00Z",
                "episode_count": len(episodes),
                "episodes": episodes,
                "implementation_lock": sut._implementation_lock(protocol_path),
                "materialization_authorized": True,
                "passive_baseline_authorized": False,
                "algorithm_authorized": False,
                "terminal": "C2_ROSTER_FROZEN_READY_FOR_ONE_MATERIALIZATION",
            }
            roster["body_sha256"] = sut._body_hash(roster)
            roster_path = root / "roster.json"
            sut._atomic_json(roster_path, roster)
            roster_file_sha256 = sut._sha256_file(roster_path)

            with mock.patch.object(sut.requests, "get", side_effect=lambda url, timeout: _Response(payloads[url])):
                report = sut.materialize(roster_path, roster_file_sha256, protocol_path, root / "formal")

            self.assertEqual("SMALL_ROSTER_MATERIALIZABLE", report["terminal"], report)
            self.assertEqual(5, report["episode_count"])
            self.assertEqual(20, report["unique_image_sha256_count"])
            barrier = json.loads((root / "formal" / "identity-lock-barrier.json").read_text(encoding="utf-8"))
            self.assertEqual(0, barrier["later_image_gets"])
            journal = json.loads((root / "formal" / "download-journal.json").read_text(encoding="utf-8"))
            first_later_dispatch = next(index for index, row in enumerate(journal) if row["role"] == "LATER_OBSERVATION")
            self.assertEqual(10, first_later_dispatch)
            self.assertTrue(all(row["role"] == "REFERENCE" for row in journal[:first_later_dispatch]))
            public_manifest = (root / "formal" / "public-manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("synthetic-instance", public_manifest)


if __name__ == "__main__":
    unittest.main()

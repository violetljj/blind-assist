import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.research.assistive_geometry.materialize_b0_arkitscenes_upsampling_train import (
    MODALITIES,
    train_video_lookup,
    upsampling_png_members,
)


class MaterializeB0ArkitScenesUpsamplingTrainTest(unittest.TestCase):
    def test_only_frozen_train_videos_are_admitted(self) -> None:
        manifest = {
            "videos": [
                {"role": "TRAIN", "video_id": str(index)}
                for index in range(16)
            ]
            + [{"role": "DEVELOPMENT", "video_id": "99"}]
        }
        result = train_video_lookup(manifest, ["1", "3"])
        self.assertEqual({"1", "3"}, set(result))
        with self.assertRaisesRegex(ValueError, "outside frozen TRAIN"):
            train_video_lookup(manifest, ["99"])

    def test_duplicate_expected_video_is_rejected(self) -> None:
        manifest = {"videos": [{"role": "TRAIN", "video_id": str(index)} for index in range(16)]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            train_video_lookup(manifest, ["1", "1"])

    def test_repeated_stem_is_scoped_by_modality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for modality in MODALITIES:
                    bundle.writestr(f"123/{modality}/123_1.000.png", modality.encode("ascii"))
            result = upsampling_png_members(archive)
        self.assertEqual(set(MODALITIES), set(result))
        self.assertTrue(all(set(members) == {"123_1.000"} for members in result.values()))

    def test_duplicate_stem_inside_one_modality_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for modality in MODALITIES:
                    bundle.writestr(f"123/{modality}/123_1.000.png", modality.encode("ascii"))
                bundle.writestr("other/wide/123_1.000.png", b"duplicate")
            with self.assertRaisesRegex(ValueError, "duplicate wide"):
                upsampling_png_members(archive)


if __name__ == "__main__":
    unittest.main()

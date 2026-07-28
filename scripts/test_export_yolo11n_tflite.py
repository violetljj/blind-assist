from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_yolo11n_tflite as subject


class ExportYolo11nTfliteTest(unittest.TestCase):
    def test_defaults_stage_under_local_models_without_promoting(self) -> None:
        args = subject.build_parser().parse_args([])

        self.assertEqual(Path("artifacts.local/models/yolo11n.pt"), args.model)
        self.assertEqual(
            Path("artifacts.local/models/yolo11n_fp16_320.tflite"),
            args.output,
        )
        self.assertFalse(args.promote_to_app)

    def test_direct_app_asset_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary).resolve()
            app_output = project_root / subject.APP_MODEL_ASSET

            with self.assertRaisesRegex(ValueError, "--promote-to-app"):
                subject.require_safe_staging_output(project_root, app_output)

    def test_validation_precedes_staging_and_explicit_app_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary).resolve()
            exported = project_root / "exported.tflite"
            staged = project_root / subject.DEFAULT_OUTPUT
            app_asset = project_root / subject.APP_MODEL_ASSET
            exported.write_bytes(b"validated-model")
            validation_calls = []

            def validator(root: Path, model: Path) -> None:
                self.assertEqual(project_root, root)
                self.assertEqual(exported, model)
                self.assertFalse(staged.exists())
                self.assertFalse(app_asset.exists())
                validation_calls.append(model)

            published, staged_sha256 = subject.publish_validated_model(
                project_root=project_root,
                exported_path=exported,
                output_path=staged,
                promote_to_app=True,
                validator=validator,
            )

            self.assertEqual([exported], validation_calls)
            self.assertEqual([staged, app_asset], published)
            self.assertEqual(b"validated-model", staged.read_bytes())
            self.assertEqual(b"validated-model", app_asset.read_bytes())
            self.assertEqual(subject.sha256_file(exported), staged_sha256)

    def test_copy_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary).resolve()
            exported = project_root / "exported.tflite"
            staged = project_root / subject.DEFAULT_OUTPUT
            exported.write_bytes(b"validated-model")

            def corrupt_copy(_source: Path, destination: Path) -> None:
                Path(destination).write_bytes(b"corrupt-copy")

            with (
                mock.patch.object(subject.shutil, "copy2", side_effect=corrupt_copy),
                self.assertRaisesRegex(IOError, "SHA-256 mismatch"),
            ):
                subject.publish_validated_model(
                    project_root=project_root,
                    exported_path=exported,
                    output_path=staged,
                    promote_to_app=False,
                    validator=lambda _root, _model: None,
                )

    def test_static_validator_invokes_existing_inspector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary).resolve()
            model_path = project_root / "candidate.tflite"
            with mock.patch.object(subject.subprocess, "run") as run:
                subject.validate_tflite(project_root, model_path)

            run.assert_called_once_with(
                [
                    subject.sys.executable,
                    str(project_root / "scripts/inspect_tflite.py"),
                    str(model_path),
                ],
                cwd=project_root,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()

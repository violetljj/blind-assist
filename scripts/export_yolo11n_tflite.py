from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


DEFAULT_MODEL = Path("artifacts.local/models/yolo11n.pt")
DEFAULT_OUTPUT = Path("artifacts.local/models/yolo11n_fp16_320.tflite")
APP_MODEL_ASSET = Path("app/src/main/assets/yolo11n_fp16_320.tflite")


def configure_local_caches(project_root: Path) -> None:
    matplotlib_cache = project_root / ".cache" / "matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    matplotlib_cache.mkdir(parents=True, exist_ok=True)


def newest_tflite(root: Path) -> Path:
    candidates = sorted(
        root.rglob("*.tflite"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Ultralytics export finished but no .tflite file was found.")
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export YOLO11n to Android-ready TFLite FP16.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Source YOLO weights. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Validated staging output. Direct app asset paths are rejected.",
    )
    parser.add_argument(
        "--promote-to-app",
        action="store_true",
        help="After static validation, also copy the staged model to the fixed app asset path.",
    )
    return parser


def resolve_project_path(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def require_safe_staging_output(project_root: Path, output_path: Path) -> None:
    app_assets_dir = (project_root / "app/src/main/assets").resolve()
    try:
        output_path.relative_to(app_assets_dir)
    except ValueError:
        return
    raise ValueError(
        "Direct app asset output is forbidden. Stage under artifacts.local/models and use "
        "--promote-to-app for an explicit, validated promotion."
    )


def validate_tflite(project_root: Path, model_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/inspect_tflite.py"),
            str(model_path),
        ],
        cwd=project_root,
        check=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_validated_model(
    project_root: Path,
    exported_path: Path,
    output_path: Path,
    promote_to_app: bool,
    validator: Callable[[Path, Path], None] = validate_tflite,
) -> tuple[list[Path], str]:
    validator(project_root, exported_path)
    exported_sha256 = sha256_file(exported_path)
    destinations = [output_path]
    if promote_to_app:
        destinations.append((project_root / APP_MODEL_ASSET).resolve())

    published = []
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if exported_path != destination:
            shutil.copy2(exported_path, destination)
        destination_sha256 = sha256_file(destination)
        if destination_sha256 != exported_sha256:
            raise IOError(
                f"Published model SHA-256 mismatch for {destination}: "
                f"expected {exported_sha256}, got {destination_sha256}"
            )
        published.append(destination)
    return published, exported_sha256


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    configure_local_caches(project_root)

    model_path = resolve_project_path(project_root, args.model)
    output_path = resolve_project_path(project_root, args.output)
    require_safe_staging_output(project_root, output_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO source weights not found: {model_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    before = {path.resolve() for path in model_path.parent.rglob("*.tflite")}
    model = YOLO(str(model_path))
    exported = model.export(
        format="tflite",
        imgsz=args.imgsz,
        half=True,
        nms=False,
        batch=1,
    )

    exported_path = Path(str(exported))
    if not exported_path.is_absolute():
        exported_path = project_root / exported_path
    exported_path = exported_path.resolve()
    if not exported_path.exists() or exported_path.suffix.lower() != ".tflite":
        after = {path.resolve() for path in model_path.parent.rglob("*.tflite")}
        created = list(after - before)
        exported_path = (
            max(created, key=lambda path: path.stat().st_mtime)
            if created
            else newest_tflite(model_path.parent)
        )

    published, staged_sha256 = publish_validated_model(
        project_root=project_root,
        exported_path=exported_path,
        output_path=output_path,
        promote_to_app=args.promote_to_app,
    )

    print(f"exported={exported_path}")
    print(f"staged_model={published[0]}")
    if args.promote_to_app:
        print(f"android_asset={published[1]}")
    print(f"staged_sha256={staged_sha256}")
    print(f"size_bytes={published[0].stat().st_size}")


if __name__ == "__main__":
    main()

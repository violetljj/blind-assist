from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve


DEFAULT_CANDIDATES = ("yolo26n.pt", "yolo12n.pt", "yolov10n.pt")
COCO8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
LAB_ROOT = Path(".downloads/detector-lab")


def configure_local_caches(project_root: Path) -> None:
    cache_root = project_root / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
    os.environ.setdefault("ULTRALYTICS_SETTINGS", str(cache_root / "ultralytics" / "settings.json"))
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "ultralytics").mkdir(parents=True, exist_ok=True)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_coco8(project_root: Path, lab_root: Path) -> dict[str, Any]:
    dataset_root = lab_root / "datasets"
    zip_path = dataset_root / "coco8.zip"
    extract_root = dataset_root / "coco8"
    dataset_root.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        print(f"download_dataset={COCO8_URL}")
        urlretrieve(COCO8_URL, zip_path)

    if not extract_root.is_dir():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dataset_root)
        nested = dataset_root / "coco8"
        if nested != extract_root and nested.is_dir():
            shutil.move(str(nested), str(extract_root))

    images = sorted(path for path in extract_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(path for path in extract_root.rglob("*.txt"))
    result = {
        "name": "coco8",
        "source": COCO8_URL,
        "zip": zip_path.relative_to(project_root),
        "root": extract_root.relative_to(project_root),
        "image_count": len(images),
        "label_count": len(labels),
    }
    print(f"dataset=coco8 images={len(images)} labels={len(labels)} root={extract_root}")
    return result


def import_ultralytics() -> tuple[Any, str]:
    from ultralytics import YOLO
    import ultralytics

    return YOLO, getattr(ultralytics, "__version__", "unknown")


def download_weight(project_root: Path, models_dir: Path, candidate: str) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / candidate
    if target.is_file():
        print(f"weight_exists={target}")
        return {"candidate": candidate, "path": target.relative_to(project_root), "downloaded": False}

    YOLO, version = import_ultralytics()
    before_cwd = Path.cwd()
    os.chdir(models_dir)
    try:
        print(f"download_weight={candidate} ultralytics={version}")
        YOLO(candidate)
    finally:
        os.chdir(before_cwd)

    if not target.is_file():
        found = sorted(models_dir.glob(candidate), key=lambda path: path.stat().st_mtime, reverse=True)
        if not found:
            raise FileNotFoundError(f"Ultralytics did not create expected weight: {target}")
        target = found[0]

    return {
        "candidate": candidate,
        "path": target.relative_to(project_root),
        "downloaded": True,
        "size_bytes": target.stat().st_size,
    }


def export_candidate(
    project_root: Path,
    models_dir: Path,
    exports_dir: Path,
    candidate: str,
    imgsz: int,
    half: bool,
) -> dict[str, Any]:
    exports_dir.mkdir(parents=True, exist_ok=True)
    weight_path = models_dir / candidate
    if not weight_path.is_file():
        raise FileNotFoundError(f"Candidate weight missing: {weight_path}")

    stem = Path(candidate).stem
    precision = "fp16" if half else "fp32"
    output_path = exports_dir / f"{stem}_{precision}_{imgsz}.tflite"
    if output_path.is_file():
        print(f"export_exists={output_path}")
        return {
            "candidate": candidate,
            "status": "exists",
            "path": output_path.relative_to(project_root),
            "size_bytes": output_path.stat().st_size,
        }

    YOLO, version = import_ultralytics()
    before = {path.resolve() for path in exports_dir.rglob("*.tflite")}
    before_cwd = Path.cwd()
    os.chdir(exports_dir)
    start = time.perf_counter()
    try:
        print(f"export_candidate={candidate} imgsz={imgsz} half={half} ultralytics={version}")
        model = YOLO(str(weight_path))
        export_kwargs: dict[str, Any] = {
            "format": "tflite",
            "imgsz": imgsz,
            "half": half,
            "nms": False,
            "batch": 1,
        }
        try:
            exported = model.export(**{**export_kwargs, "end2end": False})
        except TypeError:
            exported = model.export(**export_kwargs)
    finally:
        os.chdir(before_cwd)

    exported_path = Path(str(exported)).resolve()
    if not exported_path.is_file() or exported_path.suffix.lower() != ".tflite":
        after = {path.resolve() for path in exports_dir.rglob("*.tflite")}
        created = list(after - before)
        if not created:
            created = sorted(project_root.rglob("*.tflite"), key=lambda path: path.stat().st_mtime)
        if not created:
            raise FileNotFoundError(f"Export finished but no TFLite file was found for {candidate}")
        exported_path = max(created, key=lambda path: path.stat().st_mtime)

    if exported_path != output_path.resolve():
        shutil.copy2(exported_path, output_path)

    return {
        "candidate": candidate,
        "status": "exported",
        "path": output_path.relative_to(project_root),
        "size_bytes": output_path.stat().st_size,
        "export_seconds": round(time.perf_counter() - start, 3),
    }


def run_prepare(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    configure_local_caches(project_root)
    lab_root = resolve_project_path(project_root, args.lab_root)
    models_dir = lab_root / "models"
    exports_dir = lab_root / "exports"

    manifest: dict[str, Any] = {
        "created_at_epoch": int(time.time()),
        "lab_root": lab_root.relative_to(project_root),
        "candidates": list(args.candidate),
        "dataset": None,
        "weights": [],
        "exports": [],
    }

    if not args.skip_dataset:
        manifest["dataset"] = download_coco8(project_root, lab_root)

    for candidate in args.candidate:
        try:
            manifest["weights"].append(download_weight(project_root, models_dir, candidate))
            if args.export:
                manifest["exports"].append(
                    export_candidate(
                        project_root=project_root,
                        models_dir=models_dir,
                        exports_dir=exports_dir,
                        candidate=candidate,
                        imgsz=args.imgsz,
                        half=not args.float32,
                    )
                )
        except Exception as error:
            manifest["exports"].append(
                {
                    "candidate": candidate,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"candidate_failed={candidate} error={type(error).__name__}: {error}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    manifest_path = lab_root / "detector_lab_manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local detector benchmark assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Download COCO8 and candidate detector weights, optionally exporting TFLite models.",
    )
    prepare.add_argument("--lab-root", default=str(LAB_ROOT))
    prepare.add_argument(
        "--candidate",
        action="append",
        help="Candidate .pt filename to download/export. Repeat to override the default nano set.",
    )
    prepare.add_argument("--imgsz", type=int, default=320)
    prepare.add_argument("--float32", action="store_true", help="Export float32 TFLite instead of FP16.")
    prepare.add_argument("--export", action="store_true", help="Export candidate weights to TFLite.")
    prepare.add_argument("--skip-dataset", action="store_true")
    prepare.add_argument("--continue-on-error", action="store_true")
    prepare.set_defaults(func=run_prepare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "candidate") and args.candidate is None:
        args.candidate = list(DEFAULT_CANDIDATES)
    args.func(args)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


if __name__ == "__main__":
    main()

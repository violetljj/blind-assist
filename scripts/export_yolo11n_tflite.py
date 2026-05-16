from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def newest_tflite(root: Path) -> Path:
    candidates = sorted(
        root.rglob("*.tflite"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Ultralytics export finished but no .tflite file was found.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO11n to Android-ready TFLite FP16.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--output", default="app/src/main/assets/yolo11n_fp16_320.tflite")
    args = parser.parse_args()

    from ultralytics import YOLO

    project_root = Path(__file__).resolve().parents[1]
    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = {path.resolve() for path in project_root.rglob("*.tflite")}
    model = YOLO("yolo11n.pt")
    exported = model.export(
        format="tflite",
        imgsz=args.imgsz,
        half=True,
        nms=False,
        batch=1,
    )

    exported_path = Path(str(exported)).resolve()
    if not exported_path.exists() or exported_path.suffix.lower() != ".tflite":
        after = {path.resolve() for path in project_root.rglob("*.tflite")}
        created = list(after - before)
        exported_path = max(created, key=lambda path: path.stat().st_mtime) if created else newest_tflite(project_root)

    if exported_path != output_path:
        shutil.copy2(exported_path, output_path)

    print(f"exported={exported_path}")
    print(f"android_asset={output_path}")
    print(f"size_bytes={output_path.stat().st_size}")


if __name__ == "__main__":
    main()

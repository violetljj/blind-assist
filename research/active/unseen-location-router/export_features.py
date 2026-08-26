"""Export resumable real-image DINOv2, OCR, blur, and coarse-GPS evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


SCHEMA = "blindassist.unseen_location_router.features.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_value(salt: str, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}|{value}".encode()).digest()[:8], "big")


def load_build_manifest_module(path: Path):
    spec = importlib.util.spec_from_file_location("ulr_build_manifest_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_rows(
    rows: Iterable[dict[str, Any]], *, gallery_limit: int, query_limit: int, salt: str
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        split = str(row["split"])
        if split not in {"train", "development"}:
            continue
        buckets[(split, str(row["location_id"]), str(row["role"]))][str(row["capture_group"])].append(row)
    selected: list[dict[str, Any]] = []
    for (split, location_id, role), groups in sorted(buckets.items()):
        representatives = []
        for capture_group, items in groups.items():
            representatives.append(min(
                items,
                key=lambda item: (stable_value(salt, str(item["image_id"])), str(item["image_id"])),
            ))
        limit = gallery_limit if role == "gallery" else query_limit
        representatives.sort(key=lambda item: (
            role == "query" and str(item.get("source_kind", "unknown")) != "field_capture",
            stable_value(salt, f"{split}|{location_id}|{role}|{item['capture_group']}"),
            str(item["image_id"]),
        ))
        selected.extend(representatives[:limit])
    return sorted(selected, key=lambda item: (str(item["split"]), str(item["location_id"]), str(item["role"]), str(item["image_id"])))


def metadata_by_capture_group(module: Any, texts_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for filename in ("Metadata-Images.xlsx", "Metadata-Videos.xlsx"):
        for row in module.read_first_xlsx_sheet(texts_root / filename):
            source = Path(str(row.get("Filename", ""))).stem.upper()
            if source:
                result[f"field:{source}"] = row
    return result


def make_database(path: Path, metadata: dict[str, str]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS features (
        image_id TEXT PRIMARY KEY,
        relative_path TEXT NOT NULL,
        split TEXT NOT NULL,
        role TEXT NOT NULL,
        location_id TEXT NOT NULL,
        capture_group TEXT NOT NULL,
        illumination TEXT NOT NULL,
        descriptor BLOB NOT NULL,
        descriptor_dim INTEGER NOT NULL,
        blur_variance REAL NOT NULL,
        ocr_texts_json TEXT NOT NULL,
        ocr_scores_json TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        gps_accuracy_m REAL
        )"""
    )
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    if existing and existing != metadata:
        raise RuntimeError("feature database metadata mismatch; use a new output path")
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", metadata.items()
    )
    connection.commit()
    return connection


def normalized_descriptors(
    model: torch.nn.Module, processor: Any, images: list[Image.Image], device: torch.device
) -> list[np.ndarray]:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        descriptor = model(pixel_values=pixel_values).last_hidden_state[:, 0]
        descriptor = torch.nn.functional.normalize(descriptor.float(), dim=-1)
    return [row for row in descriptor.cpu().numpy().astype(np.float32)]


def blur_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def decode_image(path: Path) -> np.ndarray:
    # cv2.imread is not Unicode-path safe on Windows. Reading bytes through
    # NumPy preserves the dataset's Chinese filenames without renaming inputs.
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    return image


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--texts-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gallery-per-location", type=int, default=4)
    parser.add_argument("--query-per-location", type=int, default=8)
    parser.add_argument("--ocr-workers", type=int, default=3)
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--reuse-database", type=Path)
    parser.add_argument("--selection-salt", default="blindassist-ulr-v1-feature-canary")
    args = parser.parse_args()

    if args.gallery_per_location < 1 or args.query_per_location < 1:
        raise ValueError("positive gallery/query limits are required")
    if not 1 <= args.ocr_workers <= 8:
        raise ValueError("ocr-workers must be between 1 and 8")
    if not 1 <= args.dino_batch_size <= 64:
        raise ValueError("dino-batch-size must be between 1 and 64")
    try:
        from rapidocr import RapidOCR
    except ImportError as error:
        raise RuntimeError("RapidOCR must be supplied through the frozen local provider path") from error

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = select_rows(
        manifest["images"],
        gallery_limit=args.gallery_per_location,
        query_limit=args.query_per_location,
        salt=args.selection_salt,
    )
    if any(row["split"] == "test" for row in selected):
        raise RuntimeError("test images must remain unopened during Development")

    backbone_hash = sha256_file(args.backbone / "model.safetensors")
    metadata = {
        "schema": SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "backbone_sha256": backbone_hash,
        "selection_salt": args.selection_salt,
        "gallery_per_location": str(args.gallery_per_location),
        "query_per_location": str(args.query_per_location),
        "selection_policy": "query_field_capture_first_v1",
    }
    connection = make_database(args.database, metadata)
    if args.reuse_database is not None and args.reuse_database.exists():
        selected_ids = {str(row["image_id"]) for row in selected}
        source = sqlite3.connect(args.reuse_database)
        source_metadata_rows = dict(source.execute("SELECT key, value FROM metadata"))
        for key in ("schema", "manifest_sha256", "backbone_sha256", "selection_salt"):
            if source_metadata_rows.get(key) != metadata.get(key):
                raise RuntimeError(f"reuse database mismatch for {key}")
        reusable = [row for row in source.execute("SELECT * FROM features") if str(row[0]) in selected_ids]
        connection.executemany(
            "INSERT OR IGNORE INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", reusable
        )
        connection.commit()
        source.close()
    completed = {row[0] for row in connection.execute("SELECT image_id FROM features")}

    build_module = load_build_manifest_module(Path(__file__).with_name("build_manifest.py"))
    source_metadata = metadata_by_capture_group(build_module, args.texts_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.backbone, local_files_only=True)
    model = AutoModel.from_pretrained(args.backbone, local_files_only=True).eval().to(device)
    started = time.time()
    total = len(selected)
    thread_state = threading.local()

    def run_ocr(image_bgr: np.ndarray) -> tuple[list[str], list[float]]:
        if not hasattr(thread_state, "provider"):
            thread_state.provider = RapidOCR()
        output = thread_state.provider(image_bgr)
        return list(output.txts or ()), [float(value) for value in (output.scores or ())]

    newly_completed = 0
    next_report = ((len(completed) // 50) + 1) * 50

    def persist(item: tuple[Any, dict[str, Any], np.ndarray, float]) -> None:
        nonlocal newly_completed, next_report
        future, row, descriptor, blur = item
        texts, scores = future.result()
        capture_metadata = source_metadata.get(str(row["capture_group"]), {})
        connection.execute(
            """INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["image_id"], row["relative_path"], row["split"], row["role"], row["location_id"],
                row["capture_group"], row["illumination"], descriptor.tobytes(), descriptor.size,
                blur, json.dumps(texts, ensure_ascii=False), json.dumps(scores),
                finite_float(capture_metadata.get("Latitude")), finite_float(capture_metadata.get("Longitude")),
                finite_float(capture_metadata.get("Location Accuracy") or capture_metadata.get("GPS Horizontal Error")),
            ),
        )
        connection.commit()
        newly_completed += 1
        done = len(completed) + newly_completed
        if done >= next_report or done == total:
            elapsed = time.time() - started
            rate = newly_completed / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total - done)
            print(json.dumps({
                "completed": done,
                "total": total,
                "percent": round(100 * done / total, 2),
                "current": row["relative_path"],
                "rate_images_per_s": round(rate, 3),
                "eta_s": round(remaining / rate) if rate > 0 else None,
                "ocr_workers": args.ocr_workers,
            }, ensure_ascii=False), flush=True)
            next_report += 50

    pending: list[tuple[Any, dict[str, Any], np.ndarray, float]] = []
    unprocessed = [row for row in selected if row["image_id"] not in completed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.ocr_workers) as executor:
        for start in range(0, len(unprocessed), args.dino_batch_size):
            batch_rows = unprocessed[start:start + args.dino_batch_size]
            images_bgr = [decode_image(args.images_root / row["relative_path"]) for row in batch_rows]
            images_rgb = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images_bgr]
            descriptors = normalized_descriptors(model, processor, images_rgb, device)
            for row, image_bgr, descriptor in zip(batch_rows, images_bgr, descriptors):
                pending.append((executor.submit(run_ocr, image_bgr), row, descriptor, blur_variance(image_bgr)))
            while len(pending) >= args.ocr_workers * 4:
                persist(pending.pop(0))
        while pending:
            persist(pending.pop(0))

    counts = dict(connection.execute("SELECT split || ':' || role, COUNT(*) FROM features GROUP BY split, role"))
    ocr_count = int(connection.execute(
        "SELECT COUNT(*) FROM features WHERE ocr_texts_json != '[]'"
    ).fetchone()[0])
    missing_gps = int(connection.execute("SELECT COUNT(*) FROM features WHERE latitude IS NULL OR longitude IS NULL").fetchone()[0])
    receipt = {
        "schema": "blindassist.unseen_location_router.feature_receipt.v1",
        "status": "COMPLETE",
        "database": str(args.database),
        "selected_image_count": total,
        "counts": counts,
        "images_with_ocr": ocr_count,
        "images_without_gps": missing_gps,
        "manifest_sha256": metadata["manifest_sha256"],
        "backbone_sha256": backbone_hash,
        "device": str(device),
        "selection_salt": args.selection_salt,
        "selection_policy": metadata["selection_policy"],
        "ocr_workers": args.ocr_workers,
        "dino_batch_size": args.dino_batch_size,
        "test_images_read": 0,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

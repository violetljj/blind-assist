#!/usr/bin/env python3
"""Download the frozen perspective cohort and draw width-interval review overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2


SOURCE_SCHEMA = "blindassist-l10-width-first-perspective-portal-source-v1"
MANIFEST_SCHEMA = "blindassist-l10-width-first-perspective-portal-materialized-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def ensure_image(url: str, item_id: str, image_root: Path) -> Path:
    path = image_root / f"{item_id}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return path
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BlindAssist-L10-Width-Portal/1.0", "Accept": "image/jpeg"},
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            require(payload.startswith(b"\xff\xd8"), f"NOT_JPEG:{item_id}")
            atomic_write(path, payload)
            return path
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(float(attempt))
    raise RuntimeError(f"IMAGE_DOWNLOAD_FAILED:{item_id}:{last_error}")


def scaled_interval(values: list[float], source_width: int, actual_width: int) -> list[float]:
    scale = actual_width / source_width
    return [float(value) * scale for value in values]


def draw_overlay(
    image: Any,
    episode_id: str,
    role: str,
    nominal_x: list[float],
    uncertainty_x: list[float],
    review_max_width: int,
) -> tuple[Any, dict[str, Any]]:
    height, width = image.shape[:2]
    uncertainty_left = max(0, min(width - 1, round(min(uncertainty_x))))
    uncertainty_right = max(0, min(width - 1, round(max(uncertainty_x))))
    nominal_left = max(0, min(width - 1, round(min(nominal_x))))
    nominal_right = max(0, min(width - 1, round(max(nominal_x))))
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (uncertainty_left, 0),
        (uncertainty_right, height - 1),
        (0, 200, 255),
        thickness=-1,
    )
    rendered = cv2.addWeighted(overlay, 0.18, image, 0.82, 0.0)
    for x in (uncertainty_left, uncertainty_right):
        cv2.line(rendered, (x, 0), (x, height - 1), (0, 140, 255), max(2, width // 1000))
    for x in (nominal_left, nominal_right):
        cv2.line(rendered, (x, 0), (x, height - 1), (255, 255, 0), max(2, width // 800))
    cv2.putText(
        rendered,
        f"{episode_id} {role}",
        (max(12, width // 100), max(32, height // 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.8, width / 2500.0),
        (255, 255, 255),
        max(2, width // 1200),
        cv2.LINE_AA,
    )
    if width > review_max_width:
        scale = review_max_width / width
        rendered = cv2.resize(
            rendered,
            (review_max_width, round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return rendered, {
        "actual_uncertainty_x_interval": [uncertainty_left, uncertainty_right],
        "actual_nominal_x_interval": [nominal_left, nominal_right],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-max-width", type=int, default=1800)
    args = parser.parse_args()

    source_path = args.source.resolve()
    source = load_json(source_path)
    require(source.get("schema") == SOURCE_SCHEMA, "SOURCE_SCHEMA_MISMATCH")
    require(len(source.get("episodes") or []) == 3, "FROZEN_EPISODE_COUNT_MISMATCH")
    asset_root = args.asset_root.resolve()
    image_root = asset_root / "images"
    review_root = asset_root / "review"
    roles = []
    for episode in source["episodes"]:
        require(len(episode["roles"]) == 2, f"ROLE_COUNT_MISMATCH:{episode['episode_id']}")
        for role in episode["roles"]:
            image_path = ensure_image(role["hd_asset"], role["item_id"], image_root)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(image is not None, f"IMAGE_DECODE_FAILED:{role['item_id']}")
            height, width = image.shape[:2]
            sensor_width, sensor_height = role["sensor_dimensions"]
            require(
                abs(width / height - sensor_width / sensor_height) < 0.01,
                f"ASPECT_RATIO_MISMATCH:{role['item_id']}:{width}x{height}:{sensor_width}x{sensor_height}",
            )
            nominal = scaled_interval(role["nominal_sensor_x_interval"], sensor_width, width)
            uncertainty = scaled_interval(role["uncertainty_sensor_x_interval"], sensor_width, width)
            rendered, intervals = draw_overlay(
                image,
                episode["episode_id"],
                role["role"],
                nominal,
                uncertainty,
                args.review_max_width,
            )
            role_root = review_root / role["role"].lower()
            role_root.mkdir(parents=True, exist_ok=True)
            review_path = role_root / f"{episode['episode_id']}-{role['item_id']}.jpg"
            require(
                cv2.imwrite(str(review_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95]),
                f"REVIEW_IMAGE_WRITE_FAILED:{role['item_id']}",
            )
            roles.append(
                {
                    "episode_id": episode["episode_id"],
                    "role": role["role"],
                    "item_id": role["item_id"],
                    "collection_id": role["collection_id"],
                    "image": {
                        "path": str(image_path),
                        "sha256": sha256(image_path),
                        "bytes": image_path.stat().st_size,
                        "dimensions": [width, height],
                    },
                    "review_image": {
                        "path": str(review_path),
                        "sha256": sha256(review_path),
                        "bytes": review_path.stat().st_size,
                        "dimensions": [int(rendered.shape[1]), int(rendered.shape[0])],
                    },
                    **intervals,
                    "nominal_interval_color": "cyan",
                    "uncertainty_band_color": "orange",
                }
            )
            print(f"MATERIALIZED {episode['episode_id']} {role['role']}", flush=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "episode_count": 3,
        "role_image_count": len(roles),
        "roles": roles,
        "review_instruction": "Cyan lines are the nominal mapped-width endpoints. The orange band is the complete declared camera-horizontal-accuracy envelope. Judge only whether exactly one functional pedestrian portal is uniquely supported inside that band.",
    }
    write_json(args.manifest.resolve(), manifest)
    print(json.dumps({"manifest": str(args.manifest.resolve()), "images": len(roles)}, indent=2))


if __name__ == "__main__":
    main()

"""Materialize the bounded MSLS Development pixel cohort after source admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


MODULE_PATH = Path(__file__).with_name("build_msls_canary.py")
SPEC = importlib.util.spec_from_file_location("ulr_msls_canary_runtime", MODULE_PATH)
assert SPEC and SPEC.loader
MSLS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MSLS
SPEC.loader.exec_module(MSLS)

SCHEMA = "blindassist.unseen_location_router.manifest.v2"
SELECTION_SALT = "blindassist-ulr-msls-v1-development-pixels"


def stable_value(value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SELECTION_SALT}|{value}".encode()).digest()[:8], "big")


def representatives(rows: Iterable[dict[str, object]], limit: int) -> list[dict[str, object]]:
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)
    choices = [
        min(items, key=lambda row: (stable_value(str(row["key"])), str(row["key"])))
        for items in by_sequence.values()
    ]
    choices.sort(key=lambda row: (stable_value(f"{row['sequence_id']}|{row['key']}"), str(row["key"])))
    return choices[:limit]


def image_row(
    row: dict[str, object], *, city: str, partition: str, split: str, location_id: str,
    latitude: float | None, longitude: float | None, gps_accuracy_m: float | None,
) -> dict[str, object]:
    relative_path = f"train_val/{city}/{partition}/images/{row['key']}.jpg"
    return {
        "image_id": hashlib.sha256(relative_path.encode()).hexdigest()[:20],
        "relative_path": relative_path,
        "location_id": location_id,
        "capture_group": f"msls:{city}:{partition}:{row['sequence_id']}",
        "source_kind": f"msls_{partition}",
        "illumination": "night" if row["night"] else "day",
        "view_direction": row["view_direction"],
        "split": split,
        "role": "gallery" if partition == "database" else "query",
        "latitude": latitude,
        "longitude": longitude,
        "gps_accuracy_m": gps_accuracy_m,
    }


def build_manifest(
    root: Path, *, train_locations_per_city: int = 128, gallery_per_location: int = 2,
    query_per_location: int = 4,
) -> tuple[dict[str, object], list[str]]:
    locations: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    city_counts: list[dict[str, object]] = []
    for city, split in (
        [(value, "train") for value in MSLS.TRAIN_CITIES]
        + [(value, "development") for value in MSLS.DEVELOPMENT_CITIES]
    ):
        database = MSLS._load_partition(root, city, "database")
        queries = MSLS._load_partition(root, city, "query")
        usable_database = [row for row in database if row["easting"] is not None and row["northing"] is not None]
        database_tree = cKDTree(np.asarray([
            (float(row["easting"]), float(row["northing"])) for row in usable_database
        ], dtype=np.float64))
        gallery_by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in database:
            if row["latitude"] is not None and row["longitude"] is not None:
                gallery_by_cluster[str(row["cluster"])].append(row)

        query_by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in queries:
            if row["easting"] is None or row["northing"] is None or row["latitude"] is None or row["longitude"] is None:
                continue
            distance, index = database_tree.query((float(row["easting"]), float(row["northing"])), k=1)
            if float(distance) <= MSLS.POSITIVE_RADIUS_M:
                query_by_target[str(usable_database[int(index)]["cluster"])].append(row)

        eligible = sorted(
            set(gallery_by_cluster) & set(query_by_target),
            key=lambda cluster: (stable_value(f"{city}|{cluster}"), cluster),
        )
        selected_targets = set(eligible if split == "development" else eligible[:train_locations_per_city])
        cluster_centres = {
            cluster: {
                "latitude": sum(float(row["latitude"]) for row in rows) / len(rows),
                "longitude": sum(float(row["longitude"]) for row in rows) / len(rows),
                "easting": sum(float(row["easting"]) for row in rows) / len(rows),
                "northing": sum(float(row["northing"]) for row in rows) / len(rows),
            }
            for cluster, rows in gallery_by_cluster.items()
        }
        ordered_clusters = sorted(cluster_centres)
        location_tree = cKDTree(np.asarray([
            (cluster_centres[cluster]["easting"], cluster_centres[cluster]["northing"])
            for cluster in ordered_clusters
        ], dtype=np.float64))
        required_gallery_clusters = set(selected_targets)
        selected_query_rows: list[tuple[str, dict[str, object], float, float, float]] = []
        for cluster in sorted(selected_targets):
            for row in representatives(query_by_target[cluster], query_per_location):
                easting, northing = float(row["easting"]), float(row["northing"])
                coarse_easting = (math.floor(easting / MSLS.COARSE_GRID_M) + 0.5) * MSLS.COARSE_GRID_M
                coarse_northing = (math.floor(northing / MSLS.COARSE_GRID_M) + 0.5) * MSLS.COARSE_GRID_M
                _, indices = location_tree.query(
                    (coarse_easting, coarse_northing), k=min(16, len(ordered_clusters))
                )
                required_gallery_clusters.update(ordered_clusters[int(index)] for index in np.atleast_1d(indices))
                selected_query_rows.append((
                    cluster, row, coarse_easting, coarse_northing,
                    math.hypot(coarse_easting - easting, coarse_northing - northing),
                ))

        city_gallery = city_queries = 0
        for cluster in sorted(required_gallery_clusters):
            rows = gallery_by_cluster[cluster]
            location_id = f"{city}:{cluster}"
            centre = cluster_centres[cluster]
            locations.append({
                "location_id": location_id, "location_type": "msls_unique_cluster", "city": city,
                "split": split, "latitude": centre["latitude"], "longitude": centre["longitude"],
            })
            for row in representatives(rows, gallery_per_location):
                images.append(image_row(
                    row, city=city, partition="database", split=split, location_id=location_id,
                    latitude=None, longitude=None, gps_accuracy_m=None,
                ))
                city_gallery += 1

        for cluster, row, coarse_easting, coarse_northing, accuracy in selected_query_rows:
            location_id = f"{city}:{cluster}"
            easting, northing = float(row["easting"]), float(row["northing"])
            latitude = float(row["latitude"]) + (coarse_northing - northing) / 111_320.0
            longitude = float(row["longitude"]) + (coarse_easting - easting) / (
                111_320.0 * max(1e-6, math.cos(math.radians(float(row["latitude"]))))
            )
            images.append(image_row(
                row, city=city, partition="query", split=split, location_id=location_id,
                latitude=latitude, longitude=longitude, gps_accuracy_m=accuracy,
            ))
            city_queries += 1
        city_counts.append({
            "city": city, "split": split, "source_candidate_locations": len(gallery_by_cluster),
            "materialized_candidate_locations": len(required_gallery_clusters),
            "selected_query_locations": len(selected_targets), "gallery_images": city_gallery,
            "query_images": city_queries,
        })

    manifest = {
        "schema": SCHEMA,
        "source": "Mapillary Street-Level Sequences",
        "selection_salt": SELECTION_SALT,
        "coarse_gps_adapter": {"kind": "utm_grid_cell_center", "grid_m": MSLS.COARSE_GRID_M},
        "positive_radius_m": MSLS.POSITIVE_RADIUS_M,
        "train_locations_per_city": train_locations_per_city,
        "gallery_per_location": gallery_per_location,
        "query_per_location": query_per_location,
        "locations": locations,
        "images": sorted(images, key=lambda row: (str(row["split"]), str(row["role"]), str(row["relative_path"]))),
        "cities": city_counts,
        "unopened_test_cities": list(MSLS.UNOPENED_TEST_CITIES),
        "test_metadata_read": False,
        "test_images_read": 0,
    }
    return manifest, sorted(str(row["relative_path"]) for row in images)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-paths", type=Path, required=True)
    parser.add_argument("--train-locations-per-city", type=int, default=128)
    args = parser.parse_args()
    manifest, image_paths = build_manifest(args.dataset_root, train_locations_per_city=args.train_locations_per_city)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_paths.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.image_paths.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": manifest["schema"], "location_count": len(manifest["locations"]),
        "image_count": len(manifest["images"]),
        "gallery_count": sum(row["role"] == "gallery" for row in manifest["images"]),
        "query_count": sum(row["role"] == "query" for row in manifest["images"]),
        "test_images_read": 0,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

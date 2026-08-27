"""Build the metadata-only MSLS source-admission canary.

This script deliberately reads only the official MSLS train and validation
cities.  Test metadata and test pixels remain unopened.  The evaluator label is
the cluster of the nearest official database frame within the frozen 10 metre
positive radius.  Candidate construction uses a 100 metre, outcome-independent
quantization of the query GPS to model the coarse phone-location interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = "blindassist.unseen_location_router.msls_source_canary.v1"
TRAIN_CITIES = (
    "trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
    "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
    "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman",
    "nairobi", "manila",
)
DEVELOPMENT_CITIES = ("cph", "sf")
UNOPENED_TEST_CITIES = (
    "miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala",
)
POSITIVE_RADIUS_M = 10.0
COARSE_GRID_M = 100.0
CANDIDATE_SIZES = (8, 16)


def _finite(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _key(row: dict[str, str]) -> str:
    for name in ("key", "image_key"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    raise ValueError("MSLS metadata row has no key")


def _sequence_id(row: dict[str, str]) -> str:
    for name in ("sequence_id", "sequence_key"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    raise ValueError("MSLS sequence row has no sequence identifier")


def _cluster(row: dict[str, str]) -> str:
    value = str(row.get("unique_cluster", "")).strip()
    if not value:
        raise ValueError("MSLS postprocessed row has no unique_cluster")
    return value


def haversine_meters(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius = 6_371_008.8
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lambda = math.radians(lon_b - lon_a)
    value = math.sin(d_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def coarse_gps(latitude: float, longitude: float, grid_m: float = COARSE_GRID_M) -> tuple[float, float]:
    """Return the centre of a deterministic local metric grid cell."""

    lat_step = grid_m / 111_320.0
    lon_scale = max(1e-6, math.cos(math.radians(latitude)))
    lon_step = grid_m / (111_320.0 * lon_scale)
    return (
        (math.floor(latitude / lat_step) + 0.5) * lat_step,
        (math.floor(longitude / lon_step) + 0.5) * lon_step,
    )


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _load_partition(root: Path, city: str, partition: str) -> list[dict[str, object]]:
    folder = root / "train_val" / city / partition
    raw = {_key(row): row for row in _read_csv(folder / "raw.csv")}
    post = {_key(row): row for row in _read_csv(folder / "postprocessed.csv")}
    sequences = {_key(row): row for row in _read_csv(folder / "seq_info.csv")}
    keys = sorted(set(raw) | set(post) | set(sequences))
    if set(raw) != set(post) or set(raw) != set(sequences):
        raise ValueError(f"{city}/{partition} metadata keys do not align")
    rows: list[dict[str, object]] = []
    for key in keys:
        latitude = _finite(raw[key].get("lat", ""))
        longitude = _finite(raw[key].get("lon", ""))
        easting = _finite(post[key].get("easting", ""))
        northing = _finite(post[key].get("northing", ""))
        rows.append({
            "key": key,
            "latitude": latitude,
            "longitude": longitude,
            "easting": easting,
            "northing": northing,
            "cluster": _cluster(post[key]),
            "sequence_id": _sequence_id(sequences[key]),
            "night": str(post[key].get("night", "")).strip().casefold() in {"1", "true", "yes"},
            "view_direction": str(post[key].get("view_direction", "unknown")).strip() or "unknown",
            "image_exists": (folder / "images" / f"{key}.jpg").is_file(),
        })
    return rows


def build_canary(
    root: Path,
    *,
    train_cities: Iterable[str] = TRAIN_CITIES,
    development_cities: Iterable[str] = DEVELOPMENT_CITIES,
) -> dict[str, object]:
    city_splits = [(city, "train") for city in train_cities] + [
        (city, "development") for city in development_cities
    ]
    split_stats: dict[str, Counter[str]] = defaultdict(Counter)
    split_ranks: dict[str, list[int]] = defaultdict(list)
    split_errors: dict[str, list[float]] = defaultdict(list)
    city_results: list[dict[str, object]] = []

    for city, split in city_splits:
        database = _load_partition(root, city, "database")
        queries = _load_partition(root, city, "query")
        usable_database = [row for row in database if row["easting"] is not None and row["northing"] is not None]
        if not usable_database:
            raise ValueError(f"{city}/database has no usable UTM coordinates")
        database_tree = cKDTree(np.asarray([
            (float(row["easting"]), float(row["northing"])) for row in usable_database
        ], dtype=np.float64))
        cluster_coordinates: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in database:
            if row["easting"] is not None and row["northing"] is not None:
                cluster_coordinates[str(row["cluster"])].append((float(row["easting"]), float(row["northing"])))
        locations = [
            {
                "location_id": f"{city}:{cluster}",
                "easting": sum(value[0] for value in coordinates) / len(coordinates),
                "northing": sum(value[1] for value in coordinates) / len(coordinates),
            }
            for cluster, coordinates in sorted(cluster_coordinates.items())
        ]
        location_tree = cKDTree(np.asarray([
            (float(row["easting"]), float(row["northing"])) for row in locations
        ], dtype=np.float64))
        city_counts: Counter[str] = Counter(database=len(database), query=len(queries), locations=len(locations))
        for query in queries:
            city_counts["query_with_frame_gps"] += int(query["latitude"] is not None and query["longitude"] is not None)
            city_counts["query_image_present"] += int(bool(query["image_exists"]))
            if query["easting"] is None or query["northing"] is None:
                city_counts["query_without_official_positive"] += 1
                continue
            query_point = (float(query["easting"]), float(query["northing"]))
            positive_distance, database_index = database_tree.query(query_point, k=1)
            if float(positive_distance) > POSITIVE_RADIUS_M:
                city_counts["query_without_official_positive"] += 1
                continue
            target = usable_database[int(database_index)]
            city_counts["query_with_official_positive"] += 1
            if query["latitude"] is None or query["longitude"] is None:
                continue
            coarse_easting = (math.floor(query_point[0] / COARSE_GRID_M) + 0.5) * COARSE_GRID_M
            coarse_northing = (math.floor(query_point[1] / COARSE_GRID_M) + 0.5) * COARSE_GRID_M
            neighbor_count = min(max(CANDIDATE_SIZES), len(locations))
            _, neighbor_indices = location_tree.query((coarse_easting, coarse_northing), k=neighbor_count)
            neighbor_indices = np.atleast_1d(neighbor_indices)
            ranked_ids = [locations[int(index)]["location_id"] for index in neighbor_indices]
            target_id = f"{city}:{target['cluster']}"
            rank = next((index for index, location_id in enumerate(ranked_ids, start=1) if location_id == target_id), max(CANDIDATE_SIZES) + 1)
            split_ranks[split].append(rank)
            split_errors[split].append(math.hypot(coarse_easting - query_point[0], coarse_northing - query_point[1]))
            city_counts["ranked_query"] += 1
            city_counts["positive_distance_mm_sum"] += round(float(positive_distance) * 1000)
        split_stats[split].update(city_counts)
        city_results.append({"city": city, "split": split, **dict(sorted(city_counts.items()))})

    splits: dict[str, object] = {}
    for split in ("train", "development"):
        stats = split_stats[split]
        ranks = split_ranks[split]
        gps_denominator = stats["query"]
        ranked_denominator = len(ranks)
        splits[split] = {
            **dict(sorted(stats.items())),
            "query_frame_gps_rate": stats["query_with_frame_gps"] / gps_denominator if gps_denominator else None,
            "coverage": {
                str(size): {
                    "covered": sum(rank <= size for rank in ranks),
                    "denominator": ranked_denominator,
                    "rate": sum(rank <= size for rank in ranks) / ranked_denominator if ranked_denominator else None,
                }
                for size in CANDIDATE_SIZES
            },
            "target_rank": {
                "median": _percentile([float(rank) for rank in ranks], 0.5),
                "p90": _percentile([float(rank) for rank in ranks], 0.9),
                "p95": _percentile([float(rank) for rank in ranks], 0.95),
                "outside_k16_censored_as": max(CANDIDATE_SIZES) + 1,
            },
            "coarse_gps_error_m": {
                "median": _percentile(split_errors[split], 0.5),
                "p90": _percentile(split_errors[split], 0.9),
                "p95": _percentile(split_errors[split], 0.95),
            },
        }

    admission_checks = {
        "query_frame_gps_100_percent": all(splits[split]["query_frame_gps_rate"] == 1.0 for split in splits),
        "k8_coverage_at_least_90_percent": all(splits[split]["coverage"]["8"]["rate"] is not None and splits[split]["coverage"]["8"]["rate"] >= 0.90 for split in splits),
        "k16_coverage_at_least_99_percent": all(splits[split]["coverage"]["16"]["rate"] is not None and splits[split]["coverage"]["16"]["rate"] >= 0.99 for split in splits),
    }
    return {
        "schema": SCHEMA,
        "status": "ADMITTED" if all(admission_checks.values()) else "REJECTED",
        "source": "Mapillary Street-Level Sequences",
        "source_root": str(root),
        "positive_radius_m": POSITIVE_RADIUS_M,
        "coarse_gps_adapter": {"kind": "metric_grid_cell_center", "grid_m": COARSE_GRID_M},
        "candidate_sizes": list(CANDIDATE_SIZES),
        "train_cities": list(train_cities),
        "development_cities": list(development_cities),
        "unopened_test_cities": list(UNOPENED_TEST_CITIES),
        "test_metadata_read": False,
        "test_images_read": 0,
        "admission_checks": admission_checks,
        "splits": splits,
        "cities": city_results,
        "source_identity_sha256": hashlib.sha256(json.dumps(city_results, sort_keys=True).encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_canary(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ADMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

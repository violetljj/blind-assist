"""Evaluator-only audit of coarse-GPS candidate coverage before model claims."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path


def haversine_meters(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius = 6_371_008.8
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lambda = math.radians(lon_b - lon_a)
    value = math.sin(d_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    locations = {
        row["location_id"]: row for row in manifest["locations"] if row["split"] != "test"
    }
    connection = sqlite3.connect(args.database)
    query_rows = list(connection.execute(
        """SELECT split, location_id, latitude, longitude FROM features
        WHERE role = 'query' AND split != 'test'"""
    ))
    connection.close()

    result: dict[str, object] = {
        "schema": "blindassist.unseen_location_router.candidate_coverage.v1",
        "status": "EVALUATOR_ONLY",
        "candidate_sizes": [4, 8, 16],
        "splits": {},
        "test_images_read": 0,
    }
    for split in ("train", "development"):
        split_locations = [row for row in locations.values() if row["split"] == split]
        rows = [row for row in query_rows if row[0] == split]
        gps_rows = [row for row in rows if row[2] is not None and row[3] is not None]
        ranks: list[int] = []
        target_distances: list[float] = []
        for _, target_id, latitude, longitude in gps_rows:
            ranked = sorted(split_locations, key=lambda location: (
                haversine_meters(latitude, longitude, float(location["latitude"]), float(location["longitude"])),
                location["location_id"],
            ))
            ranks.append(next(index for index, location in enumerate(ranked, start=1) if location["location_id"] == target_id))
            target = locations[target_id]
            target_distances.append(haversine_meters(
                latitude, longitude, float(target["latitude"]), float(target["longitude"])
            ))
        result["splits"][split] = {
            "query_count": len(rows),
            "queries_with_gps": len(gps_rows),
            "coverage": {
                str(size): {
                    "covered": sum(rank <= size for rank in ranks),
                    "denominator": len(ranks),
                    "rate": sum(rank <= size for rank in ranks) / len(ranks) if ranks else None,
                }
                for size in (4, 8, 16)
            },
            "target_rank": {
                "median": percentile([float(rank) for rank in ranks], 0.5),
                "p90": percentile([float(rank) for rank in ranks], 0.9),
                "p95": percentile([float(rank) for rank in ranks], 0.95),
            },
            "gps_to_target_distance_m": {
                "median": percentile(target_distances, 0.5),
                "p90": percentile(target_distances, 0.9),
                "p95": percentile(target_distances, 0.95),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

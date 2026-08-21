"""Plan an outcome-blind, venue-parent-disjoint resolvable-enrichment acquisition.

The planner uses source metadata only.  It never reads RGB, proposal scores,
Brain output, or reviewed resolution labels.  Its output is Development roster
authority, not a claim that any selected venue will be visually resolvable.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


POLICY_ID = "P0-D2-RESOLVABLE-ENRICHMENT-ROSTER-V1"
BASIC_CATEGORY_FAMILY = {
    "bank_or_credit_union": "FINANCIAL",
    "books_music_and_video_store": "SPECIALTY_RETAIL",
    "casual_eatery": "HOSPITALITY",
    "coffee_shop": "HOSPITALITY",
    "fashion_and_apparel_store": "FASHION",
    "fast_food_restaurant": "HOSPITALITY",
    "flowers_and_gifts_store": "SPECIALTY_RETAIL",
    "food_and_beverage_store": "FOOD_RETAIL",
    "hotel": "LODGING",
    "lodging": "LODGING",
    "museum": "CULTURAL",
    "personal_care_and_beauty_store": "PERSONAL_CARE",
    "personal_or_beauty_service": "PERSONAL_CARE",
    "restaurant": "HOSPITALITY",
    "specialty_store": "SPECIALTY_RETAIL",
    "toys_and_games_store": "SPECIALTY_RETAIL",
}


def _category_family(primary: str, basic: str) -> str | None:
    if basic in BASIC_CATEGORY_FAMILY:
        return BASIC_CATEGORY_FAMILY[basic]
    if primary in {"pharmacy", "clinic", "dentist"} or primary.endswith("_clinic"):
        return "HEALTH"
    if primary.endswith("_restaurant") or primary in {"cafe", "restaurant"}:
        return "HOSPITALITY"
    return None


class RosterError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RosterError(message)


def _properties_by_id(places: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    features = places.get("features")
    _require(isinstance(features, list), "places features missing")
    return {str(item["id"]): item.get("properties", {}) for item in features}


def plan_roster(
    source_report: Mapping[str, Any],
    places: Mapping[str, Any],
    excluded_acquisitions: Sequence[Mapping[str, Any]],
    *,
    requested_parent_count: int = 40,
    minimum_place_confidence: float = 0.85,
    maximum_family_count: int = 30,
    maximum_anchors_per_acquisition_shard: int = 32,
    excluded_target_names: Sequence[str] = (),
) -> dict[str, Any]:
    _require(10 <= requested_parent_count <= 80, "requested_parent_count outside bounded Development range")
    _require(0.0 <= minimum_place_confidence <= 1.0, "minimum_place_confidence outside [0,1]")
    _require(1 <= maximum_anchors_per_acquisition_shard <= 40, "acquisition shard anchor cap outside [1,40]")
    properties = _properties_by_id(places)
    excluded_buildings: set[str] = set()
    excluded_frame_ids: set[str] = set()
    excluded_names = {str(value).strip().casefold() for value in excluded_target_names}
    for document in excluded_acquisitions:
        acquisition = document.get("acquisition", {})
        excluded_buildings.update(str(value) for value in acquisition.get("target_building_ids", []))
        excluded_frame_ids.update(str(item["id"]) for item in document.get("images", []))

    anchors: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_report.get("osm_entrance_building_crosswalk_candidates", []):
        if row.get("status") != "CANDIDATE_ONLY" or row.get("entrance") in {"exit", "no", "service", "emergency"}:
            continue
        anchors[str(row["overture_building_id"])].append(row)

    places_by_building: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_report.get("place_building_crosswalk_candidates", []):
        building_ids = row.get("building_ids", [])
        if row.get("status") == "CANDIDATE_ONLY" and len(building_ids) == 1:
            places_by_building[str(building_ids[0])].append(row)

    eligible = []
    for building_id, place_rows in places_by_building.items():
        if building_id in excluded_buildings or len(place_rows) != 1 or not (1 <= len(anchors.get(building_id, [])) <= 2):
            continue
        place = place_rows[0]
        props = properties.get(str(place["place_id"]), {})
        categories = props.get("categories") or {}
        primary = str(categories.get("primary") or "") if isinstance(categories, Mapping) else ""
        basic = str(props.get("basic_category") or "")
        confidence = float(props.get("confidence") or 0.0)
        name = str(place.get("place_name") or "").strip()
        family = _category_family(primary, basic)
        if family is None or confidence < minimum_place_confidence or len(name) < 4 or name.casefold() in excluded_names:
            continue
        brand = props.get("brand") or {}
        brand_names = (brand.get("names") or {}) if isinstance(brand, Mapping) else {}
        branded = bool(brand_names.get("primary")) if isinstance(brand_names, Mapping) else False
        eligible.append({
            "building_id": building_id,
            "place_id": str(place["place_id"]),
            "place_name": name,
            "primary_category": primary,
            "basic_category": basic,
            "category_family": family,
            "place_confidence": confidence,
            "brand_name_present": branded,
            "eligible_anchor_ids": sorted(str(item["osm_entrance_id"]) for item in anchors[building_id]),
        })
    eligible.sort(key=lambda row: (
        -int(row["brand_name_present"]), -float(row["place_confidence"]),
        len(row["eligible_anchor_ids"]), str(row["category_family"]), str(row["place_name"]), str(row["building_id"]),
    ))

    selected = []
    family_counts: Counter[str] = Counter()
    for row in eligible:
        family = str(row["category_family"])
        if family_counts[family] >= maximum_family_count:
            continue
        selected.append(row)
        family_counts[family] += 1
        if len(selected) == requested_parent_count:
            break
    _require(len(selected) == requested_parent_count, "insufficient metadata-eligible parent roster")
    total_anchors = sum(len(row["eligible_anchor_ids"]) for row in selected)
    shard_count = math.ceil(total_anchors / maximum_anchors_per_acquisition_shard)
    shards: list[dict[str, Any]] = [
        {"shard_id": f"shard-{index + 1:02d}", "anchor_count": 0, "building_ids": []}
        for index in range(shard_count)
    ]
    for row in sorted(selected, key=lambda item: (-len(item["eligible_anchor_ids"]), str(item["building_id"]))):
        shard = min(shards, key=lambda item: (item["anchor_count"], item["shard_id"]))
        anchor_count = len(row["eligible_anchor_ids"])
        _require(shard["anchor_count"] + anchor_count <= maximum_anchors_per_acquisition_shard, "balanced shard packing exceeded anchor cap")
        shard["anchor_count"] += anchor_count
        shard["building_ids"].append(row["building_id"])
    for shard in shards:
        shard["parent_count"] = len(shard["building_ids"])
    report = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "data_role": "DEVELOPMENT_RESOLVABLE_ENRICHMENT_NOT_ADJUDICATION",
        "selection_visibility": "SOURCE_METADATA_ONLY_NO_RGB_NO_PROPOSAL_NO_BRAIN_NO_REVIEWED_OUTCOME",
        "source_report_sha256": source_report.get("report_sha256"),
        "excluded_target_building_ids": sorted(excluded_buildings),
        "excluded_frame_ids": sorted(excluded_frame_ids),
        "criteria": {
            "unique_place_to_building": True,
            "eligible_anchor_count": [1, 2],
            "minimum_place_confidence": minimum_place_confidence,
            "allowed_basic_categories": sorted(BASIC_CATEGORY_FAMILY),
            "additional_primary_category_rules": ["*_restaurant", "*_clinic", "cafe", "restaurant", "pharmacy", "clinic", "dentist"],
            "maximum_family_count": maximum_family_count,
            "maximum_anchors_per_acquisition_shard": maximum_anchors_per_acquisition_shard,
        },
        "requested_parent_count": requested_parent_count,
        "selected_parent_count": len(selected),
        "family_counts": dict(sorted(family_counts.items())),
        "parents": selected,
        "acquisition_shards": shards,
        "claim_ceiling": "OUTCOME_BLIND_DEVELOPMENT_ACQUISITION_ROSTER_ONLY_NO_RESOLVABILITY_OR_MODEL_CLAIM",
    }
    if excluded_target_names:
        report["excluded_target_names"] = sorted(str(value) for value in excluded_target_names)
    report["report_sha256"] = materializer.content_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--places", required=True, type=Path)
    parser.add_argument("--exclude-metadata", action="append", type=Path, default=[])
    parser.add_argument("--exclude-cohort", action="append", type=Path, default=[])
    parser.add_argument("--requested-parent-count", type=int, default=40)
    parser.add_argument("--maximum-family-count", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    places = json.loads(args.places.read_text(encoding="utf-8"))
    excluded = [json.loads(path.read_text(encoding="utf-8")) for path in args.exclude_metadata]
    excluded_names = []
    for path in args.exclude_cohort:
        cohort = json.loads(path.read_text(encoding="utf-8"))
        excluded_names.extend(
            str(item["evaluator_episode"]["goal_spec"]["target_name"])
            for item in cohort.get("episodes", [])
        )
    report = plan_roster(
        source, places, excluded,
        requested_parent_count=args.requested_parent_count,
        maximum_family_count=args.maximum_family_count,
        excluded_target_names=excluded_names,
    )
    materializer.write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("selected_parent_count", "family_counts", "claim_ceiling", "report_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

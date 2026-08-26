"""Build and audit a location-disjoint MMS-VPR manifest.

The output contains evaluator identities, so it is a data/evaluation manifest,
not a model input. Provider feature exporters must emit candidate-aligned scores
without forwarding ``location_id`` into a learned arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET


SCHEMA = "blindassist.unseen_location_router.manifest.v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
LOCATION_RE = re.compile(r"^(?:N|Eh|Ev|S)-\d+(?:-\d+)?$", re.IGNORECASE)
IMAGE_ID_RE = re.compile(r"IMG[_ -]?(\d+)", re.IGNORECASE)
SOCIAL_EVENT_RE = re.compile(r"(20\d{2})\D+Event\D*(\d+)", re.IGNORECASE)
FRAME_SUFFIX_RE = re.compile(r"(?:[_ -](?:frame|f))?[_ -]?\d{3,}$", re.IGNORECASE)


@dataclass(frozen=True)
class Location:
    location_id: str
    location_type: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    location_id: str
    capture_group: str
    source_kind: str
    illumination: str


def _stable_value(salt: str, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest()[:8], "big")


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    result = 0
    for character in letters.upper():
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def read_first_xlsx_sheet(path: Path) -> list[dict[str, object]]:
    """Read the first XLSX sheet using only the Python standard library."""

    namespaces = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", namespaces):
                shared.append("".join(node.text or "" for node in item.findall(".//m:t", namespaces)))
        sheet_name = sorted(
            name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )[0]
        sheet = ET.fromstring(archive.read(sheet_name))

    rows: list[list[object]] = []
    for row in sheet.findall(".//m:sheetData/m:row", namespaces):
        values: dict[int, object] = {}
        for cell in row.findall("m:c", namespaces):
            index = _column_index(cell.attrib.get("r", "A1"))
            kind = cell.attrib.get("t")
            value_node = cell.find("m:v", namespaces)
            if kind == "inlineStr":
                value: object = "".join(node.text or "" for node in cell.findall(".//m:t", namespaces))
            elif value_node is None:
                value = ""
            elif kind == "s":
                value = shared[int(value_node.text or "0")]
            else:
                raw = value_node.text or ""
                try:
                    number = float(raw)
                    value = int(number) if number.is_integer() else number
                except ValueError:
                    value = raw
            values[index] = value
        width = max(values, default=-1) + 1
        rows.append([values.get(index, "") for index in range(width)])
    if not rows:
        return []
    headers = [str(value).strip() for value in rows[0]]
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(headers) if header}
        for row in rows[1:]
    ]


def load_location_catalog(graph_root: Path) -> dict[str, Location]:
    specs = (
        ("01 Node Features.xlsx", "Code", "node"),
        ("02 Edge Features.xlsx", "SegmentID", "edge"),
        ("04 Square Features.xlsx", "Square Code", "square"),
    )
    result: dict[str, Location] = {}
    for filename, id_column, location_type in specs:
        path = graph_root / filename
        for row in read_first_xlsx_sheet(path):
            location_id = str(row[id_column]).strip()
            result[location_id] = Location(
                location_id=location_id,
                location_type=location_type,
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
            )
    return result


def location_id_from_path(path: Path, images_root: Path) -> str:
    relative = path.relative_to(images_root)
    for part in relative.parts[:-1]:
        if LOCATION_RE.fullmatch(part):
            return part
    matches = re.findall(r"(?:N|Eh|Ev|S)-\d+(?:-\d+)?", path.stem, re.IGNORECASE)
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"cannot determine one location ID from {relative.as_posix()}")


def capture_group_from_name(name: str) -> tuple[str, str]:
    social = SOCIAL_EVENT_RE.search(name)
    if social:
        return f"social:{social.group(1)}:{social.group(2)}", "social_media"
    image = IMAGE_ID_RE.search(name)
    if image:
        return f"field:IMG_{image.group(1)}", "field_capture"
    stem = FRAME_SUFFIX_RE.sub("", Path(name).stem).strip(" _-") or Path(name).stem
    return f"fallback:{stem.casefold()}", "unknown"


def illumination_from_name(name: str) -> str:
    folded = name.casefold()
    if "night" in folded:
        return "night"
    if "day" in folded or "sunny" in folded:
        return "day"
    return "unknown"


def scan_images(images_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(item for item in images_root.rglob("*") if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES):
        relative = path.relative_to(images_root).as_posix()
        capture_group, source_kind = capture_group_from_name(path.name)
        records.append(ImageRecord(
            image_id=hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20],
            relative_path=relative,
            location_id=location_id_from_path(path, images_root),
            capture_group=capture_group,
            source_kind=source_kind,
            illumination=illumination_from_name(path.name),
        ))
    return records


def assign_location_splits(
    locations: Sequence[Location], *, salt: str, ratios: Mapping[str, float]
) -> dict[str, str]:
    if set(ratios) != {"train", "development", "test"} or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must contain train/development/test and sum to one")
    by_type: dict[str, list[Location]] = defaultdict(list)
    for location in locations:
        by_type[location.location_type].append(location)
    result: dict[str, str] = {}
    for location_type, group in sorted(by_type.items()):
        ordered = sorted(group, key=lambda item: (_stable_value(salt, item.location_id), item.location_id))
        count = len(ordered)
        development_count = round(count * ratios["development"])
        test_count = round(count * ratios["test"])
        if count >= 3:
            development_count = max(1, development_count)
            test_count = max(1, test_count)
        while development_count + test_count >= count:
            if development_count >= test_count and development_count > 0:
                development_count -= 1
            elif test_count > 0:
                test_count -= 1
        train_end = count - development_count - test_count
        development_end = train_end + development_count
        for index, location in enumerate(ordered):
            split = "train" if index < train_end else "development" if index < development_end else "test"
            result[location.location_id] = split
    return result


def partition_capture_groups(groups: Iterable[str], *, salt: str, location_id: str) -> dict[str, str]:
    unique = sorted(set(groups), key=lambda value: (_stable_value(salt, f"{location_id}|{value}"), value))
    if len(unique) < 2:
        raise ValueError("evaluation location needs at least two independent capture groups")
    gallery_count = max(1, min(len(unique) - 1, round(len(unique) * 0.3)))
    return {group: "gallery" if index < gallery_count else "query" for index, group in enumerate(unique)}


def build_manifest(
    *, images_root: Path, catalog: Mapping[str, Location], split_salt: str, group_salt: str
) -> tuple[dict[str, object], dict[str, object]]:
    records = scan_images(images_root)
    location_records: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        location_records[record.location_id].append(record)

    unknown_locations = sorted(set(location_records) - set(catalog))
    missing_locations = sorted(set(catalog) - set(location_records))
    split_by_location = assign_location_splits(
        list(catalog.values()),
        salt=split_salt,
        ratios={"train": 0.7, "development": 0.15, "test": 0.15},
    )
    rejected_locations: dict[str, str] = {}
    output_records: list[dict[str, object]] = []
    group_partitions: dict[str, dict[str, str]] = {}
    for location_id, items in sorted(location_records.items()):
        split = split_by_location.get(location_id)
        if split is None:
            continue
        try:
            group_partition = partition_capture_groups(
                (item.capture_group for item in items), salt=group_salt, location_id=location_id
            )
        except ValueError as error:
            rejected_locations[location_id] = str(error)
            continue
        group_partitions[location_id] = group_partition
        for item in items:
            row = asdict(item)
            row["split"] = split
            row["role"] = group_partition[item.capture_group]
            output_records.append(row)

    location_counts = Counter(split_by_location.values())
    image_counts = Counter(str(item["split"]) for item in output_records)
    role_counts = Counter(str(item["role"]) for item in output_records)
    valid = (
        bool(records)
        and not unknown_locations
        and not missing_locations
        and not rejected_locations
        and all(location_counts.get(split, 0) > 0 for split in ("train", "development", "test"))
    )
    audit = {
        "schema": "blindassist.unseen_location_router.data_admission.v1",
        "status": "ADMITTED" if valid else "REJECTED",
        "image_count": len(records),
        "catalog_location_count": len(catalog),
        "observed_location_count": len(location_records),
        "location_counts_by_split": dict(sorted(location_counts.items())),
        "image_counts_by_split": dict(sorted(image_counts.items())),
        "image_counts_by_role": dict(sorted(role_counts.items())),
        "capture_group_count": len({(row["location_id"], row["capture_group"]) for row in output_records}),
        "unknown_image_locations": unknown_locations,
        "catalog_locations_without_images": missing_locations,
        "rejected_locations": rejected_locations,
        "limitations": [
            "single commercial district; no unseen-city authority",
            "location classes are nodes, road segments, and squares; no named-POI authority",
            "no target boxes or masks",
            "manual store/sign annotations are forbidden as query OCR",
        ],
    }
    manifest = {
        "schema": SCHEMA,
        "split_salt": split_salt,
        "capture_group_salt": group_salt,
        "locations": [
            {
                **asdict(location),
                "split": split_by_location[location.location_id],
            }
            for location in sorted(catalog.values(), key=lambda item: item.location_id)
        ],
        "images": output_records,
    }
    return manifest, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split-salt", default="blindassist-ulr-v1-location-split")
    parser.add_argument("--group-salt", default="blindassist-ulr-v1-gallery-query")
    args = parser.parse_args()

    catalog = load_location_catalog(args.graph_root)
    manifest, audit = build_manifest(
        images_root=args.images_root,
        catalog=catalog,
        split_salt=args.split_salt,
        group_salt=args.group_salt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["status"] == "ADMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

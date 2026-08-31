#!/usr/bin/env python3
"""Materialize the exact metadata-frozen KartaView/Jakarta frames."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CANDIDATE = HERE / "l10_kartaview_jakarta_provider_disjoint_candidate_v1.json"
MANIFEST = HERE / "l10_kartaview_jakarta_materialization_v1.json"
OUTPUT = REPO / "artifacts.local" / "datasets" / "l10-kartaview-jakarta-provider-disjoint-v1" / "images"
USER_AGENT = "BlindAssist-L10-research/1.0"
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_bytes(url: str) -> tuple[bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_IMAGE_BYTES + 1)
        content_type = response.headers.get("Content-Type")
    if len(data) > MAX_IMAGE_BYTES:
        raise RuntimeError(f"response exceeded {MAX_IMAGE_BYTES} bytes: {url}")
    return data, content_type


def jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise RuntimeError("download is not a JPEG")
    offset = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0x01, 0xD8, 0xD9}:
            continue
        if offset + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            raise RuntimeError("invalid JPEG segment length")
        if marker in sof_markers:
            if segment_length < 7:
                raise RuntimeError("invalid JPEG SOF segment")
            height = struct.unpack(">H", data[offset + 3 : offset + 5])[0]
            width = struct.unpack(">H", data[offset + 5 : offset + 7])[0]
            if width < 320 or height < 240:
                raise RuntimeError(f"decoded JPEG is too small: {width}x{height}")
            return width, height
        offset += segment_length
    raise RuntimeError("JPEG dimensions were not found")


def expected_rows(candidate: dict) -> list[dict]:
    rows: list[dict] = []
    for candidate_row in candidate["rows"]:
        candidate_id = candidate_row["candidate_id"]
        query = candidate_row["query"]
        for frame in query["frame_window"]:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "role": "QUERY_WINDOW",
                    "photo_id": int(frame["photo_id"]),
                    "sequence_id": int(query["sequence_id"]),
                    "sequence_index": int(frame["sequence_index"]),
                    "contributor_user_id": int(query["contributor_user_id"]),
                    "metadata_in_fov": bool(frame["metadata_in_fov"]),
                }
            )
        negative = candidate_row["negative_control"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "role": "GEOMETRIC_NEGATIVE_CANDIDATE",
                "photo_id": int(negative["photo_id"]),
                "sequence_id": int(negative["sequence_id"]),
                "sequence_index": int(negative["sequence_index"]),
                "contributor_user_id": int(negative["contributor_user_id"]),
                "metadata_in_fov": bool(negative["metadata_in_fov"]),
            }
        )
    photo_ids = [row["photo_id"] for row in rows]
    if len(rows) != 8 or len(set(photo_ids)) != 8:
        raise RuntimeError(f"expected eight unique frozen photos, got {photo_ids}")
    return rows


def main() -> None:
    if MANIFEST.exists():
        raise RuntimeError(f"refusing existing manifest: {MANIFEST}")
    candidate_bytes = CANDIDATE.read_bytes()
    candidate = json.loads(candidate_bytes)
    if candidate["metadata_metrics"]["selected_query_or_negative_images_downloaded"] != 0:
        raise RuntimeError("candidate no longer states a zero-pixel boundary")
    rows = expected_rows(candidate)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    materialized: list[dict] = []
    try:
        for expected in rows:
            photo_id = expected["photo_id"]
            api_url = f"https://api.openstreetcam.org/2.0/photo/{photo_id}"
            api_bytes, api_content_type = get_bytes(api_url)
            payload = json.loads(api_bytes)
            actual = payload["result"]["data"]
            if isinstance(actual, list):
                if len(actual) != 1:
                    raise RuntimeError(f"photo {photo_id}: expected one API row")
                actual = actual[0]
            actual_sequence = int(actual.get("sequenceId") or actual.get("sequence", {}).get("id"))
            checks = {
                "photo_id": int(actual["id"]) == photo_id,
                "sequence_id": actual_sequence == expected["sequence_id"],
                "sequence_index": int(actual["sequenceIndex"]) == expected["sequence_index"],
                "public": actual.get("visibility") == "public",
                "active": actual.get("status") == "active",
                "processed": actual.get("autoImgProcessingStatus") == "FINISHED",
                "projection": actual.get("projection") == "PLANE",
            }
            if not all(checks.values()):
                raise RuntimeError(f"photo {photo_id}: provider identity gate failed: {checks}")
            image_url = actual.get("imageProcUrl") or actual.get("fileurlProc")
            if not image_url or not image_url.startswith("https://"):
                raise RuntimeError(f"photo {photo_id}: no HTTPS processed-image URL")
            image_bytes, image_content_type = get_bytes(image_url)
            width, height = jpeg_size(image_bytes)
            final = OUTPUT / f"{photo_id}.jpg"
            part = OUTPUT / f"{photo_id}.jpg.part"
            if final.exists() or part.exists():
                raise RuntimeError(f"photo {photo_id}: refusing existing output")
            try:
                part.write_bytes(image_bytes)
                os.replace(part, final)
            finally:
                part.unlink(missing_ok=True)
            created.append(final)
            materialized.append(
                {
                    **expected,
                    "api_url": api_url,
                    "api_response_sha256": sha256_bytes(api_bytes),
                    "api_content_type": api_content_type,
                    "provider_image_url": image_url,
                    "provider_width": int(actual["width"]),
                    "provider_height": int(actual["height"]),
                    "decoded_width": width,
                    "decoded_height": height,
                    "image_content_type": image_content_type,
                    "local_path": final.relative_to(REPO).as_posix(),
                    "bytes": len(image_bytes),
                    "sha256": sha256_bytes(image_bytes),
                    "shot_date": actual["shotDate"],
                    "heading_degrees": float(actual["heading"]),
                    "lat": float(actual["lat"]),
                    "lon": float(actual["lng"]),
                    "identity_checks": checks,
                }
            )

        manifest = {
            "schema": "blindassist-l10-kartaview-jakarta-materialization-v1",
            "decision": "L10_KARTAVIEW_JAKARTA_EXACT_FROZEN_PIXEL_MATERIALIZATION_SUCCEEDED",
            "authority": "SOURCE_MATERIALIZATION_ONLY_NO_HUMAN_TRUTH_NO_OCR_NO_MODEL",
            "candidate": {
                "path": CANDIDATE.relative_to(REPO).as_posix(),
                "bytes": len(candidate_bytes),
                "sha256": sha256_bytes(candidate_bytes),
            },
            "materializer": {
                "path": Path(__file__).resolve().relative_to(REPO).as_posix(),
                "bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "license": "KartaView imagery: CC BY-SA 4.0 according to https://kartaview.org/landing; preserve provider and contributor attribution.",
            "metrics": {
                "frozen_photos": len(materialized),
                "query_window_photos": sum(row["role"] == "QUERY_WINDOW" for row in materialized),
                "geometric_negative_candidates": sum(row["role"] == "GEOMETRIC_NEGATIVE_CANDIDATE" for row in materialized),
                "distinct_sequences": len({row["sequence_id"] for row in materialized}),
                "distinct_contributors": len({row["contributor_user_id"] for row in materialized}),
                "human_truth_rows": 0,
                "ocr_or_model_calls": 0,
            },
            "rows": materialized,
            "claim_boundary": {
                "allowed": "Eight exact metadata-frozen public KartaView images were materialized with provider identity and local hash receipts.",
                "not_allowed": [
                    "target visibility, negative validity, lexical correctness, appearance correctness, or any algorithm metric",
                    "provider-general, city-general, open-world, independent, conformal, deployment, or safety confirmation",
                    "facade, portal, entrance ownership, access, traversability, waypoint, arrival, handoff, user benefit, or safety",
                ],
            },
        }
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        manifest_part = MANIFEST.with_name(MANIFEST.name + ".part")
        try:
            manifest_part.write_bytes(manifest_bytes)
            os.replace(manifest_part, MANIFEST)
        finally:
            manifest_part.unlink(missing_ok=True)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        if OUTPUT.exists() and not any(OUTPUT.iterdir()):
            OUTPUT.rmdir()
        raise

    print(json.dumps(manifest["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()

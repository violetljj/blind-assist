"""Audit remaining SANPO official-test availability using object metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlencode

from scripts.build_sanpo_sequence_evalset import (
    GCS_API,
    GCS_PREFIX,
    fetch_json,
    fetch_text,
    frame_number,
    get_gcs_object,
    media_url,
)

from . import PROTOCOL_ID


CONSUMED = {
    "5LlqRK-hWoDLSW5MmoLjKj6uQtZMKjb9",
    "i2jglnBfoIqIIA7ojQGe-4vK07hUm4T3",
    "GxMb4zhAvoM5jbF54kfcs8wxTL4fqNnT",
    "972O8sd5HpUbGeEE_UAb1g0z1OZUtfHl",
    "ic_BpoiSOIW-7_mffGenT6yissRNiPzT",
    "eHxtA669WpN381O4ZjVAmG3-3ZUewuXr",
}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _official_session_ids(split: str, retries: int) -> list[str]:
    object_name = f"{GCS_PREFIX}/sanpo-real/splits/{split}_session_ids.txt"
    item = get_gcs_object(object_name, retries)
    values = [
        line.strip()
        for line in fetch_text(
            media_url(object_name, item.get("generation")),
            retries,
        ).splitlines()
        if line.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"official {split} session list is empty or contains duplicates")
    return values


def _frame_names_until_eligible(
    prefix: str,
    *,
    retries: int,
    minimum_aligned: int,
    counterpart: set[int] | None = None,
) -> tuple[set[int], bool]:
    """Read only object-list metadata, stopping once eligibility is established."""
    frames: set[int] = set()
    page_token: str | None = None
    exhausted = False
    while True:
        query = {
            "prefix": prefix,
            "maxResults": 256,
            "fields": "items(name),nextPageToken",
        }
        if page_token:
            query["pageToken"] = page_token
        payload = fetch_json(f"{GCS_API}?{urlencode(query)}", retries=retries)
        frames.update(
            frame_number(item["name"])
            for item in payload.get("items", [])
            if item.get("name", "").endswith(".png")
        )
        page_token = payload.get("nextPageToken")
        if counterpart is not None and len(frames & counterpart) >= minimum_aligned:
            break
        if not page_token:
            exhausted = True
            break
    return frames, exhausted


def _audit_session(
    *,
    official_position: int,
    session_id: str,
    retries: int,
    minimum_aligned: int,
) -> dict[str, Any]:
    base = f"{GCS_PREFIX}/sanpo-real/{session_id}/camera_chest/left"
    try:
        rgb_frames, rgb_exhausted = _frame_names_until_eligible(
            f"{base}/video_frames/",
            retries=retries,
            minimum_aligned=minimum_aligned,
        )
        mask_frames, mask_exhausted = _frame_names_until_eligible(
            f"{base}/segmentation_masks/",
            retries=retries,
            minimum_aligned=minimum_aligned,
            counterpart=rgb_frames,
        )
        aligned_count = len(rgb_frames & mask_frames)
        eligibility: bool | None
        if aligned_count >= minimum_aligned:
            eligibility = True
        elif rgb_exhausted and mask_exhausted:
            eligibility = False
        else:
            eligibility = None
        return {
            "official_position_zero_based": official_position,
            "session_id": session_id,
            "camera": "camera_chest",
            "lens": "left",
            "rgb_metadata_rows_observed": len(rgb_frames),
            "mask_metadata_rows_observed": len(mask_frames),
            "aligned_metadata_rows_observed": aligned_count,
            "counts_are_lower_bounds": not (rgb_exhausted and mask_exhausted),
            "metadata_eligible_at_least_50": eligibility,
            "selection_status": "NOT_SELECTED",
            "truth_access": "NONE_METADATA_ONLY",
        }
    except Exception as exc:
        return {
            "official_position_zero_based": official_position,
            "session_id": session_id,
            "camera": "camera_chest",
            "lens": "left",
            "metadata_eligible_at_least_50": None,
            "selection_status": "NOT_SELECTED",
            "truth_access": "NONE_METADATA_ONLY",
            "metadata_error": f"{type(exc).__name__}: {exc}",
        }


def audit(*, output_path: Path, retries: int, workers: int) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite metadata audit: {output_path}")
    official = _official_session_ids("test", retries)
    pending: list[tuple[int, str]] = []
    for official_position, session_id in enumerate(official):
        if session_id in CONSUMED:
            continue
        pending.append((official_position, session_id))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _audit_session,
                official_position=position,
                session_id=session_id,
                retries=retries,
                minimum_aligned=50,
            ): session_id
            for position, session_id in pending
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["official_position_zero_based"])
    eligible = [row for row in rows if row["metadata_eligible_at_least_50"] is True]
    errors = [row for row in rows if row["metadata_eligible_at_least_50"] is None]
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.holdout_metadata_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "METADATA_ONLY_AVAILABILITY_AUDITED",
        "formal_authority": False,
        "official_split": "test",
        "official_session_count": len(official),
        "consumed_session_count": len(CONSUMED),
        "remaining_session_count": len(rows),
        "metadata_eligible_session_count": len(eligible),
        "metadata_error_count": len(errors),
        "selection_status": "NO_HOLDOUT_SELECTED",
        "mask_objects_downloaded": 0,
        "mask_pixels_read": 0,
        "candidate_outputs_run": 0,
        "audit_method": "GCS_OBJECT_LIST_METADATA_PREFIXES_ONLY_STOP_AT_50_ALIGNED",
        "worker_count": max(1, workers),
        "official_session_order_sha256": _sha256_json(official),
        "eligible_identity_set_sha256": _sha256_json(
            sorted(row["session_id"] for row in eligible)
        ),
        "rows": rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(output_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = audit(
        output_path=args.output.resolve(),
        retries=args.retries,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "eligible": value["metadata_eligible_session_count"],
                "errors": value["metadata_error_count"],
            }
        )
    )

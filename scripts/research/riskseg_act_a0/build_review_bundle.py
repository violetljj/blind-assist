from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .common import (
    CELL_IDS,
    PASS_CONDITION,
    PASS_IDS,
    PROTOCOL_ID,
    A0Error,
    anchor_indices,
    assert_no_forbidden_public_fields,
    canonical_bytes,
    condition_offsets,
    load_json,
    sha256_file,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "blindassist.riskseg_act_a0.review_bundle.v1"
DEFAULT_MANIFEST_SHA256 = "6fe64391a9557fd9d80ad8f861cd4b043293955e2ff3e5979ffe7fb33d33b7ff"
EXPECTED_BUCKETS = {
    "blocking_obstacle_positive": 8,
    "boundary_level_change_positive": 8,
    "parallel_curb_negative": 7,
    "normal_walkable_negative": 7,
}


def _derive(master: bytes, purpose: str) -> bytes:
    return hmac.new(master, purpose.encode("utf-8"), hashlib.sha256).digest()


def _opaque_id(master: bytes, pass_id: str, event_index: int, anchor_slot: int) -> str:
    raw = _derive(master, f"item|{pass_id}|{event_index}|{anchor_slot}")
    return "a0_" + raw.hex()[:24]


def _parent_id(master: bytes, event_index: int) -> str:
    return "parent_" + _derive(master, f"parent|{event_index}").hex()[:20]


def _safe_source(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise A0Error(f"source path escapes manifest directory: {relative}") from error
    return candidate


def _validate_manifest(
    manifest_path: Path,
    expected_sha256: str,
    *,
    expected_event_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual = sha256_file(manifest_path)
    if actual != expected_sha256.lower():
        raise A0Error(f"manifest SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    manifest = load_json(manifest_path)
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != expected_event_count:
        raise A0Error(f"expected {expected_event_count} parent events")
    if manifest.get("event_count") != expected_event_count:
        raise A0Error("manifest event_count mismatch")
    if expected_event_count == 30 and manifest.get("bucket_counts") != EXPECTED_BUCKETS:
        raise A0Error("frozen bucket cardinality mismatch")
    sessions = [event.get("source_session_id") for event in events]
    if len(set(sessions)) != len(events) or any(not session for session in sessions):
        raise A0Error("source sessions must be present and parent-event disjoint")
    root = manifest_path.parent
    total_frames = 0
    for event_index, event in enumerate(events):
        frames = event.get("frames")
        if not isinstance(frames, list) or len(frames) < 30:
            raise A0Error(f"event {event_index} has fewer than 30 frames")
        if [frame.get("frame_index") for frame in frames] != list(range(len(frames))):
            raise A0Error(f"event {event_index} frame indices are not contiguous")
        anchor_indices(len(frames))
        total_frames += len(frames)
        for frame in frames:
            source = _safe_source(root, frame["image_path"])
            if not source.is_file() or sha256_file(source) != frame["image_sha256"]:
                raise A0Error(f"event {event_index} source RGB hash mismatch")
            with Image.open(source) as image:
                if image.size != (512, 288) or image.mode != "RGB":
                    raise A0Error(
                        f"event {event_index} source RGB must be mode RGB at 512x288"
                    )
        for anchor in anchor_indices(len(frames)):
            anchor_ms = frames[anchor].get("timestamp_ms")
            history_ms = frames[anchor - 10].get("timestamp_ms")
            future_ms = frames[anchor + 10].get("timestamp_ms")
            if (
                not all(
                    isinstance(value, int)
                    for value in (anchor_ms, history_ms, future_ms)
                )
                or anchor_ms - history_ms != 1000
                or future_ms - anchor_ms != 1000
            ):
                raise A0Error(
                    f"event {event_index} anchor {anchor} does not bind exact +/-1000 ms"
                )
    if expected_event_count == 30 and total_frames != 1920:
        raise A0Error(f"expected 1920 frozen frames, got {total_frames}")
    return manifest, events


def _grid_geometry(width: int, height: int) -> dict[str, Any]:
    top_y = round(height * 0.42)
    bottom_y = height - 1
    cx = (width - 1) / 2
    top_half = width * 0.16
    bottom_half = width * 0.42
    left_top = (round(cx - top_half), top_y)
    right_top = (round(cx + top_half), top_y)
    left_bottom = (round(cx - bottom_half), bottom_y)
    right_bottom = (round(cx + bottom_half), bottom_y)

    def interpolate(a: tuple[int, int], b: tuple[int, int], fraction: float) -> tuple[int, int]:
        return (
            round(a[0] + (b[0] - a[0]) * fraction),
            round(a[1] + (b[1] - a[1]) * fraction),
        )

    verticals = [
        (
            interpolate(left_top, right_top, fraction),
            interpolate(left_bottom, right_bottom, fraction),
        )
        for fraction in (1 / 3, 2 / 3)
    ]
    mid_left = interpolate(left_top, left_bottom, 0.5)
    mid_right = interpolate(right_top, right_bottom, 0.5)
    return {
        "polygon": [left_top, right_top, right_bottom, left_bottom],
        "verticals": verticals,
        "horizontal": (mid_left, mid_right),
    }


def overlay_corridor_grid(image: Image.Image) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    geometry = _grid_geometry(*rendered.size)
    color = (0, 255, 80)
    outline = geometry["polygon"] + [geometry["polygon"][0]]
    draw.line(outline, fill=color, width=max(2, rendered.width // 256))
    for start, end in geometry["verticals"]:
        draw.line((start, end), fill=color, width=max(2, rendered.width // 256))
    draw.line(geometry["horizontal"], fill=color, width=max(2, rendered.width // 256))
    # Cell labels are fixed ontology aids, never source semantics.
    polygon = geometry["polygon"]
    left_top, right_top, right_bottom, left_bottom = polygon
    for row, fraction_y in enumerate((0.25, 0.75)):
        left = (
            left_top[0] + (left_bottom[0] - left_top[0]) * fraction_y,
            left_top[1] + (left_bottom[1] - left_top[1]) * fraction_y,
        )
        right = (
            right_top[0] + (right_bottom[0] - right_top[0]) * fraction_y,
            right_top[1] + (right_bottom[1] - right_top[1]) * fraction_y,
        )
        for column in range(3):
            x = round(left[0] + (right[0] - left[0]) * ((column + 0.5) / 3))
            y = round(left[1])
            label = CELL_IDS[row * 3 + column].replace("_", "\n")
            draw.multiline_text((x, y), label, fill=color, anchor="mm", align="center")
    return rendered


def _write_contact_sheet(
    media_paths: list[Path],
    offsets: tuple[int, ...],
    target: Path,
    item_id: str,
) -> None:
    columns = min(7, len(media_paths))
    rows = (len(media_paths) + columns - 1) // columns
    thumb_width, thumb_height, label_height = 256, 144, 24
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height) + 28),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), item_id, fill="black")
    for index, (path, offset) in enumerate(zip(media_paths, offsets, strict=True)):
        with Image.open(path) as image:
            thumb = image.convert("RGB").resize((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = 28 + (index // columns) * (thumb_height + label_height)
        sheet.paste(thumb, (x, y))
        draw.text(
            (x + 5, y + thumb_height + 4),
            f"relative_offset={offset:+d}",
            fill="black",
        )
    sheet.save(target, format="PNG", optimize=False)


def _review_template(item_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "review_item_id": item_id,
            "hazard_aux": None,
            "intrusion_cells": {cell: None for cell in CELL_IDS},
            "boundary_relation": None,
            "alertable": None,
            "passed": None,
            "knownness": None,
            "non_actionable_reason": None,
            "quality_state": None,
            "rationale_code": None,
        }
        for item_id in item_ids
    ]


def _instructions(pass_id: str, condition: str) -> str:
    visible_context = {
        "CURRENT_ONLY": "Only the current RGB frame at relative_offset=0 is visible.",
        "CAUSAL_HISTORY": (
            "Only the causal RGB history from relative_offset=-10 through 0 is visible. "
            "Do not infer from any future frame."
        ),
        "HINDSIGHT_REFERENCE": (
            "The reference window from relative_offset=-10 through +10 is visible. "
            "Use future frames only to clarify the state at offset 0; this pass is "
            "diagnostic and can never become runtime or training truth."
        ),
    }[condition]
    return f"""# RISKSEG-ACT A0 isolated review pass {pass_id}

Review only the media and `review_template.jsonl` in this directory. Do not inspect
the source manifest, `_private`, another pass, truth/oracle masks, event labels,
intervals, model output, or earlier reviews.

Information condition: `{condition}`. {visible_context}
`relative_offset=0` is always the current frame to label. The fixed green 2x3
grid is an image-space annotation aid, not a physical safety envelope or safety
claim. Judge only whether visible evidence cuts the apparent forward corridor
at the current time. Object or non-walkable-region presence alone is not enough.

Complete every field using only these frozen values:

- `hazard_aux`: `BLOCKING_OBSTACLE`, `BOUNDARY_LEVEL_CHANGE`,
  `NONE_IN_SCOPE`, or `UNRESOLVED`.
- Each of the six `intrusion_cells`: `INTRUDING`, `NON_INTRUDING`, or
  `UNKNOWN`. `INTRUDING` means visible evidence in that cell occupies, crosses,
  or cuts the apparent forward corridor now. `NON_INTRUDING` means the cell is
  reviewable and the evidence is lateral, parallel, already cleared/behind, or
  does not cut the corridor.
- `boundary_relation`: `TRANSVERSE_CROSSING`, `PARALLEL_BOUNDARY`,
  `AMBIGUOUS`, or `NOT_APPLICABLE`. Use `NOT_APPLICABLE` only when there is no
  boundary-type evidence in scope.
- `alertable`: `YES` only when the visible situation warrants starting or
  continuing a warning now; `NO` when it does not; otherwise `UNKNOWN`.
- `passed`: `YES` only when the relevant evidence has already been passed or
  cleared at offset 0 even if it remains visible; `NO` when not passed;
  otherwise `UNKNOWN`.
- `knownness`: `KNOWN` only when the assigned visual context is sufficient to
  judge the required current state; otherwise `UNKNOWN`.
- `quality_state`: one of `STABLE`, `TURNING`, `BLURRED`, `DARK`, `OCCLUDED`,
  `MULTIPLE_PLAUSIBLE_ROUTES`, or `OTHER_NOT_EVALUABLE`.
- `rationale_code`: a short non-empty code describing the visible basis. Do
  not include source identity guesses.

The derived state is programmatic:

- `ABSTAIN_NOT_EVALUABLE` if knownness/alertable/passed is unknown, all six
  intrusion cells are unknown, or hazard_aux is unresolved.
- Otherwise `ACTIONABLE_NOW` requires alertable YES, passed NO, and at least
  one intruding cell.
- Otherwise it is `NON_ACTIONABLE_NOW`.

For a derived non-actionable item, set exactly one `non_actionable_reason`:
`CLEAR_FORWARD`, `LATERAL_OR_PARALLEL`,
`TOO_EARLY_OR_NOT_CURRENTLY_ALERTABLE`, `PASSED_CLEAR`, or
`NO_HAZARD_IN_SCOPE`. For actionable or abstained items, set it to null.
UNKNOWN is epistemic abstention and must never be converted to a negative.

Submit one completed JSONL row per opaque item plus a submission receipt copied
from `submission_receipt.template.json`. Use a fresh reviewer identity for this pass.
"""


def build_bundle(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    output_root: Path,
    master_seed: bytes | None = None,
    expected_event_count: int = 30,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    manifest, events = _validate_manifest(
        manifest_path,
        expected_manifest_sha256,
        expected_event_count=expected_event_count,
    )
    del manifest
    master = master_seed or secrets.token_bytes(32)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        private_rows: list[dict[str, Any]] = []
        public_passes: list[dict[str, Any]] = []
        all_item_ids: set[str] = set()
        for pass_id in PASS_IDS:
            condition = PASS_CONDITION[pass_id]
            pass_root = temporary / "passes" / pass_id
            media_root = pass_root / "media"
            media_root.mkdir(parents=True)
            unit_specs: list[dict[str, Any]] = []
            for event_index, event in enumerate(events):
                parent_id = _parent_id(master, event_index)
                hidden_stratum = event["bucket"]
                frames = event["frames"]
                for anchor_slot, anchor in enumerate(anchor_indices(len(frames))):
                    item_id = _opaque_id(master, pass_id, event_index, anchor_slot)
                    if item_id in all_item_ids:
                        raise A0Error("opaque item ID collision across passes")
                    all_item_ids.add(item_id)
                    unit_specs.append(
                        {
                            "review_item_id": item_id,
                            "event_index": event_index,
                            "parent_id": parent_id,
                            "hidden_source_stratum": hidden_stratum,
                            "anchor_slot": anchor_slot,
                            "anchor_frame_index": anchor,
                            "anchor_timestamp_ms": frames[anchor]["timestamp_ms"],
                            "history_start_timestamp_ms": frames[anchor - 10][
                                "timestamp_ms"
                            ],
                            "future_end_timestamp_ms": frames[anchor + 10][
                                "timestamp_ms"
                            ],
                            "history_span_ms": (
                                frames[anchor]["timestamp_ms"]
                                - frames[anchor - 10]["timestamp_ms"]
                            ),
                            "future_span_ms": (
                                frames[anchor + 10]["timestamp_ms"]
                                - frames[anchor]["timestamp_ms"]
                            ),
                        }
                    )
            rng = random.Random(int.from_bytes(_derive(master, f"order|{pass_id}"), "big"))
            rng.shuffle(unit_specs)
            public_items: list[dict[str, Any]] = []
            for order_index, spec in enumerate(unit_specs):
                item_id = spec["review_item_id"]
                event = events[spec["event_index"]]
                anchor = spec["anchor_frame_index"]
                item_media = media_root / item_id
                item_media.mkdir()
                media_rows: list[dict[str, Any]] = []
                materialized_paths: list[Path] = []
                offsets = condition_offsets(condition)
                for media_index, offset in enumerate(offsets):
                    source_frame = event["frames"][anchor + offset]
                    source_path = _safe_source(manifest_path.parent, source_frame["image_path"])
                    with Image.open(source_path) as image:
                        rendered = (
                            overlay_corridor_grid(image)
                            if offset == 0
                            else image.convert("RGB")
                        )
                        target = item_media / f"frame_{media_index:02d}.png"
                        rendered.save(target, format="PNG", optimize=False)
                    materialized_paths.append(target)
                    media_rows.append(
                        {
                            "relative_offset": offset,
                            "path": f"media/{item_id}/{target.name}",
                            "sha256": sha256_file(target),
                            "current_grid_overlay": offset == 0,
                        }
                    )
                contact_sheet = item_media / "contact_sheet.png"
                _write_contact_sheet(
                    materialized_paths, offsets, contact_sheet, item_id
                )
                public_item = {
                    "review_item_id": item_id,
                    "order_index": order_index,
                    "media": media_rows,
                    "current_relative_offset": 0,
                    "contact_sheet": {
                        "path": f"media/{item_id}/{contact_sheet.name}",
                        "sha256": sha256_file(contact_sheet),
                        "contains_only_authorized_pass_frames": True,
                    },
                }
                assert_no_forbidden_public_fields(public_item)
                public_items.append(public_item)
                private_rows.append(
                    {
                        **spec,
                        "pass_id": pass_id,
                        "condition": condition,
                    }
                )
            pass_manifest = {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "pass_id": pass_id,
                "information_condition": condition,
                "item_count": len(public_items),
                "items": public_items,
                "public_isolation": {
                    "opaque_item_ids": True,
                    "source_identity_absent": True,
                    "event_semantics_absent": True,
                    "truth_oracle_model_output_absent": True,
                    "other_pass_output_absent": True,
                },
            }
            assert_no_forbidden_public_fields(pass_manifest)
            write_json(pass_root / "pass_manifest.json", pass_manifest)
            pass_manifest_sha256 = sha256_file(pass_root / "pass_manifest.json")
            write_jsonl(
                pass_root / "review_template.jsonl",
                _review_template([row["review_item_id"] for row in public_items]),
            )
            receipt_template = {
                "schema_version": "blindassist.riskseg_act_a0.submission_receipt.v1",
                "protocol_id": PROTOCOL_ID,
                "pass_id": pass_id,
                "assigned_pass_manifest_sha256": pass_manifest_sha256,
                "reviewer_identity": None,
                "fresh_agent_for_this_pass": None,
                "isolation_declaration": {
                    "only_assigned_pass_media_used": None,
                    "source_event_session_bucket_interval_not_seen": None,
                    "truth_oracle_model_outputs_not_seen": None,
                    "other_pass_or_review_not_seen": None,
                },
                "review_jsonl_path": None,
                "review_jsonl_sha256": None,
            }
            write_json(pass_root / "submission_receipt.template.json", receipt_template)
            (pass_root / "INSTRUCTIONS.md").write_text(
                _instructions(pass_id, condition), encoding="utf-8", newline="\n"
            )
            public_passes.append(
                {
                    "pass_id": pass_id,
                    "information_condition": condition,
                    "path": f"passes/{pass_id}",
                    "item_count": len(public_items),
                    "manifest_sha256": pass_manifest_sha256,
                    "review_template_sha256": sha256_file(
                        pass_root / "review_template.jsonl"
                    ),
                }
            )

        private_root = temporary / "_private"
        private_root.mkdir()
        scoring_key = {
            "schema_version": "blindassist.riskseg_act_a0.scoring_key.v1",
            "protocol_id": PROTOCOL_ID,
            "source_manifest_sha256": expected_manifest_sha256.lower(),
            "parent_event_count": len(events),
            "anchor_count": len(events) * 4,
            "passes": list(PASS_IDS),
            "units": private_rows,
        }
        write_json(private_root / "scoring_key.json", scoring_key)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "READY_FOR_SIX_ISOLATED_REVIEWS",
            "source_manifest_sha256": expected_manifest_sha256.lower(),
            "generation_seed_sha256": hashlib.sha256(master).hexdigest(),
            "parent_event_count": len(events),
            "anchor_count_per_pass": len(events) * 4,
            "total_review_item_count": len(events) * 4 * len(PASS_IDS),
            "passes": public_passes,
            "scoring_key": {
                "path": "_private/scoring_key.json",
                "sha256": sha256_file(private_root / "scoring_key.json"),
                "must_not_be_shared_with_reviewers": True,
            },
            "atomic_new_directory": True,
            "bundle_executed_reviews": False,
        }
        write_json(temporary / "bundle_receipt.json", receipt)
        temporary.replace(output_root)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-manifest-sha256",
        default=DEFAULT_MANIFEST_SHA256,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seed-hex",
        help="Optional 32-byte deterministic test/lock seed; omit for cryptographic entropy.",
    )
    args = parser.parse_args()
    seed = bytes.fromhex(args.seed_hex) if args.seed_hex else None
    if seed is not None and len(seed) != 32:
        raise A0Error("--seed-hex must encode exactly 32 bytes")
    receipt = build_bundle(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        output_root=args.output_root,
        master_seed=seed,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SCHEMA = "blindassist.riskseg_r0.event_candidate_index.v1"
RECEIPT_SCHEMA = "blindassist.riskseg_r0.review_bundle_receipt.v1"
MINIMUM_FRAME_COUNT = 30
MAXIMUM_FRAME_COUNT = 120
THUMBNAIL = (240, 135)
COLUMNS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def excluded_sessions(ledger: dict[str, Any]) -> set[str]:
    roles = ledger["roles"]
    result: set[str] = set()
    for role in ("train", "dev"):
        result.update(key.split(":", 1)[1] for key in roles[role]["sessions"])
    result.update(item["source_session_id"] for item in roles["fixed_regression"]["events"])
    return result


def frame_path(root: Path, row: dict[str, Any], directory: str) -> Path:
    candidates = list((root / directory).rglob(f"{row['id']}.png"))
    if len(candidates) != 1:
        raise ValueError(
            f"{root.name}: expected exactly one {directory} asset for {row['id']}, "
            f"got {len(candidates)}"
        )
    return candidates[0]


def load_candidate(root: Path, blocked: set[str]) -> dict[str, Any]:
    manifest = root / "manifest.draft.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"draft manifest missing: {manifest}")
    rows = load_jsonl(manifest)
    if not MINIMUM_FRAME_COUNT <= len(rows) <= MAXIMUM_FRAME_COUNT:
        raise ValueError(
            f"{root.name}: frame count must be "
            f"{MINIMUM_FRAME_COUNT}..{MAXIMUM_FRAME_COUNT}, got {len(rows)}"
        )
    rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    if [int(item["frame_index"]) for item in rows] != list(range(len(rows))):
        raise ValueError(f"{root.name}: frame_index must be contiguous from zero")
    sessions = {item["source"]["session_id"] for item in rows}
    sequences = {item["sequence_id"] for item in rows}
    if len(sessions) != 1 or len(sequences) != 1:
        raise ValueError(f"{root.name}: draft mixes sessions or sequences")
    session_id = next(iter(sessions))
    if session_id in blocked:
        raise ValueError(f"{root.name}: excluded source session {session_id}")
    rgb_paths = [frame_path(root, item, "images") for item in rows]
    mask_paths = [frame_path(root, item, "source_masks") for item in rows]
    source_frames = [int(item["source_frame_index"]) for item in rows]
    if any(current <= previous for previous, current in zip(source_frames, source_frames[1:])):
        raise ValueError(f"{root.name}: source frame order is not strictly increasing")
    return {
        "root": root,
        "rows": rows,
        "rgb_paths": rgb_paths,
        "mask_paths": mask_paths,
        "session_id": session_id,
        "sequence_id": next(iter(sequences)),
        "source_frame_start": source_frames[0],
        "source_frame_end": source_frames[-1],
    }


def render_contact_sheet(candidate: dict[str, Any], destination: Path, ordinal: int) -> None:
    header = 52
    frame_count = len(candidate["rgb_paths"])
    row_count = (frame_count + COLUMNS - 1) // COLUMNS
    canvas = Image.new(
        "RGB",
        (THUMBNAIL[0] * COLUMNS, header + THUMBNAIL[1] * row_count),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"{ordinal:03d} {candidate['root'].name}", fill="black")
    draw.text(
        (5, 26),
        f"{candidate['session_id']} | {candidate['sequence_id']}",
        fill="black",
    )
    for frame_index, source in enumerate(candidate["rgb_paths"]):
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail(THUMBNAIL)
            cell = Image.new("RGB", THUMBNAIL, "black")
            cell.paste(
                image,
                (
                    (THUMBNAIL[0] - image.width) // 2,
                    (THUMBNAIL[1] - image.height) // 2,
                ),
            )
        x = (frame_index % COLUMNS) * THUMBNAIL[0]
        y = header + (frame_index // COLUMNS) * THUMBNAIL[1]
        canvas.paste(cell, (x, y))
        draw.text((x + 3, y + 3), f"f{frame_index:02d}", fill="yellow")
    canvas.save(destination, quality=86)


def prepare(ledger_path: Path, draft_roots: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    blocked = excluded_sessions(ledger)
    candidates = [load_candidate(root.resolve(), blocked) for root in draft_roots]
    by_sequence: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        sequence_id = candidate["sequence_id"]
        if sequence_id in by_sequence:
            raise ValueError(f"duplicate sequence_id in requested roots: {sequence_id}")
        by_sequence[sequence_id] = candidate
    ordered = sorted(
        candidates,
        key=lambda item: (
            item["session_id"],
            item["source_frame_start"],
            item["sequence_id"],
        ),
    )
    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    (temporary / "contact_sheets").mkdir(parents=True)
    index_items: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(ordered, 1):
        contact_name = f"{ordinal:03d}-{candidate['session_id']}.jpg"
        contact_path = temporary / "contact_sheets" / contact_name
        render_contact_sheet(candidate, contact_path, ordinal)
        rows = candidate["rows"]
        index_items.append(
            {
                "event_candidate_id": f"riskseg-event-candidate-{ordinal:03d}",
                "source_session_id": candidate["session_id"],
                "sequence_id": candidate["sequence_id"],
                "source_frame_start": candidate["source_frame_start"],
                "source_frame_end": candidate["source_frame_end"],
                "frame_count": len(candidate["rows"]),
                "draft_manifest_path": candidate["root"].as_posix(),
                "draft_manifest_sha256": sha256_file(
                    candidate["root"] / "manifest.draft.jsonl"
                ),
                "rgb_sha256s": [
                    sha256_file(path) for path in candidate["rgb_paths"]
                ],
                "source_mask_sha256s": [
                    sha256_file(path) for path in candidate["mask_paths"]
                ],
                "contact_sheet": f"contact_sheets/{contact_name}",
                "contact_sheet_sha256": sha256_file(contact_path),
                "candidate_output_visible": False,
                "selection_is_truth": False,
                "official_split": rows[0]["source"].get("official_split"),
                "camera": rows[0]["source"].get("camera"),
                "lens": rows[0]["source"].get("lens"),
            }
        )
    index = {
        "schema_version": SCHEMA,
        "protocol_id": "RISKSEG_R0_EVENT_EVAL_V1",
        "status": "OUTPUT_BLIND_RGB_REVIEW_BUNDLE",
        "candidate_output_visible": False,
        "excluded_source_sessions": sorted(blocked),
        "items": index_items,
    }
    index_path = temporary / "candidate_index.json"
    index_path.write_text(canonical_json(index), encoding="utf-8")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": "RISKSEG_R0_EVENT_EVAL_V1",
        "status": "REVIEW_BUNDLE_READY",
        "candidate_count": len(index_items),
        "source_session_count": len(
            {item["source_session_id"] for item in index_items}
        ),
        "contract_ledger_sha256": sha256_file(ledger_path),
        "candidate_index_sha256": sha256_file(index_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "candidate_output_visible": False,
        "next_action": "two isolated RGB-only reviews before cohort adjudication",
    }
    (temporary / "review_bundle_receipt.json").write_text(
        canonical_json(receipt),
        encoding="utf-8",
    )
    temporary.replace(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-ledger", type=Path, required=True)
    parser.add_argument("--draft-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.contract_ledger, args.draft_root, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

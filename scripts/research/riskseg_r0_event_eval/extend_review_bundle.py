from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .prepare_review_bundle import (
    RECEIPT_SCHEMA,
    SCHEMA,
    canonical_json,
    excluded_sessions,
    load_candidate,
    render_contact_sheet,
    sha256_file,
)


ID_PATTERN = re.compile(r"^riskseg-event-candidate-(\d+)$")


def extend(
    ledger_path: Path,
    base_index_path: Path,
    draft_roots: list[Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    blocked = excluded_sessions(ledger)
    base = json.loads(base_index_path.read_text(encoding="utf-8"))
    if (
        base.get("schema_version") != SCHEMA
        or base.get("candidate_output_visible") is not False
        or not isinstance(base.get("items"), list)
    ):
        raise ValueError("base candidate index contract mismatch")
    base_root = base_index_path.parent
    base_items: list[dict[str, Any]] = base["items"]
    ordinals: list[int] = []
    identities: set[tuple[str, int, int]] = set()
    for item in base_items:
        match = ID_PATTERN.fullmatch(str(item.get("event_candidate_id")))
        if not match:
            raise ValueError("base candidate id format mismatch")
        ordinals.append(int(match.group(1)))
        identity = (
            str(item["source_session_id"]),
            int(item["source_frame_start"]),
            int(item["source_frame_end"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate base candidate identity: {identity}")
        identities.add(identity)
        sheet = base_root / item["contact_sheet"]
        if sha256_file(sheet) != item["contact_sheet_sha256"]:
            raise ValueError(f"base contact sheet hash mismatch: {sheet}")
    if sorted(ordinals) != list(range(1, len(base_items) + 1)):
        raise ValueError("base candidate ids must be contiguous")

    new_candidates = [
        load_candidate(root.resolve(), blocked)
        for root in draft_roots
    ]
    new_candidates.sort(
        key=lambda item: (
            item["session_id"],
            item["source_frame_start"],
            item["sequence_id"],
        )
    )
    for candidate in new_candidates:
        identity = (
            candidate["session_id"],
            candidate["source_frame_start"],
            candidate["source_frame_end"],
        )
        if identity in identities:
            raise ValueError(f"duplicate extension candidate identity: {identity}")
        identities.add(identity)

    temporary = output.with_name(f"{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    (temporary / "contact_sheets").mkdir(parents=True)
    extended_items = [dict(item) for item in base_items]
    for item in extended_items:
        source = base_root / item["contact_sheet"]
        destination = temporary / item["contact_sheet"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for offset, candidate in enumerate(new_candidates, start=1):
        ordinal = len(base_items) + offset
        contact_name = f"{ordinal:03d}-{candidate['session_id']}.jpg"
        contact_path = temporary / "contact_sheets" / contact_name
        render_contact_sheet(candidate, contact_path, ordinal)
        rows = candidate["rows"]
        extended_items.append({
            "event_candidate_id": f"riskseg-event-candidate-{ordinal:03d}",
            "source_session_id": candidate["session_id"],
            "sequence_id": candidate["sequence_id"],
            "source_frame_start": candidate["source_frame_start"],
            "source_frame_end": candidate["source_frame_end"],
            "frame_count": len(rows),
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
        })
    index = {
        **{key: value for key, value in base.items() if key != "items"},
        "status": "OUTPUT_BLIND_RGB_REVIEW_BUNDLE_EXTENDED",
        "items": extended_items,
    }
    index_path = temporary / "candidate_index.json"
    index_path.write_text(canonical_json(index), encoding="utf-8")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": "RISKSEG_R0_EVENT_EVAL_V1",
        "status": "REVIEW_BUNDLE_EXTENDED_READY",
        "candidate_count": len(extended_items),
        "added_candidate_count": len(new_candidates),
        "source_session_count": len(
            {item["source_session_id"] for item in extended_items}
        ),
        "contract_ledger_sha256": sha256_file(ledger_path),
        "base_candidate_index_sha256": sha256_file(base_index_path),
        "candidate_index_sha256": sha256_file(index_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "candidate_output_visible": False,
        "next_action": "isolated review extensions before adjudication",
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
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--draft-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        extend(
            args.contract_ledger,
            args.base_index,
            args.draft_root,
            args.output,
        ),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

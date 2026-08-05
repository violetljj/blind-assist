#!/usr/bin/env python3
"""Freeze P3 R0.2 parent roles and label-blind clip identities."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_p3_r0_2_data_role_and_sealing_protocol"
LOCK_SCHEMA = "blindassist_p3_r0_2_role_identity_lock"
MANIFEST_SCHEMA = "blindassist_p3_r0_2_identity_manifest"
RECEIPT_SCHEMA = "blindassist_p3_r0_2_role_freeze_receipt"
SHA_LENGTH = 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def bound_file(repo_root: Path, binding: dict[str, Any]) -> Path:
    require(set(binding) == {"path", "sha256"}, "binding field drift")
    path = (repo_root / binding["path"]).resolve()
    require(path.is_file(), f"bound file missing: {path}")
    require(sha256_file(path) == str(binding["sha256"]).upper(), f"bound SHA mismatch: {path}")
    return path


def timestamp_ns(frame_id: str) -> int:
    return int(round(float(frame_id.rsplit("_", 1)[-1]) * 1_000_000_000))


def select_nonoverlap_clips(rows: list[dict[str, Any]], maximum: int | None) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: int(row["timestamp_ns"]))
    output: list[list[dict[str, Any]]] = []
    cursor = 0
    while cursor + 4 <= len(ordered) and (maximum is None or len(output) < maximum):
        window = ordered[cursor : cursor + 4]
        times = [int(row["timestamp_ns"]) for row in window]
        if all(0 < right - left <= 500_000_000 for left, right in zip(times, times[1:])):
            output.append(window)
            cursor += 4
        else:
            cursor += 1
    return output


def parse_index(sequence: Path, name: str, members_required: bool) -> list[tuple[float, str, Path | None]]:
    rows = []
    for number, raw in enumerate((sequence / name).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"invalid {name}:{number}")
        candidate = (sequence / parts[1]).resolve()
        require(candidate.is_relative_to(sequence.resolve()), "indexed member escaped sequence")
        if not candidate.is_file():
            require(not members_required, f"indexed member missing: {candidate}")
            candidate = None
        rows.append((float(parts[0]), parts[1], candidate))
    return rows


def paired_rgb_rows(sequence: Path) -> list[dict[str, Any]]:
    rgb = parse_index(sequence, "rgb.txt", True)
    depth = parse_index(sequence, "depth.txt", False)
    depth_times = [timestamp for timestamp, _relative, path in depth if path is not None]
    used: set[int] = set()
    output = []
    for timestamp, relative, path in rgb:
        insertion = bisect.bisect_left(depth_times, timestamp)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(depth_times) and index not in used]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(depth_times[index] - timestamp))
        if abs(depth_times[nearest] - timestamp) > 0.05:
            continue
        used.add(nearest)
        assert path is not None
        output.append({
            "frame_id": f"{sequence.name}:{timestamp:.5f}",
            "video_id": sequence.name,
            "parent_id": sequence.name,
            "timestamp_ns": int(round(timestamp * 1_000_000_000)),
            "rgb_identity": relative.replace("\\", "/"),
            "rgb_sha256": sha256_file(path),
        })
    return output


def arkit_rows(manifest: dict[str, Any], allowed: dict[str, set[str]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output = {"train": {}, "validation": {}}
    for video in manifest["videos"]:
        role = str(video["role"])
        if role not in output:
            continue
        parent = str(video["visit_id"])
        require(parent in allowed[role], f"unexpected {role} parent: {parent}")
        assets = {Path(row["member"]).stem: row for row in video["extracted"]["lowres_wide"]}
        rows = []
        for stem in video["selected_frame_stems"]:
            require(stem in assets, f"selected RGB missing: {stem}")
            asset = assets[stem]
            path = Path(asset["path"])
            require(path.is_file() and sha256_file(path) == asset["sha256"], f"ARKit RGB binding mismatch: {stem}")
            rows.append({
                "frame_id": stem, "video_id": str(video["video_id"]), "parent_id": parent,
                "timestamp_ns": timestamp_ns(stem), "rgb_identity": asset["member"], "rgb_sha256": asset["sha256"],
            })
        require(parent not in output[role], f"duplicate ARKit parent: {parent}")
        output[role][parent] = rows
    for role in output:
        require(set(output[role]) == allowed[role], f"{role} parent roster mismatch")
    return output


def make_manifest(role: str, clips: list[dict[str, Any]], protocol_sha: str) -> dict[str, Any]:
    value: dict[str, Any] = {"schema": MANIFEST_SCHEMA, "protocol_sha256": protocol_sha, "role": role, "clips": clips}
    if role == "public_holdout":
        value["outcomes_opened"] = False
    return value


def freeze(repo_root: Path, protocol_path: Path, output_dir: Path, source_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(source_path), "producer SHA drift")
    protocol_sha = sha256_file(protocol_path)
    sources = protocol["sources"]
    arkit = load_json(bound_file(repo_root, sources["arkitscenes_manifest"]))
    bonn_receipt = load_json(bound_file(repo_root, sources["bonn_identity_receipt"]))
    bound_file(repo_root, sources["public_source_admission_result"])
    bound_file(repo_root, sources["legacy_p1_exclusion_ledger"])
    bound_file(repo_root, sources["r0_1_closure"])
    bound_file(repo_root, sources["bonn_ancestry_exclusions"])
    archive_path = bound_file(repo_root, sources["bonn_full_archive"])
    require(bonn_receipt["archive"]["sha256"] == sha256_file(archive_path), "Bonn archive/receipt drift")

    roles = protocol["parent_roles"]
    allowed = {role: set(roles[role]) for role in ("train", "validation")}
    holdout_parents = set(roles["public_holdout"])
    require(allowed["train"].isdisjoint(allowed["validation"]), "train/validation overlap")
    require((allowed["train"] | allowed["validation"]).isdisjoint(holdout_parents), "development/holdout overlap")
    require(holdout_parents.isdisjoint(set(protocol["permanent_exclusions"]["bonn_parent_ids"])), "Bonn ancestry overlap")
    require((allowed["train"] | allowed["validation"] | holdout_parents).isdisjoint(set(protocol["permanent_exclusions"]["legacy_p1_parent_ids"])), "legacy P1 overlap")
    require((allowed["train"] | allowed["validation"] | holdout_parents).isdisjoint(set(protocol["permanent_exclusions"]["r0_1_attempted_holdout_parent_ids"])), "R0.1 attempted holdout overlap")

    arkit_by_role = arkit_rows(arkit, allowed)
    manifests: dict[str, dict[str, Any]] = {}
    all_clip_ids: set[str] = set()
    all_frame_ids: set[str] = set()
    for role in ("train", "validation"):
        clips = []
        for parent in roles[role]:
            for index, window in enumerate(select_nonoverlap_clips(arkit_by_role[role][parent], None)):
                clip_id = f"P3R02:{role}:{parent}:{index:04d}"
                clips.append({"clip_id": clip_id, "video_id": window[0]["video_id"], "parent_id": parent, "frames": window})
        manifests[role] = make_manifest(role, clips, protocol_sha)

    receipt_parents = {row["parent_id"]: row for row in bonn_receipt["parents"]}
    require(holdout_parents <= set(receipt_parents), "Bonn holdout parent absent from receipt")
    bonn_root = (repo_root / sources["bonn_dataset_root"]).resolve()
    holdout_clips = []
    for parent in roles["public_holdout"]:
        detail = receipt_parents[parent]
        require(detail["ancestry_excluded"] is False, f"Bonn parent excluded: {parent}")
        windows = select_nonoverlap_clips(paired_rgb_rows(bonn_root / parent), protocol["clip_selection"]["holdout_clips_per_parent"])
        require(len(windows) == protocol["clip_selection"]["holdout_clips_per_parent"], f"holdout clip capacity insufficient: {parent}")
        for index, window in enumerate(windows):
            clip_id = f"P3R02:public_holdout:{parent}:{index:04d}"
            public_frames = []
            for frame in window:
                public_frames.append(frame | {"sealed_target_id": f"P3R02:{frame['frame_id']}"})
            holdout_clips.append({"clip_id": clip_id, "video_id": parent, "parent_id": parent, "frames": public_frames})
    manifests["public_holdout"] = make_manifest("public_holdout", holdout_clips, protocol_sha)

    for manifest in manifests.values():
        for clip in manifest["clips"]:
            require(clip["clip_id"] not in all_clip_ids, "duplicate clip ID")
            all_clip_ids.add(clip["clip_id"])
            for frame in clip["frames"]:
                key = f"{frame['video_id']}:{frame['frame_id']}"
                require(key not in all_frame_ids, "frame reused across clips")
                all_frame_ids.add(key)

    output_values = {
        "train-identity-manifest.json": manifests["train"],
        "validation-identity-manifest.json": manifests["validation"],
        "public-holdout-identity-manifest.json": manifests["public_holdout"],
    }
    output_bytes = {name: canonical_bytes(value) for name, value in output_values.items()}
    lock = {
        "schema": LOCK_SCHEMA, "protocol_sha256": protocol_sha, "clip_length": 4,
        "parents_by_role": {role: roles[role] for role in ("train", "validation", "public_holdout")},
        "clip_counts_by_role": {role: len(manifests[role]["clips"]) for role in manifests},
        "manifest_sha256_by_role": {
            "train": hashlib.sha256(output_bytes["train-identity-manifest.json"]).hexdigest().upper(),
            "validation": hashlib.sha256(output_bytes["validation-identity-manifest.json"]).hexdigest().upper(),
            "public_holdout": hashlib.sha256(output_bytes["public-holdout-identity-manifest.json"]).hexdigest().upper(),
        },
        "zero_parent_overlap_proven": True, "frame_reuse": False, "holdout_outcomes_opened": False,
    }
    output_bytes["role-identity-lock.json"] = canonical_bytes(lock)
    receipt = {
        "schema": RECEIPT_SCHEMA, "protocol_sha256": protocol_sha, "producer_sha256": sha256_file(source_path),
        "source_sha256": {key: value["sha256"] for key, value in sources.items() if isinstance(value, dict) and set(value) == {"path", "sha256"}},
        "outputs": {name: hashlib.sha256(data).hexdigest().upper() for name, data in output_bytes.items()},
        "parent_counts": {role: len(roles[role]) for role in ("train", "validation", "public_holdout")},
        "clip_counts": lock["clip_counts_by_role"], "label_fields_read": False, "model_outputs_read": False,
        "holdout_outcomes_opened": False, "sealed_targets_created": False, "model_loaded": False,
        "optimizer_constructed": False, "training_started": False,
        "terminal": "P3_R0_2_DATA_ROLES_FROZEN_HOLDOUT_IDENTITIES_LOCKED_TARGETS_UNOPENED",
    }
    output_bytes["role-freeze-receipt.json"] = canonical_bytes(receipt)
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for name, data in output_bytes.items():
        (output_dir / name).write_bytes(data)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze(args.repo_root, args.protocol.resolve(), args.output_dir.resolve(), Path(__file__).resolve())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

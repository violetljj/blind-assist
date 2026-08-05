#!/usr/bin/env python3
"""Build the label-blind ARKit role manifest for P3 R0.2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_p3_r0_2_1_arkit_role_manifest_protocol"
OUTPUT_SCHEMA = "blindassist_p3_r0_2_1_arkit_role_manifest"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_bound(repo_root: Path, binding: dict[str, str]) -> dict[str, Any]:
    require(set(binding) == {"path", "sha256"}, "binding field drift")
    path = (repo_root / binding["path"]).resolve()
    require(path.is_file(), f"bound source missing: {path}")
    require(sha256_file(path) == binding["sha256"].upper(), f"bound source SHA mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "source manifest must be an object")
    return value


def build(protocol: dict[str, Any], primary: dict[str, Any], extension: dict[str, Any], protocol_sha256: str) -> dict[str, Any]:
    roles = protocol["parent_roles"]
    requested = {role: set(roles[role]) for role in ("train", "validation")}
    require(requested["train"].isdisjoint(requested["validation"]), "train/validation overlap")
    by_parent: dict[str, dict[str, Any]] = {}
    for manifest in (primary, extension):
        require(isinstance(manifest.get("videos"), list), "videos list missing")
        for video in manifest["videos"]:
            parent = str(video["visit_id"])
            require(parent not in by_parent, f"duplicate parent across manifests: {parent}")
            by_parent[parent] = video

    output_videos = []
    for role in ("train", "validation"):
        for parent in roles[role]:
            require(parent in by_parent, f"requested parent missing: {parent}")
            video = dict(by_parent[parent])
            video["role"] = role
            output_videos.append(video)
    require({str(v["visit_id"]) for v in output_videos} == requested["train"] | requested["validation"], "output roster drift")
    return {
        "schema": OUTPUT_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "parent_counts": {role: len(roles[role]) for role in ("train", "validation")},
        "labels_opened": False,
        "model_outputs_read": False,
        "videos": output_videos,
        "terminal": "P3_R0_2_1_ARKIT_ROLE_MANIFEST_MATERIALIZED_LABEL_BLIND",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    protocol_sha256 = sha256_file(protocol_path)
    producer = (repo_root / protocol["producer"]["path"]).resolve()
    require(sha256_file(producer) == protocol["producer"]["sha256"], "producer SHA drift")
    primary = load_bound(repo_root, protocol["sources"]["primary_manifest"])
    extension = load_bound(repo_root, protocol["sources"]["validation_extension_manifest"])
    require(not args.output.exists(), f"output already exists: {args.output}")
    value = build(protocol, primary, extension, protocol_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "terminal": value["terminal"]}, indent=2))


if __name__ == "__main__":
    main()

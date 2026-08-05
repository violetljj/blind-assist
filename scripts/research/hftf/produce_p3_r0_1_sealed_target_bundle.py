#!/usr/bin/env python3
"""Produce the deterministic authenticated P3 sealed target bundle."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from p3_r0_1_asset_common import (
    commit_outputs,
    exact_fields,
    load_json,
    output_receipt,
    request_sha256,
    require,
    validate_protocol,
    verify_bound_file,
    verify_producer_sha,
)


REQUEST_SCHEMA = "blindassist_p3_r0_1_sealed_target_bundle_request"
PRIVATE_SCHEMA = "blindassist_p3_r0_1_private_holdout_targets"
RECEIPT_SCHEMA = "blindassist_p3_r0_1_sealed_target_bundle_materialization_receipt"
MAGIC = b"P3R01HMACS1"
PRIVATE_FRAME_FIELDS = {
    "frame_id", "video_id", "parent_id", "timestamp_ns", "sealed_target_id",
    "rgb_identity", "rgb_sha256", "clearance_m", "geometry_state",
    "geometry_target_valid", "teacher_valid", "tof_valid", "truth_depth_ref",
    "truth_depth_sha256",
}


def _validate_private(private: dict[str, Any], public: dict[str, Any]) -> None:
    exact_fields(private, {"schema", "clips"}, "private targets")
    require(private["schema"] == PRIVATE_SCHEMA, "private target schema drift")
    public_ids = {
        frame["sealed_target_id"]
        for clip in public["clips"]
        for frame in clip["frames"]
    }
    private_ids = set()
    for clip in private["clips"]:
        exact_fields(clip, {"clip_id", "video_id", "parent_id", "frames"}, "private clip")
        require(len(clip["frames"]) == 4, "private clip length drift")
        for frame in clip["frames"]:
            exact_fields(frame, PRIVATE_FRAME_FIELDS, "private target frame")
            require(frame["sealed_target_id"] not in private_ids, "duplicate private sealed target")
            private_ids.add(frame["sealed_target_id"])
    require(private_ids == public_ids, "private/public sealed identity mismatch")


def _seal(key: bytes, nonce: bytes, plaintext: bytes, associated: bytes) -> bytes:
    """Deterministic encrypt-then-MAC envelope using only stdlib primitives."""
    stream = bytearray()
    counter = 0
    while len(stream) < len(plaintext):
        stream.extend(hmac.new(key, b"ENC" + nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    ciphertext = bytes(left ^ right for left, right in zip(plaintext, stream))
    tag = hmac.new(key, b"MAC" + associated + nonce + ciphertext, hashlib.sha256).digest()
    return MAGIC + nonce + ciphertext + tag


def build(repo_root: Path, request: dict[str, Any], source_path: Path) -> None:
    exact_fields(request, {"schema", "protocol", "public_holdout_manifest", "private_targets", "encryption_key", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    producer_sha = verify_producer_sha(request["producer_sha256"], source_path)
    _, protocol_sha = validate_protocol(repo_root, request["protocol"])
    public_path = verify_bound_file(repo_root, request["public_holdout_manifest"], "public holdout manifest")
    private_path = verify_bound_file(repo_root, request["private_targets"], "private targets")
    key_path = verify_bound_file(repo_root, request["encryption_key"], "sealing key")
    public = load_json(public_path)
    require(public.get("role") == "public_holdout" and public.get("outcomes_opened") is False, "public holdout boundary drift")
    private = load_json(private_path)
    _validate_private(private, public)
    key = key_path.read_bytes()
    require(len(key) == 32, "AES-256-GCM key must contain exactly 32 bytes")
    plaintext = private_path.read_bytes()
    request_sha = request_sha256(request)
    nonce = hashlib.sha256((request_sha + request["private_targets"]["sha256"]).encode("ascii")).digest()[:12]
    associated = (protocol_sha + request["public_holdout_manifest"]["sha256"].upper()).encode("ascii")
    bundle = _seal(key, nonce, plaintext, associated)
    exact_fields(request["outputs"], {"bundle", "receipt"}, "outputs")
    outputs = {"sealed_target_bundle": (str(request["outputs"]["bundle"]), bundle)}
    receipt = output_receipt(
        schema=RECEIPT_SCHEMA,
        producer_sha256=producer_sha,
        request_sha256=request_sha,
        input_sha256={
            "protocol": protocol_sha,
            "public_holdout_manifest": request["public_holdout_manifest"]["sha256"].upper(),
            "private_targets": request["private_targets"]["sha256"].upper(),
            "encryption_key": request["encryption_key"]["sha256"].upper(),
        },
        outputs=outputs,
    )
    commit_outputs(repo_root, outputs=outputs, receipt_relative=str(request["outputs"]["receipt"]), receipt=receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a deterministic authenticated encrypted P3 R0.2.1 target bundle."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from p3_r0_2_1_sealing_common import canonical_bytes, exact_fields, exclusive_write, load_json, materialization_receipt, pretty_bytes, require, sha256_file, verify_bound_file


REQUEST_SCHEMA = "blindassist_p3_r0_2_1_sealed_target_bundle_request"
PRIVATE_SCHEMA = "blindassist_p3_r0_2_1_private_holdout_targets"
RECEIPT_SCHEMA = "blindassist_p3_r0_2_1_sealed_target_bundle_materialization_receipt"
MAGIC = b"P3R021HMACS1"


def seal(key: bytes, nonce: bytes, plaintext: bytes, associated: bytes) -> bytes:
    stream = bytearray()
    for counter in range((len(plaintext) + 31) // 32):
        stream.extend(hmac.new(key, b"ENC" + nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
    ciphertext = bytes(left ^ right for left, right in zip(plaintext, stream))
    tag = hmac.new(key, b"MAC" + associated + nonce + ciphertext, hashlib.sha256).digest()
    return MAGIC + nonce + ciphertext + tag


def build(repo_root: Path, request: dict[str, Any], source_path: Path) -> None:
    exact_fields(request, {"schema", "protocol", "public_holdout_manifest", "private_targets", "sealing_key", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    require(sha256_file(source_path) == request["producer_sha256"], "producer SHA mismatch")
    protocol = verify_bound_file(repo_root, request["protocol"], "protocol")
    public_path = verify_bound_file(repo_root, request["public_holdout_manifest"], "public manifest")
    private_path = verify_bound_file(repo_root, request["private_targets"], "private targets")
    key_path = verify_bound_file(repo_root, request["sealing_key"], "sealing key")
    public, private = load_json(public_path), load_json(private_path)
    require(public.get("outcomes_opened") is False and private.get("schema") == PRIVATE_SCHEMA, "sealing boundary drift")
    public_ids = {f["sealed_target_id"] for c in public["clips"] for f in c["frames"]}
    private_ids = {f["sealed_target_id"] for c in private["clips"] for f in c["frames"]}
    require(public_ids == private_ids, "private/public identity mismatch")
    key = key_path.read_bytes()
    require(len(key) == 32, "sealing key must contain exactly 32 bytes")
    request_sha = hashlib.sha256(canonical_bytes(request)).hexdigest().upper()
    nonce = hashlib.sha256((request_sha + request["private_targets"]["sha256"]).encode("ascii")).digest()[:12]
    associated = (sha256_file(protocol) + sha256_file(public_path)).encode("ascii")
    bundle = seal(key, nonce, private_path.read_bytes(), associated)
    exact_fields(request["outputs"], {"bundle", "receipt"}, "outputs")
    outputs = {"sealed_target_bundle": (request["outputs"]["bundle"], bundle)}
    receipt = materialization_receipt(RECEIPT_SCHEMA, request["producer_sha256"], {"protocol": sha256_file(protocol), "public_holdout_manifest": sha256_file(public_path), "private_targets": sha256_file(private_path), "sealing_key": sha256_file(key_path)}, outputs)
    exclusive_write(repo_root / request["outputs"]["bundle"], bundle)
    exclusive_write(repo_root / request["outputs"]["receipt"], pretty_bytes(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()

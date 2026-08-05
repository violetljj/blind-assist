#!/usr/bin/env python3
"""Generate one local 256-bit P3 R0.2.1 sealing key and its receipt."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from p3_r0_2_1_sealing_common import exact_fields, exclusive_write, materialization_receipt, pretty_bytes, require, sha256_file


REQUEST_SCHEMA = "blindassist_p3_r0_2_1_sealing_key_request"
RECEIPT_SCHEMA = "blindassist_p3_r0_2_1_sealing_key_materialization_receipt"


def build(repo_root: Path, request: dict, source_path: Path, random_bytes=secrets.token_bytes) -> None:
    exact_fields(request, {"schema", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    require(sha256_file(source_path) == request["producer_sha256"], "producer SHA mismatch")
    exact_fields(request["outputs"], {"key", "receipt"}, "outputs")
    key = random_bytes(32)
    require(len(key) == 32, "random source must return 32 bytes")
    outputs = {"sealing_key": (request["outputs"]["key"], key)}
    receipt = materialization_receipt(RECEIPT_SCHEMA, request["producer_sha256"], {}, outputs)
    receipt["key_bytes_disclosed"] = False
    exclusive_write(repo_root / request["outputs"]["key"], key)
    exclusive_write(repo_root / request["outputs"]["receipt"], pretty_bytes(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()

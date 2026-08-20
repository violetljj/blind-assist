"""AES-256-GCM sealing for the fresh cohort; keys are never written to disk."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AAD = b"GOAL-COPILOT-1-SKY-PILOT/FRESH/V1"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def seal(payload: dict[str, Any], key: bytes, nonce: bytes) -> dict[str, Any]:
    plaintext = canonical(payload)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, AAD)
    families: dict[str, int] = {}
    for scenario in payload["scenarios"]:
        family = scenario["task_family"]
        families[family] = families.get(family, 0) + 1
    return {
        "schema_version": 1,
        "algorithm": "AES-256-GCM",
        "aad": AAD.decode(),
        "nonce_b64": base64.b64encode(nonce).decode(),
        "ciphertext_b64": base64.b64encode(ciphertext).decode(),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        "scenario_count": len(payload["scenarios"]),
        "family_counts": dict(sorted(families.items())),
    }


def unseal(envelope: dict[str, Any], key_hex: str) -> dict[str, Any]:
    if envelope.get("algorithm") != "AES-256-GCM" or envelope.get("aad") != AAD.decode():
        raise ValueError("fresh envelope algorithm or AAD mismatch")
    key = bytes.fromhex(key_hex)
    if len(key) != 32:
        raise ValueError("fresh key must be 32 bytes")
    plaintext = AESGCM(key).decrypt(
        base64.b64decode(envelope["nonce_b64"]),
        base64.b64decode(envelope["ciphertext_b64"]),
        AAD,
    )
    if hashlib.sha256(plaintext).hexdigest() != envelope["plaintext_sha256"]:
        raise ValueError("fresh plaintext digest mismatch")
    payload = json.loads(plaintext)
    if len(payload["scenarios"]) != envelope["scenario_count"]:
        raise ValueError("fresh scenario count mismatch")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seal", type=Path)
    parser.add_argument("envelope", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.seal.read_text())
    key = os.urandom(32)
    envelope = seal(payload, key, os.urandom(12))
    args.envelope.write_bytes(canonical(envelope))
    args.manifest.write_bytes(canonical({
        "protocol_id": "GOAL-COPILOT-1-SKY-PILOT",
        "status": "SEALED_ENCRYPTED_NOT_EXPOSED_TO_SEARCH",
        "algorithm": envelope["algorithm"],
        "plaintext_sha256": envelope["plaintext_sha256"],
        "ciphertext_sha256": hashlib.sha256(args.envelope.read_bytes()).hexdigest(),
        "scenario_count": envelope["scenario_count"],
        "family_counts": envelope["family_counts"],
    }))
    print(key.hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

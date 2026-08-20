"""Materialize BA-only encrypted held-out corruption schedule permutations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AAD = b"GOAL-COPILOT-2B/HELDOUT-CORRUPTION-SCHEDULES/V1"
SEEDS = (8627, 11939)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def material() -> dict[str, Any]:
    schedules = []
    for seed in SEEDS:
        order = list(range(12))
        random.Random(seed).shuffle(order)
        schedules.append({"seed": seed, "scenario_order": order})
    return {
        "schema_version": 1,
        "protocol_id": "GOAL-COPILOT-2B",
        "evidence_role": "HELD_OUT_CORRUPTION_SCHEDULES_OVER_CONSUMED_TASK_SEMANTICS_NOT_FRESH_TASKS",
        "conditions": ["CLEAN", "COMBINED_MILD", "COMBINED_MODERATE"],
        "schedules": schedules,
    }


def seal(payload: dict[str, Any], key: bytes, nonce: bytes) -> dict[str, Any]:
    plaintext = canonical(payload)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, AAD)
    return {
        "schema_version": 1,
        "algorithm": "AES-256-GCM",
        "aad": AAD.decode(),
        "nonce_b64": base64.b64encode(nonce).decode(),
        "ciphertext_b64": base64.b64encode(ciphertext).decode(),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        "schedule_count": len(payload["schedules"]),
        "schedule_seeds": [item["seed"] for item in payload["schedules"]],
    }


def unseal(envelope: dict[str, Any], key_hex: str) -> dict[str, Any]:
    if envelope.get("algorithm") != "AES-256-GCM" or envelope.get("aad") != AAD.decode():
        raise ValueError("held-out envelope identity mismatch")
    key = bytes.fromhex(key_hex)
    plaintext = AESGCM(key).decrypt(
        base64.b64decode(envelope["nonce_b64"]),
        base64.b64decode(envelope["ciphertext_b64"]),
        AAD,
    )
    if hashlib.sha256(plaintext).hexdigest() != envelope["plaintext_sha256"]:
        raise ValueError("held-out plaintext digest mismatch")
    return json.loads(plaintext)


def create_once(output_root: Path) -> tuple[Path, Path, Path]:
    if output_root.exists():
        raise FileExistsError(f"held-out root already exists: {output_root}")
    output_root.mkdir(parents=True)
    key = os.urandom(32)
    envelope = seal(material(), key, os.urandom(12))
    envelope_path = output_root / "heldout_schedules.enc.json"
    key_path = output_root / "heldout_key.hex"
    manifest_path = output_root / "heldout_manifest.json"
    envelope_path.write_bytes(canonical(envelope))
    key_path.write_text(key.hex() + "\n", encoding="ascii")
    manifest_path.write_bytes(
        canonical(
            {
                "schema_version": 1,
                "protocol_id": "GOAL-COPILOT-2B",
                "status": "SEALED_BA_ONLY_NOT_EXPOSED_TO_SKY",
                "algorithm": envelope["algorithm"],
                "envelope_sha256": hashlib.sha256(envelope_path.read_bytes()).hexdigest(),
                "plaintext_sha256": envelope["plaintext_sha256"],
                "schedule_count": envelope["schedule_count"],
                "schedule_seeds": envelope["schedule_seeds"],
                "key_exported_to_sky": False,
                "not_fresh_task_evidence": True,
            }
        )
    )
    if unseal(envelope, key.hex()) != material():
        raise RuntimeError("held-out encryption roundtrip failed")
    return envelope_path, manifest_path, key_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in create_once(args.output_root.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

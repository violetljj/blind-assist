#!/usr/bin/env python3
"""Load the exact hash-bound R3 fit-only uncertainty model for O1R."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.taro_o0r_factor_headroom_runtime.uncertainty_loader import load_factory_bound_uncertainty_model
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3"
ARTIFACT_PATH = SOURCE_ROOT / "uncertainty-model-artifact.json.gz"
ARTIFACT_BYTES = 58084
ARTIFACT_SHA256 = "833CA7074E178D3D2FE6FEB66A386985C0CAEB8AA6878089E7C1B08984FD5E59"
RECEIPT_PATH = SOURCE_ROOT / "uncertainty-model-receipt.json"
RECEIPT_BYTES = 14820
RECEIPT_SHA256 = "8E1C97C0961DEF6C4E7B3FCF2EADA6ECA022F0DFF2140221D221FB4C8B6A8CAE"
MODEL_SHA256 = "3FB93AC1A7FF4F21456766E9D0CC5D4EA37AC15D2F8DF0DDD49F936C7AF65365"


class LockedUncertaintyError(RuntimeError):
    pass


def _verify(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise LockedUncertaintyError(f"locked uncertainty input missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise LockedUncertaintyError(f"locked uncertainty byte count drift: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest().upper() != expected_sha256:
        raise LockedUncertaintyError(f"locked uncertainty SHA-256 drift: {path}")


def load_locked_uncertainty_model() -> Any:
    """Hydrate and re-register only the exact 8-parent/211-frame model."""

    _verify(ARTIFACT_PATH, ARTIFACT_BYTES, ARTIFACT_SHA256)
    _verify(RECEIPT_PATH, RECEIPT_BYTES, RECEIPT_SHA256)
    with gzip.open(ARTIFACT_PATH, "rt", encoding="utf-8") as stream:
        package = json.load(stream)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    hydrated = materializer.hydrate_content_addressed_artifact(
        package,
        lambda relative: materializer.safe_join(SOURCE_ROOT, relative).read_bytes(),
    )
    model = load_factory_bound_uncertainty_model(
        hydrated,
        receipt,
        expected_artifact_canonical_sha256=package["artifact_canonical_sha256"],
        expected_model_sha256=MODEL_SHA256,
    )
    if model.content_sha256 != MODEL_SHA256:
        raise LockedUncertaintyError("locked uncertainty model identity drift")
    return model


__all__ = ["MODEL_SHA256", "LockedUncertaintyError", "load_locked_uncertainty_model"]

"""Small fail-closed schema for NamedReferentProviderV0.

This is a provider/input schema, not an evaluator schema.  It deliberately has
no truth, fused score, decision, navigation action, or arrival field.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "blindassist_named_referent_provider_v0"
CLAIM_CEILING = "ENGINEERING_MECHANICS_ONLY"
CHANNELS = (
    "text_evidence",
    "visual_reference_evidence",
    "proposal_evidence",
    "bearing_evidence",
)
CHANNEL_STATUSES = {"AVAILABLE", "NOT_EVALUABLE"}


class SchemaError(ValueError):
    """Input or output violates the provider-only contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_image(value: str | Path, field: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise SchemaError(f"{field} is not a file: {path}")
    return path


@dataclass(frozen=True)
class GoalReferencePack:
    name: str
    aliases: tuple[str, ...]
    reference_images: tuple[Path, ...]
    logo: Path | None = None
    map_bearing_degrees: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GoalReferencePack":
        name = str(value.get("name", "")).strip()
        if not name:
            raise SchemaError("goal reference pack name is required")
        aliases_raw = value.get("aliases", [])
        if not isinstance(aliases_raw, Sequence) or isinstance(aliases_raw, (str, bytes)):
            raise SchemaError("aliases must be an array")
        aliases = tuple(str(item).strip() for item in aliases_raw)
        if any(not item for item in aliases):
            raise SchemaError("aliases cannot contain empty values")
        refs_raw = value.get("reference_images", [])
        if not isinstance(refs_raw, Sequence) or isinstance(refs_raw, (str, bytes)):
            raise SchemaError("reference_images must be an array")
        references = tuple(_existing_image(item, "reference image") for item in refs_raw)
        logo_raw = value.get("logo")
        logo = _existing_image(logo_raw, "logo") if logo_raw else None
        bearing_raw = value.get("map_bearing_degrees")
        bearing = None if bearing_raw is None else float(bearing_raw)
        if bearing is not None and (not math.isfinite(bearing) or not 0.0 <= bearing < 360.0):
            raise SchemaError("map_bearing_degrees must be finite and in [0, 360)")
        return cls(name=name, aliases=aliases, reference_images=references, logo=logo, map_bearing_degrees=bearing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "reference_images": [str(path) for path in self.reference_images],
            "logo": str(self.logo) if self.logo else None,
            "map_bearing_degrees": self.map_bearing_degrees,
        }


@dataclass(frozen=True)
class CurrentFrame:
    frame_id: str
    image_path: Path
    captured_at_ms: int | None = None
    heading_degrees: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CurrentFrame":
        frame_id = str(value.get("frame_id", "")).strip()
        if not frame_id:
            raise SchemaError("frame_id is required")
        image = _existing_image(value.get("image_path", ""), "current RGB")
        captured_raw = value.get("captured_at_ms")
        captured = None if captured_raw is None else int(captured_raw)
        heading_raw = value.get("heading_degrees")
        heading = None if heading_raw is None else float(heading_raw)
        if heading is not None and (not math.isfinite(heading) or not 0.0 <= heading < 360.0):
            raise SchemaError("heading_degrees must be finite and in [0, 360)")
        return cls(frame_id=frame_id, image_path=image, captured_at_ms=captured, heading_degrees=heading)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_path": str(self.image_path),
            "image_sha256": sha256_file(self.image_path),
            "captured_at_ms": self.captured_at_ms,
            "heading_degrees": self.heading_degrees,
        }


def provider_identity(
    *,
    provider: str,
    implementation_version: str,
    model_repository: str | None = None,
    model_revision: str | None = None,
    artifact_sha256: Mapping[str, str] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "implementation_version": implementation_version,
        "model_repository": model_repository,
        "model_revision": model_revision,
        "artifact_sha256": dict(artifact_sha256 or {}),
        "runtime": dict(runtime or {}),
    }


def evidence_item(
    *,
    item_id: str,
    source: Mapping[str, Any],
    raw_match: Mapping[str, Any],
    normalized_match: Mapping[str, Any],
) -> dict[str, Any]:
    if not item_id:
        raise SchemaError("evidence item_id is required")
    if not any(key in source for key in ("image_path", "crop", "polygon", "bbox_xyxy", "map_bearing_degrees")):
        raise SchemaError("evidence source needs image/crop/polygon/bbox/bearing provenance")
    return {
        "item_id": item_id,
        "source": dict(source),
        "raw_match": dict(raw_match),
        "normalized_match": dict(normalized_match),
    }


def available_channel(
    *, channel: str, identity: Mapping[str, Any], latency_ms: float, items: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if channel not in CHANNELS:
        raise SchemaError(f"unknown evidence channel: {channel}")
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise SchemaError("latency_ms must be finite and non-negative")
    return {
        "channel": channel,
        "status": "AVAILABLE",
        "provider_identity": dict(identity),
        "latency_ms": latency_ms,
        "items": [dict(item) for item in items],
        "error": None,
    }


def not_evaluable_channel(
    *,
    channel: str,
    identity: Mapping[str, Any],
    latency_ms: float,
    code: str,
    message: str,
    retryable: bool,
) -> dict[str, Any]:
    if channel not in CHANNELS:
        raise SchemaError(f"unknown evidence channel: {channel}")
    return {
        "channel": channel,
        "status": "NOT_EVALUABLE",
        "provider_identity": dict(identity),
        "latency_ms": max(0.0, float(latency_ms)),
        "items": [],
        "error": {"code": code, "message": message, "retryable": bool(retryable)},
    }


def validate_output(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError("provider output schema mismatch")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise SchemaError("claim ceiling drift")
    if any(key in value for key in ("decision", "fused_score", "selected_candidate", "arrival", "action")):
        raise SchemaError("provider output contains a forbidden fusion/decision field")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or len(evidence) != len(CHANNELS) or set(evidence) != set(CHANNELS):
        raise SchemaError("evidence channels must be complete and independent")
    for channel in CHANNELS:
        result = evidence[channel]
        if not isinstance(result, Mapping) or result.get("channel") != channel:
            raise SchemaError(f"channel binding mismatch: {channel}")
        if result.get("status") not in CHANNEL_STATUSES:
            raise SchemaError(f"invalid channel status: {channel}")
        if result["status"] == "NOT_EVALUABLE" and (result.get("items") or not result.get("error")):
            raise SchemaError(f"NOT_EVALUABLE channel fabricated evidence: {channel}")

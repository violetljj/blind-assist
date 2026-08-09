"""Validate the AG-DUE SANPO metadata-only initial source-manifest lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import validate_due_r0 as r0


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "docs/research/assistive-geometry-data-upgrade/BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_SOURCE_MANIFEST_LOCK_2026-08-10.json"


class LockError(ValueError):
    """Raised when the manifest lock drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LockError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bootstrap(receipt: dict[str, Any]) -> None:
    require(receipt.get("schema") == "blindassist.assistive_geometry_due.sanpo_metadata_bootstrap_receipt.v1", "bootstrap schema drift")
    require(receipt.get("status") == "DIAGNOSTIC_METADATA_RECONNAISSANCE_ONLY", "bootstrap status drift")
    require(receipt.get("official_repository_head") == "11FACA999B5C223B804CD3196541A1427834918B", "official repository head drift")
    require(receipt.get("payload_opened") is False, "bootstrap opened payload")
    require(receipt.get("payload_downloaded") is False, "bootstrap downloaded payload")
    require(receipt.get("formal_prescreen_executed") is False, "bootstrap executed prescreen")
    require(receipt.get("selection_or_tuning_influence") is False, "bootstrap influenced selection")
    sources = receipt.get("sources")
    require(isinstance(sources, dict) and set(sources) == {"SANPO_REAL", "SANPO_SYNTHETIC"}, "bootstrap source set drift")
    expected = {
        "SANPO_REAL": (560, "1BAD2EFA02690F455D5F27B59319EB9EF31C0EFB6E1A314B056B63D836FB3CEC", "HfUrNcBq2jQ_Q5m5cp373cUtJPtFactP"),
        "SANPO_SYNTHETIC": (1560, "F9C5DC4C289FA87342ABC0D2CC49F112FCC78C7E02E0B6B081E296A99344173C", "17c7d6bc6d4d4573afecc730cabf4db65f66b04ced504396a71d1185920179cb"),
    }
    for source_name, (count, object_sha, selected_id) in expected.items():
        item = sources[source_name]
        require(item.get("official_train_session_count") == count, f"{source_name} split count drift")
        require(item.get("official_train_split_sha256") == object_sha, f"{source_name} split SHA drift")
        require(item.get("selected_session_id") == selected_id, f"{source_name} selected identity drift")
        require(item.get("tracked_reference_hits") == 0, f"{source_name} selected identity is not discovery-fresh")
        require(item.get("source_frame_count_observed") == 0, f"{source_name} frame outcome leaked into bootstrap")


def validate_manifest_lock(manifest: dict[str, Any], *, expected_source_id: str, expected_session_id: str) -> None:
    hard_reasons = r0.validate_source_manifest(manifest)
    require(not hard_reasons, f"manifest governance hard rejection: {hard_reasons}")
    require(manifest["source"]["source_id"] == expected_source_id, "manifest source identity drift")
    expected_namespaced = f"{expected_source_id}:{expected_session_id}"
    roster = manifest["identity_roster"]
    require(roster["parent_ids"] == [expected_namespaced], "manifest parent roster drift")
    require(roster["session_ids"] == [expected_namespaced], "manifest session roster drift")
    require(roster["history_roles"] == ["SOURCE_DISCOVERY", "TRAIN"], "manifest history role drift")
    require(roster["claimed_freshness"] == "FRESH_SOURCE_DISCOVERY", "manifest freshness drift")
    require(manifest["source"]["payload_presence"] == "UNKNOWN", "payload presence was inferred")
    require(all(value is False for value in manifest["access_receipt"].values()), "manifest access receipt drift")
    for capability_name, observation in manifest["capabilities"].items():
        require(observation["total_frames"] == 0, f"pre-payload frame count asserted: {capability_name}")
        require(observation["orientation_frame_counts"] == {"portrait": 0, "landscape": 0}, f"pre-payload orientation asserted: {capability_name}")
        require(observation["parent_frame_counts"] == {expected_namespaced: 0}, f"pre-payload parent count asserted: {capability_name}")
        require(observation["quality_status"] == "CHARACTERIZED_NOT_VALIDATED", f"metadata upgraded to claim truth: {capability_name}")
        require(observation["orientation_and_camera_basis"] == "UNKNOWN", f"camera/upright basis asserted: {capability_name}")
        require(observation["derivation_receipt"] is None, f"unexecuted derivation receipt asserted: {capability_name}")
        require(observation["teacher_receipts"] == [], f"teacher receipt present: {capability_name}")


def validate_protocol(protocol: dict[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    require(protocol.get("schema") == "blindassist.assistive_geometry_due.sanpo_manifest_lock.v1", "lock schema drift")
    require(protocol.get("status") == "SANPO_INITIAL_SOURCE_MANIFESTS_LOCKED_NOT_PRESCREENED", "lock status drift")
    require(protocol.get("research_mode") == "REVERSIBLE_EXPLORATION", "research mode drift")
    expected_authority = {
        "metadata_bootstrap_disclosed": True,
        "source_manifests_locked": True,
        "formal_source_prescreen": False,
        "payload_download_or_open": False,
        "source_specific_integrity_audit": False,
        "teacher_or_pseudo_label_generation": False,
        "data_materialization": False,
        "model_or_training": False,
        "development_or_confirmation": False,
        "android_or_default_app": False,
    }
    require(protocol.get("execution_authority") == expected_authority, "execution authority drift")
    bindings = protocol.get("bindings")
    require(isinstance(bindings, dict), "bindings missing")
    loaded: dict[str, dict[str, Any]] = {}
    for name, binding in bindings.items():
        path = repo_root / binding["path"]
        require(path.is_file(), f"binding missing: {name}")
        require(sha256_file(path) == binding["sha256"], f"binding SHA drift: {name}")
        if path.suffix == ".json":
            loaded[name] = load_json(path)
    r0.validate_protocol(loaded["due_r0_protocol"])
    validate_bootstrap(loaded["bootstrap_receipt"])
    validate_manifest_lock(
        loaded["sanpo_real_manifest"],
        expected_source_id="sanpo_real_v0_train_discovery",
        expected_session_id="HfUrNcBq2jQ_Q5m5cp373cUtJPtFactP",
    )
    validate_manifest_lock(
        loaded["sanpo_synthetic_manifest"],
        expected_source_id="sanpo_synthetic_v0_train_discovery",
        expected_session_id="17c7d6bc6d4d4573afecc730cabf4db65f66b04ced504396a71d1185920179cb",
    )
    require(protocol.get("local_successor") == "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_MANIFEST_STATIC_PRESCREEN_EXECUTION", "successor drift")
    return {"protocol_id": protocol["protocol_id"], "status": "VALID"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    try:
        result = validate_protocol(load_json(args.protocol))
    except (LockError, r0.ProtocolError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

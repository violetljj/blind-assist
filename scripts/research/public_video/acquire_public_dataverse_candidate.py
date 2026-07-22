#!/usr/bin/env python3
"""Acquire a bounded Dataverse archive as a quarantined unlabeled candidate.

This tool deliberately does not turn public geometry, tracks, or annotations into
BlindAssist risk labels. A receipt proves the source and the quarantine boundary,
not authorization to train or replace a production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Callable

import requests


class AcquisitionError(ValueError):
    """The public source could not pass the quarantine gate."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AcquisitionError(f"JSON root must be an object: {path}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    required_text = (
        "source_id",
        "dataset_persistent_id",
        "dataset_metadata_url",
        "candidate_file_name",
        "source_role",
    )
    if config.get("schema") != "blindassist_public_dataverse_unlabeled_candidate_source_v1":
        raise AcquisitionError("unexpected public Dataverse source schema")
    if not all(isinstance(config.get(key), str) and config[key] for key in required_text):
        raise AcquisitionError("source config has missing text fields")
    if not isinstance(config.get("maximum_file_bytes"), int) or config["maximum_file_bytes"] <= 0:
        raise AcquisitionError("source config maximum_file_bytes must be positive")
    if not isinstance(config.get("source_constraints"), list) or not config["source_constraints"]:
        raise AcquisitionError("source config requires constraints")
    if config.get("training_execution_authorized") is not False:
        raise AcquisitionError("candidate source cannot authorize training")
    if config.get("production_model_replacement_authorized") is not False:
        raise AcquisitionError("candidate source cannot authorize production replacement")


def select_file(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        version = metadata["data"]["latestVersion"]
        files = version["files"]
    except (KeyError, TypeError) as error:
        raise AcquisitionError("Dataverse metadata lacks files") from error
    license_info = version.get("license")
    license_name = license_info.get("name") if isinstance(license_info, dict) else None
    if not isinstance(files, list):
        raise AcquisitionError("Dataverse metadata files must be a list")
    matches = [
        item.get("dataFile")
        for item in files
        if isinstance(item, dict)
        and isinstance(item.get("dataFile"), dict)
        and item["dataFile"].get("filename") == config["candidate_file_name"]
    ]
    if len(matches) != 1:
        raise AcquisitionError("expected exactly one configured candidate file")
    data_file = matches[0]
    if data_file.get("restricted") is True:
        raise AcquisitionError("candidate file is restricted")
    if not isinstance(data_file.get("id"), int) or not isinstance(data_file.get("filesize"), int):
        raise AcquisitionError("candidate file metadata lacks id or size")
    if data_file["filesize"] <= 0 or data_file["filesize"] > config["maximum_file_bytes"]:
        raise AcquisitionError("candidate file violates configured byte bound")
    if not isinstance(data_file.get("md5"), str) or len(data_file["md5"]) != 32:
        raise AcquisitionError("candidate file metadata lacks an MD5 checksum")
    return {**data_file, "observed_license": license_name}


def download_file(opener: Callable[[str], BinaryIO], url: str, destination: Path, *, expected_md5: str) -> tuple[str, str]:
    if destination.exists():
        raise AcquisitionError(f"refusing to overwrite existing candidate: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise AcquisitionError(f"partial download exists; inspect or remove it explicitly: {partial}")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with opener(url) as response, partial.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
            sha256.update(chunk)
            md5.update(chunk)
    actual_md5 = md5.hexdigest()
    if actual_md5.lower() != expected_md5.lower():
        raise AcquisitionError("downloaded candidate MD5 does not match Dataverse metadata")
    partial.replace(destination)
    return sha256.hexdigest(), actual_md5


def download_dataverse_file(url: str, destination: Path, *, expected_md5: str) -> tuple[str, str]:
    """Download through Requests because Dataverse S3 redirects reject urllib here."""
    if destination.exists():
        raise AcquisitionError(f"refusing to overwrite existing candidate: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    resume_bytes = partial.stat().st_size if partial.exists() else 0
    if resume_bytes:
        with partial.open("rb") as existing:
            for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                sha256.update(chunk)
                md5.update(chunk)
    headers = {"User-Agent": "BlindAssist-public-source-verifier/1.0"}
    if resume_bytes:
        headers["Range"] = f"bytes={resume_bytes}-"
    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(45, 120),
        ) as response:
            response.raise_for_status()
            if resume_bytes and response.status_code != 206:
                raise AcquisitionError("remote server refused a safe range resume")
            with partial.open("ab" if resume_bytes else "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        sha256.update(chunk)
                        md5.update(chunk)
    except requests.RequestException as error:
        raise AcquisitionError(f"download failed: {error}") from error
    actual_md5 = md5.hexdigest()
    if actual_md5.lower() != expected_md5.lower():
        raise AcquisitionError("downloaded candidate MD5 does not match Dataverse metadata")
    partial.replace(destination)
    return sha256.hexdigest(), actual_md5


def build_receipt(config: dict[str, Any], data_file: dict[str, Any], *, local_name: str | None, local_sha256: str | None) -> dict[str, Any]:
    return {
        "format": "blindassist_public_dataverse_unlabeled_candidate_receipt_v1",
        "source_id": config["source_id"],
        "dataset_persistent_id": config["dataset_persistent_id"],
        "expected_license": config.get("expected_license"),
        "observed_license": data_file.get("observed_license"),
        "license_status": "recorded_nonblocking" if data_file.get("observed_license") else "unknown_recorded_nonblocking",
        "source_role": config["source_role"],
        "remote_file": {
            "id": data_file["id"],
            "name": data_file["filename"],
            "bytes": data_file["filesize"],
            "md5": data_file["md5"],
        },
        "local_file_name": local_name,
        "local_file_sha256": local_sha256,
        "download_status": "raw_candidate_pending_deidentification" if local_name else "metadata_verified_not_downloaded",
        "privacy_processing_required": bool(config.get("privacy_processing_required", True)),
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "production_model_replacement_authorized": False,
        "constraints": config["source_constraints"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        validate_config(config)
        response = requests.get(
            config["dataset_metadata_url"],
            headers={"User-Agent": "BlindAssist-public-source-verifier/1.0"},
            timeout=(45, 45),
        )
        response.raise_for_status()
        metadata = response.json()
        data_file = select_file(config, metadata)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = args.output_dir / "public_candidate_receipt.json"
        if receipt_path.exists():
            raise AcquisitionError(f"refusing to overwrite existing receipt: {receipt_path}")
        local_name = local_sha256 = None
        if args.download:
            destination = args.output_dir / str(data_file["filename"])
            local_sha256, local_md5 = download_dataverse_file(
                f"https://dataverse.harvard.edu/api/access/datafile/{data_file['id']}",
                destination,
                expected_md5=data_file["md5"],
            )
            local_name = destination.name
        receipt_path.write_text(
            json.dumps(build_receipt(config, data_file, local_name=local_name, local_sha256=local_sha256), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "receipt": str(receipt_path.resolve()), "downloaded": bool(args.download), "training_execution_authorized": False}, ensure_ascii=False))
    except (AcquisitionError, OSError, json.JSONDecodeError, requests.RequestException) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

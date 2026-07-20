#!/usr/bin/env python3
"""Acquire one public GND package as a non-evidentiary candidate source.

The resulting receipt deliberately keeps this material outside the counterfactual
episode manifest and outside all production-training paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, BinaryIO
from urllib.request import urlopen


class AcquisitionError(ValueError):
    """The public candidate source cannot be acquired safely."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AcquisitionError(f"JSON root must be an object: {path}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    text_fields = ("source_id", "dataset_persistent_id", "dataset_metadata_url", "expected_license", "candidate_file_name", "source_role")
    if config.get("schema") != "blindassist_public_video_candidate_source_v1":
        raise AcquisitionError("unexpected public candidate source schema")
    if not all(isinstance(config.get(key), str) and config[key] for key in text_fields):
        raise AcquisitionError("source config has missing text fields")
    if not isinstance(config["maximum_file_bytes"], int) or config["maximum_file_bytes"] <= 0:
        raise AcquisitionError("source config maximum_file_bytes must be positive")
    if not isinstance(config["source_constraints"], list) or not config["source_constraints"]:
        raise AcquisitionError("source config requires source constraints")
    if config.get("production_model_replacement_authorized") is not False:
        raise AcquisitionError("public candidate source must never authorize a production replacement")


def metadata_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        files = metadata["data"]["latestVersion"]["files"]
    except (KeyError, TypeError) as error:
        raise AcquisitionError("Dataverse metadata lacks latestVersion files") from error
    if not isinstance(files, list):
        raise AcquisitionError("Dataverse metadata files must be a list")
    return [item for item in files if isinstance(item, dict)]


def select_file(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    matches = [item.get("dataFile") for item in metadata_files(metadata) if isinstance(item.get("dataFile"), dict) and item["dataFile"].get("filename") == config["candidate_file_name"]]
    if len(matches) != 1:
        raise AcquisitionError(f"expected exactly one public file named {config['candidate_file_name']}")
    data_file = matches[0]
    if data_file.get("restricted") is True:
        raise AcquisitionError("candidate file is restricted")
    file_id, size = data_file.get("id"), data_file.get("filesize")
    if not isinstance(file_id, int) or not isinstance(size, int) or size <= 0:
        raise AcquisitionError("candidate file metadata lacks id or size")
    if size > config["maximum_file_bytes"]:
        raise AcquisitionError(f"candidate file exceeds configured limit: {size}")
    return data_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(opener: Callable[[str], BinaryIO], url: str, destination: Path) -> str:
    if destination.exists():
        raise AcquisitionError(f"refusing to overwrite existing candidate: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise AcquisitionError(f"partial download exists; inspect or remove it explicitly: {partial}")
    digest = hashlib.sha256()
    try:
        with opener(url) as response, partial.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                digest.update(chunk)
        partial.replace(destination)
    except OSError as error:
        raise AcquisitionError(f"download failed: {error}") from error
    return digest.hexdigest()


def receipt(config: dict[str, Any], data_file: dict[str, Any], *, local_name: str | None, local_sha256: str | None) -> dict[str, Any]:
    return {
        "format": "blindassist_public_video_candidate_receipt_v1",
        "source_id": config["source_id"],
        "dataset_persistent_id": config["dataset_persistent_id"],
        "expected_license": config["expected_license"],
        "source_role": config["source_role"],
        "remote_file": {"id": data_file["id"], "name": data_file["filename"], "bytes": data_file["filesize"], "md5": data_file.get("md5")},
        "local_file_name": local_name,
        "local_file_sha256": local_sha256,
        "download_status": "downloaded_candidate_only" if local_name else "metadata_verified_not_downloaded",
        "model_assisted_candidate_screening_allowed": True,
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "production_model_replacement_authorized": False,
        "constraints": config["source_constraints"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download the configured public candidate after metadata validation.")
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        validate_config(config)
        with urlopen(config["dataset_metadata_url"], timeout=45) as response:
            metadata = json.loads(response.read().decode("utf-8"))
        data_file = select_file(config, metadata)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = args.output_dir / "public_candidate_receipt.json"
        if receipt_path.exists():
            raise AcquisitionError(f"refusing to overwrite existing receipt: {receipt_path}")
        local_name = local_sha256 = None
        if args.download:
            destination = args.output_dir / str(data_file["filename"])
            local_sha256 = download_file(lambda url: urlopen(url, timeout=90), f"https://dataverse.orc.gmu.edu/api/access/datafile/{data_file['id']}", destination)
            local_name = destination.name
        receipt_path.write_text(json.dumps(receipt(config, data_file, local_name=local_name, local_sha256=local_sha256), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "receipt": str(receipt_path.resolve()), "downloaded": bool(args.download), "training_execution_authorized": False}, ensure_ascii=False))
    except (AcquisitionError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

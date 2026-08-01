#!/usr/bin/env python3
"""Run the offline filesystem canary for the future T0 SANPO transport."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from acquire_stage_c_t0_sanpo_short_path_transport import (
    AcquisitionConfig,
    TRANSPORT_SCHEMA,
    _tmp_path,
    download_verified,
    enumerate_content_paths,
    plan_layout,
    preflight_layout,
)


CANARY_SCHEMA = "blindassist_hftf_stage_c_t0_sanpo_short_path_canary"
CANARY_READY = "T0_SANPO_SHORT_PATH_FILESYSTEM_CANARY_READY"


def _fixture_receipt(name: str, payload: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "generation": "fixture-generation-7",
        "size": len(payload),
        "md5Hash": base64.b64encode(
            hashlib.md5(payload, usedforsecurity=False).digest()
        ).decode("ascii"),
        "crc32c": "fixture-only",
    }


def run_canary(parent: Path | None = None) -> dict[str, Any]:
    long_session_id = "future-synthetic-session-" + ("x" * 512)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        transport_root = Path(directory) / ("nested-" + ("n" * 48)) / "t"
        config = AcquisitionConfig(
            session_id=long_session_id,
            camera="camera_chest",
            lens="left",
            official_split="test",
            start_frame=0,
            target_fps=10.0,
            frame_count=25,
        )
        layout = plan_layout(transport_root, config)
        path_report = preflight_layout(layout)
        paths = enumerate_content_paths(layout)
        if any(long_session_id in str(path) for path in paths):
            raise RuntimeError("long session identity leaked into content path")

        payload = b'{"session_type":"synthetic"}\n'
        target = (
            layout.staging_root
            / "source_metadata/source_session_description.json"
        )
        observed: dict[str, Any] = {}

        def fake_download(url: str, destination: Path, retries: int) -> None:
            observed.update(
                {"url": url, "destination": str(destination), "retries": retries}
            )
            temporary = _tmp_path(destination)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(payload)
            temporary.replace(destination)

        receipt = _fixture_receipt(
            "sanpo_dataset/v0/fixture/description.json", payload
        )
        download_verified(
            receipt, target, 1, downloader=fake_download
        )
        if (
            not target.is_file()
            or _tmp_path(target).exists()
            or "?generation=fixture-generation-7"
            not in str(observed.get("url"))
        ):
            raise RuntimeError("generation-bound temporary download canary failed")
        return {
            "schema": CANARY_SCHEMA,
            "terminal": CANARY_READY,
            "transport_schema": TRANSPORT_SCHEMA,
            "long_session_id_length": len(long_session_id),
            "long_session_id_excluded_from_content_paths": True,
            "path_preflight": path_report,
            "temporary_download_path_exercised": str(_tmp_path(target)),
            "temporary_download_path_length": len(
                str(_tmp_path(target).resolve())
            ),
            "generation_bound_url_observed": True,
            "size_and_md5_verified": True,
            "network_opened": False,
            "source_opened": False,
            "fresh_or_reserved_source_opened": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"canary output must stay under {artifacts_root}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporary-parent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_canary(args.temporary_parent)
        if args.output is not None:
            output = _require_artifacts_output(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema": CANARY_SCHEMA,
                    "terminal": "T0_SANPO_SHORT_PATH_FILESYSTEM_CANARY_FAILED",
                    "error": str(error),
                    "network_opened": False,
                    "source_opened": False,
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

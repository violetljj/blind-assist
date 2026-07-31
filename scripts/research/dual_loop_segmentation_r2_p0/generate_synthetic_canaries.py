"""Generate deterministic PNG decoder/mapping canaries without source claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import PROTOCOL_ID
from .canonicalizer import (
    CanonicalizationError,
    canonicalize_array,
    load_contract,
    sha256_file,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def generate(*, contract_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite canary root: {output_root}")
    output_root.mkdir(parents=True)
    contract = load_contract(contract_path)
    native = np.arange(31, dtype=np.uint8).reshape(1, 31)
    rows: list[dict[str, Any]] = []
    for mode in ("L", "P", "RGB", "RGBA"):
        if mode in {"L", "P"}:
            image = Image.fromarray(native, mode=mode)
            if mode == "P":
                palette = []
                for value in range(256):
                    palette.extend(((value * 17) % 256, (value * 31) % 256, (value * 47) % 256))
                image.putpalette(palette)
        else:
            channels = [
                native,
                np.full(native.shape, 4, dtype=np.uint8),
                np.arange(31, dtype=np.uint8).reshape(1, 31),
            ]
            if mode == "RGBA":
                channels.append(np.full(native.shape, 127, dtype=np.uint8))
            image = Image.fromarray(np.stack(channels, axis=-1), mode=mode)
        path = output_root / f"source_native_{mode.lower()}.png"
        image.save(path, format="PNG")
        canonical = canonicalize_array(path, "source_native", contract)
        expected_ids = sorted(set(contract["_parsed_mapping"].values()))
        actual_ids = sorted(int(value) for value in np.unique(canonical))
        if actual_ids != expected_ids:
            raise CanonicalizationError(f"{mode} canary mapped IDs differ: {actual_ids}")
        rows.append(
            {
                "id": f"synthetic_source_native_{mode.lower()}",
                "split": "synthetic",
                "source_id": f"synthetic:{mode.lower()}",
                "session_id": f"synthetic:{mode.lower()}",
                "sequence_id": "synthetic_all_native_ids",
                "frame_id": len(rows),
                "scene_bucket": "synthetic_decoder_canary",
                "semantic_mask_path": path.name,
                "semantic_mask_sha256": sha256_file(path),
                "label_authority": "synthetic_canary",
            }
        )
    invalid_native = output_root / "invalid_native_id_31.png"
    Image.fromarray(np.asarray([[31]], dtype=np.uint8), mode="L").save(invalid_native)
    invalid_canonical = output_root / "invalid_canonical_id_4.png"
    Image.fromarray(np.asarray([[4]], dtype=np.uint8), mode="L").save(invalid_canonical)
    invalid_mode = output_root / "invalid_mode_i16.png"
    Image.fromarray(np.asarray([[1]], dtype=np.uint16), mode="I;16").save(invalid_mode)
    failure_canaries = [
        (invalid_native, "source_native", "unmapped IDs"),
        (invalid_canonical, "canonical_passthrough", "invalid IDs"),
        (invalid_mode, "source_native", "mode"),
    ]
    failures: list[dict[str, Any]] = []
    for path, decoder, expected_fragment in failure_canaries:
        try:
            canonicalize_array(path, decoder, contract)
        except CanonicalizationError as exc:
            if expected_fragment not in str(exc):
                raise CanonicalizationError(
                    f"{path.name}: unexpected failure message {exc!s}"
                ) from exc
            failures.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "decoder": decoder,
                    "expected": "FAIL_CLOSED",
                    "observed_error": str(exc),
                }
            )
        else:
            raise CanonicalizationError(f"{path.name}: invalid canary did not fail closed")
    manifest_path = output_root / "manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    sources_config = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.synthetic_sources.v1",
        "protocol_id": PROTOCOL_ID,
        "sources": [
            {
                "role": "synthetic_canary",
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "dataset_root": str(output_root),
                "split": "synthetic",
                "decoder": "source_native",
                "expected_rows": len(rows),
            }
        ],
    }
    config_path = output_root / "sources_config.json"
    _write_json(config_path, sources_config)
    receipt = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.synthetic_canary_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "SYNTHETIC_CANARIES_VALID",
        "accepted_modes": ["L", "P", "RGB", "RGBA"],
        "accepted_row_count": len(rows),
        "all_native_ids_covered": list(range(31)),
        "invalid_canaries": failures,
        "manifest_sha256": sha256_file(manifest_path),
        "sources_config_sha256": sha256_file(config_path),
        "formal_authority": False,
    }
    _write_json(output_root / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = generate(contract_path=args.contract.resolve(), output_root=args.output_root.resolve())
    print(json.dumps({"status": result["status"], "accepted": result["accepted_row_count"]}))

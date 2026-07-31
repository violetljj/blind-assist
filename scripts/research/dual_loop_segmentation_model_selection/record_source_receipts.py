#!/usr/bin/env python3
"""Create the hash-bound R1 official-source initialization receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddrnet-source", type=Path, required=True)
    parser.add_argument("--ddrnet-checkpoint", type=Path, required=True)
    parser.add_argument("--segformer-source-commit", required=True)
    parser.add_argument("--segformer-config", type=Path, required=True)
    parser.add_argument("--segformer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {
        "ddrnet_architecture": {
            "path": str(args.ddrnet_source.resolve()),
            "sha256": sha256(args.ddrnet_source),
            "repository": "https://github.com/ydhongHIT/DDRNet",
            "commit": "de0db317c5af4b0946231f475cf74ca4f51b8ac2",
        },
        "ddrnet_imagenet_checkpoint": {
            "path": str(args.ddrnet_checkpoint.resolve()),
            "sha256": sha256(args.ddrnet_checkpoint),
            "source_url": "https://drive.google.com/file/d/1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4/view",
            "source_repository": "https://github.com/ydhongHIT/DDRNet",
            "model": "DDRNet_23_slim ImageNet classification checkpoint",
        },
        "segformer_architecture_config": {
            "path": str(args.segformer_config.resolve()),
            "sha256": sha256(args.segformer_config),
            "repository": "https://github.com/NVlabs/SegFormer",
            "commit": args.segformer_source_commit,
            "model": "MiT-B0 / SegFormer-B0 backbone config",
        },
        "segformer_mit_b0_checkpoint": {
            "path": str(args.segformer_checkpoint.resolve()),
            "sha256": sha256(args.segformer_checkpoint),
            "source_url": "https://huggingface.co/nvidia/mit-b0",
            "source_repository": "https://github.com/NVlabs/SegFormer",
            "model": "NVIDIA nvidia/mit-b0 ImageNet backbone checkpoint",
        },
    }
    payload: dict[str, Any] = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.source_pretrained_receipt.v1",
        "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
        "status": "SOURCE_PRETRAINED_CHECKPOINTS_HASHED",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_attestation": {
            "ddrnet_repository": "https://github.com/ydhongHIT/DDRNet",
            "segformer_repository": "https://github.com/NVlabs/SegFormer",
            "pretrained_policy": "official_or_official-publisher_source_attested_backbone_then_new_four_class_head",
        },
        "files": files,
        "fresh_holdout_consumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "files": files}, ensure_ascii=False))


if __name__ == "__main__":
    main()

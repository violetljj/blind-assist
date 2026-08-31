#!/usr/bin/env python3
"""Fresh SceneNN confirmation of full-context, target-mask-gated RoMa."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_roma_temporal_local_fresh as fresh_base  # noqa: E402
import l10_scenenn_roma_full_context_mask_posthoc as context_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-roma-full-context-mask-fresh-protocol-v1"
ADMISSION_SCHEMA = "blindassist-l10-scenenn-roma-full-context-mask-fresh-source-admission-v1"
COHORT_SCHEMA = "blindassist-l10-scenenn-roma-full-context-mask-fresh-cohort-v1"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-roma-full-context-mask-fresh-rgb-receipt-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-roma-full-context-mask-fresh-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return fresh_base.base.sha256(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(protocol["implementation"]["sha256"] == sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    prior_path = HERE / protocol["predecessor"]["result_path"]
    require(sha256(prior_path) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    require(load_json(prior_path)["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    model_root = ROOT / protocol["matcher"]["path"]
    require(sha256(model_root / "roma_indoor.pth") == protocol["matcher"]["roma_weights_sha256"], "ROMA_WEIGHTS")
    require(sha256(model_root / "dinov2_vitl14_pretrain.pth") == protocol["matcher"]["dinov2_weights_sha256"], "ROMA_DINOV2_WEIGHTS")
    require(sha256(ROOT / protocol["matcher"]["wheel_path"]) == protocol["matcher"]["wheel_sha256"], "ROMA_WHEEL")
    protocol = dict(protocol)
    protocol["renderer"] = protocol["pre_rgb_selector"]["renderer"]
    return protocol


@contextmanager
def fresh_surface():
    names = ("PROTOCOL_SCHEMA", "ADMISSION_SCHEMA", "COHORT_SCHEMA", "RECEIPT_SCHEMA", "RESULT_SCHEMA", "load_protocol", "__file__")
    saved = {name: getattr(fresh_base, name) for name in names}
    fresh_base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    fresh_base.ADMISSION_SCHEMA = ADMISSION_SCHEMA
    fresh_base.COHORT_SCHEMA = COHORT_SCHEMA
    fresh_base.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    fresh_base.RESULT_SCHEMA = RESULT_SCHEMA
    fresh_base.load_protocol = load_protocol
    fresh_base.__file__ = __file__
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(fresh_base, name, value)


def freeze(protocol_path: Path, admission_path: Path, source_root: Path, output_path: Path) -> None:
    with fresh_surface():
        fresh_base.freeze(protocol_path, admission_path, source_root, output_path)
    cohort = load_json(output_path)
    cohort["authority"] = "FRESH_PRE_RGBD_FULL_CONTEXT_MASK_GATED_ROMA_CONFIRMATION_COHORT"
    cohort["selection_policy"] = "unchanged minimum qualifying frame gap, then minimum qualifying baseline, visibility and time tie-breaks"
    fresh_base.base.predecessor.parent.write_json(output_path, cohort)


def seal(cohort_path: Path, source_root: Path, extraction_root: Path, extractor_exe: Path, output_path: Path) -> None:
    with fresh_surface():
        fresh_base.seal(cohort_path, source_root, extraction_root, extractor_exe, output_path)
    receipt = load_json(output_path)
    receipt["authority"] = "POST_FRESH_COHORT_FREEZE_FULL_CONTEXT_SPARSE_RGB_ONLY_RECEIPT"
    fresh_base.base.predecessor.parent.write_json(output_path, receipt)


@contextmanager
def context_surface(predecessor_name: str):
    saved = {
        "PROTOCOL_SCHEMA": context_base.PROTOCOL_SCHEMA,
        "RESULT_SCHEMA": context_base.RESULT_SCHEMA,
        "load_protocol": context_base.load_protocol,
        "load_json": context_base.load_json,
        "__file__": context_base.__file__,
    }
    original_load = context_base.load_json

    def compatible_load(path: Path) -> dict[str, Any]:
        data = original_load(path)
        if path.name == predecessor_name and "predecessor_dinov2_baseline" in data:
            data = dict(data)
            data["metrics"] = dict(data["metrics"])
            data["metrics"]["aggregate"] = {
                "dinov2_reciprocal_no_none_support": data["predecessor_dinov2_baseline"]
            }
        return data

    context_base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    context_base.RESULT_SCHEMA = RESULT_SCHEMA
    context_base.load_protocol = load_protocol
    context_base.load_json = compatible_load
    context_base.__file__ = __file__
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(context_base, name, value)


def replay(protocol_path: Path, cohort_path: Path, receipt_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "COHORT_PROTOCOL_HASH")
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__)), "COHORT_ENTRYPOINT_HASH")
    require(receipt["cohort_sha256"] == sha256(cohort_path), "RECEIPT_COHORT_HASH")
    require(cohort_path.name == protocol["source"]["cohort_path"], "COHORT_PATH")
    require(receipt_path.name == protocol["source"]["rgb_receipt_path"], "RECEIPT_PATH")
    with context_surface(protocol["predecessor"]["result_path"]):
        context_base.replay(protocol_path, source_root, output_path)
    result = load_json(output_path)
    result["authority"] = "FRESH_SOURCE_DISJOINT_FULL_CONTEXT_MASK_GATED_ROMA_NONE_DEVELOPMENT_RESULT"
    result["conclusion"] = (
        "L10_SCENENN_ROMA_FULL_CONTEXT_MASK_FRESH_CONFIRMATION_GATE_MET"
        if result["gate_met"]
        else "L10_SCENENN_ROMA_FULL_CONTEXT_MASK_FRESH_CONFIRMATION_GATE_NOT_MET"
    )
    fresh_base.base.predecessor.parent.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--admission", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cohort", type=Path, required=True)
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--extractor-exe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.admission, args.source_root, args.output)
    elif args.action == "seal":
        seal(args.cohort, args.source_root, args.extraction_root, args.extractor_exe, args.output)
    else:
        replay(args.protocol, args.cohort, args.receipt, args.source_root, args.output)


if __name__ == "__main__":
    main()

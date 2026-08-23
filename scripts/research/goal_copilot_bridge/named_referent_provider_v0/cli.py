"""CLI for the NamedReferentProviderV0 provider seam and public canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .canary import acquire_public_canary, run_canary, run_vision_canary
from .provider import (
    DINOv2ReferenceAdapter,
    GroundingDINOProposalAdapter,
    MapBearingAdapter,
    NamedReferentProviderV0,
    PaddleOCRV5Adapter,
)
from .schema import CurrentFrame, GoalReferencePack


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire-canary")
    acquire.add_argument("--artifact-root", type=Path, required=True)

    run = subparsers.add_parser("run-canary")
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--grounding-dino-model-dir", type=Path, required=True)
    run.add_argument("--reference-model-dir", type=Path, required=True)
    run.add_argument("--ocr-devices", default="cpu,gpu:0")

    vision = subparsers.add_parser("run-vision-canary")
    vision.add_argument("--artifact-root", type=Path, required=True)
    vision.add_argument("--grounding-dino-model-dir", type=Path, required=True)
    vision.add_argument("--reference-model-dir", type=Path, required=True)

    provider = subparsers.add_parser("run-provider")
    provider.add_argument("--frame-json", type=Path, required=True)
    provider.add_argument("--goal-pack-json", type=Path, required=True)
    provider.add_argument("--output", type=Path, required=True)
    provider.add_argument("--ocr-device", default="cpu")
    provider.add_argument("--paddlex-cache", type=Path)
    provider.add_argument("--grounding-dino-model-dir", type=Path)
    provider.add_argument("--reference-model-dir", type=Path)
    provider.add_argument("--reference-device", default="cuda")

    args = parser.parse_args()
    if args.command == "acquire-canary":
        path = acquire_public_canary(args.artifact_root)
        print(json.dumps({"manifest": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "run-canary":
        devices = tuple(item.strip() for item in args.ocr_devices.split(",") if item.strip())
        result, closeout = run_canary(
            artifact_root=args.artifact_root,
            dino_model_dir=args.grounding_dino_model_dir,
            reference_model_dir=args.reference_model_dir,
            ocr_devices=devices,
        )
        print(json.dumps({"result": str(result), "closeout": str(closeout)}, ensure_ascii=False))
        return 0
    if args.command == "run-vision-canary":
        path = run_vision_canary(
            artifact_root=args.artifact_root,
            dino_model_dir=args.grounding_dino_model_dir,
            reference_model_dir=args.reference_model_dir,
        )
        print(json.dumps({"vision_result": str(path)}, ensure_ascii=False))
        return 0

    frame = CurrentFrame.from_mapping(_read_json(args.frame_json))
    goal = GoalReferencePack.from_mapping(_read_json(args.goal_pack_json))
    adapters = {
        "text_evidence": PaddleOCRV5Adapter(device=args.ocr_device, cache_dir=args.paddlex_cache)
        if args.paddlex_cache else None,
        "visual_reference_evidence": DINOv2ReferenceAdapter(model_dir=args.reference_model_dir, device=args.reference_device)
        if args.reference_model_dir else None,
        "proposal_evidence": GroundingDINOProposalAdapter(model_dir=args.grounding_dino_model_dir)
        if args.grounding_dino_model_dir else None,
        "bearing_evidence": MapBearingAdapter(),
    }
    output = NamedReferentProviderV0(adapters).run(frame, goal)
    _write_json(args.output, output)
    print(json.dumps({"output": str(args.output), "statuses": {key: value["status"] for key, value in output["evidence"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

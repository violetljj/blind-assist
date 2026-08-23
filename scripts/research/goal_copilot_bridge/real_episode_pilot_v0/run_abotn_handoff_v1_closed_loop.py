"""Run one frozen ABotN episode with HANDOFF_READY as the visual terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .run_abotn_v0_closed_loop import run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-graph", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--prospective-freeze", type=Path, required=True)
    parser.add_argument("--pixel-receipt", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codex-exe", type=Path, default=Path("E:/codex-tools/bin/codex.exe"))
    parser.add_argument("--grounding-dino", type=Path, required=True)
    parser.add_argument("--provider-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(
        public_graph_path=args.public_graph.resolve(),
        private_truth_path=args.private_truth.resolve(),
        freeze_path=args.freeze_receipt.resolve(),
        prospective_freeze_path=args.prospective_freeze.resolve(),
        pixel_receipt_path=args.pixel_receipt.resolve(),
        qualification_path=args.qualification.resolve(),
        output_dir=args.output_dir.resolve(),
        codex_exe=args.codex_exe.resolve(),
        grounding_dino=args.grounding_dino.resolve(),
        provider_lock_path=args.provider_lock.resolve(),
        termination_mode="HANDOFF_V1",
    )
    print(json.dumps({
        "terminal": receipt["terminal"],
        "episode": receipt.get("episode"),
        "next_action": receipt.get("next_action"),
    }, ensure_ascii=False, indent=2))
    return 0 if receipt["terminal"] == "ABOTN_HANDOFF_V1_CLOSED_LOOP_ENGINEERING_RUN_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .simulator import (
    run_q0,
    write_exclusive,
    write_text_exclusive,
    write_jsonl_exclusive,
)


def _ensure_artifacts_local(path: Path) -> None:
    if "artifacts.local" not in path.resolve().parts:
        raise ValueError("Q0 output must be under artifacts.local")


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Causal Event-Preserving Semantic Refresh Scheduling Q0",
        "",
        f"- status: `{result['status']}`",
        f"- authority: `{result['authority']}`",
        f"- propagation: `{result['propagation_mode']}`",
        f"- Pareto front: `{', '.join(result['pareto_front']) or 'NONE'}`",
        "",
        "本报告是固定模型全频参考保持的 Development-only 反事实筛查，不是真实语义真值、能效或安全结果。",
        "",
        "| policy | status | detector calls | call rate | Level-3 event/feedback divergence | reference event misses | max delay (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in result["policies"]:
        if metrics["status"] != "VALID":
            lines.append(
                f"| `{metrics['policy']}` | `{metrics['status']}` | - | - | - | - | - |"
            )
            continue
        truth = metrics["truth_item_metrics"]
        lines.append(
            "| `{policy}` | `{status}` | {calls} | {rate:.4f} | {divergence} | {misses} | {delay} |".format(
                policy=metrics["policy"],
                status=metrics["status"],
                calls=metrics["detector_call_count"],
                rate=metrics["detector_call_rate"],
                divergence=metrics["divergence"]["level3_event_or_feedback_frame_count"],
                misses=truth["reference_event_missed_count"],
                delay=(
                    "NA"
                    if truth["first_feedback_delay_max_ms"] is None
                    else f"{truth['first_feedback_delay_max_ms']:.3f}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `FULL_RATE_REFERENCE` is a parity control; any mismatch invalidates the run.",
            "- Other arms hold the last semantic snapshot. This is an explicit R0 propagation baseline, not a production tracker implementation.",
            "- Feature-rule arms require a separately supplied current-frame-only fast-feature trace and are not evaluable without it.",
            "- Two source sessions are insufficient for learned-policy generalization or inferential claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development-only Q0 semantic refresh counterfactual simulator"
    )
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--dump-receipt", type=Path)
    parser.add_argument("--fast-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--interval-ms",
        type=int,
        nargs="+",
        default=[33, 66, 100, 167, 267],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    _ensure_artifacts_local(output_dir)
    result, traces = run_q0(
        dump_path=args.dump.resolve(),
        baseline_path=args.baseline.resolve(),
        baseline_evaluation_path=args.baseline_evaluation.resolve(),
        dump_receipt_path=args.dump_receipt.resolve() if args.dump_receipt else None,
        fast_features_path=args.fast_features.resolve() if args.fast_features else None,
        intervals_ms=tuple(args.interval_ms),
    )
    write_exclusive(output_dir / "result.json", result)
    write_text_exclusive(output_dir / "report.md", render_report(result))
    for policy_name, rows in traces.items():
        safe_name = policy_name.lower().replace("/", "_")
        write_jsonl_exclusive(output_dir / "traces" / f"{safe_name}.jsonl", rows)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(output_dir),
                "pareto_front": result["pareto_front"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    constrained = result["constrained_operating_point"]
    lines = [
        "# Causal Event-Preserving Semantic Refresh Scheduling Q0 R0.1",
        "",
        f"- status: `{result['status']}`",
        f"- authority: `{result['authority']}`",
        f"- propagation: `{result['propagation_mode']}`",
        f"- raw nondominated set: `{', '.join(result['raw_nondominated_set']) or 'NONE'}`",
        f"- admissible set: `{', '.join(constrained['admissible_set']) or 'NONE'}`",
        f"- constrained best operating point: `{constrained['best_operating_point'] or 'NONE'}`",
        "",
        "本报告是固定模型全频参考保持与风险 episode 对齐的 Development-only 反事实筛查，不是真实语义真值、能效或安全结果。",
        "",
        "episode admission constraints: "
        + ", ".join(
            f"`{key}={value}`"
            for key, value in constrained["constraints"].items()
        ),
        "",
        "| policy | status | detector calls | call rate | Level-3 divergence | ref event misses | episode coverage | mean IoU | onset abs p95 (ms) | feedback signed p50/p90/p95 (ms) | stale max (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in result["policies"]:
        if metrics["status"] != "VALID":
            lines.append(
                f"| `{metrics['policy']}` | `{metrics['status']}` | - | - | - | - | - | - | - | - | - |"
            )
            continue
        truth = metrics["truth_item_metrics"]
        episodes = metrics["episode_alignment"]
        onset_p95 = episodes["absolute_onset_delay_ms"]["p95"]
        signed = truth["first_feedback_delay"]["signed_ms"]
        lines.append(
            "| `{policy}` | `{status}` | {calls} | {rate:.4f} | {divergence} | {misses} | {coverage:.3f} | {iou:.3f} | {onset} | {signed_p50}/{signed_p90}/{signed_p95} | {stale:.3f} |".format(
                policy=metrics["policy"],
                status=metrics["status"],
                calls=metrics["detector_call_count"],
                rate=metrics["detector_call_rate"],
                divergence=metrics["divergence"]["level3_event_or_feedback_frame_count"],
                misses=truth["reference_event_missed_count"],
                coverage=episodes["reference_episode_match_recall"],
                iou=episodes["temporal_iou"]["mean"] or 0.0,
                onset=(
                    "NA"
                    if onset_p95 is None
                    else f"{onset_p95:.3f}"
                ),
                signed_p50=(
                    "NA" if signed["p50"] is None else f"{signed['p50']:.3f}"
                ),
                signed_p90=(
                    "NA" if signed["p90"] is None else f"{signed['p90']:.3f}"
                ),
                signed_p95=(
                    "NA" if signed["p95"] is None else f"{signed['p95']:.3f}"
                ),
                stale=episodes["candidate_longest_stale_duration_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `FULL_RATE_REFERENCE` is a parity control; any mismatch invalidates the run.",
            "- Other arms use zero-order hold semantic propagation. This is the simplest propagation baseline, not a theoretical lower bound or production tracker implementation.",
            "- Risk episodes are contiguous active risk-signature runs; episode alignment is a Development diagnostic and not a validated runtime event lifecycle.",
            "- Signed feedback delay reports early/late timing; missing positive events and unmatched episodes remain separate failures.",
            "- Feature-rule arms require a separately supplied current-frame-only fast-feature trace and are not evaluable without it.",
            "- Two source sessions are insufficient for learned-policy generalization or inferential claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development-only Q0 R0.1 semantic refresh evaluation repair"
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
                "raw_nondominated_set": result["raw_nondominated_set"],
                "admissible_set": result["constrained_operating_point"]["admissible_set"],
                "best_operating_point": result["constrained_operating_point"]["best_operating_point"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

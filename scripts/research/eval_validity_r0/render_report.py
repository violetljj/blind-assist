from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json


def cell(value: Any) -> str:
    if value is None:
        return "NOT_EVALUABLE"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(result: dict[str, Any]) -> str:
    if result.get("schema_version") != "blindassist.eval_validity_r0.report.v1" or result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("result schema/protocol mismatch")
    status = str(result.get("status"))
    lines = [
        "# EVAL-VALIDITY R0 result",
        "",
        f"状态：`{status}`",
        "",
        "## 结论",
        "",
        str(result.get("next_action", "No next action recorded.")),
        "",
        "该结果只审计评价构造；不构成模型质量、训练、默认 App 或安全结论。",
        "",
        "## P0 事件事实一致性",
        "",
    ]
    consistency = result.get("actionability_consistency", {})
    metrics = consistency.get("metrics", {}) if isinstance(consistency, dict) else {}
    lines += [
        f"- passed：`{consistency.get('passed')}`",
        f"- reminder_now exact：`{cell(metrics.get('reminder_now_exact_agreement'))}`",
        f"- cleared exact：`{cell(metrics.get('cleared_exact_agreement'))}`",
        f"- knownness exact：`{cell(metrics.get('knownness_exact_agreement'))}`",
        f"- parent-event sequence exact：`{cell(metrics.get('parent_event_sequence_exact_agreement'))}`",
        f"- unknown anchor burden：`{cell(metrics.get('unknown_anchor_burden'))}`",
        "",
    ]
    representation = result.get("scene_representation")
    if isinstance(representation, dict):
        lines += [
            "## 表征层（研究排序，不作晋级）",
            "",
            "| arm | coverage | false area | component recall | false components/frame | fragmentation | temporal stability |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ("current_yolo", "truth_box", "truth_mask"):
            row = representation.get(arm, {})
            lines.append(
                "| " + arm + " | " + " | ".join(cell(row.get(key)) for key in (
                    "coverage", "false_area_rate", "component_recall", "false_components_per_frame", "fragmentation_ratio", "temporal_stability_median_iou",
                )) + " |"
            )
        lines.append("")
    events = result.get("event_quality")
    if isinstance(events, dict):
        lines += [
            "## 事件层（后续候选资格）",
            "",
            "| arm | hits | critical misses | premature alerts | negative false alerts | cleared | median response delay |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ("current_yolo", "truth_box", "truth_mask", "synthetic_oracle"):
            row = events.get(arm, {})
            aggregate = row.get("aggregate", {}) if isinstance(row, dict) else {}
            lines.append(
                f"| {arm} | {cell(aggregate.get('hits', 0))} | {cell(aggregate.get('critical_misses', 0))} | "
                f"{cell(aggregate.get('premature_alerts', 0))} | {cell(aggregate.get('false_alerts', 0))} | "
                f"{cell(aggregate.get('cleared', 0))} | {cell(row.get('median_response_delay_frames'))} |"
            )
        lines.append("")
    ladder = result.get("oracle_monotonicity")
    if isinstance(ladder, dict):
        lines += ["## Oracle 单调性", ""]
        for check in ladder.get("checks", []):
            lines.append(
                f"- `{check.get('candidate')} >= {check.get('baseline')}`：`{check.get('passed')}`"
                + (f"；{', '.join(check.get('problems', []))}" if check.get("problems") else "")
            )
        lines += [
            f"- synthetic integrity：`{ladder.get('synthetic_integrity_passed')}`",
            "",
        ]
    lines += [
        "## 证据限制",
        "",
        "- parent event/source session 是独立单位；frame 仅为纵向重复观测。",
        "- 通过只说明这个预冻结评价路径具备可解释的 oracle 阶梯；之后的模型仍必须在新合同下接受 event-quality 检验。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    text = render(read_json(args.result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .local_expansion import fit_fixed_grid_local_affine
from .protocol import TrialSpec
from .rotation_compensation import compensate_current_to_previous
from .sparse_flow import detect_fixed_grid_features, track_features
from .synthetic_generator import generate_sequence


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_paired_rotation_leakage(
    rows: Sequence[dict[str, Any]], output: Path
) -> None:
    selected = [
        row
        for row in rows
        if row["split"] == "clean"
        and row["motion_family"] == "pure_rotation"
        and row["axis"] in {"yaw", "pitch", "roll"}
        and row["evaluable"]
    ]
    colors = {"yaw": "#3264a8", "pitch": "#e07a2d", "roll": "#568c55"}
    plt.figure(figsize=(6.6, 5.6))
    maximum = max(
        [
            max(
                row["raw_rotation_leakage_per_s"],
                row["compensated_rotation_leakage_per_s"],
            )
            for row in selected
        ]
        + [0.05]
    )
    for axis in ("yaw", "pitch", "roll"):
        subset = [row for row in selected if row["axis"] == axis]
        plt.scatter(
            [row["raw_rotation_leakage_per_s"] for row in subset],
            [row["compensated_rotation_leakage_per_s"] for row in subset],
            s=12,
            alpha=0.45,
            label=axis,
            color=colors[axis],
        )
    plt.plot([0.0, maximum], [0.0, maximum], "k--", linewidth=1.0)
    plt.xlabel("Raw rotation leakage $L_{rot}$ (s$^{-1}$)")
    plt.ylabel("Compensated rotation leakage $L_{rot}$ (s$^{-1}$)")
    plt.title("Trial-level pure-rotation leakage")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    _save_figure(output)


def plot_closing_error(
    rows: Sequence[dict[str, Any]], output: Path
) -> None:
    scale = [
        row["compensated_closing_error_per_s"]
        for row in rows
        if row["split"] == "clean"
        and row["motion_family"] == "scale"
        and row["evaluable"]
    ]
    mixed = [
        row["compensated_closing_error_per_s"]
        for row in rows
        if row["split"] == "clean"
        and row["motion_family"] == "rotation_plus_scale_up"
        and row["evaluable"]
    ]
    plt.figure(figsize=(6.4, 4.8))
    parts = plt.violinplot([scale, mixed], showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_alpha(0.55)
    plt.xticks([1, 2], ["Scale only", "Rotation + scale"])
    plt.ylabel("Absolute closing-rate error (s$^{-1}$)")
    plt.title("Closing signal retention")
    plt.grid(axis="y", alpha=0.2)
    _save_figure(output)


def plot_expansion_curve(row: dict[str, Any], output: Path) -> None:
    trace = [item for item in row["pair_trace"] if item["evaluable"]]
    timestamps = [item["timestamp_seconds"] for item in trace]
    raw = [item["raw_expansion_median_per_s"] for item in trace]
    compensated = [
        item["compensated_expansion_median_per_s"] for item in trace
    ]
    truth = [row["truth_scale_rate_per_s"]] * len(trace)
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(timestamps, raw, label="Raw", color="#d55e00", linewidth=1.7)
    plt.plot(
        timestamps,
        compensated,
        label="Rotation compensated",
        color="#0072b2",
        linewidth=1.7,
    )
    plt.plot(
        timestamps,
        truth,
        label="Synthetic truth",
        color="#202020",
        linestyle="--",
        linewidth=1.2,
    )
    plt.axhline(0.0, color="#777777", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Median local expansion (s$^{-1}$)")
    plt.title(row["trial_id"])
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    _save_figure(output)


def _draw_tracks(
    image: np.ndarray,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    stride = max(1, previous_points.shape[0] // 180)
    for start, end in zip(
        previous_points[::stride], current_points[::stride], strict=True
    ):
        start_point = tuple(int(round(value)) for value in start)
        end_point = tuple(int(round(value)) for value in end)
        cv2.arrowedLine(
            canvas,
            start_point,
            end_point,
            color,
            1,
            line_type=cv2.LINE_AA,
            tipLength=0.25,
        )
    return canvas


def render_flow_comparison(
    spec: TrialSpec, protocol: dict[str, Any], output: Path
) -> None:
    sequence = generate_sequence(spec, protocol)
    pair_index = max(0, (len(sequence.frames) - 1) // 2)
    previous = sequence.frames[pair_index]
    current = sequence.frames[pair_index + 1]
    previous_valid = sequence.valid_masks[pair_index]
    current_valid = sequence.valid_masks[pair_index + 1]
    compensation = compensate_current_to_previous(
        current,
        current_valid,
        previous_valid,
        sequence.rotation_homography_previous_to_current,
    )
    points = detect_fixed_grid_features(
        previous, previous_valid, protocol["sparse_lk"]
    )
    raw = track_features(
        previous,
        current,
        points,
        current_valid,
        protocol["sparse_lk"],
    )
    compensated = track_features(
        previous,
        compensation.image,
        points,
        compensation.valid_mask,
        protocol["sparse_lk"],
    )
    raw_canvas = _draw_tracks(
        previous, raw.previous_points, raw.current_points, (0, 110, 255)
    )
    compensated_canvas = _draw_tracks(
        previous,
        compensated.previous_points,
        compensated.current_points,
        (255, 130, 0),
    )
    combined = np.hstack((raw_canvas, compensated_canvas))
    cv2.putText(
        combined,
        "Raw LK flow",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        "Rotation-compensated LK flow",
        (previous.shape[1] + 15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), combined):
        raise OSError(f"failed to write {output}")


def save_representative_sequence(
    spec: TrialSpec,
    protocol: dict[str, Any],
    directory: Path,
) -> dict[str, Any]:
    sequence = generate_sequence(spec, protocol)
    directory.mkdir(parents=True, exist_ok=True)
    selected_indices = sorted(
        {0, (len(sequence.frames) - 1) // 2, len(sequence.frames) - 1}
    )
    files: list[str] = []
    for index in selected_indices:
        name = f"frame_{index:03d}.png"
        path = directory / name
        if not cv2.imwrite(str(path), sequence.frames[index]):
            raise OSError(f"failed to write {path}")
        files.append(name)
    truth = {
        "trial": spec.to_dict(),
        "timestamps_seconds": sequence.timestamps_seconds,
        "rotation_current_from_previous": (
            sequence.rotation_current_from_previous.tolist()
        ),
        "rotation_homography_previous_to_current": (
            sequence.rotation_homography_previous_to_current.tolist()
        ),
        "pair_homography_previous_to_current": (
            sequence.pair_homography_previous_to_current.tolist()
        ),
        "scale_factor_per_pair": sequence.scale_factor_per_pair,
        "base_sha256": sequence.base_sha256,
        "sequence_sha256": sequence.sequence_sha256,
        "saved_frames": files,
    }
    (directory / "ground_truth.json").write_text(
        json.dumps(truth, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return truth


def write_main_tables(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output_root: Path,
) -> None:
    table_path = output_root / "tables" / "main_results.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    component_rows: list[dict[str, Any]] = []
    for name, component in summary["components"].items():
        component_rows.append(
            {
                "component": name,
                "pass": component["pass"],
                "details_json": json.dumps(
                    component, sort_keys=True, ensure_ascii=False
                ),
            }
        )
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["component", "pass", "details_json"]
        )
        writer.writeheader()
        writer.writerows(component_rows)

    coverage_path = output_root / "tables" / "coverage_and_abstention.csv"
    reasons: dict[str, int] = {}
    for row in rows:
        for reason, count in row["abstention_counts"].items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in sorted(reasons.items()):
            writer.writerow({"reason": reason, "count": count})


def write_markdown_report(
    summary: dict[str, Any],
    runtime_summary: dict[str, Any],
    output: Path,
) -> None:
    components = summary["components"]
    rotation = components["clean_rotation_yaw_pitch"]
    roll = components["clean_rotation_roll"]
    closing = components["clean_closing"]
    coverage = components["coverage"]
    stress = components["stress"]
    lines = [
        "# RCLE-Minimal Phase A — Synthetic Signal Audit R0",
        "",
        f"最终判定：**{summary['verdict']}**",
        "",
        "> 权限：仅为程序生成数据上的机制、坐标与实现证据；不证明 Bonn、真实场景、视障用户、安全、Android 或生产有效性。",
        "",
        "## 主结果",
        "",
        "| Gate | 结果 | 关键统计 |",
        "| --- | --- | --- |",
        (
            "| Clean yaw/pitch suppression | "
            f"{'PASS' if rotation['pass'] else 'FAIL'} | "
            f"comp median={rotation['compensated_leakage']['estimate']}, "
            f"95% CI={rotation['compensated_leakage']['ci95']}; "
            f"paired reduction={rotation['paired_reduction']['estimate']}, "
            f"95% CI={rotation['paired_reduction']['ci95']} |"
        ),
        (
            "| Clean roll non-degradation | "
            f"{'PASS' if roll['pass'] else 'FAIL'} | "
            f"comp median={roll['compensated_leakage']['estimate']}, "
            f"95% CI={roll['compensated_leakage']['ci95']} |"
        ),
        (
            "| Clean closing retention | "
            f"{'PASS' if closing['pass'] else 'FAIL'} | "
            f"scale MAE={closing['scale_error']['estimate']}, "
            f"mixed MAE={closing['mixed_error']['estimate']}, "
            f"sign={closing['sign_accuracy']['estimate']} |"
        ),
        (
            "| FPS consistency | "
            f"{'PASS' if components['fps_consistency']['pass'] else 'FAIL'} | "
            f"range median={components['fps_consistency']['range']['estimate']}, "
            f"95% CI={components['fps_consistency']['range']['ci95']} |"
        ),
        (
            "| Coverage | "
            f"{'PASS' if coverage['pass'] else 'FAIL'} | "
            f"clean={coverage['clean']['evaluable']}/{coverage['clean']['planned']}, "
            f"stress={coverage['stress']['evaluable']}/{coverage['stress']['planned']} |"
        ),
        (
            "| Stress guardrail | "
            f"{'PASS' if stress['pass'] else 'FAIL'} | "
            + ", ".join(
                f"{name}={'PASS' if item['pass'] else 'FAIL'}"
                for name, item in stress["profiles"].items()
            )
            + " |"
        ),
        "",
        "## Trial 与不可评价",
        "",
        f"- 计划 / 实际：{summary['planned_trials']} / {summary['actual_trials']} trials。",
        f"- 可评价 / 不可评价：{summary['evaluable_trials']} / {summary['not_evaluable_trials']}。",
        "- 统计单位始终是 trial；frame 与 grid 只用于 trial 内稳健汇总。",
        "- RSR/CRR 只作诊断，低分母保持 NOT_EVALUABLE，不使用 epsilon。",
        "",
        "## Runtime",
        "",
        "| Module | Mean ms/pair | Median ms/pair | P95 ms/pair |",
        "| --- | ---: | ---: | ---: |",
    ]
    for module, values in runtime_summary["modules"].items():
        lines.append(
            f"| {module} | {values['mean_ms_per_pair']:.3f} | "
            f"{values['median_ms_per_pair']:.3f} | "
            f"{values['p95_ms_per_pair']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 结果包",
            "",
            "- `trial_manifest.jsonl`：全部计划 trial 与合成序列 hash。",
            "- `trial_metrics.jsonl`：trial-level 主指标、pair 曲线、覆盖与 abstention。",
            "- `scientific_summary.json`：95% CI、全部 gate component 与最终判定。",
            "- `tables/`：主结果与不可评价原因。",
            "- `figures/`：paired leakage、closing error、flow 与 expansion 曲线。",
            "- `receipt.json`：协议、源码、环境与全部产物 hash；可用 runner 的 `--validate-existing` 独立复算。",
            "",
            "## 限制与下一边界",
            "",
            "本轮图像、旋转、尺度与退化均为程序生成；没有真实相机 pose 误差、滚动快门、真实深度不连续或真实用户证据。只有 PASS 且获得单独授权后，才可决定 Phase B 或 Replay Demo；本报告本身不开放后继。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")

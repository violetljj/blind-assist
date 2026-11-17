"""Render one Mobilio confirmation error and one PING route-progress example."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from mobilio_turn_confirmation import trial_path, wrap_degrees


def select(path, person, trial, id_field):
    with path.open(newline="", encoding="utf-8") as file:
        return next(
            row for row in csv.DictReader(file)
            if row["person"] == person and int(row[id_field]) == trial
        )


def draw_error(axis, validation, diagnosis):
    row = select(diagnosis, "VI06", 209, "trial")
    raw = json.loads(trial_path(validation, "VI06", 209).read_bytes())
    time = np.array(raw["time"]) - raw["time"][0]
    error = np.array(list(map(wrap_degrees, raw["target"])))
    clip = (time >= 1.4) & (time <= 4.35)
    trigger = float(row["trigger_s"])
    exit_time = trigger + float(row["after_trigger_exit_s"])
    reacquired = exit_time + float(row["after_exit_next_stable_s"])
    axis.axhspan(-10, 10, color="#ddedea", label="对准范围（±10°）", zorder=0)
    axis.plot(time[clip], error[clip], color="#243d50", linewidth=2.3, label="记录的相对角误差")
    for stamp, offset, label, color in (
        (trigger, (-30, -38), "2.10 秒回放确认", "#136861"),
        (exit_time, (15, 18), "2.33 秒离带", "#ae421d"),
        (reacquired, (30, 18), "3.50 秒重新进带\n并持续至少半秒", "#136861"),
    ):
        value = error[np.argmin(abs(time - stamp))]
        axis.scatter(stamp, value, s=48, color=color, zorder=3)
        axis.annotate(
            label, (stamp, value), xytext=offset, textcoords="offset points",
            color=color, fontsize=10, arrowprops={"arrowstyle": "-", "color": color},
        )
    axis.set(xlim=(1.45, 4.3), ylim=(-20, 41), xlabel="转向记录开始后的时间（秒）", ylabel="相对角误差（°）")
    axis.set_title("Mobilio · 回放确认后又偏离", loc="left", fontsize=14, pad=13)
    axis.legend(loc="upper right", frameon=False, fontsize=9)


def draw_route(axis, progress):
    row = select(progress, "P8", 130, "number")
    origin = np.array([float(row["corner_x_m"]), float(row["corner_y_m"])])
    next_vertex = np.array([float(row["next_vertex_x_m"]), float(row["next_vertex_y_m"])])
    tangent = next_vertex - origin
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])

    def coordinates(x, y):
        delta = np.array([float(row[x]), float(row[y])]) - origin
        return np.array([delta @ tangent, delta @ normal])

    start = coordinates("gt_x_now_m", "gt_y_now_m")
    end = coordinates("gt_x_future_m", "gt_y_future_m")
    estimated = coordinates("estimated_x_now_m", "estimated_y_now_m")
    axis.fill_between((1, 4.2), -1.5, 1.5, color="#ddedea", label="计划出弯路段评价范围")
    axis.plot((0, 4.2), (0, 0), color="#136861", linewidth=1.7, linestyle="--")
    axis.scatter(0, 0, color="#136861", s=37, zorder=4)
    axis.annotate("计划转弯点", (0, 0), xytext=(8, -16), textcoords="offset points", fontsize=10, color="#136861")
    axis.quiver(*start, *(end - start), angles="xy", scale_units="xy", scale=1, width=0.014, color="#ae421d")
    axis.scatter(*start, color="#ae421d", s=38, zorder=4)
    axis.annotate(
        "两秒后的标注位置\n离路线中心 2.37 米", end, xytext=(-26, -46), textcoords="offset points",
        fontsize=10, color="#ae421d", arrowprops={"arrowstyle": "-", "color": "#ae421d"},
    )
    axis.scatter(*estimated, s=70, marker="D", facecolors="white", edgecolors="#263d53", zorder=5)
    axis.annotate(
        "手机估计位置\n朝向差 2.3°", estimated, xytext=(14, 33), textcoords="offset points", fontsize=10,
        color="#263d53", arrowprops={"arrowstyle": "-", "color": "#263d53"},
    )
    axis.set(
        xlim=(-0.78, 4.25), ylim=(-1.85, 3.25), xlabel="沿计划出弯路线（米）",
        ylabel="相对路线的横向距离（米）", aspect="equal",
    )
    axis.set_title("PING · 朝向接近，位置仍偏离路线", loc="left", fontsize=14, pad=13)
    axis.legend(loc="lower right", frameon=False, fontsize=9)


def draw_status(axis, status_path):
    status = json.loads(status_path.read_bytes())
    row = next(r for r in status["trials"] if r["person"] == "P2" and r["number"] == 12)
    for slot in row["slots"]:
        if slot["correct"] is not None:
            color = "#d0e9e1" if slot["correct"] else "#f2e2d9"
            axis.axvspan(slot["time_s"] - 0.20, slot["time_s"] + 0.20, color=color, linewidth=0)
    for index, (rule, label) in enumerate((
        ("angle", "只看角度"), ("after_6s", "固定等 6 秒"), ("angle_progress", "角度＋出弯前进"),
    )):
        event = row["outputs"][rule]
        timestamp = event["time_s"]
        color = "#136861" if event["correct"] else "#ae421d"
        axis.hlines(index, 2, timestamp, color="#6f7f87", linewidth=1.0)
        axis.scatter(timestamp, index, color=color, s=56, zorder=3)
        alignment = "right" if timestamp > 10 else "left"
        offset = (-9, 6) if timestamp > 10 else (9, 6)
        axis.annotate(f"{timestamp:g} 秒", (timestamp, index), xytext=offset, textcoords="offset points",
                      ha=alignment, fontsize=10, color=color)
        axis.text(1.84, index, label, ha="right", va="center", fontsize=10)
    axis.set(
        xlim=(0.4, 12.35), ylim=(2.56, -0.6), yticks=[], xticks=(2, 4, 6, 8, 10, 12),
        xlabel="转弯指令发出后的时间（秒）",
    )
    axis.spines["left"].set_visible(False)
    axis.set_title("PING · 同一次转弯的三种确认时机", loc="left", fontsize=14, pad=13)
    axis.text(4.8, -0.43, "尚未达到前进条件", color="#ae421d", fontsize=9)
    axis.text(9.2, -0.43, "已达到前进条件", color="#136861", fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", help="Installed font family with Chinese glyphs.")
    args = parser.parse_args()
    installed = set(font_manager.get_font_names())
    font = args.font or next(
        (name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC") if name in installed),
        None,
    )
    if font not in installed:
        parser.error("Install a font with Chinese glyphs or select an installed family with --font")
    plt.rcParams.update({
        "font.family": [font, "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b6c2c7", "text.color": "#223641", "axes.labelcolor": "#223641",
    })
    fig = plt.figure(figsize=(12, 7.2), layout="constrained")
    layout = fig.add_gridspec(2, 2, height_ratios=(4.8, 2.65))
    angular, route, status = (fig.add_subplot(spec) for spec in (layout[0, 0], layout[0, 1], layout[1, :]))
    draw_error(angular, args.validation_root, args.extension_root / "diagnosis/exit_audit.csv")
    draw_route(route, args.extension_root / "ping-progress/turn_progress.csv")
    draw_status(status, args.extension_root / "ping-progress/status_replay.json")
    for axis in (angular, route, status):
        axis.grid(color="#e4eaec", linewidth=0.65)
        axis.set_axisbelow(True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()

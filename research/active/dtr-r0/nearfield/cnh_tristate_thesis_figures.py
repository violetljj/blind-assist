"""Render thesis figures from frozen event curves.csv; no statistical rerun.

Selected indices are the A-variant all-group points already reported in
CNH_TRISTATE_EVENT_DEV_20261006.md. Mode panels inherit those same thresholds;
they do not reselect a threshold per mode. Percentages are display conversions
of saved CSV fractions. Nothing imports model, geometry or bootstrap code.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CSV = ROOT / "artifacts.local/work/cnh-tristate-event-dev-20261006/curves.csv"
DEFAULT_OUT = ROOT / "artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures"
ARMS = ("single/A", "dual/A")
COLORS = {"single/A": "#C17428", "dual/A": "#138B88"}
NAMES = {"single/A": "单路", "dual/A": "双路"}
BUDGETS = (0, 2, 10)
SELECTED = {"single/A": {0: 5, 2: 13, 10: 19}, "dual/A": {0: 0, 2: 7, 10: 11}}
MARKERS = {0: "o", 2: "s", 10: "D"}
MODE_NAMES = {"mode0": "mode0：恒偏 15°", "mode1": "mode1：±20° 扫视", "mode2": "mode2：转弯"}


def schematic_config():
    paths = {
        "r3_plan": ROOT / "artifacts.local/work/cnh-tristate-dev-r3-20261006/PLAN.json",
        "event_plan": ROOT / "artifacts.local/work/cnh-tristate-event-dev-20261006/PLAN.json",
        "arm_source": Path(__file__).with_name("cnh_tristate_dev_r3_truth.py"),
        "nominal_source": Path(__file__).with_name("cnh_tristate_dev_r2.py"),
        "core_source": Path(__file__).with_name("cnh_tristate_dev_r3_geometry.py"),
    }
    before = {key: sha(path) for key, path in paths.items()}
    plan = json.loads(paths["r3_plan"].read_text(encoding="utf-8"))
    event = json.loads(paths["event_plan"].read_text(encoding="utf-8"))
    definition = plan["gate"]["definition"]
    half = float(re.search(r"halfFOV([\d.]+)deg", definition).group(1))
    actual = float(re.search(r"unshrunk([\d.]+)deg", definition).group(1))
    margin = float(re.search(r"noise m([\d.]+)", event["output"]).group(1))
    pitch = float(re.search(r"pitch(-?[\d.]+)deg", plan["calibration"]["trajectory"]).group(1))
    offsets = None
    tree = ast.parse(paths["arm_source"].read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ARMS" for target in node.targets):
            offsets = ast.literal_eval(node.value)
    assert offsets == ((0,), (-15, 15)), "Frozen arm configuration changed"
    assert actual == 2 * half and margin == 3
    cfg = {"actual_fov_deg": actual, "nominal_half_fov_deg": half, "margin_deg": margin,
           "nominal_pitch_deg": pitch, "head_yaw_demo_deg": 15,
           "demo_only": "Chosen schematic head yaw; not an evaluated scene or trajectory",
           "arm_offsets_deg": offsets, "forward_m": plan["gate"]["query_forward_m"],
           "check_half_width_m": plan["gate"]["width_m"], "label_half_width_m": plan["label"]["width_m"],
           "radial_limit_m": plan["gate"]["range_m"],
           "history_seconds": plan["gate"]["age_seconds"],
           "schematic_scope": "Eye-height horizontal section at a single illustrative pose; the gate actually uses recent pose union and two height slices",
           "core_definition": "Check rectangle intersected with nominal direction FOV contracted by m, not contracted actual head FOV",
           "sources": {key: {"path": str(path), "sha256": before[key]} for key, path in paths.items()}}
    return cfg, paths, before


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_curves(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        if row["group"] in ("all", *MODE_NAMES) and row["arm_variant"] in ARMS:
            groups.setdefault((row["group"], row["arm_variant"]), []).append(row)
    for (group, arm), curve in groups.items():
        assert len(curve) == 20, f"Expected frozen 20-point curve: {group}/{arm}"
        taus = [float(row["tau"]) for row in curve]
        assert taus == sorted(set(taus)), f"CSV threshold order changed: {group}/{arm}"
        assert len({row["events"] for row in curve}) == 1, "Event denominator changed within curve"
    assert len(groups) == 8, "Missing all or mode A-variant curves"
    for arm in ARMS:
        assert float(groups[("all", arm)][0]["events"]) == 960
        for budget, index in SELECTED[arm].items():
            # This validates the already reported choice; it does not choose a new point.
            row = groups[("all", arm)][index]
            assert float(row["silent"]) == budget, f"Reported selected point changed: {arm}/{budget}"
    return groups


def style(plt, font_manager):
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        font_name = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        font_name = "sans-serif"
    plt.rcParams.update({
        "font.family": font_name, "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#87939C", "axes.labelcolor": "#243444",
        "text.color": "#243444", "xtick.color": "#526272", "ytick.color": "#526272",
        "grid.color": "#DFE5E9", "grid.linewidth": .7,
        "savefig.facecolor": "white", "svg.fonttype": "path",
        "axes.unicode_minus": False, "figure.dpi": 120,
    })


def export(fig, out, stem, plt):
    paths = [out / f"{stem}.png", out / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight", pad_inches=.16)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=.16)
    plt.close(fig)
    return paths


def main_figure(groups, out, plt, ticker):
    fig, ax = plt.subplots(figsize=(11.3, 6.7))
    for arm in ARMS:
        curve = groups[("all", arm)]
        ax.plot([float(row["silent"]) for row in curve],
                [100 * float(row["unknown_time"]) for row in curve],
                color=COLORS[arm], lw=2.4, marker=".", ms=5, label=NAMES[arm], zorder=3)
        for budget in BUDGETS:
            row = curve[SELECTED[arm][budget]]
            x, y = float(row["silent"]), 100 * float(row["unknown_time"])
            ax.scatter(x, y, marker=MARKERS[budget], color=COLORS[arm],
                       edgecolor="white", linewidth=1, s=90, zorder=5)
            offsets = {("single/A", 0): (80, -8), ("dual/A", 0): (22, 2),
                       ("single/A", 2): (20, 11), ("dual/A", 2): (145, 35),
                       ("single/A", 10): (17, 6), ("dual/A", 10): (24, 12)}
            label = "静默 = 0" if budget == 0 else f"预算 ≤{budget}"
            ax.annotate(f"{label}：{y:.2f}%", (x, y),
                        xytext=offsets[(arm, budget)], textcoords="offset points",
                        va="center", fontsize=10.5, color=COLORS[arm],
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1, "pad": 2},
                        arrowprops={"arrowstyle": "-", "color": COLORS[arm], "lw": .65})
    for budget in (2, 10):
        ax.axvline(budget, color="#B5C0C9", ls=(0, (3, 4)), lw=.8, zorder=0)
    ax.set(xlim=(-.8, 31), ylim=(0, 104),
           xlabel="截止时静默漏报计数 / 960 个接触事件", ylabel="畅通对照序列中的无法判断时间（%）")
    ax.set_xticks([0, 2, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return export(fig, out, "fig1_tristate_A_silent_unknown", plt)


def mode_figure(groups, out, plt, ticker, Line2D):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.2), sharey=True)
    for ax, group in zip(axes, MODE_NAMES):
        denominator = int(float(groups[(group, ARMS[0])][0]["events"]))
        for arm in ARMS:
            curve = groups[(group, arm)]
            ax.plot([100 * float(row["silent_rate"]) for row in curve],
                    [100 * float(row["unknown_time"]) for row in curve],
                    color=COLORS[arm], lw=2.2, marker=".", ms=4, zorder=3)
            positions = {}
            for budget in BUDGETS:
                row = curve[SELECTED[arm][budget]]
                point = (100 * float(row["silent_rate"]), 100 * float(row["unknown_time"]))
                positions.setdefault(point, []).append(budget)
            for (x, y), budgets in positions.items():
                # Same coordinates may represent several inherited global thresholds.
                ax.scatter(x, y, marker=MARKERS[budgets[-1]], s=68, color=COLORS[arm],
                           edgecolor="white", linewidth=.8, zorder=5)
                label = " / ".join(str(budget) for budget in budgets)
                dx, dy = 10, 9 if arm == "single/A" else -16
                if group == "mode0" and arm == "single/A":
                    label = "单路：" + label
                    dy = -27
                if group == "mode0" and arm == "dual/A" and budgets == [0]:
                    label = "双路：0"
                    dy = 12
                if group == "mode1" and budgets == [0]:
                    dy = -13 if arm == "single/A" else 11
                if group == "mode2" and arm == "single/A" and budgets == [10]:
                    dy = 11
                ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                            color=COLORS[arm], fontsize=9,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": .6})
            endpoint = curve[-1]
            end_x, end_y = 100 * float(endpoint["silent_rate"]), 100 * float(endpoint["unknown_time"])
            ax.scatter(end_x, end_y, s=55, facecolors="none", edgecolors=COLORS[arm], linewidth=1.3, zorder=6)
            if group == "mode2":
                ax.annotate(f"最大 τ：{int(float(endpoint['silent']))}/{denominator}",
                            (end_x, end_y), xytext=(-2, 29 if arm == "single/A" else 13),
                            textcoords="offset points", ha="center", color=COLORS[arm], fontsize=9,
                            arrowprops={"arrowstyle": "-", "lw": .6, "color": COLORS[arm]})
        ax.set_title(f"{MODE_NAMES[group]}\n接触事件 n={denominator}", loc="left", fontsize=12)
        ax.set(xlim=(-.3, 10.2), ylim=(-2, 106), xlabel="本模式静默漏报率（%）")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("本模式畅通对照序列中的无法判断时间（%）")
    handles = [Line2D([0], [0], color=COLORS[arm], lw=2.3, label=NAMES[arm]) for arm in ARMS]
    handles += [Line2D([0], [0], marker=MARKERS[budget], color="none", markerfacecolor="#637380",
                       markeredgecolor="#637380", label="静默 = 0" if budget == 0 else f"预算 ≤{budget}") for budget in BUDGETS]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor="#637380", label="最大 τ"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .01), ncol=6, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, 1))
    return export(fig, out, "figA1_tristate_A_mode_curves", plt)


def mode_summary(groups, out, plt, ticker):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.1]})
    unknown_color, silent_color = "#B9CBD8", "#BE6650"
    labels = []
    for mi, group in enumerate(MODE_NAMES):
        n = int(float(groups[(group, ARMS[0])][0]["events"]))
        for ai, arm in enumerate(ARMS):
            y = mi * 2.6 + ai
            row = groups[(group, arm)][SELECTED[arm][10]]
            unknown, silent = int(float(row["unknown_miss"])), int(float(row["silent"]))
            percent = 100 * float(row["unknown_time"])
            axes[0].barh(y, unknown, height=.65, color=unknown_color, zorder=3)
            axes[0].barh(y, silent, left=unknown, height=.65, color=silent_color, zorder=3)
            if unknown:
                axes[0].text(unknown / 2, y, str(unknown), ha="center", va="center", fontsize=11)
            if silent:
                axes[0].text(unknown + silent / 2, y, str(silent), ha="center", va="center", fontsize=11, color="white")
            axes[1].barh(y, percent, height=.65, color=COLORS[arm], zorder=3)
            axes[1].text(percent + 1.3, y, f"{percent:.2f}%", va="center", fontsize=11, color=COLORS[arm])
            labels.append(f"{MODE_NAMES[group]}｜{NAMES[arm]}（接触 n={n}）")
    positions = [mi * 2.6 + ai for mi in range(3) for ai in range(2)]
    axes[0].set(yticks=positions, yticklabels=labels, xlim=(0, 38), xlabel="截止前未及时提醒的事件数")
    axes[0].invert_yaxis()
    axes[0].set_title("(a) 漏报组成", loc="left")
    axes[1].set(xlim=(0, 112), xlabel="畅通对照序列中的无法判断时间（%）")
    axes[1].set_title("(b) 无法判断时间", loc="left")
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(20))
    for ax in axes:
        ax.grid(axis="x", zorder=0)
        ax.tick_params(axis="y", length=0)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=unknown_color, label="无法判断漏报"),
                            plt.Rectangle((0, 0), 1, 1, color=silent_color, label="静默漏报")],
                   loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    return export(fig, out, "fig2_tristate_A_mode_summary_budget10", plt)


def method_figure(cfg, out, plt):
    from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 6.6))
    near, far = cfg["forward_m"]
    width, label_width = cfg["check_half_width_m"], cfg["label_half_width_m"]
    pitch = math.radians(cfg["nominal_pitch_deg"])
    half = math.degrees(math.atan(math.tan(math.radians(cfg["nominal_half_fov_deg"])) * math.cos(pitch)))
    core_half = math.degrees(math.atan(math.tan(math.radians(cfg["nominal_half_fov_deg"] - cfg["margin_deg"])) * math.cos(pitch)))
    radius = cfg["radial_limit_m"]
    head = cfg["head_yaw_demo_deg"]

    def sector(angle, spread):
        theta = [math.radians(angle - spread + 2 * spread * i / 96) for i in range(97)]
        return [(0, 0)] + [(radius * math.sin(t), radius * math.cos(t)) for t in theta] + [(0, 0)]

    for ax, arm, offsets in zip(axes, ARMS, cfg["arm_offsets_deg"]):
        color = COLORS[arm]
        # Recent-pose union is intentionally not simulated in this explanatory figure.
        for offset in offsets:
            ax.add_patch(Polygon(sector(head + offset, half), facecolor=color, edgecolor=color,
                                 alpha=.12 if arm == "dual/A" else .19, linewidth=1.3, zorder=1))
            axis_angle = math.radians(head + offset)
            ax.plot([0, radius * math.sin(axis_angle)], [0, radius * math.cos(axis_angle)],
                    color=color, lw=.8, ls=":", alpha=.75, zorder=2)
        # Nominal direction (not current head direction) defines the contracted core.
        t = math.tan(math.radians(core_half))
        left_near, right_near = max(-width, -near * t), min(width, near * t)
        left_far, right_far = max(-width, -far * t), min(width, far * t)
        core = [(left_near, near), (right_near, near), (right_far, far), (left_far, far)]
        ax.add_patch(Polygon(core, facecolor="#DFE8EF", edgecolor="#465E76", lw=1.6, zorder=4))
        ax.add_patch(Rectangle((-width, near), 2 * width, far - near, fill=False,
                               edgecolor="#465E76", lw=1.1, ls="--", zorder=5))
        for side in (-1, 1):
            x = -label_width if side < 0 else width
            ax.add_patch(Rectangle((x, near), label_width - width, far - near,
                                  facecolor="#BE6650", edgecolor="#BE6650", alpha=.75, zorder=6))
            ax.plot([0, side * radius * math.sin(math.radians(core_half))],
                    [0, radius * math.cos(math.radians(core_half))], color="#637589", ls="--", lw=.9, zorder=3)
        ax.hlines(near, -1.02, 1.4, color="#BE6650", lw=1.2, ls=(0, (4, 3)), zorder=7)
        ax.text(-1.04, near + .055, "0.9 m 截止", ha="left", va="bottom", fontsize=10, color="#A55544")
        ax.annotate("行进方向", (0, 2.52), (0, -.37), ha="center", va="top",
                    arrowprops={"arrowstyle": "-|>", "color": "#344B63", "lw": 1.5}, fontsize=10, color="#344B63", zorder=8)
        angle = math.radians(head)
        ax.annotate("头部方向", (.76 * math.sin(angle), .76 * math.cos(angle)), (.72, .48),
                    arrowprops={"arrowstyle": "-", "color": "#344B63", "lw": .8}, ha="left", fontsize=10, zorder=9)
        ax.arrow(0, 0, .74 * math.sin(angle), .74 * math.cos(angle), width=.004,
                 head_width=.07, head_length=.09, color="#344B63", length_includes_head=True, zorder=8)
        ax.add_patch(Ellipse((0, -.16), .43, .23, facecolor="#596C7E", edgecolor="white", lw=1, zorder=10))
        ax.add_patch(Circle((0, 0), .09, facecolor="#EBCBB1", edgecolor="#596C7E", lw=1.2, zorder=11))
        ax.text(-.62, -.18, "行人", fontsize=10, va="center", color="#344B63")
        ax.annotate("收缩检查核心\n前向 0.9–2.1 m\n横向 ±0.29 m", (.07, 1.48), (-1.08, 1.65),
                    ha="left", va="center", fontsize=10, color="#465E76",
                    arrowprops={"arrowstyle": "-", "color": "#465E76", "lw": .8}, zorder=12)
        ax.annotate("未检查侧带\n0.29 < |横向| ≤ 0.30 m", (width + .005, 1.08), (.53, 1.13),
                    ha="left", va="center", fontsize=9, color="#A55544",
                    arrowprops={"arrowstyle": "-", "color": "#A55544", "lw": .75}, zorder=12)
        ax.annotate("参考半角 22.5° − 3°", (-.42, 1.23), (-1.08, 2.38),
                    ha="left", va="center", fontsize=9, color="#637589",
                    arrowprops={"arrowstyle": "-", "color": "#637589", "lw": .7}, zorder=12)
        sensor_label = "单路标称视场 45°" if arm == "single/A" else "双路各 45° 视场的并集\n相对头部方向 −15° / +15°"
        ax.text(.7, 2.19, sensor_label, ha="left", va="center", fontsize=10, color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2}, zorder=12)
        ax.text(-1.08, -.55, "头部相对行进方向偏转 15°（示意）", fontsize=9, color="#637589")
        ax.set_title(f"{'(a)' if arm == 'single/A' else '(b)'} {NAMES[arm]}覆盖", loc="left", fontsize=13)
        ax.set(xlim=(-1.15, 1.82), ylim=(-.64, 2.68), aspect="equal")
        ax.axis("off")
    fig.tight_layout()
    return export(fig, out, "fig3_coverage_method_topview_schematic", plt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    csv_path, out = args.csv.resolve(), args.out.resolve()
    assert csv_path.name == "curves.csv", "Use saved curves.csv as the only numeric input"
    assert out != csv_path.parent, "Do not write into the frozen source payload"
    assert ROOT.joinpath("artifacts.local").resolve() in out.parents, "Outputs must stay under canonical artifacts.local"
    initial = sha(csv_path)
    groups = load_curves(csv_path)
    config, config_paths, config_hashes = schematic_config()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out / "mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    from matplotlib.lines import Line2D
    style(plt, font_manager)
    files = main_figure(groups, out, plt, ticker)
    files += mode_summary(groups, out, plt, ticker)
    files += mode_figure(groups, out, plt, ticker, Line2D)
    files += method_figure(config, out, plt)
    extracted = {
        "source": "curves.csv", "variant": "A", "selected_global_indices": SELECTED,
        "selection_rule": "Fixed indices reported in CNH_TRISTATE_EVENT_DEV_20261006.md; no reselection",
        "mode_selection_rule": "Inherit the all-group selected threshold index per arm",
        "mode_summary_selection": {arm: SELECTED[arm][10] for arm in ARMS},
        "schematic_config": config,
        "curves": {f"{group}|{arm}": rows for (group, arm), rows in groups.items()},
    }
    extracted_path = out / "extracted_plot_data.json"
    dump(extracted_path, extracted)
    files.append(extracted_path)
    assert sha(csv_path) == initial, "Frozen CSV changed during figure generation"
    assert {key: sha(path) for key, path in config_paths.items()} == config_hashes, "Schematic configuration input changed"
    manifest = {
        "input": {"path": str(csv_path), "sha256": initial}, "inputs_unchanged": True,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "outputs": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in files},
        "selected_global_indices": SELECTED, "png_dpi": 300,
        "schematic_config_sources": config["sources"],
        "svg_font": "glyph paths for portability", "matplotlib_version": matplotlib.__version__,
        "scope": "Statistical plots read saved CSV only; method diagram reads frozen PLAN and source configuration; no model, statistic, bootstrap or threshold optimization",
        "display_conversion": "Saved fractional unknown_time and silent_rate multiplied by 100 for percent axes",
        "uncertainty": "CSV has no paired reselection intervals; none invented or plotted",
    }
    dump(out / "hash_manifest.json", manifest)
    print(json.dumps({"out": str(out), "inputs_unchanged": True, "outputs": [path.name for path in files] + ["hash_manifest.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

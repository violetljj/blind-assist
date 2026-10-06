"""Render thesis figures from frozen event curves.csv; no statistical rerun.

Selected indices are the A-variant all-group points already reported in
CNH_TRISTATE_EVENT_DEV_20261006.md. Mode panels inherit those same thresholds;
they do not reselect a threshold per mode. Percentages are display conversions
of saved CSV fractions. Nothing imports model, geometry or bootstrap code.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CSV = ROOT / "artifacts.local/work/cnh-tristate-event-dev-20261006/curves.csv"
DEFAULT_OUT = ROOT / "artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures"
ARMS = ("single/A", "dual/A")
COLORS = {"single/A": "#C17428", "dual/A": "#138B88"}
NAMES = {"single/A": "single", "dual/A": "dual"}
BUDGETS = (0, 2, 10)
SELECTED = {"single/A": {0: 5, 2: 13, 10: 19}, "dual/A": {0: 0, 2: 7, 10: 11}}
MARKERS = {0: "o", 2: "s", 10: "D"}
MODE_NAMES = {"mode0": "mode0：恒偏 15°", "mode1": "mode1：±20° 扫视", "mode2": "mode2：转弯"}


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
            offsets = {("single/A", 0): (22, -8), ("dual/A", 0): (22, 2),
                       ("single/A", 2): (20, 11), ("dual/A", 2): (20, -20),
                       ("single/A", 10): (17, 6), ("dual/A", 10): (17, -3)}
            ax.annotate(f"预算 ≤{budget}：{y:.2f}%", (x, y),
                        xytext=offsets[(arm, budget)], textcoords="offset points",
                        va="center", fontsize=10.5, color=COLORS[arm],
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 1})
    for budget in (2, 10):
        ax.axvline(budget, color="#B5C0C9", ls=(0, (3, 4)), lw=.8, zorder=0)
    ax.set(xlim=(-.8, 31), ylim=(0, 104),
           xlabel="截止时静默漏报计数 / 960 个接触 episode", ylabel="采样畅通 control 的 unknown 时间负担（%）")
    ax.set_xticks([0, 2, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, loc="upper right", title="A 变体：无保持", title_fontsize=10)
    fig.suptitle("三态事件级权衡：静默漏报与无法判断时间", fontsize=17, y=.98)
    fig.text(.5, .082, "标记沿用已报告的全体最小负担点；每臂使用同一冻结 20 点 τ 网格。", ha="center", fontsize=10)
    fig.text(.5, .049, "预算 2 / 10 的配对重选区间跨零；经验零静默端负担反转。曲线仅作描述。", ha="center", fontsize=10)
    fig.text(.5, .016, "模拟 Development；unknown 不是安全证明，采样控制时间不是实际用户提示负担。", ha="center", fontsize=9.5, color="#627181")
    fig.tight_layout(rect=(0, .11, 1, .95))
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
                    label = "single: " + label
                    dy = -27
                if group == "mode0" and arm == "dual/A" and budgets == [0]:
                    label = "dual: 0"
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
        ax.set_title(f"{MODE_NAMES[group]}\n接触 episode 分母 n={denominator}", loc="left", fontsize=12)
        ax.set(xlim=(-.3, 10.2), ylim=(-2, 106), xlabel="本模式静默漏报率（%）")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("本模式采样畅通 control 的 unknown 时间（%）")
    handles = [Line2D([0], [0], color=COLORS[arm], lw=2.3, label=NAMES[arm]) for arm in ARMS]
    handles += [Line2D([0], [0], marker=MARKERS[budget], color="none", markerfacecolor="#637380",
                       markeredgecolor="#637380", label=f"全体预算 ≤{budget}") for budget in BUDGETS]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor="#637380", label="最大 τ"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .105), ncol=6, frameon=False, fontsize=10)
    fig.suptitle("模式分层：全体选中阈值在不同头部运动中的表现", fontsize=16, y=.98)
    fig.text(.5, .064, "标记数字为全体静默预算 0 / 2 / 10；各模式沿用全体选中的 τ，不在模式内重选。重合点合并标签。", ha="center", fontsize=10)
    fig.text(.5, .024, "A 变体；横轴使用各模式 CSV 保存的静默率。分层是描述性结果，不作模式因果解释。", ha="center", fontsize=10, color="#627181")
    fig.tight_layout(rect=(0, .19, 1, .92))
    return export(fig, out, "fig2_tristate_A_by_mode", plt)


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
    out.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out / "mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    from matplotlib.lines import Line2D
    style(plt, font_manager)
    files = main_figure(groups, out, plt, ticker)
    files += mode_figure(groups, out, plt, ticker, Line2D)
    extracted = {
        "source": "curves.csv", "variant": "A", "selected_global_indices": SELECTED,
        "selection_rule": "Fixed indices reported in CNH_TRISTATE_EVENT_DEV_20261006.md; no reselection",
        "mode_selection_rule": "Inherit the all-group selected threshold index per arm",
        "curves": {f"{group}|{arm}": rows for (group, arm), rows in groups.items()},
    }
    extracted_path = out / "extracted_plot_data.json"
    dump(extracted_path, extracted)
    files.append(extracted_path)
    assert sha(csv_path) == initial, "Frozen CSV changed during figure generation"
    manifest = {
        "input": {"path": str(csv_path), "sha256": initial}, "inputs_unchanged": True,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "outputs": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in files},
        "selected_global_indices": SELECTED, "png_dpi": 300,
        "svg_font": "glyph paths for portability", "matplotlib_version": matplotlib.__version__,
        "scope": "Read and render saved CSV only; no model, statistic, bootstrap or threshold optimization",
        "display_conversion": "Saved fractional unknown_time and silent_rate multiplied by 100 for percent axes",
        "uncertainty": "CSV has no paired reselection intervals; none invented or plotted",
    }
    dump(out / "hash_manifest.json", manifest)
    print(json.dumps({"out": str(out), "inputs_unchanged": True, "outputs": [path.name for path in files] + ["hash_manifest.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

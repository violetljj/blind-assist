"""Plot two annotated examples from the Mobilio turn-confirmation replay."""

import argparse
import json
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


TRACE_COLOR = "#213950"
BAND_COLOR = "#e9f2fb"
STABLE_COLOR = "#2b59a3"
HELD_COLOR = "#007364"
EXIT_COLOR = "#ad363e"
GRID_COLOR = "#e3e9ee"
SECONDARY_TEXT = "#526474"
METHODS = (
    ("first_entry", "First entry"),
    ("dwell_500ms", "Fixed dwell (0.50 s)"),
    ("linear_projection", "Linear projection"),
)
CUES = {"default": "Generic audio", "personalized": "Personalized audio"}
EXAMPLES = (("transient", "A   Brief alignment"), ("stable", "B   Sustained alignment"))


def plot_example(trace, decision, example, title, window, tolerance, hold, y_limit):
    first_stable = example["first_stable_s"]
    cue = CUES[example["cue"]]
    trial = f'{example["person"]}, trial {example["trial"]}'
    trace.set_title(f"{title}   ·   {trial}   ·   {cue}", loc="left", fontsize=11, pad=9)
    trace.axhspan(-tolerance, tolerance, color=BAND_COLOR, zorder=0)
    trace.axhline(0, color="#95a6b3", linewidth=0.8)
    trace.plot(example["time_s"], example["error_deg"], color=TRACE_COLOR, linewidth=1.55)
    trace.set(xlim=(0, window), ylim=(-y_limit, y_limit), ylabel="Heading error (°)")
    trace.yaxis.set_major_locator(MaxNLocator(nbins=5))
    trace.grid(axis="x", color=GRID_COLOR)
    trace.tick_params(axis="x", labelbottom=False, length=0)

    if first_stable is not None:
        trace.fill_between(
            [first_stable, first_stable + hold], -tolerance, tolerance,
            color="#9bc4eb", zorder=1,
        )
        for axis in (trace, decision):
            axis.axvline(first_stable, color=STABLE_COLOR, linewidth=1, linestyle="--")
        trace.text(
            1.015, 0.90, f"First sustained window\n{first_stable:.2f} s",
            transform=trace.transAxes, va="top", color=STABLE_COLOR, fontsize=9, clip_on=False,
        )
    else:
        trace.text(
            0.98, 0.88, "No sustained window", transform=trace.transAxes,
            ha="right", va="top", fontsize=9, color=SECONDARY_TEXT,
        )

    decision.set_yticks(range(len(METHODS)), [label for _, label in METHODS], fontsize=9)
    decision.set_ylim(len(METHODS) - 0.5, -0.5)
    decision.xaxis.set_major_locator(MaxNLocator(nbins=5))
    decision.grid(color=GRID_COLOR)
    decision.tick_params(axis="y", length=0)
    for row, (rule, _) in enumerate(METHODS):
        result = example["decisions"][rule]
        time = result["time_s"]
        if time is None:
            status, color = "No confirmation", SECONDARY_TEXT
        else:
            correct = result["correct"]
            if not isinstance(correct, bool):
                raise ValueError(f"Confirmation for {trial} and {rule} has no outcome")
            status = f"{time:.2f} s   " + ("held" if correct else "left the band")
            color, marker = (HELD_COLOR, "o") if correct else (EXIT_COLOR, "x")
            decision.plot(
                time, row, marker=marker, color=color, markeredgewidth=1.8,
                markersize=6.5, linestyle="none", zorder=3,
            )
        decision.text(
            1.015, row, status, transform=decision.get_yaxis_transform(),
            va="center", color=color, fontsize=9, clip_on=False,
        )

    for axis in (trace, decision):
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#b3c0ca")
        axis.tick_params(colors=TRACE_COLOR, labelsize=9)
    decision.spines["left"].set_visible(False)


def plot_replay(replay, output):
    examples = {item["kind"]: item for item in replay["examples"]}
    if len(replay["examples"]) != 2 or examples.keys() != {"transient", "stable"}:
        raise ValueError("Expected one transient and one stable example")

    window = replay["window_seconds"]
    tolerance = replay["tolerance_deg"]
    hold = replay["hold_seconds"]
    extent = max(
        abs(error)
        for example in examples.values()
        for time, error in zip(example["time_s"], example["error_deg"], strict=True)
        if 0 <= time <= window
    )
    y_limit = 10 * math.ceil(max(extent * 1.04, 2 * tolerance) / 10)

    figure = Figure(figsize=(11.8, 7.5), facecolor="white")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        5, 1, height_ratios=(2.7, 1, 0.55, 2.7, 1), hspace=0.16,
        left=0.17, right=0.82, top=0.79, bottom=0.10,
    )
    figure.text(
        0.17, 0.955, "Mobilio turn confirmation", color=TRACE_COLOR,
        fontsize=22, fontweight="bold", ha="left", va="top",
    )
    figure.text(
        0.17, 0.906, f"Offline replay · Each confirmation is checked over the next {hold:.2f} s",
        color=SECONDARY_TEXT, fontsize=11, ha="left", va="top",
    )
    figure.legend(
        handles=[
            Line2D([], [], color=TRACE_COLOR, label="Recorded heading error"),
            Patch(facecolor=BAND_COLOR, edgecolor="#b3c0ca", label=f"±{tolerance:g}° tolerance"),
            Line2D([], [], color=STABLE_COLOR, linestyle="--", label="First sustained window"),
        ],
        loc="upper left", bbox_to_anchor=(0.165, 0.873), ncol=3, frameon=False, fontsize=9,
    )
    for row, (kind, title) in zip((0, 3), EXAMPLES):
        trace = figure.add_subplot(grid[row])
        decision = figure.add_subplot(grid[row + 1], sharex=trace)
        plot_example(trace, decision, examples[kind], title, window, tolerance, hold, y_limit)
        if row == 3:
            decision.set_xlabel("Time from trial start (s)")
    figure.savefig(output, dpi=175)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_replay(json.loads(args.examples.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()

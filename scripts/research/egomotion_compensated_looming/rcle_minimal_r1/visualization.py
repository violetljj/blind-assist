from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    TrialSpec,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.rotation_compensation import (
    compensate_current_to_previous,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.synthetic_generator import (
    generate_sequence,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.visualization import (
    plot_closing_error,
    plot_expansion_curve,
    plot_paired_rotation_leakage,
    save_representative_sequence,
    write_main_tables,
    write_markdown_report as write_r0_markdown_report,
)

from .sparse_flow import detect_fixed_grid_features, track_features


def _draw_tracks(
    image: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for source, target in zip(previous, current, strict=True):
        start = tuple(np.rint(source).astype(int))
        end = tuple(np.rint(target).astype(int))
        cv2.arrowedLine(
            canvas, start, end, color, 1, cv2.LINE_AA, tipLength=0.25
        )
    return canvas


def render_flow_comparison(
    spec: TrialSpec, protocol: dict[str, Any], output: Path
) -> None:
    sequence = generate_sequence(spec, protocol)
    pair_index = max(0, (len(sequence.frames) - 2) // 2)
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
        previous, current, points, current_valid, protocol["sparse_lk"]
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
        "Raw LK flow (R1 cycle)",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        "Rotation-compensated LK flow (R1 cycle)",
        (previous.shape[1] + 15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), combined):
        raise OSError(f"failed to write {output}")


def write_markdown_report(
    summary: dict[str, Any],
    runtime_summary: dict[str, Any],
    output: Path,
) -> None:
    write_r0_markdown_report(summary, runtime_summary, output)
    text = output.read_text(encoding="utf-8")
    text = text.replace(
        "# RCLE-Minimal Phase A — Synthetic Signal Audit R0",
        "# RCLE-Minimal Phase A — Coverage Revision R1",
        1,
    )
    text += (
        "\n## Revision boundary\n\n"
        "- Implementation revision: "
        "`RCLE_MINIMAL_PHASE_A_COVERAGE_REVISION_R1`.\n"
        "- R0 protocol, 2520 trial inventory, thresholds and gates are unchanged.\n"
        "- R0 negative rows and receipt remain retained and independently valid.\n"
        "- This result remains synthetic mechanism and implementation evidence only.\n"
    )
    output.write_text(text, encoding="utf-8", newline="\n")


__all__ = [
    "plot_closing_error",
    "plot_expansion_curve",
    "plot_paired_rotation_leakage",
    "render_flow_comparison",
    "save_representative_sequence",
    "write_main_tables",
    "write_markdown_report",
]

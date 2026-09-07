"""Procedural, single-frame depth-space Development probes.

These are analytic fixtures, not rendered UE observations or natural sensor data.
They probe representation and sampling behavior; they do not establish recognition
of unknown object categories. Arrays stay in memory and callers own all outputs.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


CAMERA = {
    "width": 320,
    "height": 240,
    "hfov_deg": 90.0,
    "camera_height_m": 1.6,
    "pitch_deg": 0.0,
}
HEIGHT_EDGES_M = (0.065, 0.65, 1.4, 1.85)
ATTENTION_FORWARD_M = 3.0
FX = FY = 160.0
CX, CY = 159.5, 119.5
_V, _U = np.mgrid[:240, :320]
_RIGHT_RAY = (_U - CX) / FX
_UP_RAY = -(_V - CY) / FY
_ANGLE_DEG = np.degrees(np.arctan(_RIGHT_RAY))


def _background() -> np.ndarray:
    """Nearest intersection of each forward ray with floor z=0 or wall x=8."""
    depth = np.full((240, 320), 8.0, dtype=np.float32)
    downward = _UP_RAY < 0.0
    floor_depth = np.full((240, 320), np.inf, dtype=np.float64)
    floor_depth[downward] = -CAMERA["camera_height_m"] / _UP_RAY[downward]
    depth[downward] = np.minimum(depth[downward], floor_depth[downward])
    return depth


def _world_at(forward_m: float) -> tuple[np.ndarray, np.ndarray]:
    return _RIGHT_RAY * forward_m, CAMERA["camera_height_m"] + _UP_RAY * forward_m


def _rectangle(
    forward_m: float, right_min: float, right_max: float, z_min: float, z_max: float
) -> np.ndarray:
    right, height = _world_at(forward_m)
    return (right >= right_min) & (right < right_max) & (height >= z_min) & (height < z_max)


def _expected_geometry(visible: np.ndarray, forward_m: float) -> np.ndarray:
    """Analytic truth from intended visible surfaces, with no support threshold.

    The declared attention boundary is forward depth, matching the raster's x
    coordinate. Pixel rays establish direction and world height independently of
    any encoder. Every nonempty intended surface in a cell makes it positive.
    """
    expected = np.zeros((3, 3), dtype=bool)
    if forward_m > ATTENTION_FORWARD_M:
        return expected
    _, height = _world_at(forward_m)
    directions = (_ANGLE_DEG < -10.0, (_ANGLE_DEG >= -10.0) & (_ANGLE_DEG < 10.0), _ANGLE_DEG >= 10.0)
    for direction, direction_mask in enumerate(directions):
        for band, (lower, upper) in enumerate(zip(HEIGHT_EDGES_M[:-1], HEIGHT_EDGES_M[1:])):
            expected[direction, band] = np.any(visible & direction_mask & (height >= lower) & (height < upper))
    return expected


def _case(
    name: str, family: str, depth: np.ndarray, expected: np.ndarray, note: str,
    *, expected_unknown: bool = False,
) -> dict:
    result = {
        "name": name,
        "family": family,
        "depth": np.asarray(depth, dtype=np.float32),
        "expected": np.asarray(expected, dtype=bool),
        "camera": dict(CAMERA),
        "note": note,
    }
    if expected_unknown:
        result["expected_unknown"] = True
    return result


def _inserted(name: str, family: str, shape: np.ndarray, forward_m: float, note: str) -> dict:
    depth = _background()
    visible = shape & (forward_m < depth)
    depth[visible] = forward_m
    return _case(name, family, depth, _expected_geometry(visible, forward_m), note)


def generate_cases() -> Iterator[dict]:
    """Yield 88 deterministic fixtures, ordered independently of model outcomes.

    Counts: six clear/invalid/artifact controls, 32 thin poles, 32 overhead
    bars, and six each of near walls, low blocks and irregular L silhouettes.
    Thickness and alignment cases cover each residue of a four-pixel sampler.
    """
    negative = np.zeros((3, 3), dtype=bool)
    yield _case("clear_floor_far_wall", "clear", _background(), negative.copy(),
                "Raycast floor and 8 m wall only; no inserted near-field obstacle.")
    for kind, value in (("zeros", 0.0), ("nan", np.nan)):
        yield _case(f"all_invalid_{kind}", "invalid", np.full((240, 320), value, dtype=np.float32),
                    negative.copy(), "No valid observations: expected UNKNOWN, not observed clear.",
                    expected_unknown=True)

    # Separated source-error samples cannot accidentally form a connected object.
    rng = np.random.default_rng(20260907)
    depth = _background()
    candidates = [(row, col) for row in range(100, 220, 12) for col in range(40, 280, 12)]
    for index in rng.choice(len(candidates), size=24, replace=False):
        row, col = candidates[int(index)]
        depth[row, col] = np.float32(2.2)
    yield _case("isolated_depth_speckles_seed20260907", "sensor_artifact", depth, negative.copy(),
                "24 spatially isolated near-depth errors; no inserted physical geometry.")
    for size in (2, 3):
        depth = _background()
        depth[156:156 + size, 156:156 + size] = np.float32(2.2)
        yield _case(f"one_frame_depth_artifact_{size}x{size}", "sensor_artifact", depth, negative.copy(),
                    f"One-frame {size}x{size} correlated sensor artifact, not real geometry. "
                    "Sensor-artifact ambiguity: an identical single-frame patch could be a small real object; "
                    "this provenance label does not make that distinction visually observable.")

    for forward_m in (2.2, 2.8):
        depth_tag = str(forward_m).replace(".", "p")
        for width_px in (1, 2, 3, 4):
            for phase in (0, 1, 2, 3):
                # Half-pixel world boundaries rasterize exactly width_px columns.
                col = 156 + phase
                right_min = (col - 0.5 - CX) * forward_m / FX
                right_max = (col + width_px - 0.5 - CX) * forward_m / FX
                pole = _rectangle(forward_m, right_min, right_max, 0.08, 1.80)
                yield _inserted(
                    f"thin_pole_d{depth_tag}_w{width_px}_phase{phase}", "thin_pole", pole, forward_m,
                    f"Physical fronto-parallel pole, {width_px} projected columns at column {col}; "
                    f"sampling phase {phase}, heights 0.08–1.80 m, forward depth {forward_m} m.",
                )

                # A high horizontal bar has the same thickness/phase sweep by row.
                row = 108 + phase
                z_min = CAMERA["camera_height_m"] - (row + width_px - 0.5 - CY) * forward_m / FY
                z_max = CAMERA["camera_height_m"] - (row - 0.5 - CY) * forward_m / FY
                bar = _rectangle(forward_m, -0.9, 0.9, z_min, z_max)
                yield _inserted(
                    f"overhead_bar_d{depth_tag}_h{width_px}_phase{phase}", "overhead_bar", bar, forward_m,
                    f"Physical overhead bar, {width_px} projected rows at row {row}; sampling phase {phase}, "
                    f"right extent -0.9–0.9 m, forward depth {forward_m} m.",
                )

        for direction, angle_deg in (("left", -22.0), ("center", 0.0), ("right", 22.0)):
            center = forward_m * np.tan(np.radians(angle_deg))
            wall = _rectangle(forward_m, center - 0.4, center + 0.4, 0.08, 1.80)
            yield _inserted(f"near_wall_{direction}_d{depth_tag}", "near_wall", wall, forward_m,
                            "Visible 0.8 m wide vertical rectangle; world-height extent 0.08–1.80 m.")
            block = _rectangle(forward_m, center - 0.22, center + 0.22, 0.09, 0.38)
            yield _inserted(f"low_block_{direction}_d{depth_tag}", "low_block", block, forward_m,
                            "Visible low block face; world-height extent 0.09–0.38 m.")
            stem = _rectangle(forward_m, center - 0.24, center - 0.10, 0.10, 1.78)
            foot = _rectangle(forward_m, center - 0.24, center + 0.24, 0.10, 0.46)
            yield _inserted(f"irregular_l_{direction}_d{depth_tag}", "irregular_l", stem | foot, forward_m,
                            "Union of a tall narrow face and short wider foot creates an L silhouette. "
                            "Shape geometry is known here; no natural-category generalization is implied.")

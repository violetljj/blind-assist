"""Exact largest adjacent-ray chord on a discrete pinhole image, given calibrated intrinsics."""
import json
import math

import numpy as np
from silicon_run import WORK


def largest_chord(fx, fy, cx, cy, width, height):
    if width < 2 or height < 1 or min(fx, fy) < 1:
        raise ValueError("Expected a finite pinhole grid with positive clamped focal lengths")
    left = min(max(round(cx - 1), 0), width - 2)
    a = (left + .5 - cx) / fx
    b = a + 1 / fx
    y_star = math.sqrt(max(a*b - 1, 0))
    rows = set()
    for y in (-y_star, y_star):
        target = cy - .5 + fy*y
        for row in (math.floor(target), math.ceil(target)):
            rows.add(min(max(row, 0), height-1))
    maximum = 0.
    for row in rows:
        r = math.hypot(1, (row+.5-cy)/fy)
        theta = math.atan2(r/fx, r*r + a*b)
        maximum = max(maximum, 2*math.sin(theta/2))
    return maximum


def verify():
    rng = np.random.default_rng(20261124)
    largest_residual = 0.
    interior_vertical_optima = 0
    for _ in range(1200):
        w, h = map(int, rng.integers(3, 61, size=2))
        fx, fy = np.exp(rng.uniform(0, 6, size=2))
        cx, cy = rng.uniform(-3, 4, size=2)*[w, h]
        v, u = np.mgrid[:h,:w]
        rays = np.stack(((u+.5-cx)/fx, (v+.5-cy)/fy, np.ones((h,w))))
        rays /= np.linalg.norm(rays, axis=0)
        chords = np.linalg.norm(np.diff(rays, axis=2), axis=0)
        maximum = largest_chord(fx,fy,cx,cy,w,h)
        largest_residual = max(largest_residual, abs(maximum-chords.max()))
        row = np.unravel_index(chords.argmax(), chords.shape)[0]
        if 0 < row < h-1 and abs(row-(cy-.5)) > 1:
            interior_vertical_optima += 1
    assert largest_residual < 2e-14 and interior_vertical_optima > 0
    panel = WORK / "panel_inputs"
    entries = {}
    fp16_equal = 0
    for tag, k in json.loads((panel/"intrinsics.json").read_text()).items():
        chord = largest_chord(k[0][0],k[1][1],k[0][2],k[1][2],384,288)
        value = np.clip(350*math.sqrt(chord*chord+1e-12),.2,4.)
        used = float(np.fromfile(WORK/"focal_inputs"/(tag+".raw"),"<f4").item())
        equal_half = bool(np.float16(used) == np.float16(value))
        fp16_equal += equal_half
        entries[tag] = {"from_intrinsics": value, "from_rotated_fp32_rays": used,
                            "relative_pct": 100*(value/used-1), "same_ieee_fp16": equal_half}
    report = {"synthetic_grids": 1200, "max_chord_abs_diff": largest_residual,
                  "interior_off_axis_optima": interior_vertical_optima, "panel": entries,
                  "panel_max_abs_pct": max(abs(r["relative_pct"]) for r in entries.values()),
                  "panel_matching_half": fp16_equal}
    (WORK/"focal_inputs"/"intrinsic_check.json").write_text(json.dumps(report,indent=2))
    print(report["panel_max_abs_pct"], fp16_equal, largest_residual, interior_vertical_optima)


if __name__ == "__main__":
    verify()

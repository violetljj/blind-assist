"""Real L8CH raw zone order -> frozen model zone convention.

The frozen projector (dtr-r0/nearfield/cnh_cvr_projection.py) and the simulator
(angular_rays, camera X right, Y down) index zones row-major with row 0 at the
top of the scene and column 0 at its left. The current head-worn XIAO/L8CH
assembly emits raw zones with row 0 at the bottom and column 0 at the right
(receiver optics invert the scene, ST UM3183 section 2.2). Evidence for the
present mounting:

* installation-view.json: raw_to_current_display "row=7-raw_row,col=7-raw_col",
  accepted against operator up/right probes (cnh-h3-wifi-20261011 screenshots:
  display top-left is zone 63; the up probe gives near returns in the top rows).
* dorm pilot crossbar segment: raw column 0 shows a 0.6-1.2 m return over all
  eight raw rows, i.e. a vertical structure at one side (the ladder, right side
  of the aisle); a floor cannot appear at that range from 1.55 m.

Real CNH must pass through ``to_model`` before any frozen projection, M3, LOCAL,
ordinary or S feature. Remounting invalidates this mapping (``check_mounting``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
INSTALLATION = ROOT / 'artifacts.local/work/real-labeled-capture-20261011/installation-view.json'
EXPECTED_DISPLAY_MAP = 'row=7-raw_row,col=7-raw_col'
MAPPING_ID = 'raw-rot180-to-model-20261011'


def check_mounting(path=INSTALLATION):
    """Refuse to map unless the recorded installation still has the verified map."""
    record = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if record.get('raw_to_current_display') != EXPECTED_DISPLAY_MAP:
        raise ValueError('Installation view changed; re-verify raw zone orientation')
    return record


def model_zone_for_raw(raw_zone):
    """Model row-major zone index (row 0 top, col 0 left) of a raw zone index."""
    raw_zone = int(raw_zone)
    if not 0 <= raw_zone < 64:
        raise ValueError('raw zone must be 0..63')
    row, col = divmod(raw_zone, 8)
    return (7 - row) * 8 + (7 - col)


def to_model(values, zone_axis=None):
    """Reorder raw-zone arrays into the model convention.

    Accepts [..., 64, K] / [..., 64] (zone axis inferred as the last 64-sized axis
    before an optional bin axis) or [..., 8, 8, K] / [..., 8, 8]. Returns a new
    contiguous array with the same layout. The mapping is an involution.
    """
    a = np.asarray(values)
    if zone_axis is None:
        if a.ndim >= 2 and a.shape[-1] == 64:
            zone_axis = a.ndim - 1
        elif a.ndim >= 2 and a.shape[-2] == 64:
            zone_axis = a.ndim - 2
        elif a.ndim >= 2 and a.shape[-2:] == (8, 8):
            return np.ascontiguousarray(a[..., ::-1, ::-1])
        elif a.ndim >= 3 and a.shape[-3:-1] == (8, 8):
            return np.ascontiguousarray(a[..., ::-1, ::-1, :])
        else:
            raise ValueError('Cannot locate a 64-zone or 8x8 zone axis')
    if a.shape[zone_axis] != 64:
        raise ValueError('Zone axis must have 64 entries')
    order = np.array([model_zone_for_raw(z) for z in range(64)])
    # model[m] = raw[r] with m = order[r]; the map is its own inverse.
    return np.ascontiguousarray(np.take(a, order, axis=zone_axis))

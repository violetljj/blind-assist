"""C29 dense-motion authority policy searched by SkyDiscover.

The evaluator passes only truth-blind, current/past causal row features.  The
policy returns indices of residual rows that may extend the sealed M1-PDC alert
baseline for the current frame.
"""

from __future__ import annotations

from typing import Any, Sequence


# EVOLVE-BLOCK-START
def select_extension(rows: Sequence[dict[str, Any]]) -> list[int]:
    """Authorize stable raw motion through lineage or strong local consensus."""

    def num(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def vec(row: dict[str, Any], key: str) -> tuple[float, float]:
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                pass
        if key == "position":
            return num(row, "forward"), num(row, "left")
        return num(row, "vf"), num(row, "vl")

    raw = [
        index
        for index, row in enumerate(rows)
        if row.get("status") == "RAW_PD_RESIDUAL"
        and row.get("visibility")
        not in {"VISIBILITY_KNOWN_FREE", "VISIBILITY_UNSENSED"}
        and max(num(row, "q"), num(row, "quality")) >= 0.55
        and num(row, "flow_support") > 0
        and num(row, "source_point_count") >= 2
    ]
    observed = [
        index
        for index, row in enumerate(rows)
        if row.get("status") == "OBSERVED_PD_HIT"
        and max(num(row, "motion_support"), num(row, "support_count")) > 0
    ]

    def agrees(first: int, second: int) -> bool:
        a, b = rows[first], rows[second]
        position_a, position_b = vec(a, "position"), vec(b, "position")
        velocity_a, velocity_b = vec(a, "velocity"), vec(b, "velocity")
        distance = (
            (position_a[0] - position_b[0]) ** 2
            + (position_a[1] - position_b[1]) ** 2
        ) ** 0.5
        velocity_delta = (
            (velocity_a[0] - velocity_b[0]) ** 2
            + (velocity_a[1] - velocity_b[1]) ** 2
        ) ** 0.5
        speed = max(0.2, num(a, "speed_mps"), num(b, "speed_mps"))
        locality = 0.27 + 0.05 * max(
            0.0, min(abs(position_a[0]), abs(position_b[0]))
        )
        return distance <= locality and velocity_delta <= 0.18 + 0.28 * speed

    selected: list[int] = []
    for index in raw:
        row = rows[index]
        speed = max(0.2, num(row, "speed_mps"))
        temporal = (
            num(row, "dp_m") <= 0.22 + 0.18 * speed
            and num(row, "dv_mps") <= 0.18 + 0.30 * speed
        )
        if not temporal:
            continue

        peers = sum(agrees(index, other) for other in raw if other != index)
        lineage = any(agrees(index, other) for other in observed)
        strong = (
            max(num(row, "q"), num(row, "quality")) >= 0.68
            and num(row, "source_point_count") >= 3
            and num(row, "flow_support") >= 2
        )
        if lineage or peers >= 2 or (strong and peers >= 1):
            selected.append(index)

    return selected
# EVOLVE-BLOCK-END


def choose(rows: Sequence[dict[str, Any]]) -> list[int]:
    selected = select_extension(rows)
    if not isinstance(selected, (list, tuple, set)):
        raise TypeError("select_extension must return a list/tuple/set of row indices")
    output = [int(value) for value in selected]
    if len(output) != len(set(output)):
        raise ValueError("duplicate selected row index")
    if any(value < 0 or value >= len(rows) for value in output):
        raise ValueError("selected row index out of bounds")
    return sorted(output)

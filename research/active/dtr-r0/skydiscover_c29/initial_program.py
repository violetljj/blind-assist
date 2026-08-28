"""C29 dense-motion authority policy searched by SkyDiscover.

The evaluator passes only truth-blind, current/past causal row features.  The
policy returns indices of residual rows that may extend the sealed M1-PDC alert
baseline for the current frame.
"""

from __future__ import annotations

from typing import Any, Sequence


# EVOLVE-BLOCK-START
def select_extension(rows: Sequence[dict[str, Any]]) -> list[int]:
    """Return residual row indices authorized to carry route-risk evidence.

    The baseline is intentionally conservative: M1-PDC already supplies the
    alert path, so no residual row receives authority until executable evidence
    shows a Pareto improvement.  Useful causal fields include ``status``,
    ``visibility``, ``age_s``, ``seed_confidence``, ``q``, ``h``, ``w``,
    ``dp_m``, ``dv_mps``, ``position=(forward,left)``,
    ``velocity=(velocity_forward,velocity_left)``, ``speed_mps``, and remembered
    height support. ``h`` is memory decay and ``w`` is fusion weight; neither is
    an object width or height.
    Sequence identity, frame id, timestamps, labels, and future truth are not
    provided.
    """

    return []
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

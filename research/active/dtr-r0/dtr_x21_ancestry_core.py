"""Source-neutral state machine for frozen X21 component ancestry.

This module contains only the two representation rules that must survive a
source adapter: refresh-first one-step ancestry transport without current-cell
absorption, and the unchanged 0.50-second emitted-row continuation window.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TypeVar


Key = TypeVar("Key", bound=Hashable)
Payload = TypeVar("Payload")


def advance(
    states: Mapping[Key, Payload],
    refreshes: Mapping[Key, Payload],
    *,
    live_ids: set[object],
    identity: Callable[[Key], object],
    carry_filter: Callable[[Key, Payload], Payload | None],
) -> tuple[dict[Key, Payload], list[Payload], dict[str, int]]:
    """Apply X21 refresh-first state advance with no source-cell absorption."""

    next_states = dict(refreshes)
    emitted = list(refreshes.values())
    transported = dropped = 0
    for key, previous in states.items():
        if key in refreshes:
            continue
        if identity(key) not in live_ids:
            dropped += 1
            continue
        carried = carry_filter(key, previous)
        if carried is None:
            dropped += 1
            continue
        next_states[key] = carried
        emitted.append(carried)
        transported += 1
    return next_states, emitted, {
        "refreshed_states": len(refreshes),
        "transported_states": transported,
        "dropped_states": dropped,
    }


def continue_window(
    history: Sequence[tuple[float, Payload]],
    *,
    target_time_s: float,
    transport: Callable[[Payload, float], Payload],
    window_s: float = 0.50,
    epsilon: float = 1e-9,
) -> list[Payload]:
    """Return current-to-older X21 continuation rows through the frozen window."""

    output: list[Payload] = []
    for source_time_s, payload in reversed(history):
        delta_s = float(target_time_s) - float(source_time_s)
        if delta_s > float(window_s) + float(epsilon):
            break
        if delta_s < -float(epsilon):
            raise ValueError(f"x21_future_history:{delta_s}")
        output.append(transport(payload, delta_s))
    return output

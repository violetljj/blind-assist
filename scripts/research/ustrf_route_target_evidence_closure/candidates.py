from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any


ACTIVE = {"route_intersecting", "approaching_route"}
RELEASE = {"receding", "adjacent_safe"}


def relation_observation(
    *, track_id: int, frame_number: int, box: list[float], route: dict[str, Any],
    width: int, height: int, route_intersects: bool,
    histories: dict[int, deque[tuple[int, float]]],
) -> str | None:
    if route.get("status") != "known" or route.get("uv") is None:
        histories[track_id].clear()
        return None
    u, v = [float(value) for value in route["uv"]]
    x1, _, x2, y2 = [float(value) for value in box]
    distance = math.hypot((x1 + x2) / 2.0 - u, y2 - v) / math.hypot(width, height)
    history = histories[track_id]
    if history and frame_number != history[-1][0] + 1:
        history.clear()
    history.append((frame_number, distance))
    if route_intersects:
        return "route_intersecting"
    if len(history) == 3:
        values = [row[1] for row in history]
        if values[0] > values[1] > values[2]:
            return "approaching_route"
        if values[0] < values[1] < values[2]:
            return "receding"
    return "adjacent_safe"


class C1PerPersonRelationFSM:
    def __init__(self, min_alert: int, min_clear: int) -> None:
        self.min_alert = min_alert
        self.min_clear = min_clear
        self.risk_runs: dict[int, int] = defaultdict(int)
        self.clear_runs: dict[int, int] = defaultdict(int)
        self.active: set[int] = set()

    def update(self, frame_number: int, route_known: bool, relations: dict[int, str | None]) -> dict[str, list[int]]:
        deliveries, closures = [], []
        track_ids = set(relations) | self.active
        for track_id in track_ids:
            relation = relations.get(track_id)
            if not route_known or relation is None:
                self.risk_runs[track_id] = 0
                continue
            if relation in ACTIVE:
                self.risk_runs[track_id] += 1
                self.clear_runs[track_id] = 0
                if track_id not in self.active and self.risk_runs[track_id] >= self.min_alert:
                    self.active.add(track_id)
                    deliveries.append(track_id)
            else:
                self.risk_runs[track_id] = 0
                self.clear_runs[track_id] += 1
                if track_id in self.active and self.clear_runs[track_id] >= self.min_clear:
                    self.active.remove(track_id)
                    closures.append(track_id)
        return {"deliveries": deliveries, "closures": closures}


class C2RouteOccupancyEpisodeFSM:
    def __init__(self, min_alert: int, min_clear: int) -> None:
        self.min_alert = min_alert
        self.min_clear = min_clear
        self.risk_run = 0
        self.clear_run = 0
        self.active = False
        self.delivered = False
        self.episode_number = 0

    def update(self, frame_number: int, route_known: bool, relations: dict[int, str | None]) -> dict[str, list[int]]:
        deliveries, closures = [], []
        if not route_known:
            self.risk_run = 0
            return {"deliveries": deliveries, "closures": closures}
        risk = any(relation in ACTIVE for relation in relations.values())
        if risk:
            self.risk_run += 1
            self.clear_run = 0
            if not self.active and self.risk_run >= self.min_alert:
                self.active = True
                self.delivered = True
                self.episode_number += 1
                deliveries.append(self.episode_number)
        else:
            self.risk_run = 0
            self.clear_run += 1
            if self.active and self.clear_run >= self.min_clear:
                self.active = False
                self.delivered = False
                closures.append(self.episode_number)
                self.clear_run = 0
        return {"deliveries": deliveries, "closures": closures}


class C3DualKeyClearanceFSM(C2RouteOccupancyEpisodeFSM):
    def __init__(self, min_alert: int, min_clear: int) -> None:
        super().__init__(min_alert, min_clear)
        self.episode_lineages: set[int] = set()
        self.release_runs: dict[int, int] = defaultdict(int)

    def update(self, frame_number: int, route_known: bool, relations: dict[int, str | None]) -> dict[str, list[int]]:
        deliveries, closures = [], []
        if not route_known:
            self.risk_run = 0
            return {"deliveries": deliveries, "closures": closures}
        active_now = {track_id for track_id, relation in relations.items() if relation in ACTIVE}
        if active_now:
            self.risk_run += 1
            self.clear_run = 0
            if not self.active and self.risk_run >= self.min_alert:
                self.active = True
                self.episode_number += 1
                self.episode_lineages = set(active_now)
                self.release_runs.clear()
                deliveries.append(self.episode_number)
            elif self.active:
                self.episode_lineages.update(active_now)
        else:
            self.risk_run = 0
        if self.active:
            for track_id in self.episode_lineages:
                relation = relations.get(track_id)
                if relation in RELEASE:
                    self.release_runs[track_id] += 1
                elif relation in ACTIVE:
                    self.release_runs[track_id] = 0
                else:
                    # Missing and unknown evidence do not clear a lineage.
                    self.release_runs[track_id] = 0
            route_released = not active_now
            all_released = bool(self.episode_lineages) and all(
                self.release_runs[track_id] >= self.min_clear for track_id in self.episode_lineages
            )
            if route_released and all_released:
                self.active = False
                closures.append(self.episode_number)
                self.episode_lineages.clear()
                self.release_runs.clear()
        return {"deliveries": deliveries, "closures": closures}

"""DTR R2: robust early filtering with a fixed imminent-collision guard.

R1 deliberately rejects motion that is not supported by the robust Theil-Sen
track.  That is useful for early warnings but can react late to a sudden
incursion.  R2 keeps R1 unchanged and admits one narrow fallback: the R0
least-squares route intersection may trigger only inside the already frozen
ESCALATE half-horizon.  No new threshold is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from dtr_r0 import Arm, CausalFrame, DTRConfig, DTRR0Arm, Prediction
from dtr_r1 import DTRR1Arm, FROZEN_R1_CONFIG, R1Config, RiskEventLifecycle


@dataclass(frozen=True)
class R2Config:
    imminent_horizon_fraction: float = 0.50

    def to_dict(self) -> dict[str, float | str]:
        values: dict[str, float | str] = {
            "imminent_horizon_fraction": self.imminent_horizon_fraction,
            "early_branch": "frozen_r1_robust_occupancy_consensus",
            "guard_branch": "r0_least_squares_route_intersection",
            "fusion": "r1_or_r0_inside_escalate_half_horizon",
        }
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
        values["fingerprint_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return values


FROZEN_R2_CONFIG = R2Config()


class DTRR2Arm:
    def __init__(
        self,
        r0_config: DTRConfig | None = None,
        r1_config: R1Config = FROZEN_R1_CONFIG,
        r2_config: R2Config = FROZEN_R2_CONFIG,
    ) -> None:
        if not 0.0 < r2_config.imminent_horizon_fraction <= 1.0:
            raise ValueError("imminent horizon fraction must be in (0, 1]")
        self.arm = Arm.E_R2_GUARDED_CONSENSUS
        self.r0_config = r0_config or DTRConfig()
        self.r1_config = r1_config
        self.r2_config = r2_config
        self._route = DTRR0Arm(Arm.C_ROUTE_INTERSECTION, self.r0_config)
        self._robust = DTRR1Arm(self.r1_config)
        self._lifecycle = RiskEventLifecycle(self.r1_config.clear_grace_s)

    def step(self, frame: CausalFrame) -> Prediction:
        route = self._route.step(frame)
        robust = self._robust.step(frame)
        future_s = route.diagnostic.get("future_s")
        guard_boundary_s = (
            self.r0_config.route_horizon_s
            * self.r2_config.imminent_horizon_fraction
        )
        guard = bool(
            route.raw_alert is True
            and isinstance(future_s, (int, float))
            and float(future_s) <= guard_boundary_s + 1e-9
        )
        if robust.raw_alert is True or guard:
            raw_alert: bool | None = True
        elif robust.raw_alert is False:
            raw_alert = False
        else:
            raw_alert = None
        robust_entry = robust.diagnostic.get("median_entry_s")
        robust_urgent = bool(
            robust.raw_alert is True
            and isinstance(robust_entry, (int, float))
            and float(robust_entry) <= guard_boundary_s + 1e-9
        )
        urgent = guard or robust_urgent
        return Prediction(
            time_s=frame.time_s,
            signal=self._lifecycle.update(frame.time_s, raw_alert, urgent=urgent),
            raw_alert=raw_alert,
            reason=(
                "imminent_route_guard"
                if guard and robust.raw_alert is not True
                else "robust_route_occupancy_consensus"
                if robust.raw_alert is not None
                else "insufficient_causal_track"
            ),
            track_id=(
                route.track_id
                if guard and robust.raw_alert is not True
                else robust.track_id
            ),
            diagnostic={
                "r1_raw_alert": str(robust.raw_alert).lower(),
                "r0_route_raw_alert": str(route.raw_alert).lower(),
                "r0_future_s": future_s if isinstance(future_s, (int, float)) else "none",
                "guard_boundary_s": guard_boundary_s,
                "imminent_guard_active": str(guard).lower(),
                **{
                    f"r1_{key}": value
                    for key, value in robust.diagnostic.items()
                },
            },
        )


def run_r2_arm(
    frames: Iterable[CausalFrame],
    r0_config: DTRConfig | None = None,
    r1_config: R1Config = FROZEN_R1_CONFIG,
    r2_config: R2Config = FROZEN_R2_CONFIG,
) -> list[Prediction]:
    runner = DTRR2Arm(r0_config, r1_config, r2_config)
    return [runner.step(frame) for frame in frames]

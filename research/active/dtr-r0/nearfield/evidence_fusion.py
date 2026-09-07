"""Optional ground enhancement; unavailable height never vetoes a near alert.

Cell unions are diagnostic hypotheses, not verified object-height assignments.
User-facing output is directional, with ground height evidence kept separate.
"""
from dataclasses import dataclass
import numpy as np
from near_field import DIRECTIONS,HEIGHTS


@dataclass(frozen=True)
class BranchEvidence:
    alerts: np.ndarray
    states: tuple

    def __post_init__(self):
        alerts=np.asarray(self.alerts,dtype=bool).copy()
        states=np.asarray(self.states)
        if alerts.shape!=(3,3) or states.shape!=(3,3):
            raise ValueError("Require three direction by three height hypotheses")
        if not np.isin(states,["OBSTACLE","UNKNOWN","NO_NEAR_OBSERVED"]).all():
            raise ValueError("Invalid observation state")
        if not np.array_equal(alerts,states=="OBSTACLE"):
            raise ValueError("Alert and state disagree")
        alerts.flags.writeable=False
        object.__setattr__(self,"alerts",alerts)
        object.__setattr__(self,"states",tuple(tuple(r) for r in states.tolist()))

    @classmethod
    def from_encoder(cls,evidence):
        return cls(evidence.alerts(),evidence.region_state)


def fuse(raw:BranchEvidence,ground:BranchEvidence|None,*,mode="union"):
    """No weights, scene labels, score tuning or cross-frame state."""
    if mode not in ("union","conditional"):
        raise ValueError("Unknown fusion mode")
    a=raw.alerts
    b=ground.alerts if ground is not None else np.zeros((3,3),bool)
    ag=np.asarray(raw.states)
    bg=np.asarray(ground.states) if ground is not None else np.full((3,3),"UNKNOWN")
    if mode=="union":
        alerts=a|b
        unknown=(ag=="UNKNOWN")|(bg=="UNKNOWN")
    else:
        alerts=b.copy() if ground is not None else a.copy()
        unknown=(bg=="UNKNOWN") if ground is not None else (ag=="UNKNOWN")
    states=np.where(alerts,"OBSTACLE",np.where(unknown,"UNKNOWN","NO_NEAR_OBSERVED"))
    provenance=np.where(a&b,"BOTH",np.where(a,"RAW_ONLY",np.where(b,"GROUND_ONLY","NONE")))
    directions=[]
    used_raw=a if mode=="union" or ground is None else np.zeros_like(a)
    for d,name in enumerate(DIRECTIONS):
        positive=bool(alerts[d].any())
        # No-ground absence is not proof of absence. In particular, it must
        # not convert unavailable low-height sensing into full-direction clear.
        uncertain=ground is None or bool((states[d]=="UNKNOWN").any())
        state="OBSTACLE" if positive else "UNKNOWN" if uncertain else "NO_NEAR_OBSERVED"
        ground_bands=[HEIGHTS[k] for k in range(3) if b[d,k]]
        raw_unassigned=bool((a[d]&~b[d]).any()) and (mode=="union" or ground is None)
        directions.append(dict(direction=name,existence="OBSERVED" if positive else "NOT_ESTABLISHED",
            observation_state=state,range_state="NEAR" if positive else state,
            height_status="GROUND_REFERENCED_WITH_UNASSIGNED_EVIDENCE" if ground_bands and raw_unassigned else
                          "GROUND_REFERENCED" if ground_bands else "UNKNOWN",
            ground_height_bands=ground_bands,unassigned_raw_height_evidence=raw_unassigned,
            source="BOTH" if used_raw[d].any() and b[d].any() else "RAW_ONLY" if used_raw[d].any() else "GROUND_ONLY" if b[d].any() else "NONE"))
    return dict(mode=mode,diagnostic_alerts=alerts.tolist(),diagnostic_states=states.tolist(),
                provenance=provenance.tolist(),directions=directions,ground_available=ground is not None,
                height_set_disagreement=[DIRECTIONS[d] for d in range(3) if a[d].any() and b[d].any() and not np.array_equal(a[d],b[d])],
                physical_height_conflict="NOT_EVALUABLE_WITHOUT_SURFACE_ASSOCIATION",
                cell_score_scope="Branch height hypotheses only; not user-facing verified heights")

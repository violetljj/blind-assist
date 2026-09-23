"""Per-image query adapter, preserving the frozen contact model's parameterization."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from contact_boundary_model import ContactBoundaryModel


class ContactBoundarySamplingModel(ContactBoundaryModel):
    """Only query batching changes; inherited parameters and state keys are exact."""

    def forward(self, features: Tensor, queries: Tensor) -> Tensor:
        if queries.ndim == 2:
            return super().forward(features, queries)
        if queries.ndim != 3 or queries.shape[-1] != 3:
            raise ValueError("queries must have shape [Q,3] or [B,Q,3]")
        if features.ndim != 2 or queries.shape[0] != features.shape[0]:
            raise ValueError("per-image queries must match the feature batch")
        self._validate_queries(queries.reshape(-1, 3))
        scene = self.encode_scene(features)
        queries = queries.to(device=scene.device, dtype=scene.dtype)
        batch, count = queries.shape[:2]
        if self.mode == "direct":
            condition = torch.cat((queries[..., :1] / 1.2, queries[..., 1:2] / 3.0,
                                   F.one_hot(queries[..., 2].long(), 2).to(scene.dtype)), dim=-1)
            pair = torch.cat((scene[:, None, :].expand(batch, count, 128), condition), dim=-1)
            return self.direct_head(pair).squeeze(-1)
        intensity = self._geometry_intensity_from_scene(scene).flatten(1)
        overlap = self.query_overlap(queries.reshape(-1, 3)).reshape(batch, count, 1200)
        mass = torch.einsum("bi,bqi->bq", intensity, overlap)
        return mass + torch.log((-torch.expm1(-mass)).clamp_min(torch.finfo(mass.dtype).tiny))


def operator_counterexample() -> dict:
    """Tiny CPU linear-algebra diagnostic; no fit, observations or source labels.

    Original float32 query/edge coordinates are promoted to float64 for SVD.
    Also inspect an ideal decimal-grid operator to expose representational
    aliasing separately from inherited floating-point endpoint roundoff.
    """
    from contact_boundary_data import queries

    model = ContactBoundaryModel("geometry").double()
    old_query = torch.from_numpy(queries()["seen"]).double()
    original_operator = model.query_overlap(old_query).flatten(1)
    singular = torch.linalg.svdvals(original_operator)
    tolerance = max(original_operator.shape) * torch.finfo(torch.float64).eps * singular[0]
    grids = torch.full((2, 2, 20, 30), 1e-6, dtype=torch.float64)
    grids[0, 0, 6, 6] += 3.0  # BODY, |X| .18-.21, Z .6-.7
    grids[1, 0, 9, 8] += 3.0  # BODY, |X| .27-.30, Z .8-.9
    cells = [{"layer": "BODY", "x_index": x, "z_index": z,
              "actual_x_edges_m": model.width_edges[x:x+2].tolist(),
              "actual_z_edges_m": model.depth_edges[z:z+2].tolist()}
             for x, z in ((6, 6), (9, 8))]
    novel = torch.tensor([[.48, .75, 0.]], dtype=torch.float64)
    mass = grids.flatten(1) @ original_operator.T
    probability = -torch.expm1(-mass)
    novel_mass = grids.flatten(1) @ model.query_overlap(novel).flatten(1).T
    novel_probability = -torch.expm1(-novel_mass)
    threshold = .5281916856765747
    curves = queries()
    crossings = {}
    for kind in ("width", "horizon"):
        q = torch.from_numpy(curves[kind + "_curve"]).double()
        # float32(1.2) exceeds decimal 1.2 after promotion; restore only this
        # nominal domain endpoint before the inherited float64 validation.
        q[:, 0].clamp_(max=1.2)
        p = -torch.expm1(-(grids.flatten(1) @ model.query_overlap(q).flatten(1).T))
        rows = []
        for row in p:
            body = q[:, 2] == 0
            on = (row >= threshold) & body
            rows.append(float(q[on][0, 0 if kind == "width" else 1]) if on.any() else None)
        crossings[kind] = rows
    # Integer-scaled arithmetic represents the nominal decimal grid exactly.
    # Widths: 36/60/84/108 cm; |X| cells 3 cm; depth cells 10 cm.
    ideal_query = torch.tensor([(w, h, layer) for layer in range(2)
                                for w in (36, 60, 84, 108)
                                for h in (60, 90, 120, 150, 180, 210, 240, 270, 300)])
    x0 = torch.arange(20) * 6  # doubled centimetres, matches full query width
    z0 = torch.arange(30) * 10
    xf = ((torch.minimum(ideal_query[:, 0, None], x0 + 6) - x0).clamp_min(0) / 6).double()
    zf = ((torch.minimum(ideal_query[:, 1, None], z0 + 10)
           - torch.maximum(z0, torch.tensor(30))).clamp_min(0) / 10).double()
    layer = F.one_hot(ideal_query[:, 2], 2).double()
    ideal_operator = (layer[:, :, None, None] * xf[:, None, :, None] * zf[:, None, None, :]).flatten(1)
    ideal_mass = grids.flatten(1) @ ideal_operator.T
    return dict(schema="contact-boundary-operator-proof-v1", backend="cpu",
        backend_reason="TASK_NOT_GPU_SUITABLE", dtype="float64", shape=list(original_operator.shape),
        rank=int((singular > tolerance).sum()), rank_tolerance=float(tolerance),
        smallest_singular_value=float(singular[-1]), nominal_decimal_operator_rank=int(torch.linalg.matrix_rank(ideal_operator)),
        cells=cells, background_intensity=1e-6, added_intensity=3.0,
        original_max_mass_difference=float((mass[0]-mass[1]).abs().max()),
        original_max_probability_difference=float((probability[0]-probability[1]).abs().max()),
        nominal_decimal_max_mass_difference=float((ideal_mass[0]-ideal_mass[1]).abs().max()),
        novel_query=novel[0].tolist(), novel_probabilities=novel_probability[:, 0].tolist(),
        novel_probability_difference=float((novel_probability[0]-novel_probability[1]).abs().item()),
        threshold=threshold, novel_decisions=(novel_probability[:, 0] >= threshold).tolist(),
        original_threshold_decision_disagreements=int(((probability[0] >= threshold) != (probability[1] >= threshold)).sum()),
        body_threshold_crossings_m=crossings,
        interpretation="Nominal operator aliasing; actual old-coordinate residual is reported, not assumed zero. This is no learned-model performance claim.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = operator_counterexample()
    path = Path(__file__).resolve().parents[4] / "artifacts.local/evidence/ba-contact-sampling-20260923-proof.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))

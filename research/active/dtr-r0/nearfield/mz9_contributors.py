"""Evaluator/training-only contributors to the frozen clean zone returns.

Never pass these source counts, native XYZ or query counts into model inference.
An echo's contributing pixels can fall outside every body query. Their measured
depth, not the echo mean placed on a representative ray, determines membership.
"""
from functools import lru_cache

import torch

from multizone64_observation import _check, geometry, EVENT_ORDER
from mz8_attribution import query_membership


@lru_cache(maxsize=4)
def _cached_geometry(device):
    g = geometry(device)
    keep = g['crop'].flatten().nonzero().flatten()
    rays = g['rays'].reshape(-1, 3)[keep]
    zone = g['zone_ids'].flatten()[keep]
    az = torch.rad2deg(torch.atan(rays[:, 1]))
    el = torch.rad2deg(torch.atan(rays[:, 2]))
    # Midpoint quadrature49 cells: label the whole cell containing each source
    # pixel, rather than requiring a pixel to coincide with its midpoint ray.
    col = torch.floor(((az+22.5)/(45/8)-zone.remainder(8))*7).long().clamp(0, 6)
    row = torch.floor(((22.5-el)/(45/8)-zone.div(8, rounding_mode='floor'))*7).long().clamp(0, 6)
    return keep, rays, g['radial_factor'].flatten()[keep], zone, row*7+col


@torch.no_grad()
def reconstruct(depth):
    """Reproduce multi_surface packet and selected-bin contributor counts.

    depth: floating CUDA[B,360,640] axial metres, same calibration as MZ0.
    Returns range_m[B,64,2] float64 (NaN invalid), valid[B,64,2], int64
    source_counts[B,64,2,49], query_counts[B,64,2,49,4], cell_known_counts
    [B,64,49] independent of bin selection, selected_bins and
    event_order. Counts describe selected supported0.1m bins only, exactly the
    first/last bins producing the packet. A single supported bin has one echo.
    Parallel scatter sums may differ from observe by floating roundoff only.
    """
    _check(depth)
    keep, rays, factor, zone, subcell = _cached_geometry(depth.device)
    axial = depth.flatten(1)[:, keep].double()
    radial = axial*factor
    good = torch.isfinite(axial) & (axial > 0) & (axial < 100) & (radial <= 4)
    edges = torch.linspace(0., 4., 41, dtype=torch.float64, device=depth.device)
    bins = torch.bucketize(radial.contiguous(), edges[1:-1], right=True)
    address = zone[None]*40+bins
    batch = depth.shape[0]
    counts = torch.zeros(batch, 64*40, dtype=torch.int64, device=depth.device)
    counts.scatter_add_(1, address, good.long())
    sums = torch.zeros(batch, 64*40, dtype=torch.float64, device=depth.device)
    sums.scatter_add_(1, address, torch.where(good, radial, torch.zeros_like(radial)))
    counts, sums = counts.reshape(batch, 64, 40), sums.reshape(batch, 64, 40)
    supported = counts >= 3
    first = supported.int().argmax(-1)
    last = 39-supported.flip(-1).int().argmax(-1)
    any_bin = supported.any(-1)
    valid = torch.stack((any_bin, any_bin & (last != first)), -1)
    selected = torch.stack((first, last), -1)
    means = sums/counts.clamp_min(1)
    ranges = means.gather(2, selected).masked_fill(~valid, torch.nan)
    points = axial[..., None]*rays
    points[..., 2] += 1.7
    membership = query_membership(points)
    sources, queries = [], []
    cell_address = (zone*49+subcell)[None].expand(batch, -1)
    cell_known = torch.zeros(batch, 64*49, dtype=torch.int64, device=depth.device)
    cell_known.scatter_add_(1, cell_address, good.long())
    for echo in range(2):
        chosen = good & valid[:, zone, echo] & (bins == selected[:, zone, echo])
        count = torch.zeros(batch, 64*49, dtype=torch.int64, device=depth.device)
        count.scatter_add_(1, cell_address, chosen.long())
        sources.append(count.reshape(batch, 64, 49))
        per_query = []
        for q in range(4):
            count = torch.zeros(batch, 64*49, dtype=torch.int64, device=depth.device)
            count.scatter_add_(1, cell_address, (chosen & membership[..., q]).long())
            per_query.append(count.reshape(batch, 64, 49))
        queries.append(torch.stack(per_query, -1))
    return dict(range_m=ranges, valid=valid, source_counts=torch.stack(sources, 2),
                query_counts=torch.stack(queries, 2), selected_bins=selected,
                cell_known_counts=cell_known.reshape(batch, 64, 49),
                selected_bin_counts=counts.gather(2, selected).masked_fill(~valid, 0),
                event_order=EVENT_ORDER, interpretation='EVALUATOR_ONLY_ACTUAL_SOURCE_CONTRIBUTORS')

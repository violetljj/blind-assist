"""Learned matching for the ORBIT part memory (EXPLORE, 2026-10-16; plan: PLAN.md addendum 5).

The training-free method scores object o in a frame as the mean of the top-8 patch similarities S[t, o] to o's part
memory (k-means parts of the discriminative context patches). Two small learned pieces replace hand-set rules; both
are trained on the ORBIT *train* users only. The backbone stays frozen, the memory is still built training-free from
the user's clean videos, and nothing at test time needs boxes or masks:

  adapter  g(x) = normalise(x + U V x), rank r, U = 0 at start (identity)                         [optional, r > 0]
           S[t, o] = max_k cos(g(x_t), g(p_ok))   (query patches and memory parts in the learned matching space)
  a_t      = objectness logit of patch t: MLP(patch token) + learned per-position prior            [optional]
  C[t, o]  = S[t, o] - max_{o' != o} S[t, o']   (how much better o explains the patch than the user's other objects)
  e[t, o]  = S[t, o] + MLP([S, C, a])          (per patch-object evidence; MLP output layer starts at zero)
  pool_o   = tau * logsumexp_t(e[t, o] / tau) - tau * log T     (learned temperature: max ... mean)
  logit_o  = w0 * z(pool)_o + w1 * z(full)_o     (z over the user's objects, as the training-free fusion; w1 starts
                                                  at 0, so the untrained model is a soft top-k of S)

Training targets: the frame's object label (cross-entropy over the user's objects) and, optionally, the train-split
ORBIT boxes as patch-level targets (objectness BCE; in-box patch classification for the adapter).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    def __init__(self, D=384, T=256, hid=64, objectness=True, contrast=True, ehid=16, rank=0):
        super().__init__()
        self.objectness, self.contrast, self.T, self.rank = objectness, contrast, T, rank
        if rank:
            self.V = nn.Linear(D, rank, bias=False); self.U = nn.Linear(rank, D, bias=False); nn.init.zeros_(self.U.weight)
        if objectness:
            self.obj = nn.Sequential(nn.Linear(D, hid), nn.GELU(), nn.Linear(hid, 1))
            self.pos = nn.Parameter(torch.zeros(T))
        nin = 1 + int(contrast) + int(objectness)
        self.ev = nn.Sequential(nn.Linear(nin, ehid), nn.GELU(), nn.Linear(ehid, 1))
        nn.init.zeros_(self.ev[2].weight); nn.init.zeros_(self.ev[2].bias)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.02)))
        self.w = nn.Parameter(torch.tensor([3., 0.]))

    def g(self, x):
        x = x.float()
        return nn.functional.normalize(x + self.U(self.V(x)), dim=-1) if self.rank else x

    def sim_maps(self, X, parts, pvalid):
        """X [N,T,D]; parts [N,O,K,D] (each frame's own user's memory); pvalid [N,O] -> S [N,T,O] in g-space."""
        gx, gp = self.g(X), self.g(parts)
        s = torch.einsum('ntd,nokd->ntok', gx, gp).amax(3)
        return s.masked_fill(~pvalid[:, None, :], 0.)

    def objectness_logits(self, X):
        return self.obj(X.float()).squeeze(-1) + self.pos

    def forward(self, X, S, full, valid=None):
        """X [N,T,D] patch tokens (normalised, may be fp16); S [N,T,O] part similarities; full [N,O] whole-frame
        cosine; valid [N,O] bool for padded object slots. Returns (logits [N,O], objectness logits [N,T] or None)."""
        S = S.float(); full = full.float()
        if valid is None:
            valid = torch.ones(full.shape, dtype=torch.bool, device=full.device)
        feats = [S]
        if self.contrast:
            Sm = S.masked_fill(~valid[:, None, :], -2.)
            top2 = Sm.topk(min(2, S.shape[2]), dim=2).values
            other = top2[..., 1:2] if top2.shape[2] > 1 else torch.full_like(top2[..., :1], -2.)
            other = torch.where(Sm >= top2[..., :1], other, top2[..., :1])            # best of the *other* objects
            feats.append(S - other)
        a = None
        if self.objectness:
            a = self.objectness_logits(X)
            feats.append(a[..., None].expand_as(S))
        e = S + self.ev(torch.stack(feats, -1)).squeeze(-1)                             # [N,T,O]
        tau = self.log_tau.exp()
        pool = tau * torch.logsumexp(e.masked_fill(~valid[:, None, :], -1e4) / tau, dim=1) - tau * math.log(S.shape[1])
        logits = self.w[0] * zscore(pool, valid) + self.w[1] * zscore(full, valid)
        return logits.masked_fill(~valid, -1e4), a


def zscore(s, valid):
    """z-score over the valid object slots of each row (padding ignored)."""
    n = valid.sum(1, keepdim=True).clamp(min=1)
    mu = (s * valid).sum(1, keepdim=True) / n
    sd = (((s - mu) * valid) ** 2).sum(1, keepdim=True).div(n).sqrt()
    return ((s - mu) / (sd + 1e-6)) * valid


def load(path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    m = EvidenceHead(**ck['kw']).cuda().eval()
    m.load_state_dict(ck['state'])
    return m


def scores(m, fpat, S, full, allp=None, owner=None):
    """Inference inside orbit_dump: fpat [N,T,D] half; S [N,T,O] training-free similarity maps; full [N,O];
    allp [P,D] / owner [P] the user's memory parts (needed when the model has an adapter). Returns logits [N,O]."""
    out = []
    with torch.inference_mode():
        if m.rank:
            O = int(owner.max()) + 1
            gp = m.g(allp)
        for i in range(0, len(fpat), 64):
            Si = S[i:i + 64]
            if m.rank:
                sim = m.g(fpat[i:i + 64]) @ gp.T
                Si = torch.stack([sim[:, :, owner == o].amax(2) for o in range(O)], 2)
            out.append(m(fpat[i:i + 64], Si, full[i:i + 64])[0].float())
    return torch.cat(out)

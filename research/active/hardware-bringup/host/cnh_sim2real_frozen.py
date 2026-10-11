"""Observation-only frozen M3/old5/S replay under explicit stationary geometry.

Accept already normalized real CNH; never read labels, authored geometry, captures,
or protected outcomes. Two input paths must use the same precision. FP32 engineering
preserves finite real amplitudes, whereas the nominal simulated cache was FP16.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
NEAR = ROOT / 'research/active/dtr-r0/nearfield'
MANIFEST = ROOT / 'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json'
SEEDS = (2026100955, 2026100956, 2026100957)
M3_TAU = .8557642486787612
RAISED_TAU = .9404184587540165
LOCAL_TAU = 4.625390338985158


def segments(seq, ms):
    """Strict sequence continuity and 0 < sensor dt <= 500 ms; no gap filling."""
    seq, ms = np.asarray(seq), np.asarray(ms)
    if seq.ndim != 1 or ms.shape != seq.shape or not all(np.isfinite(a).all() for a in (seq, ms)):
        raise ValueError('Matching finite seq[N], ms[N] required')
    if any((a != np.floor(a)).any() for a in (seq, ms)):
        raise ValueError('Integral sequence and sensor time required')
    reset = np.r_[True, (np.diff(seq) != 1) | (np.diff(ms) <= 0) | (np.diff(ms) > 500)] if len(seq) else np.zeros(0, bool)
    return np.cumsum(reset)-1, reset


def temporal(raw, groups):
    """Frozen last-five exponential smoothing and raw linear trend, generalized
    to replay length. Only inference slots beginning at the fourth exposure enter
    this history, matching simulator f3..15; all histories reset at actual gaps.
    """
    smooth = np.full(raw.shape, np.nan, np.float64)
    slope, residual = np.zeros(raw.shape, np.float64), np.zeros(raw.shape, np.float64)
    for group in np.unique(groups):
        ids = np.flatnonzero((groups == group) & np.isfinite(raw).all(-1))
        for j, i in enumerate(ids):
            y = raw[ids[max(0, j-4):j+1]]
            weights = 2. ** np.arange(len(y))
            smooth[i] = (y.astype(np.float64)*weights[:, None]).sum(0)/weights.sum()
            x = np.arange(len(y), dtype=float); x -= x.mean()
            if len(x) > 1:
                slope[i] = (y*x[:, None]).sum(0)/(x*x).sum()
                residual[i] = np.sqrt(((y-y.mean(0)-slope[i]*x[:, None])**2).mean(0))
    return smooth, slope, residual


class FrozenReplay:
    """Owns frozen networks and cached stationary maps, no hardware or files.

    infer(z[N,8,8,16], seq[N], ms[N], check=...) returns per-frame scores,
    full8 readiness, four-exposure inference eligibility, and reset flags. Flags
    are masked by ready; raw scores preserve shorter warmup windows for audit.
    """
    def __init__(self, device='cuda', feature_precision='float32-engineering', batch_size=32):
        if feature_precision not in ('float16-frozen', 'float32-engineering'):
            raise ValueError('Explicit FP16 frozen or FP32 engineering precision required')
        if str(NEAR) not in sys.path:
            sys.path.insert(0, str(NEAR))
        import torch
        from cnh_cvr_pilot import CVR, rotation
        from cnh_cvr_projection import Projector, query_masks, SHAPE, SUB, EDGE, WIDTH
        import cnh_boundary_token_model as M
        import cnh_counterfactual_train_dev as T
        import cnh_bar_local_readout_dev as L
        import cnh_graded_peak_tracks_dev as P
        self.torch, self.M, self.T, self.L, self.P = torch, M, T, L, P
        self.device = torch.device(device)
        self.precision = feature_precision
        self.np_dtype = np.float16 if feature_precision == 'float16-frozen' else np.float32
        self.batch = int(batch_size)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.manifest = json.loads(MANIFEST.read_text())
        self.s_tau = float(self.manifest['threshold']['tau'])
        bindings = self.manifest['models'] + self.manifest['strong_source']['frozen_dependencies']
        # Validate used frozen models/thresholds, without opening unused evidence.
        hashes = {r['path']: r['sha256'] for r in bindings}
        self.used_hashes = {}
        def require(path):
            rel = path.relative_to(ROOT).as_posix()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes.get(rel) != actual:
                raise ValueError('Frozen dependency mismatch: '+rel)
            self.used_hashes[rel] = actual
            return path
        self.m3, self.ordinary, self.s_models = [], [], []
        for seed in range(5):
            path = require(ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{seed}.pt')
            model = CVR().to(self.device).eval()
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.m3.append(model)
        for seed in SEEDS:
            path = require(ROOT/f'artifacts.local/work/cnh-counterfactual-dev-20261009/models/ordinary_seed{seed}.pt')
            state = torch.load(path, map_location=self.device, weights_only=True)
            if state['arm'] != 'ordinary' or state['seed'] != seed or state['steps'] != 7488:
                raise ValueError('Frozen ordinary state mismatch')
            model = M.BoundaryTokenReadout().to(self.device).eval(); model.load_state_dict(state['state_dict'])
            self.ordinary.append(model)
            models = []
            for height in ('HEAD', 'BODY'):
                path = require(ROOT/f'artifacts.local/work/cnh-task-cost-retrain-dev-20261010/models/S_{seed}_{height}.pickle')
                with path.open('rb') as f: models.append(pickle.load(f))
            self.s_models.append(models)
        cuts = require(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
        cuts = json.loads(cuts.read_text()); self.single = np.array([cuts[str(s)]['single'] for s in SEEDS])
        self.transform = np.eye(4); self.transform[:3, :3] = rotation(-10, 'x')
        projector = Projector(device=self.device)
        # Exactly the frozen FP32 map contract (not live Engine's FP64 map).
        pts = projector.points.float(); volumes = projector.volumes.float()
        t = torch.as_tensor(self.transform, device=self.device, dtype=torch.float32)
        xyz = torch.bmm(pts[None]-t[None, None, :3, 3], t[None, :3, :3])[0]
        radius = torch.linalg.vector_norm(xyz, dim=1)
        ij = torch.floor((xyz[:, :2]/xyz[:, 2:3].clamp_min(1e-30)+EDGE)/(2*EDGE)*8).long()
        bins = torch.floor(radius/WIDTH).long()
        valid = (xyz[:, 2] > 0)&(ij >= 0).all(1)&(ij < 8).all(1)&(bins >= 0)&(bins < 16)
        self.index = (ij[:, 1].clamp(0, 7)*8+ij[:, 0].clamp(0, 7))*16+bins.clamp(0, 15)
        self.weight = valid*projector.voxel_volume/(SUB**3)/volumes[self.index]
        self.shape, self.sub = SHAPE, SUB
        self.coverage = valid.reshape(-1, SUB**3).float().mean(1).reshape(SHAPE)
        self.masks = torch.as_tensor(query_masks(), device=self.device)
        self.qs, self.locations = L.patches()
        from scipy import sparse
        rows = np.repeat(np.arange(np.prod(SHAPE)), SUB**3)
        ix, w = self.index.cpu().numpy(), self.weight.cpu().numpy()
        mapping = sparse.coo_matrix((w[w != 0].astype(float), (rows[w != 0], ix[w != 0])), shape=(np.prod(SHAPE), 1024)).tocsr()
        self.norms = {}
        for length in range(1, 9):
            fullmap = sparse.hstack([mapping]*length, format='csr')
            self.norms[length] = []
            for q in self.qs:
                projected = q@fullmap
                variance = np.asarray(projected.multiply(projected).sum(1)).ravel()
                self.norms[length].append(np.where(variance > 0, np.sqrt(variance), np.inf))
        self.geometry = {}
        for length in range(1, 9):
            transforms = np.repeat(np.eye(4)[None, None], 8, 1); transforms[0, -length:] = self.transform
            self.geometry[length] = M.feature_geometry(torch.as_tensor(transforms, device=self.device, dtype=torch.float32), torch.tensor([length], device=self.device), 'center')
        sensor = np.repeat(np.eye(4)[None], 16, 0); query = np.repeat(self.transform[None], 16, 0)
        self.current_geometry = P.native_geometry(sensor, query)

    def close(self):
        self.m3.clear(); self.ordinary.clear(); self.s_models.clear(); self.geometry.clear()
        self.index = self.weight = self.coverage = self.masks = None
        if self.device.type == 'cuda': self.torch.cuda.empty_cache()

    def infer(self, z, seq, ms, check=lambda: None):
        from cnh_temporal_readout_model import prepare_voxels
        from cnh_graded_peak_joint_features_dev import build
        from threadpoolctl import threadpool_limits
        began = time.monotonic(); check()
        z = np.asarray(z)
        if z.shape != (len(seq), 8, 8, 16) or not np.isfinite(z).all():
            raise ValueError('Finite z[N,8,8,16] required')
        z = z.astype(self.np_dtype)
        if not np.isfinite(z).all(): raise ValueError('Explicit feature precision overflow; no clipping/fallback')
        groups, reset = segments(seq, ms); n = len(z)
        starts = np.maximum.accumulate(np.where(reset, np.arange(n), 0))
        lengths = np.minimum(np.arange(n)-starts+1, 8)
        eligible, ready = lengths >= 4, lengths == 8
        raw = np.full((n, 2), np.nan, np.float32)
        local = np.full((n, 2), np.nan, np.float64)
        ordinary = np.full((3, n, 2), np.nan, np.float32)
        torch = self.torch
        with torch.inference_mode():
            for length in range(4, 9):
                ids = np.flatnonzero(eligible & (lengths == length))
                for offset in range(0, len(ids), self.batch):
                    check(); chosen = ids[offset:offset+self.batch]; b = len(chosen)
                    history = np.zeros((b, 8, 8, 8, 16), self.np_dtype)
                    indexes = chosen[:, None] + np.arange(1-length, 1)[None]
                    history[:, -length:] = z[indexes]
                    values = torch.as_tensor(history[:, -length:].copy(), device=self.device, dtype=torch.float32).reshape(b, length, 1024)
                    mass = values[:, :, self.index]*self.weight
                    evidence = mass.reshape(b, length, -1, self.sub**3).sum(3).reshape(b, length, *self.shape)
                    count = self.coverage[None].expand(length, -1, -1, -1).sum(0)
                    voxels = torch.stack((evidence.sum(1), count[None].expand(b, -1, -1, -1), evidence[:, -1]), 1)
                    voxels = voxels.half() if self.np_dtype == np.float16 else voxels.float()
                    if not bool(torch.isfinite(voxels).all()): raise ValueError('Projected feature overflow')
                    x = prepare_voxels(voxels, self.masks)
                    raw[chosen] = torch.stack([model(x).float() for model in self.m3]).mean(0).cpu().numpy()
                    local[chosen] = self.L.scan(voxels.cpu().numpy(), self.qs, self.norms[length], self.locations)[0]
                    le = torch.full((b,), length, device=self.device, dtype=torch.int64)
                    h = torch.as_tensor(history, device=self.device)
                    g = self.geometry[length].expand(b, -1, -1, -1, -1, -1, -1)
                    compact = self.M.build_features(h, g, le)[:, :, :, list(self.T.KEPT_CHANNELS)]
                    compact = compact.half() if self.np_dtype == np.float16 else compact.float()
                    features = self.T.restore_features(compact)
                    for si, model in enumerate(self.ordinary): ordinary[si, chosen] = model(features, le).cpu().numpy()
        check()
        smoothed, _, _ = temporal(raw, groups); local_smooth, _, _ = temporal(local, groups)
        # Repack independent current observations into the inherited 13-slot
        # extractor. No temporal association or fabricated observation is used.
        slots = np.zeros(((n+12)//13, 16, 8, 8, 16), self.np_dtype)
        slots[:, 3:] = np.pad(z, ((0, (-n)%13), (0, 0), (0, 0), (0, 0))).reshape(-1, 13, 8, 8, 16)
        candidates = self.P.current_candidates(slots, self.current_geometry)
        current, validity = build(dict(candidate_amplitude=candidates['amplitude'], candidate_valid=candidates['valid'], candidate_inner_share=candidates['inner_share'], candidate_xyz=candidates['xyz'], candidate_support_low_xyz=candidates['support_low_xyz'], candidate_support_high_xyz=candidates['support_high_xyz']))
        current = current.reshape(-1, 2, 22)[:n].astype(float); validity = validity.reshape(-1, 2, 22)[:n]&np.isfinite(current)
        spatial = np.concatenate((np.where(validity, current, np.nan), (~validity).astype(float)), -1)
        learned = np.full((3, n, 2), np.nan, np.float32)
        with threadpool_limits(limits=2):
            for si in range(3):
                osmooth, slope, residual = temporal(ordinary[si], groups)
                features = np.concatenate((np.stack((osmooth-self.single[si], slope, residual), -1), spatial), -1).astype(np.float32)
                for q in range(2):
                    check(); learned[si, eligible, q] = self.s_models[si][q].decision_function(features[eligible, q])
        s_score = learned.mean(0)
        old5 = (smoothed >= RAISED_TAU)|(local_smooth >= LOCAL_TAU)
        m3 = smoothed >= M3_TAU; sensemble = old5|(s_score >= self.s_tau)
        return dict(m3_raw=raw, m3_score=smoothed, local_raw=local, local_score=local_smooth, ordinary_raw=ordinary, s_seed_score=learned, s_score=s_score, m3=m3&ready[:, None], old5=old5&ready[:, None], Sensemble=sensemble&ready[:, None], S_grade=np.where(ready[:, None], np.where(old5, 2, np.where(s_score >= self.s_tau, 1, 0)), 0).astype(np.int8), ready=ready, eligible=eligible, reset=reset, segment=groups, history=lengths, seconds=np.array(time.monotonic()-began))

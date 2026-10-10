"""Frozen M3 on stationary real CNH, with an explicit engineering residual input.

The reference is the current scene, potentially containing obstacles. Subtracting
its mean measures scene changes, unlike M3's synthetic crosstalk-only subtraction.
Neither these standardized residuals nor the retained synthetic threshold are a
physical SNR calibration, a probability, or established real obstacle detection.
"""
from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
NEAR = ROOT / 'research/active/dtr-r0/nearfield'
MODEL_DIRECTORY = ROOT / 'artifacts.local/work/cnh-margin-labels-20261002/models/M3'
MODEL_HASHES = (
    'abda90d193390e5743c0cebd0d942231ac929f171a2b0452845c47c911247eb7',
    'f97ab9b831ab698ef3d9b14df2504dfbd8df16bf74425148bb6c126d55cc5819',
    '91ef643e6185a9ee577939bc2772cc208b2a78a4c8a6189e916a76218d924647',
    'a9e71dc648972a0a0eb9014c9e939e1b111f382ab5e879193d07d5181e3bbca4',
    '884bea0c6dc207c8b33c75dfad8deb1b74ad5973c13a2175d66b1a9d51413d9d',
)
THRESHOLD = 0.8557642486787612
QUERY_ORDER = ('HEAD', 'BODY')
INPUT_MODE = 'current-scene-background-residual-engineering'


def _histogram(value):
    h = np.asarray(value, dtype=np.float64)
    if h.shape == (64, 16):
        h = h.reshape(8, 8, 16)
    if h.shape != (8, 8, 16) or not np.isfinite(h).all():
        raise ValueError('Finite decoded CNH [8,8,16] or [64,16] required')
    return h


class Engine:
    """A stationary bench adapter; owns no hardware, files, capture or workers."""

    def __init__(self, device='cuda', pitch=-10, feature_precision='float16-frozen'):
        if not np.isfinite(pitch):
            raise ValueError('Finite nominal mount pitch required')
        if feature_precision not in ('float16-frozen', 'float32-engineering'):
            raise ValueError('Select float16-frozen or float32-engineering explicitly')
        self.feature_precision = feature_precision
        self.input_dtype = np.float16 if feature_precision == 'float16-frozen' else np.float32
        if str(NEAR) not in sys.path:
            sys.path.insert(0, str(NEAR))
        import torch
        from cnh_cvr_pilot import CVR, rotation
        from cnh_cvr_projection import Projector, query_masks
        from cnh_temporal_readout_model import prepare_voxels

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; select CPU explicitly')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.pitch = float(pitch)
        self.transform = np.eye(4)
        self.transform[:3, :3] = rotation(self.pitch, 'x')
        self.models = []
        self.model_hashes = {}
        for seed, expected in enumerate(MODEL_HASHES):
            path = MODEL_DIRECTORY / f'model_seed{seed}.pt'
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Frozen M3 checkpoint identity mismatch: {path}')
            self.model_hashes[str(path)] = actual
            model = CVR().to(self.device).eval()
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.models.append(model)
        self.projector = Projector(device=self.device)
        self.masks = torch.as_tensor(query_masks(), device=self.device)
        self.prepare_voxels = prepare_voxels
        self.mean = self.std = None
        self.reference = None
        self.history = deque(maxlen=8)
        self.logits = deque(maxlen=5)
        self.last = None
        self.resets = 0

    def reset(self):
        """Clear causal history and scores while retaining the reference."""
        self.history.clear()
        self.logits.clear()
        self.last = None
        self.resets += 1

    def clear_reference(self):
        self.mean = self.std = self.reference = None
        self.reset()

    def normalized(self, hist):
        """Engineering residual before selected caching; None without a reference."""
        h = _histogram(hist)
        if self.reference is None:
            return None
        z = (h - self.mean) / self.std
        with np.errstate(over='ignore', invalid='ignore'):
            stored = z.astype(self.input_dtype)
        if not np.isfinite(stored).all():
            raise ValueError(f'Engineering residual overflows {np.dtype(self.input_dtype).name}; no clipping or fabricated score')
        return z

    def close(self):
        """Release only this adapter's model and projection tensors."""
        self.reset()
        self.models.clear()
        self.projector = None
        self.masks = None
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()

    def set_reference(self, histograms):
        """Set a scene reference; caller controls stationary acquisition and lineage.

        Forty-frame probes are allowed. This routine does not infer that the scene
        is obstacle-free. Sample SD uses ddof=1; every bin has an explicit floor of
        max(median(positive sample SD)*1e-3, 1e-9 decoded CNH units).
        """
        h = np.asarray([_histogram(item) for item in histograms], dtype=np.float64)
        if len(h) < 2:
            raise ValueError('At least two actual reference frames required')
        mean = h.mean(axis=0)
        std = h.std(axis=0, ddof=1)
        positive = std[std > 0]
        floor = max(float(np.median(positive)) * 1e-3 if len(positive) else 0., 1e-9)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError('Reference statistics must be finite')
        self.mean, self.std = mean, np.maximum(std, floor)
        self.reference = dict(
            frames=int(len(h)),
            decoded_histograms_sha256=hashlib.sha256(np.ascontiguousarray(h, dtype='<f8').tobytes()).hexdigest(),
            standard_deviation_ddof=1,
            std_floor=floor,
            std_floor_rule='max(median(positive sample SD)*1e-3, 1e-9 decoded CNH units)',
            floored_bins=int((std < floor).sum()),
            median_standard_deviation=float(np.median(std)),
            scene_mean_subtracted=True,
            physical_noise_or_snr_calibration=False,
            obstacle_free_scene_verified=False,
        )
        self.reset()
        return dict(self.reference)

    def metadata(self):
        return dict(
            algorithm='frozen M3 five-seed CVR', model_hashes=dict(self.model_hashes),
            device=str(self.device), query_order=list(QUERY_ORDER), nominal_pitch_deg=self.pitch,
            stationary_sensor=True, relative_transport='identity; fixed nominal mount',
            past_frames=8, feature_precision=self.feature_precision,
            native_z_dtype=f'{np.dtype(self.input_dtype).name} engineering residual',
            projection='float64 geometry / float32 sequential accumulation / '+np.dtype(self.input_dtype).name+' voxel cache',
            simulation_precision_match=self.feature_precision == 'float16-frozen',
            precision_scope=('Retained simulation FP16 representation' if self.feature_precision == 'float16-frozen'
                             else 'Explicit FP32 engineering representation avoids FP16 range overflow; no clipping or automatic fallback'),
            channels=['total residual mass', 'coverage count', 'last residual mass'],
            network_scaling='signed log1p(total,last), coverage/8, two retained query masks',
            smoothing='causal last-five logits; oldest-to-newest weights 1,2,4,8,16',
            reference_threshold=THRESHOLD, threshold_operator='>=', threshold_domain='smoothed logit',
            threshold_scope='Retained simulation reference only; not calibrated on real CNH',
            input_mode=INPUT_MODE, reference=dict(self.reference) if self.reference else None,
            input_scope='Current-scene residual; existing reference surfaces are subtracted',
            score_scope='Engineering model response; not probability or obstacle accuracy',
        )

    def step(self, hist, seq, ms):
        """Consume exactly one actual frame; gaps reset rather than fill histories."""
        started = time.perf_counter()
        h = _histogram(hist)
        if isinstance(seq, bool) or isinstance(ms, bool) or int(seq) != seq or int(ms) != ms:
            raise ValueError('Integer sensor sequence and millisecond timestamp required')
        seq, ms = int(seq), int(ms)
        reason = None
        if self.last is not None:
            previous_seq, previous_ms = self.last
            if seq != previous_seq + 1:
                reason = 'sequence-discontinuity'
            elif not 0 < ms - previous_ms <= 500:
                reason = 'timestamp-discontinuity-or-gap-over-500ms'
        if reason:
            self.reset()
        self.last = (seq, ms)
        result = dict(seq=seq, ms=ms, input_mode=INPUT_MODE, reset=bool(reason), reset_reason=reason,
                      ready=False, history=0, raw=None, smoothed=None, reference_crossing=None,
                      threshold=THRESHOLD, threshold_is_simulation_reference=True,
                      query_order=list(QUERY_ORDER), maxz=None, max_abs_residual=None,
                      zone_max_residual=None, reference_frames=self.reference['frames'] if self.reference else 0)
        if self.reference is None:
            result.update(status='REFERENCE_REQUIRED', compute_ms=(time.perf_counter()-started)*1000)
            return result
        try:
            z = self.normalized(h)
        except ValueError:
            self.reset()
            raise
        self.history.append(z.astype(self.input_dtype))
        torch = self.torch
        with torch.inference_mode():
            transforms = np.repeat(self.transform[None], len(self.history), axis=0)
            voxels = self.projector.sequence(np.asarray(self.history), transforms)
            voxels = (voxels.half() if self.feature_precision == 'float16-frozen' else voxels.float())[None]
            if not bool(torch.isfinite(voxels).all()):
                self.reset()
                raise ValueError(f'M3 {np.dtype(self.input_dtype).name} projected features are nonfinite')
            x = self.prepare_voxels(voxels, self.masks)
            raw = torch.stack([model(x).float() for model in self.models]).mean(0)[0].cpu().numpy()
        if not np.isfinite(raw).all():
            self.reset()
            raise ValueError('Frozen M3 produced nonfinite logits')
        self.logits.append(raw)
        weights = 2. ** np.arange(len(self.logits))
        smoothed = (np.asarray(self.logits) * weights[:, None]).sum(axis=0) / weights.sum()
        ready = len(self.history) == 8
        result.update(status='READY' if ready else 'WARMING_UP', ready=ready, history=len(self.history),
                      feature_precision=self.feature_precision,
                      raw=[float(v) for v in raw], smoothed=[float(v) for v in smoothed],
                      reference_crossing=[bool(v >= THRESHOLD) for v in smoothed] if ready else None,
                      maxz=float(z.max()), max_abs_residual=float(np.abs(z).max()),
                      zone_max_residual=z.max(axis=-1).tolist(),
                      compute_ms=(time.perf_counter()-started)*1000)
        return result

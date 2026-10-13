"""Infer OctoNet VL53L8CH CNH bin width/offset from its own data (static pickle decode, no pickle.load).

Regresses the reported per-zone depth (unknown unit) on the sub-bin histogram peak position.
A slope of k*37.5 mm/bin identifies binning factor k (datasheet: core bin 37.5 mm); ours is 8 (0.3003 m/bin, 16 bins).
"""
import io, json, math, pickletools, struct, sys
from pathlib import Path
import numpy as np

src = Path(sys.argv[1]); out = Path(sys.argv[2])
raw = src.read_bytes(); stream = io.BytesIO(raw); H, Dp = [], []
while stream.tell() < len(raw):
    ops = list(pickletools.genops(stream))
    byteargs = [arg for op, arg, pos in ops if isinstance(arg, bytes)]
    assert list(map(len, byteargs)) == [10, 1, 9216, 512]
    H.append([v[0] for v in struct.iter_unpack('<d', byteargs[2])]); Dp.append([v[0] for v in struct.iter_unpack('<q', byteargs[3])])
H = np.array(H).reshape(-1, 64, 18); Dp = np.array(Dp, float).reshape(-1, 64)
k = H[:, :, 2:].argmax(2) + 2   # skip bins 0-1 (near crosstalk dominates bin 0)
def centroid(h, k):
    i = np.clip(k, 1, 16); l, c, r = [np.take_along_axis(h, (i + o)[..., None], 2)[..., 0] for o in (-1, 0, 1)]
    den = l - 2 * c + r
    return i + np.where(np.abs(den) > 1e-9, .5 * (l - r) / den, 0.)
peak = centroid(H, k)
res = {}
for name, perm in {'same': np.arange(64), 'transpose': np.arange(64).reshape(8, 8).T.ravel(),
                   'flip_rows': np.arange(64).reshape(8, 8)[::-1].ravel(), 'rot180': np.arange(64)[::-1]}.items():
    d = Dp[:, perm]; m = (d > 0) & (H.max(2) > 0) & (k >= 2) & (k <= 16)
    x, y = peak[m], d[m]
    A = np.stack([x, np.ones_like(x)], 1); (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - (a * x + b); keep = np.abs(r) < 3 * np.median(np.abs(r)) * 1.4826 + 1e-9
    (a2, b2), *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    res[name] = dict(n=int(m.sum()), corr=float(np.corrcoef(x, y)[0, 1]), slope=float(a2), intercept=float(b2),
                     inlier=float(keep.mean()), resid_mad=float(np.median(np.abs(y[keep] - (a2 * x[keep] + b2)))))
best = max(res, key=lambda n: res[n]['corr'])
s = res[best]['slope']
summary = dict(source=src.name, frames=len(H), zone_mapping=res, best=best,
               binning_factor_if_mm=s / 37.5348, peak_bin_hist=np.bincount(k.ravel(), minlength=18).tolist(),
               depth_percentiles=np.percentile(Dp[Dp > 0], [5, 50, 95]).tolist(),
               note='Empirical only; assumes depth in mm and core bin 37.5348 mm; no ambient/status fields exist in this file.')
out.write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))

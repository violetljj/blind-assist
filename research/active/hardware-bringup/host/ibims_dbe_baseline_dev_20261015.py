"""iBims-1 baseline for the depth-boundary thesis route (EXPLORE, 2026-10-15; plan: <work>/PLAN_BOUNDARY.md).

Runs Depth Anything V2 variants on the 100 iBims-1 core images (640x480) and scores them with the official
evaluation functions (evaluate_ibims_error_metrics.py from the dataset's evaluation_scripts.zip, unmodified):
global errors, depth boundary error (DBE acc/comp, px) and planarity error, following evaluate_ibims.py masking.
  dav2s_rel  Depth-Anything-V2-Small relative disparity, least-squares scale+shift to GT inverse depth
  dav2s_hyp  Depth-Anything-V2-Metric-Hypersim-Small, per-image median scale (raw metric reported too)
  dav2l_hyp  Depth-Anything-V2-Metric-Hypersim-Large, same
  gt         ground truth itself (sanity of the scorer)
Note from the official script: its Python Canny differs from the paper's Matlab one, so DBE values are only comparable
between runs of this same scorer, not with the paper's table.
Usage: ibims_dbe_baseline_dev_20261015.py <ibims dir with *.mat, imagelist.txt, eval/evaluation_scripts> <out json> [models]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
UP = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream/Depth-Anything-V2'
MODELS = ROOT / 'artifacts.local/models'
CFG = {'s': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
       'l': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])}


def load(name):
    import torch
    for k in [k for k in sys.modules if k.startswith('depth_anything_v2')]:   # relative and metric packages share a name
        del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'Depth-Anything-V2' not in p]
    if name == 'dav2s_rel':
        sys.path.insert(0, str(UP))
        from depth_anything_v2.dpt import DepthAnythingV2
        m = DepthAnythingV2(**CFG['s'])
        sd = torch.load(MODELS / 'depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True)
    else:
        sys.path.insert(0, str(UP / 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
        size = name[4]
        m = DepthAnythingV2(**CFG[size], max_depth=20)
        sd = torch.load(MODELS / f'depth-anything--Depth-Anything-V2-Metric-Hypersim-{"Small" if size == "s" else "Large"}/'
                        f'depth_anything_v2_metric_hypersim_vit{size}.pth', map_location='cpu', weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m.cuda().eval(), sum(p.numel() for p in m.parameters())


def predict(model, bgr):
    import torch
    import torch.nn.functional as Fn
    t, _ = model.image2tensor(bgr, 518)
    with torch.inference_mode():
        out = model(t)
    return Fn.interpolate(out[:, None], bgr.shape[:2], mode='bilinear', align_corners=True)[0, 0].cpu().numpy().astype(np.float64)


def main(ibims, out, models='gt,dav2s_rel,dav2s_hyp,dav2l_hyp'):
    import torch
    from scipy import io
    ibims = Path(ibims)
    sys.path.insert(0, str(ibims / 'eval/evaluation_scripts'))
    from evaluate_ibims_error_metrics import compute_global_errors, compute_depth_boundary_error, compute_planarity_error
    names = [x.strip() for x in (ibims / 'imagelist.txt').read_text().splitlines() if x.strip()]
    data = {}
    for n in names:
        d = io.loadmat(str(ibims / 'ibims1_core_mat' / n))['data']
        data[n] = {k: d[k][0][0] for k in ('rgb', 'depth', 'edges', 'calib', 'mask_invalid', 'mask_transp',
                                            'mask_wall', 'mask_wall_paras', 'mask_table', 'mask_table_paras',
                                            'mask_floor', 'mask_floor_paras')}
    res = {}
    for name in models.split(','):
        model, npar = (None, 0) if name == 'gt' else load(name)
        per, raw_rel, times = [], [], []
        for n in names:
            D = data[n]; gt = D['depth'].astype(np.float64)
            valid = (D['mask_transp'] * D['mask_invalid'] * (gt != 0)) > 0
            if name == 'gt':
                pred = gt.copy()
            else:
                bgr = np.ascontiguousarray(D['rgb'][:, :, ::-1])
                torch.cuda.synchronize(); t0 = time.time()
                y = predict(model, bgr)
                torch.cuda.synchronize(); times.append(time.time() - t0)
                if name == 'dav2s_rel':                                # disparity: LS scale+shift to GT inverse depth
                    A = np.stack([y[valid], np.ones(valid.sum())], 1)
                    s, b = np.linalg.lstsq(A, 1. / gt[valid], rcond=None)[0]
                    disp = s * y + b
                    pred = np.where(disp > 1e-3, 1. / np.maximum(disp, 1e-3), 0.)
                else:
                    raw_rel.append(float(np.mean(np.abs(y[valid] - gt[valid]) / gt[valid])))
                    pred = y * np.median(gt[valid] / np.maximum(y[valid], 1e-6))
            pred = np.where(np.isfinite(pred) & (pred > 0), pred, 0.)
            pv = pred > 0; mv = valid & pv
            g, p = gt * mv, pred * mv
            rel, sq, rms, log10, t1, t2, t3 = compute_global_errors(g.flatten(), p.flatten())
            acc, com = compute_depth_boundary_error(D['edges'], pred.copy())
            pf, po = [], []
            for kind in ('wall', 'table', 'floor'):
                paras = D[f'mask_{kind}_paras']
                if paras.size:
                    a, o = compute_planarity_error(g.copy(), p.copy(), paras, D[f'mask_{kind}'] * mv, D['calib'])
                    pf += list(a); po += list(o)
            per.append(dict(img=n, rel=float(rel), rms=float(rms), log10=float(log10), d1=float(t1), dbe_acc=float(acc),
                            dbe_com=float(com), pe_fla=pf, pe_ori=po))
        pf = np.concatenate([np.array(r['pe_fla'], float) for r in per]); po = np.concatenate([np.array(r['pe_ori'], float) for r in per])
        res[name] = dict(
            params=npar, n=len(per),
            **{k: round(float(np.nanmean([r[k] for r in per])), 4) for k in ('rel', 'rms', 'log10', 'd1', 'dbe_acc', 'dbe_com')},
            pe_fla=round(float(np.nanmean(pf)), 3), pe_ori=round(float(np.nanmean(po)), 3),
            raw_metric_rel=round(float(np.mean(raw_rel)), 4) if raw_rel else None,
            ms_per_img_gpu=round(1000 * float(np.median(times)), 1) if times else None, per_image=per)
        print(name, {k: v for k, v in res[name].items() if k != 'per_image'}, flush=True)
        del model; torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(res), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:])

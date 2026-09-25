"""Frozen, obstacle-free volume quadrature for Track A. No sensor/data access."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc


def rx(deg):
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rz(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    poses = qmc.Sobol(2, scramble=True, seed=20260925).random_base2(8)
    poses = poses * [25, 20] + [-15, -10]
    rotations = np.array([rx(p) @ rz(r) for p, r in poses])
    cells = (np.arange(16) + .5) / 16
    unit = np.stack(np.meshgrid(cells, cells, cells, indexing='ij'), -1).reshape(-1, 3)
    rows, scores = [], []
    for mount in range(0, -31, -5):
        rot = rotations @ rx(mount)
        groups = {}
        for zi, (loz, hiz) in enumerate(((.3, 1), (1, 2), (2, 3))):
            for group, (loy, hiy) in {'HEAD': (-.2, .42), 'BODY': (.42, .9), 'HEAD_ABOVE': (-.2, 0)}.items():
                fractions = []
                for side, cx in zip(('L', 'C', 'R'), (-.3, 0, .3)):
                    pts = unit * [.6, hiy-loy, hiz-loz] + [cx-.3, loy, loz]
                    local = np.einsum('pi,sij->spj', pts, rot)
                    visible = (local[..., 2] > 0) & (np.abs(local[..., :2]) <= local[..., 2:3]*np.tan(np.pi/8)).all(-1)
                    f = visible.mean(-1)
                    fractions.append(f)
                    rows.append(dict(mount=mount, group=group, side=side, z=[loz, hiz], median=float(np.median(f)), p10=float(np.quantile(f,.1)), p90=float(np.quantile(f,.9))))
                f = np.mean(fractions, axis=0)
                groups[group, zi] = float(np.median(f))
                rows.append(dict(mount=mount, group=group, side='ALL', z=[loz, hiz], median=float(np.median(f)), p10=float(np.quantile(f,.1)), p90=float(np.quantile(f,.9))))
        ch, cb = groups['HEAD',1], groups['BODY',1]
        scores.append(dict(mount=mount, cH=ch, cB=cb, score=min(ch,cb)))
    best = sorted(scores, key=lambda r:(-r['score'], abs(r['mount']+15), abs(r['mount'])))[0]
    ref = next(r for r in scores if r['mount']==-15)
    selected = -15 if best['score']-ref['score'] < .05 else best['mount']
    result = dict(schema='cnh.track-a.fov.v1', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), pose_count=256, points_per_box_slice=4096, seed=20260925, rows=rows, selection=scores, argmax=best['mount'], advantage=best['score']-ref['score'], M1=selected)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines = ['# Track A 纯几何视场覆盖结果', '', '256个固定Sobol姿态；每盒/距离段4096个等体积中点。体积覆盖不含遮挡。', '', '|俯仰|HEAD [1,2)中位数|BODY [1,2)中位数|min|', '|---:|---:|---:|---:|']
    for r in scores:
        lines.append(f"|{r['mount']}|{r['cH']:.6f}|{r['cB']:.6f}|{r['score']:.6f}|")
    lines += ['', f"argmax={best['mount']}°，相对-15°增量={result['advantage']:.6f}；冻结规则选 M1={selected}°，M0=0°。仅为模拟对照，产品角度未定。", '', '以下每格为 median [p10,p90]；三个横向盒ALL是等体积合并后取姿态分位数。', '', '|俯仰|组|横向|Z(m)|覆盖|', '|---:|---|---|---|---|']
    for r in rows:
        lines.append(f"|{r['mount']}|{r['group']}|{r['side']}|{r['z']}|{r['median']:.4f} [{r['p10']:.4f},{r['p90']:.4f}]|")
    (out/'coverage.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax,group in zip(axes,('HEAD','BODY','HEAD_ABOVE')):
        for mount in range(0,-31,-5):
            rr=[r for r in rows if r['mount']==mount and r['group']==group and r['side']=='ALL']
            ax.plot([.65,1.5,2.5],[r['median'] for r in rr],label=f'{mount} deg',marker='o')
        ax.axhline(.3,color='gray',linestyle='--'); ax.set_title(group); ax.set_xlabel('Forward distance (m)'); ax.set_ylim(0,1)
    axes[0].set_ylabel('Median volume coverage'); axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'coverage.png',dpi=160); plt.close(fig)
    print(json.dumps({k:result[k] for k in ('selection','M1','advantage')}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)

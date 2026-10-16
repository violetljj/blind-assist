"""Thesis figures: what the part memory keeps, and where the zoomed second look goes (EXPLORE, 2026-10-16).

Usage: orbit_figures.py <split dir> <out dir> <user> [n_clutter=6] [zoom=0.5] [seed=0] [loc=<loc_all.pt>]
  <user>_memory.png   context frames of several objects; the kept discriminative patches are highlighted (patches
                      that look like the object's other frames but unlike the user's other objects)
  <user>_clutter.png  clutter frames: evidence map (max over the user's objects of the part similarity), the
                      annotated box (dashed, figure only - never used by the method), the zoom box from the evidence
                      peak (yellow) and, with loc, from the learned localizer (cyan), and the predicted object from the
                      whole frame and after each second look
Examples are picked at random (seeded) among valid frames, not selected for success.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main(root, out, user, n_clutter=6, zoom=0.5, seed=0, loc=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    from orbit_dump import fnum, guard, part_memory_from, per_object, read_gpu, zoom_boxes
    from orbit_eval import CFG, forward
    from orbit_proto_pilot import load
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    guard()
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True); n_clutter, zoom = int(n_clutter), float(zoom)
    rng = random.Random(int(seed))
    plan = json.loads((root / 'plan.json').read_text()); vids = [v for v in plan['videos'] if v['user'] == user]
    objs = sorted({v['obj'] for v in vids if v['type'] == 'clean'})
    ctx = {o: [f for v in vids if v['type'] == 'clean' and v['obj'] == o for f in v['context'] if fnum(f) <= 1000] for o in objs}
    names = [f for o in objs for f in ctx[o]]; L = torch.as_tensor([i for i, o in enumerate(objs) for _ in ctx[o]]).cuda()
    m, size, mean, std = load('dinov2_s14'); m = m.to(memory_format=torch.channels_last)
    xc = read_gpu(root / 'frames', names)
    _, p1 = forward(m, size, mean, std, xc, bs=192)
    grid = int(round(p1.shape[1] ** .5)); cfg = {**CFG, 'keep': 0.15}
    keeps = []
    mem = part_memory_from([p1[L == i].reshape(-1, p1.shape[-1]) for i in range(len(objs))], cfg, 'mixed', keep_out=keeps)
    allp = torch.cat(mem).half(); owner = torch.cat([torch.full((len(q),), i) for i, q in enumerate(mem)]).cuda()

    # memory figure: up to 4 objects x 3 context frames
    show = objs[:4]; fig, ax = plt.subplots(len(show), 3, figsize=(7.5, 2.6 * len(show)))
    ax = np.atleast_2d(ax)
    for r, o in enumerate(show):
        i = objs.index(o); km = keeps[i].view(-1, grid, grid).float()
        idx = torch.nonzero(L == i).squeeze(1)
        for c, k in enumerate(np.linspace(0, len(idx) - 1, 3).round().astype(int)):
            img = xc[idx[k]].permute(1, 2, 0).cpu().numpy() / 255.
            mk = F.interpolate(km[k][None, None], img.shape[:2], mode='nearest')[0, 0].numpy()
            shade = img * (0.35 + 0.65 * mk[..., None]); shade[..., 0] = np.maximum(shade[..., 0], 0.9 * mk)
            ax[r, c].imshow(shade); ax[r, c].axis('off')
        ax[r, 0].set_title(f'{o}：保留 {100 * float(keeps[i].float().mean()):.0f}% 图块', fontsize=9, loc='left')
    fig.suptitle(f'{user}：局部记忆保留的判别图块（红）', fontsize=11); fig.tight_layout()
    fig.savefig(out / f'{user}_memory.png', dpi=150); plt.close(fig)
    del xc, p1; torch.cuda.empty_cache()

    # clutter figure
    pool = [(v, f) for v in vids if v['type'] == 'clutter' for f in v['valid'] if fnum(f) <= 1000 and v['boxes'].get(f)]
    pick = rng.sample(pool, min(n_clutter, len(pool)))
    x = read_gpu(root / 'frames', [f for _, f in pick])
    cls, pat = forward(m, size, mean, std, x, bs=64)
    S = per_object(pat @ allp.T, owner, len(objs), 'mixed'); part = S.topk(8, dim=1).values.mean(1)
    side = x.shape[-1]

    def z(s):
        return (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)

    def look(bx):
        crop = roi_align(x.float(), torch.cat([torch.arange(len(x), device=bx.device)[:, None].float(), bx], 1), (size, size), 1.0, 2, True)
        _, zpat = forward(m, size, mean, std, crop, bs=64)
        return z(part) + z(per_object(zpat @ allp.T, owner, len(objs), 'mixed').topk(8, dim=1).values.mean(1))
    bx = zoom_boxes(S, grid, side, zoom); fused = look(bx)
    if loc:
        import train_localizer as TL
        with torch.inference_mode():
            lbx = zoom_boxes(TL.load_loc(loc)(pat, S)[..., None], grid, side, zoom)
        lfused = look(lbx)
    cols = 3; rows = (len(pick) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.1 * rows)); ax = np.atleast_1d(ax).ravel()
    for k, (v, f) in enumerate(pick):
        img = x[k].permute(1, 2, 0).cpu().numpy() / 255.
        ev = F.interpolate(S[k].amax(1).view(1, 1, grid, grid), img.shape[:2], mode='bilinear')[0, 0].cpu().numpy()
        ax[k].imshow(img); ax[k].imshow(ev, cmap='inferno', alpha=0.45); ax[k].axis('off')
        b = v['boxes'][f]; sc = side / 1080
        ax[k].add_patch(plt.Rectangle((b['x'] * sc, b['y'] * sc), b['w'] * sc, b['h'] * sc, fill=False, ec='lime', ls='--', lw=1.2))
        x0, y0, x1, y1 = bx[k].tolist()
        ax[k].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='yellow', lw=1.8))
        gt = objs.index(v['obj']); p0 = int(part[k].argmax()); p1_ = int(fused[k].argmax())
        mark = lambda p: '(对)' if p == gt else '(错)'
        title = f'真值 {v["obj"]}\n整图 {objs[p0]} {mark(p0)}｜峰值放大 {objs[p1_]} {mark(p1_)}'
        if loc:
            x0, y0, x1, y1 = lbx[k].tolist(); p2 = int(lfused[k].argmax())
            ax[k].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='cyan', lw=1.8))
            title += f'\n学到的放大 {objs[p2]} {mark(p2)}'
        ax[k].set_title(title, fontsize=7.5)
    for k in range(len(pick), len(ax)):
        ax[k].axis('off')
    fig.suptitle(f'{user}：证据热图、标注框（绿虚线，仅作图）、证据峰值放大框（黄）' + ('与学到的放大框（青）' if loc else ''),
                 fontsize=10); fig.tight_layout()
    fig.savefig(out / f'{user}_clutter.png', dpi=150); plt.close(fig)
    print('wrote', out / f'{user}_memory.png', out / f'{user}_clutter.png', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])

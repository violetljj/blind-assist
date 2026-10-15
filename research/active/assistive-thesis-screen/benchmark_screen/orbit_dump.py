"""Dump per-frame ORBIT scores once per backbone so temporal/scale/keep variants are evaluated offline.
EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md (training-free upgrades).

For every clutter video with >= 50 valid frames: all frames in order (official cap: first 1000 frames), one forward
pass each; scores against each user's objects for the whole-frame prototype ('full') and for every part-memory
config ('part@j'). Configs may add context scales (e.g. [1, 0.5]): context frames are shrunk and centred on a grey
canvas, so the memory also holds the object at clutter-like sizes (personalisation cost only; per-frame cost unchanged).
Decoding of the next video overlaps GPU work on the current one. Output: <out>/<video>.npz (names, valid, y, scores)
and <out>/meta.json. Usage: orbit_dump.py <split dir with plan.json, frames/> <out dir> <backbone> <cfgs json list>
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orbit_eval import CFG, forward, kmeans, read_many  # noqa: E402
from orbit_proto_pilot import load, macs_per_frame  # noqa: E402


def fnum(f):
    return int(f.rsplit('-', 1)[1].split('.')[0])


def shrink(x, s):
    """Centre each frame at scale s on a grey canvas of the same size (uint8 NHWC)."""
    import cv2
    if s == 1:
        return x
    n, H, W, _ = x.shape; h, w = int(round(H * s)), int(round(W * s)); out = np.full_like(x, 124)
    y0, x0 = (H - h) // 2, (W - w) // 2
    for i in range(n):
        out[i, y0:y0 + h, x0:x0 + w] = cv2.resize(x[i], (w, h), interpolation=cv2.INTER_AREA)
    return out


def part_memory_from(flat, cfg):
    """flat: list of [n_o, D] normalised context patches (all scales). Discriminative keep + k-means, as orbit_eval."""
    import torch
    mem = []
    for o, P in enumerate(flat):
        others = torch.cat([q for j, q in enumerate(flat) if j != o])
        g = torch.Generator(device=P.device).manual_seed(o)
        others = others[torch.randperm(len(others), generator=g, device=P.device)[:30000]]
        own = P[torch.randperm(len(P), generator=g, device=P.device)[:30000]]
        s_other = torch.cat([(P[i:i + 4096] @ others.T).amax(1) for i in range(0, len(P), 4096)])
        s_own = torch.cat([(P[i:i + 4096] @ own.T).topk(6, dim=1).values[:, 1:].mean(1) for i in range(0, len(P), 4096)])
        d = s_own - s_other
        keep = P[d >= d.quantile(1 - cfg['keep'])]
        mem.append(kmeans(keep, min(cfg['parts'], len(keep))))
    return mem


def main(root, out, backbone, cfgs):
    import torch
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    cfgs = [{**CFG, 'scales': [1.0], **c} for c in json.loads(cfgs)]
    plan = json.loads((root / 'plan.json').read_text())
    m, size, mean, std = load(backbone); m = m.to(memory_format=torch.channels_last)
    mpf = macs_per_frame(m, size)
    users = {}
    for v in plan['videos']:
        users.setdefault(v['user'], []).append(v)
    t0 = time.time(); meta = dict(backbone=backbone, cfgs=cfgs, gmacs_per_frame=mpf / 1e9, users={})
    pool = ThreadPoolExecutor(1)
    for user, vids in sorted(users.items()):
        objs = sorted({v['obj'] for v in vids if v['type'] == 'clean'})
        ctx = {o: [f for v in vids if v['type'] == 'clean' and v['obj'] == o for f in v['context'] if fnum(f) <= 1000] for o in objs}
        names = [f for o in objs for f in ctx[o]]; lab = np.array([i for i, o in enumerate(objs) for _ in ctx[o]])
        xc = read_many(root / 'frames', names)
        L = torch.as_tensor(lab).cuda()
        cls1, _ = forward(m, size, mean, std, xc)
        protos = torch.nn.functional.normalize(torch.stack([cls1[L == i].mean(0) for i in range(len(objs))]), dim=1)
        scale_pat = {}
        for s in sorted({s for c in cfgs for s in c['scales']}):
            _, p = forward(m, size, mean, std, shrink(xc, s)); scale_pat[s] = p
        mems = []
        for c in cfgs:
            flat = [torch.cat([scale_pat[s][L == i].reshape(-1, scale_pat[s].shape[-1]).float() for s in c['scales']])
                    for i in range(len(objs))]
            mem = part_memory_from(flat, c)
            mems.append((torch.cat(mem).half(), torch.cat([torch.full((len(q),), i) for i, q in enumerate(mem)]).cuda()))
        del scale_pat, xc
        meta['users'][user] = dict(objects=objs, context_frames=len(names),
                                   context_forwards={j: len(names) * len(c['scales']) for j, c in enumerate(cfgs)})
        clut = []
        for v in vids:
            if v['type'] != 'clutter':
                continue
            allf = sorted(v['boxes'])                                      # every frame of the video
            allf = [f for f in allf if fnum(f) <= 1000]
            valid = set(f for f in v['valid'] if fnum(f) <= 1000)
            if len(valid) >= 50:
                clut.append((v, allf, np.array([f in valid for f in allf])))
        nxt = pool.submit(read_many, root / 'frames', clut[0][1]) if clut else None
        for k, (v, allf, valid) in enumerate(clut):
            x = nxt.result()
            if k + 1 < len(clut):
                nxt = pool.submit(read_many, root / 'frames', clut[k + 1][1])
            res = {'full': [], **{f'part@{j}': [] for j in range(len(cfgs))}}
            for a in range(0, len(x), 256):
                fcls, fpat = forward(m, size, mean, std, x[a:a + 256])
                res['full'].append((fcls @ protos.T).cpu().numpy())
                for j, (allp, owner) in enumerate(mems):
                    sim = (fpat @ allp.T).float()
                    per_obj = torch.stack([sim[:, :, owner == i].amax(2) for i in range(len(objs))], 2)
                    res[f'part@{j}'].append(per_obj.topk(cfgs[j]['topk'], dim=1).values.mean(1).cpu().numpy())
            np.savez_compressed(out / f"{v['video']}.npz", names=np.array(allf), valid=valid, y=objs.index(v['obj']),
                                user=user, **{a: np.concatenate(b).astype(np.float32) for a, b in res.items()})
        print(user, len(objs), 'objects', len(clut), 'videos', round(time.time() - t0), 's', flush=True)
    meta['wall_s'] = round(time.time() - t0, 1)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])

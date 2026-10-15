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


def read_bytes(frames_dir, names):
    """CPU-only file reads (safe to run in a prefetch thread)."""
    from torchvision.io import read_file
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda f: read_file(str(frames_dir / f)), names))


def decode_gpu(data):
    """nvJPEG decode into CUDA uint8 NCHW. Must run on the consuming (main) thread: nvJPEG writes on its own stream,
    and decoding from a prefetch thread let it overwrite blocks the main thread had just freed but still used."""
    import torch
    from torchvision.io import decode_jpeg
    out = []
    for i in range(0, len(data), 256):
        out += decode_jpeg(data[i:i + 256], device='cuda')
    out = torch.stack(out)
    torch.cuda.synchronize()
    return out


def read_gpu(frames_dir, names):
    return decode_gpu(read_bytes(frames_dir, names))


def part_memory_from(flat, cfg, prec='fp32'):
    """flat: list of [n_o, D] normalised context patches (fp16, all scales). Discriminative keep + k-means, as
    orbit_eval; converts one object at a time and samples the other objects before gathering (bounded memory)."""
    import torch
    mem, sizes = [], [len(q) for q in flat]
    for o in range(len(flat)):
        g = torch.Generator(device='cpu').manual_seed(o)
        oth = [j for j in range(len(flat)) if j != o]; tot = sum(sizes[j] for j in oth)
        pick = torch.randperm(tot, generator=g)[:30000]
        offs = torch.tensor([0] + [sizes[j] for j in oth]).cumsum(0)
        dt = torch.half if prec == 'fp16' else torch.float          # 'mixed': fp32 storage, TF32 tensor-core matmuls
        torch.backends.cuda.matmul.allow_tf32 = prec == 'mixed'
        others = torch.cat([flat[j][pick[(pick >= offs[i]) & (pick < offs[i + 1])].to(flat[j].device) - offs[i]]
                            for i, j in enumerate(oth)]).to(dt)
        P = flat[o].to(dt)
        own = P[torch.randperm(len(P), generator=g)[:30000].to(P.device)]
        s_other = torch.cat([(P[i:i + 4096] @ others.T).amax(1) for i in range(0, len(P), 4096)])
        s_own = torch.cat([(P[i:i + 4096] @ own.T).topk(6, dim=1).values[:, 1:].mean(1) for i in range(0, len(P), 4096)])
        d = (s_own - s_other).float()
        keep = P[d >= d.quantile(1 - cfg['keep'])].float()
        mem.append(kmeans(keep, min(cfg['parts'], len(keep))))
        torch.backends.cuda.matmul.allow_tf32 = False
        del others, P, own, s_other, s_own, d, keep
    return mem


def guard(frac=0.75, ram_free_min_gb=2.0, wait_s=1200):
    """Fail fast instead of freezing Windows: cap this process's CUDA allocator (an overrun raises OOM rather than
    spilling into shared system memory); when free host RAM is low (often other apps), pause until it recovers and
    give up only after wait_s."""
    import psutil
    import torch
    torch.cuda.set_per_process_memory_fraction(float(frac))
    t = time.time()
    while psutil.virtual_memory().available / 2 ** 30 < ram_free_min_gb:
        if time.time() - t > wait_s:
            raise MemoryError(f'host RAM below {ram_free_min_gb} GB for {wait_s} s')
        print(f'paused: {psutil.virtual_memory().available / 2 ** 30:.1f} GB host RAM free', flush=True)
        time.sleep(30)


def peaks():
    import psutil
    import torch
    return dict(gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                gpu_reserved_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
                rss_gb=round(psutil.Process().memory_info().rss / 2 ** 30, 2),
                ram_free_gb=round(psutil.virtual_memory().available / 2 ** 30, 2))


def main(root, out, backbone, cfgs, baselines='', limit=0, only_users='', decode='cpu', prec='fp32'):
    """baselines: comma list of Swiss DINO support masks, e.g. 'full,center' -> arms swiss_full, swiss_center."""
    import torch
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    cfgs = [{**CFG, 'scales': [1.0], **c} for c in json.loads(cfgs)]
    plan = json.loads((root / 'plan.json').read_text())
    guard(); limit = int(limit)
    m, size, mean, std = load(backbone); m = m.to(memory_format=torch.channels_last)
    tok = (size // 14) ** 2; bs = max(16, int(192 * 256 / tok)); qbs = max(32, int(256 * 256 / tok))   # chunks shrink with tokens
    mpf = macs_per_frame(m, size)
    users = {}
    for v in plan['videos']:
        users.setdefault(v['user'], []).append(v)
    t0 = time.time(); meta = dict(backbone=backbone, cfgs=cfgs, gmacs_per_frame=mpf / 1e9, users={}, decode=decode, prec=prec)
    # prefetch thread does CPU work only; GPU decode (if any) happens on the main thread in `finish`
    reader = (lambda names: read_bytes(root / 'frames', names)) if decode == 'gpu' else (lambda names: read_many(root / 'frames', names))
    finish = decode_gpu if decode == 'gpu' else (lambda x: x)
    T = dict(read_wait=0., ctx=0., forward=0., full=0., part=0., swiss=0., save=0.)

    def tick(key, t):
        torch.cuda.synchronize(); T[key] += time.time() - t; return time.time()
    pool = ThreadPoolExecutor(1)
    for user, vids in sorted(users.items()):
        if only_users and user not in only_users.split(','):
            continue
        todo = [v for v in vids if v['type'] == 'clutter' and len([f for f in v['valid'] if fnum(f) <= 1000]) >= 50]
        if not limit and todo and all((out / f"{v['video']}.npz").exists() for v in todo):
            print(user, 'already dumped, skipped', flush=True); continue
        objs = sorted({v['obj'] for v in vids if v['type'] == 'clean'})
        ctx = {o: [f for v in vids if v['type'] == 'clean' and v['obj'] == o for f in v['context'] if fnum(f) <= 1000] for o in objs}
        names = [f for o in objs for f in ctx[o]]; lab = np.array([i for i, o in enumerate(objs) for _ in ctx[o]])
        tc = time.time()
        xc = finish(reader(names))
        L = torch.as_tensor(lab).cuda()
        cls1, p1 = forward(m, size, mean, std, xc, bs=bs)                 # one pass gives CLS and scale-1 patches
        protos = torch.nn.functional.normalize(torch.stack([cls1[L == i].mean(0) for i in range(len(objs))]), dim=1)
        scale_pat = {}
        scale_pat = {1.0: p1}; del p1
        for s in sorted({s for c in cfgs for s in c['scales']} - {1.0}):
            _c, p = forward(m, size, mean, std, shrink(xc, s), bs=bs); scale_pat[s] = p; del _c, p
        mems = []
        for c in cfgs:
            flat = [torch.cat([scale_pat[s][L == i].reshape(-1, scale_pat[s].shape[-1]) for s in c['scales']])
                    for i in range(len(objs))]
            mem = part_memory_from(flat, c, prec)
            mems.append((torch.cat(mem).half(), torch.cat([torch.full((len(q),), i) for i, q in enumerate(mem)]).cuda()))
        import swiss_dino_core as SW
        grid = int(round(scale_pat[1.0].shape[1] ** .5)); sw = {}
        for mode in [b for b in baselines.split(',') if b]:
            pt = [SW.support(scale_pat[1.0][L == i], grid, mode) for i in range(len(objs))]
            sw[mode] = (torch.stack([p for p, _ in pt]), torch.stack([t for _, t in pt]))
        del scale_pat, xc; torch.cuda.empty_cache(); tick('ctx', tc)
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
        nxt = pool.submit(reader, clut[0][1]) if clut else None
        if limit:
            clut = clut[:limit]
        for k, (v, allf, valid) in enumerate(clut):
            t = time.time(); x = finish(nxt.result()); t = tick('read_wait', t)
            if k + 1 < len(clut):
                nxt = pool.submit(reader, clut[k + 1][1])
            res = {'full': [], **{f'part@{j}': [] for j in range(len(cfgs))}, **{f'swiss_{b}': [] for b in sw}}
            for a in range(0, len(x), qbs):
                t = time.time(); fcls, fpat = forward(m, size, mean, std, x[a:a + qbs], bs=bs); t = tick('forward', t)
                res['full'].append((fcls @ protos.T).cpu().numpy()); t = tick('full', t)
                for b, (P_, T_) in sw.items():
                    res[f'swiss_{b}'].append(SW.query_scores(fpat, P_, T_, grid, prec='fp16' if prec in ('fp16', 'mixed') else prec).cpu().numpy())
                t = tick('swiss', t)
                for j, (allp, owner) in enumerate(mems):
                    sim = fpat @ allp.T                                          # fp16 tensor-core matmul in both modes
                    counts = torch.bincount(owner, minlength=len(objs))
                    if prec in ('fp16', 'mixed') and bool((counts == counts[0]).all()):   # equal part counts: one reshape-max
                        per_obj = sim.view(*sim.shape[:2], len(objs), int(counts[0])).amax(3).float()
                    else:
                        sim = sim.float()
                        per_obj = torch.stack([sim[:, :, owner == i].amax(2) for i in range(len(objs))], 2)
                    res[f'part@{j}'].append(per_obj.topk(cfgs[j]['topk'], dim=1).values.mean(1).cpu().numpy())
                t = tick('part', t)
            t = time.time()
            np.savez_compressed(out / f"{v['video']}.npz", names=np.array(allf), valid=valid, y=objs.index(v['obj']),
                                user=user, **{a: np.concatenate(b).astype(np.float32) for a, b in res.items()})
            tick('save', t)
        print(user, len(objs), 'objects', len(clut), 'videos', round(time.time() - t0), 's', json.dumps(peaks()), flush=True)
        del mems, sw, protos, cls1, L
        torch.cuda.empty_cache(); guard()
        if limit:
            break
    meta['wall_s'] = round(time.time() - t0, 1); meta['peaks'] = peaks(); meta['stage_s'] = {k: round(v, 1) for k, v in T.items()}
    print('stage seconds', json.dumps(meta['stage_s']), 'wall', meta['wall_s'], flush=True)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])

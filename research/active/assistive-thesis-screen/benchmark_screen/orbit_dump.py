"""Dump per-frame ORBIT scores once per backbone so temporal/scale/keep variants are evaluated offline.
EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md (training-free upgrades).

For every clutter video with >= 50 valid frames: all frames in order (official cap: first 1000 frames), one forward
pass each; scores against each user's objects for the whole-frame prototype ('full') and for every part-memory
config ('part@j'). Configs may add context scales (e.g. [1, 0.5]): context frames are shrunk and centred on a grey
canvas, so the memory also holds the object at clutter-like sizes (personalisation cost only; per-frame cost unchanged).
Decoding of the next video overlaps GPU work on the current one. Output: <out>/<video>.npz (names, valid, y, scores)
and <out>/meta.json. Usage: orbit_dump.py <split dir with plan.json, frames/> <out dir> <backbone> <cfgs json list>
patch_k > 0 also stores, for k seeded random valid frames per video, the patch tokens, the per-object part similarity
maps of cfg 0 and the patch-in-box mask (boxes only used as training targets / diagnostics) in <out>/patch/<video>.npz.
A cfg with 'color': beta > 0 makes the part match colour-aware: DINOv2's colour-jitter training leaves its patches
nearly colour-blind, while personal objects often differ only in colour (failure analysis: five markers of different
colours absorbed by one). Each patch gets its mean opponent chroma (O1, O2); a memory part carries the mean chroma of
its kept patches, and the match becomes max_k [cos(x_t, p_k) - beta * ||chroma_t - chroma_k||].
zoom=<s1,s2,...> adds "look again" arms 'zoom<100 s>' (PLAN addendum 5): from cfg 0's evidence map (max over objects of
the part similarity, 3x3 smoothed) take the square of side s x frame around the strongest evidence, crop it from the
already-decoded GPU frame with roi_align to the model size, forward once more and score it with the same memory
(heads, if any, also get '<head>~zoom<100 s>'); the crop box is saved as 'box_zoom<100 s>' (pixels, x0 y0 x1 y1).
zoom entry 'cand<s>' = candidate verification: each of the two best objects of the whole-frame part score gets its own
crop around *its* evidence peak and is rescored there (2 extra forwards); arm 'cand<100 s>' keeps the whole-frame scores
but puts the two candidates on top, ordered by their crop scores (aux_cand*: idx0, idx1, crop0, crop1).
zoom entry 'gt' = DIAGNOSTIC upper bound only: crop around the annotated box (evaluator truth, never a method arm);
'gtc<s>' = annotated centre with a fixed side s, 'gts' = evidence-peak centre with the annotated size (both DIAGNOSTIC:
they separate where to look from how much to zoom).
loc=<loc_all.pt | loc_folds.json> + zoom entries 'loc<s>' / 'locadapt': the crop centre comes from the learned
localizer (train_localizer.py, trained on train-split boxes; a folds json maps each user to the model that never saw
them, for honest dev runs); 'locadapt' sizes the square from the localizer's predicted object area (1.25 x sqrt, side
0.35-0.8 of the frame).
head=<a.pt,b.pt,...> adds one arm 'head:<file stem>' per learned matching model (evidence_head.py) on cfg 0's memory;
an entry <name>.json maps each user to the fold model that never saw them (honest dev runs; arm 'head:<name>').
"""
from __future__ import annotations

import json
import sys
import time
import zlib
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


def patch_colors(x, size, grid, chunk=256):
    """Mean opponent colour (L, O1, O2) of every patch: x uint8 CUDA NCHW tensor or NHWC numpy -> [N, grid*grid, 3]."""
    import torch
    import torch.nn.functional as F
    out = []
    for i in range(0, len(x), chunk):
        b = x[i:i + chunk] if torch.is_tensor(x) else torch.from_numpy(x[i:i + chunk]).cuda().permute(0, 3, 1, 2)
        b = b.float().div(255.)
        if b.shape[-1] != size:
            b = F.interpolate(b, (size, size), mode='bilinear', align_corners=False, antialias=True)
        r, g, bl = F.adaptive_avg_pool2d(b, grid).unbind(1)
        out.append(torch.stack([(r + g + bl) / 3, (r - g) / 2, (r + g - 2 * bl) / 4], -1).flatten(1, 2))
    return torch.cat(out)


def part_memory_from(flat, cfg, prec='fp32', cols=None, keep_out=None):
    """flat: list of [n_o, D] normalised context patches (fp16, all scales). Discriminative keep + k-means, as
    orbit_eval; converts one object at a time and samples the other objects before gathering (bounded memory).
    cols: optional list of [n_o, 3] patch colours -> also returns each part's mean chroma (O1, O2).
    keep_out: optional list that receives each object's kept-patch mask (for figures)."""
    import torch
    mem, sizes, pcols = [], [len(q) for q in flat], []
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
        sel = d >= d.quantile(1 - cfg['keep']); keep = P[sel].float()
        if keep_out is not None:
            keep_out.append(sel.cpu())
        mem.append(kmeans(keep, min(cfg['parts'], len(keep))))
        if cols is not None:
            a = (keep @ mem[-1].T).argmax(1); ch = cols[o][sel][:, 1:]
            pcols.append(torch.zeros(len(mem[-1]), 2, device=ch.device).index_add_(0, a, ch)
                         / torch.bincount(a, minlength=len(mem[-1])).clamp(min=1)[:, None])
        torch.backends.cuda.matmul.allow_tf32 = False
        del others, P, own, s_other, s_own, d, keep
    return (mem, pcols) if cols is not None else mem


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


def box_mask(box, grid, side=1080):
    """Patch-centre-in-box mask [grid*grid] for an ORBIT box in original 1080 px coordinates (None -> all False)."""
    if not box:
        return np.zeros(grid * grid, bool)
    c = (np.arange(grid) + .5) * side / grid
    inx = (c >= box['x']) & (c < box['x'] + box['w']); iny = (c >= box['y']) & (c < box['y'] + box['h'])
    return (iny[:, None] & inx[None]).reshape(-1)


def per_object(sim, owner, n_obj, prec):
    """sim [N,T,P] patch-part similarities -> [N,T,n_obj] max over each object's parts."""
    import torch
    counts = torch.bincount(owner, minlength=n_obj)
    if prec in ('fp16', 'mixed') and bool((counts == counts[0]).all()):   # equal part counts: one reshape-max
        return sim.view(*sim.shape[:2], n_obj, int(counts[0])).amax(3).float()
    sim = sim.float()
    return torch.stack([sim[:, :, owner == i].amax(2) for i in range(n_obj)], 2)


def zoom_boxes(S, grid, side, s):
    """S [N,T,O] part similarities -> square boxes [N,4] (x0,y0,x1,y1 px) of side s*side around the smoothed
    strongest evidence over all of the user's objects, kept inside the frame."""
    import torch
    import torch.nn.functional as F
    E = F.avg_pool2d(S.amax(2).view(-1, 1, grid, grid), 3, 1, 1, count_include_pad=False).flatten(1)
    k = E.argmax(1); half = torch.as_tensor(s * side / 2, device=S.device, dtype=torch.float).expand(len(k))
    cy = ((k // grid).float() + .5) * side / grid; cx = ((k % grid).float() + .5) * side / grid
    cx = torch.minimum(torch.maximum(cx, half), side - half); cy = torch.minimum(torch.maximum(cy, half), side - half)
    return torch.stack([cx - half, cy - half, cx + half, cy + half], 1)


def zoom_tag(z):
    if z in ('gt', 'gts'):
        return 'zoom' + z
    if z.startswith('gtc'):
        return f'zoomgtc{round(100 * float(z[3:]))}'
    if z.startswith('loc'):
        return z if z == 'locadapt' else f'loc{round(100 * float(z[3:]))}'
    return f'cand{round(100 * float(z[4:]))}' if z.startswith('cand') else f'zoom{round(100 * float(z))}'


def gt_boxes(names, boxes, side, src=1080, fixed=None, sizes_only=False):
    """DIAGNOSTIC: square crop around each frame's annotated box (1.25x its longer side, >= a quarter of the frame;
    or a fixed side fraction); frames without a box get the whole frame. sizes_only -> the side fractions."""
    import torch
    out = []
    for f in names:
        b = boxes.get(f)
        if not b:
            out.append([0., 0., float(side), float(side)]); continue
        sc = side / src
        L = fixed * side if fixed else min(max(max(b['w'], b['h']) * sc * 1.25, side / 4), side); h = L / 2
        cx = min(max((b['x'] + b['w'] / 2) * sc, h), side - h); cy = min(max((b['y'] + b['h'] / 2) * sc, h), side - h)
        out.append([cx - h, cy - h, cx + h, cy + h])
    t = torch.tensor(out, device='cuda')
    return (t[:, 2] - t[:, 0]) / side if sizes_only else t


def main(root, out, backbone, cfgs, baselines='', limit=0, only_users='', decode='cpu', prec='fp32', patch_k=0, head='',
         zoom='', loc=''):
    """baselines: comma list of Swiss DINO support masks, e.g. 'full,center' -> arms swiss_full, swiss_center."""
    import torch
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    patch_k = int(patch_k); zooms = [z for z in str(zoom).split(',') if z]
    assert not zooms or decode == 'gpu', 'zoom crops from the GPU-decoded frames'
    if patch_k:
        (out / 'patch').mkdir(exist_ok=True)
    hds = {}
    if head:
        import evidence_head as EH
        hds = {f'head:{Path(h).stem}': ({u: EH.load(pth) for u, pth in json.loads(Path(h).read_text()).items()}
                                         if h.endswith('.json') else EH.load(h)) for h in head.split(',')}
    cfgs = [{**CFG, 'scales': [1.0], **c} for c in json.loads(cfgs)]
    plan = json.loads((root / 'plan.json').read_text())
    guard(); limit = int(limit)
    m, size, mean, std = load(backbone); m = m.to(memory_format=torch.channels_last)
    tok = (size // 14) ** 2; bs = max(16, int(192 * 256 / tok)); qbs = max(32, int(256 * 256 / tok))   # chunks shrink with tokens
    mpf = macs_per_frame(m, size)
    users = {}
    for v in plan['videos']:
        users.setdefault(v['user'], []).append(v)
    locs = {}
    if loc:
        import train_localizer as TL
        if loc.endswith('.json'):
            mp = json.loads(Path(loc).read_text()); locs = {u: TL.load_loc(pth) for u, pth in mp.items()}
        else:
            locs = {'*': TL.load_loc(loc)}
    t0 = time.time(); meta = dict(loc=loc, backbone=backbone, cfgs=cfgs, gmacs_per_frame=mpf / 1e9, users={}, decode=decode, prec=prec,
                                  patch_k=patch_k, head=head, zoom=zooms)
    # prefetch thread does CPU work only; GPU decode (if any) happens on the main thread in `finish`
    reader = (lambda names: read_bytes(root / 'frames', names)) if decode == 'gpu' else (lambda names: read_many(root / 'frames', names))
    finish = decode_gpu if decode == 'gpu' else (lambda x: x)
    T = dict(read_wait=0., ctx=0., forward=0., full=0., part=0., swiss=0., head=0., zoom=0., save=0.)

    def tick(key, t):
        torch.cuda.synchronize(); T[key] += time.time() - t; return time.time()
    pool = ThreadPoolExecutor(1)
    for user, vids in sorted(users.items()):
        if only_users and user not in only_users.split(','):
            continue
        todo = [v for v in vids if v['type'] == 'clutter' and len([f for f in v['valid'] if fnum(f) <= 1000]) >= 50]
        if not limit and todo and all((out / f"{v['video']}.npz").exists() for v in todo):
            print(user, 'already dumped, skipped', flush=True); continue
        torch.cuda.empty_cache(); guard()                               # before each user's work, not after the last
        objs = sorted({v['obj'] for v in vids if v['type'] == 'clean'})
        ctx = {o: [f for v in vids if v['type'] == 'clean' and v['obj'] == o for f in v['context'] if fnum(f) <= 1000] for o in objs}
        names = [f for o in objs for f in ctx[o]]; lab = np.array([i for i, o in enumerate(objs) for _ in ctx[o]])
        tc = time.time()
        xc = finish(reader(names))
        L = torch.as_tensor(lab).cuda()
        cls1, p1 = forward(m, size, mean, std, xc, bs=bs)                 # one pass gives CLS and scale-1 patches
        protos = torch.nn.functional.normalize(torch.stack([cls1[L == i].mean(0) for i in range(len(objs))]), dim=1)
        scale_pat = {}
        scale_pat = {1.0: p1}; p_tok = p1.shape[1]; del p1
        for s in sorted({s for c in cfgs for s in c['scales']} - {1.0}):
            _c, p = forward(m, size, mean, std, shrink(xc, s), bs=bs); scale_pat[s] = p; del _c, p
        mems, pcol = [], []
        ccol = patch_colors(xc, size, int(round(p_tok ** .5))) if any(c.get('color', 0) for c in cfgs) else None
        for c in cfgs:
            # context frames are grouped by object, so each object's patches are a slice (a view, no copy)
            bounds = np.cumsum([0] + [len(ctx[o]) for o in objs])
            view = lambda s_, i: scale_pat[s_][bounds[i]:bounds[i + 1]].reshape(-1, scale_pat[s_].shape[-1])
            flat = [view(1.0, i) if c['scales'] == [1.0] else torch.cat([view(s_, i) for s_ in c['scales']])
                    for i in range(len(objs))]
            if c.get('color', 0):
                assert c['scales'] == [1.0], 'colour-aware memory is single-scale'
                mem, pc = part_memory_from(flat, c, prec, [ccol[L == i].reshape(-1, 3) for i in range(len(objs))])
                pcol.append(torch.cat(pc))
            else:
                mem = part_memory_from(flat, c, prec); pcol.append(None)
            mems.append((torch.cat(mem).half(), torch.cat([torch.full((len(q),), i) for i, q in enumerate(mem)]).cuda()))
        del ccol
        import swiss_dino_core as SW
        grid = int(round(scale_pat[1.0].shape[1] ** .5)); sw = {}
        for mode in [b for b in baselines.split(',') if b]:
            pt = [SW.support(scale_pat[1.0][L == i], grid, mode) for i in range(len(objs))]
            sw[mode] = (torch.stack([p for p, _ in pt]), torch.stack([t for _, t in pt]))
        del scale_pat, xc; torch.cuda.empty_cache(); tick('ctx', tc)
        if patch_k:                                                     # cfg-0 memory, for training the matching head
            np.savez(out / 'patch' / f'_mem_{user}.npz', parts=mems[0][0].float().cpu().numpy(),
                     owner=mems[0][1].cpu().numpy(), objects=np.array(objs))
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
            res = {'full': [], **{f'part@{j}': [] for j in range(len(cfgs))}, **{f'swiss_{b}': [] for b in sw},
                   **{k: [] for k in hds}}
            for z in zooms:
                tag = zoom_tag(z); res[tag] = []; res[f'box_{tag}'] = []
                if z.startswith('cand'):
                    res[f'aux_{tag}'] = []
                for k in hds:
                    res[f'{k}~{tag}'] = []
            sel = np.zeros(len(allf), bool); store = dict(X=[], S=[], idx=[])
            if patch_k:
                vi = np.flatnonzero(valid); rs = np.random.default_rng(zlib.crc32(v['video'].encode()))
                sel[rs.choice(vi, min(patch_k, len(vi)), replace=False)] = True
            for a in range(0, len(x), qbs):
                t = time.time(); fcls, fpat = forward(m, size, mean, std, x[a:a + qbs], bs=bs); t = tick('forward', t)
                qcol = patch_colors(x[a:a + qbs], size, grid)[..., 1:] if any(pc is not None for pc in pcol) else None
                full_s = fcls @ protos.T
                res['full'].append(full_s.cpu().numpy()); t = tick('full', t)
                for b, (P_, T_) in sw.items():
                    res[f'swiss_{b}'].append(SW.query_scores(fpat, P_, T_, grid, prec='fp16' if prec in ('fp16', 'mixed') else prec).cpu().numpy())
                t = tick('swiss', t)
                for j, (allp, owner) in enumerate(mems):
                    sim = fpat @ allp.T                                          # fp16 tensor-core matmul in both modes
                    if pcol[j] is not None:
                        sim = sim - (cfgs[j]['color'] * torch.cdist(qcol, pcol[j][None].expand(len(qcol), -1, -1))).to(sim.dtype)
                    per_obj = per_object(sim, owner, len(objs), prec)
                    res[f'part@{j}'].append(per_obj.topk(cfgs[j]['topk'], dim=1).values.mean(1).cpu().numpy())
                    if j == 0:
                        po0 = per_obj
                t = tick('part', t)
                hsel = {k: (hd[user] if isinstance(hd, dict) else hd) for k, hd in hds.items()}
                for k, hd in hsel.items():
                    res[k].append(EH.scores(hd, fpat, po0, full_s, *mems[0]).cpu().numpy())
                t = tick('head', t)
                if zooms:
                    from torchvision.ops import roi_align
                    src = x[a:a + qbs].float(); side = src.shape[-1]; n = len(src)
                    def look(bx):
                        crop = roi_align(src, torch.cat([torch.arange(n, device=bx.device)[:, None].float(), bx], 1),
                                         (size, size), 1.0, 2, True)                    # 0-255 float, like the frames
                        zcls, zpat = forward(m, size, mean, std, crop, bs=bs); del crop
                        return zcls, zpat, per_object(zpat @ mems[0][0].T, mems[0][1], len(objs), prec)
                    k8 = cfgs[0]['topk']
                    if any(z.startswith('loc') for z in zooms):
                        lm = locs.get(user, locs.get('*'))
                        with torch.inference_mode():
                            lmap = lm(fpat, po0)
                        lprob = torch.sigmoid(lmap)
                    for z in zooms:
                        tag = zoom_tag(z)
                        if z.startswith('cand'):                                   # verify the two best candidates
                            p224 = po0.topk(k8, dim=1).values.mean(1); top2 = p224.topk(2, dim=1).indices
                            ar = torch.arange(n, device=p224.device); crops, bxs = [], []
                            for r in range(2):
                                o = top2[:, r]; bx = zoom_boxes(po0[ar, :, o][..., None], grid, side, float(z[4:]))
                                _, _, zpo = look(bx); crops.append(zpo.topk(k8, dim=1).values.mean(1)[ar, o]); bxs.append(bx)
                            sc = p224 - 1.; sc[ar, top2[:, 0]] = crops[0]; sc[ar, top2[:, 1]] = crops[1]
                            res[tag].append(sc.cpu().numpy()); res[f'box_{tag}'].append(torch.stack(bxs, 1).cpu().numpy())
                            res[f'aux_{tag}'].append(torch.stack([top2[:, 0].float(), top2[:, 1].float(), crops[0], crops[1]], 1).cpu().numpy())
                            continue
                        fr = [allf[a + i] for i in range(n)]
                        if z.startswith('loc'):
                            sz = ((1.25 * lprob.sum(1).div(lprob.shape[1]).sqrt()).clamp(0.35, 0.8) if z == 'locadapt'
                                  else float(z[3:]))
                            bx = zoom_boxes(lmap[..., None], grid, side, sz)
                        else:
                            bx = None
                        bx = bx if bx is not None else (gt_boxes(fr, v['boxes'], side) if z == 'gt' else
                              gt_boxes(fr, v['boxes'], side, fixed=float(z[3:])) if z.startswith('gtc') else
                              zoom_boxes(po0, grid, side, gt_boxes(fr, v['boxes'], side, sizes_only=True)) if z == 'gts' else
                              zoom_boxes(po0, grid, side, float(z)))
                        zcls, zpat, zpo = look(bx)
                        res[tag].append(zpo.topk(k8, dim=1).values.mean(1).cpu().numpy())
                        res[f'box_{tag}'].append(bx.cpu().numpy())
                        for k, hd in hsel.items():
                            res[f'{k}~{tag}'].append(EH.scores(hd, zpat, zpo, zcls @ protos.T, *mems[0]).cpu().numpy())
                        del zcls, zpat, zpo
                    del src
                t = tick('zoom', t)
                ii = np.flatnonzero(sel[a:a + len(fcls)])
                if len(ii):
                    it = torch.as_tensor(ii, device=fpat.device)
                    store['X'].append(fpat[it].cpu().numpy()); store['S'].append(po0[it].half().cpu().numpy())
                    store['idx'].append(ii + a)
                del po0
            t = time.time()
            np.savez_compressed(out / f"{v['video']}.npz", names=np.array(allf), valid=valid, y=objs.index(v['obj']),
                                user=user, **{a: np.concatenate(b).astype(np.float32) for a, b in res.items()})
            if store['idx']:
                idx = np.concatenate(store['idx']); full_all = np.concatenate(res['full'])
                np.savez(out / 'patch' / f"{v['video']}.npz", X=np.concatenate(store['X']), S=np.concatenate(store['S']),
                         full=full_all[idx].astype(np.float32), idx=idx, names=np.array(allf)[idx], y=objs.index(v['obj']),
                         user=user, mask=np.stack([box_mask(v['boxes'][allf[i]], grid) for i in idx]))
            tick('save', t)
        print(user, len(objs), 'objects', len(clut), 'videos', round(time.time() - t0), 's', json.dumps(peaks()), flush=True)
        del mems, sw, protos, cls1, L
        torch.cuda.empty_cache()
        if limit:
            break
    meta['wall_s'] = round(time.time() - t0, 1); meta['peaks'] = peaks(); meta['stage_s'] = {k: round(v, 1) for k, v in T.items()}
    print('stage seconds', json.dumps(meta['stage_s']), 'wall', meta['wall_s'], flush=True)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])

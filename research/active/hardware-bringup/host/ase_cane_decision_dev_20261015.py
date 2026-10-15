"""Fixed evaluation (labels v2: body +-0.25 m, cane sweep +-0.35 m) and a per-frame decision head on the side-view grid.
(EXPLORE, 2026-10-15; rules frozen in <work>/PLAN_DECISION.md before any result.)
Usage:
  ase_cane_decision_dev_20261015.py relabel <scenes> <work> <first> <last>
  ase_cane_decision_dev_20261015.py probs   <work> <seed>          (occupancy-net probabilities for all scenes)
  ase_cane_decision_dev_20261015.py train   <work> <seed>
  ase_cane_decision_dev_20261015.py eval    <work>
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ase_cane_reach_dev_20261015 as A  # noqa: E402
import ase_cane_sideview_dev_20261015 as V  # noqa: E402

BODY_HALF, CANE_HALF = .25, .35
TEST, VAL = list(range(0, 20)), list(range(55, 60))
TRAIN = [k for k in range(20, 200) if k not in VAL]


# ---------------------------------------------------------------- labels v2
def relabel_scene(args):
    scene, work = Path(args[0]), Path(args[1])
    out = work / 'labels_v2' / f'{scene.name}.json'
    if out.exists():
        return scene.name, 'cached'
    R = np.load(work / 'rays.npz'); dirs, sa, T_dc = R['dirs'], R['sa'], R['T_dc']
    frames = A.load_scene(scene, T_dc)
    cmap = scene / 'object_instances_to_classes.json'
    if cmap.exists():
        C, I = A.voxel_map(scene, frames, dirs)
    else:                                                              # no class map: occupancy only, skip instance decoding
        import cv2
        keys = []
        for i, _, T, _ in frames:
            d = cv2.imread(str(scene / 'depth' / f'depth{i:07d}.png'), cv2.IMREAD_UNCHANGED)[::A.PIX, ::A.PIX]
            ok = (d > 0) & (d < A.MAX_MM)
            keys.append(np.unique(A.pack(np.floor(((dirs[ok] * (d[ok, None] / 1000.)) @ T[:3, :3].T + T[:3, 3]) / A.VOX).astype(np.int64))))
            if len(keys) > 60:
                keys = [np.unique(np.concatenate(keys))]
        C = ((A.unpack(np.unique(np.concatenate(keys))) + .5) * A.VOX).astype(np.float32)
    if cmap.exists():
        cl = {int(a): b for a, b in json.loads(cmap.read_text()).items()}
        floor = float(np.median(C[np.array([cl.get(int(v), '?') for v in I]) == 'floor', 2]))
    else:
        floor = .015                                                   # ASE floor plane z = 0 (as the side-view grids)
    H = C[:, 2] - floor
    sup = C[(H >= A.BAND_LO) & (H <= A.TOPS['R2'])]
    fl = np.unique(np.floor(C[np.abs(H) <= A.FLOOR_TOL][:, :2] / A.VOX).astype(np.int64), axis=0); flc = (fl + .5) * A.VOX
    t = np.array([f[1] for f in frames]); pos = np.array([f[3][:2] for f in frames]); n = len(frames)
    lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    vel = (pos[hi] - pos[lo]) / np.maximum(t[hi] - t[lo], 1e-3)[:, None]; speed = np.linalg.norm(vel, axis=1)
    rows = []
    for j, (i, _, T, _) in enumerate(frames):
        o = T[:3, 3]; f = T[:3, 2] - A.UP * (T[:3, 2] @ A.UP); nf = np.linalg.norm(f)
        row = dict(frame=i, speed=round(float(speed[j]), 3))
        if nf < .2:
            rows.append(dict(row, skip='vertical')); continue
        f /= nf; xr = np.cross(f, A.UP)
        row['align_deg'] = round(float(np.degrees(np.arccos(np.clip(vel[j] @ f[:2] / max(speed[j], 1e-6), -1, 1)))), 1)
        P, k, ok = A.points(scene, i, T, dirs)
        lat, fwd, hgt = (P - o) @ xr, (P - o) @ f, P[:, 2] - floor
        body = (np.abs(lat) <= BODY_HALF) & (fwd >= A.Z_RANGE[0]) & (fwd <= A.Z_RANGE[1]) & (hgt >= A.BODY[0]) & (hgt <= A.BODY[1])
        w = sa[ok][body]
        if float(w.sum()) < A.MIN_SR:
            rows.append(dict(row, pos=False)); continue
        d = float(np.percentile(fwd[body], 5)); inst = int(np.bincount(k[body], weights=w).argmax()); reach = d + A.MARGIN

        def region(Q):
            a, b = (Q[:, :2] - o[:2]) @ xr[:2], (Q[:, :2] - o[:2]) @ f[:2]
            return (np.abs(a) <= CANE_HALF) & (b >= 0) & (b <= reach)
        nv = int(region(sup).sum())
        cover = region(np.c_[flc, np.zeros(len(flc))]).sum() * A.VOX * A.VOX / (2 * CANE_HALF * reach)
        cat = 'CANE' if nv >= A.MIN_VOX else ('UNOBSERVED' if cover < A.FLOOR_COVER else 'MISS')
        rows.append(dict(row, pos=True, d_body=round(d, 3), inst=inst, cat=cat))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(scene=scene.name, floor=floor, rows=rows)), encoding='utf8')
    return scene.name, f'{n} frames'


def relabel(scenes, work, first, last):
    todo = [Path(scenes) / str(k) for k in range(int(first), int(last) + 1)]
    with ProcessPoolExecutor(10) as ex:
        for name, msg in ex.map(relabel_scene, [(str(s), str(work)) for s in todo]):
            print('relabel', name, msg, flush=True)


def strata(row):
    return 'skip' not in row and row['speed'] >= A.WALK and row.get('align_deg', 180) <= A.ALIGN_DEG


def frame_labels(work, k, frames):
    """Per grid frame: 'MISS'/'CANE'/'UNOBS' + '_near'/'_far', 'NEG', or None (skip); primary-stratum flag; instance."""
    rows = {r['frame']: r for r in json.loads((Path(work) / 'labels_v2' / f'{k}.json').read_text())['rows']}
    lab, prim, inst = [], [], []
    for fr in frames:
        r = rows.get(int(fr))
        if r is None or 'skip' in r:
            lab.append(None); prim.append(False); inst.append(None); continue
        if r['pos']:
            c = {'MISS': 'MISS', 'CANE': 'CANE'}.get(r['cat'], 'UNOBS')
            lab.append(c + ('_near' if r['d_body'] <= A.NEAR else '_far')); inst.append(r['inst'])
        else:
            lab.append('NEG'); inst.append(None)
        prim.append(strata(r))
    return lab, np.array(prim), inst


# ---------------------------------------------------------------- occupancy-net probabilities (175-scene nets)
def _occ_net(work, seed):
    import torch
    net = V.make_net(8).cuda()
    net.load_state_dict(torch.load(Path(work) / f'sideview_net{V._tag(False, seed, "big")}.pt')); net.eval()
    return net


def probs(work, seed):
    import torch
    work = Path(work); net = _occ_net(work, seed); od = work / f'probs_s{seed}'; od.mkdir(exist_ok=True)
    for k in range(200):
        if (od / f'{k}.npy').exists():
            continue
        X = np.load(work / 'grids' / f'{k}.npz')['X']
        with torch.no_grad():
            P = np.concatenate([torch.sigmoid(net(V.pad(torch.from_numpy(X[b:b + 512]).cuda().float()))[:, :V.NH]).cpu().numpy()
                                for b in range(0, len(X), 512)])
        np.save(od / f'{k}.npy', P.astype(np.float16))
    print('probs done', seed)


# ---------------------------------------------------------------- decision head
def make_head(cin=9):
    import torch.nn as nn

    def c(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(c(cin, 32), c(32, 32), nn.MaxPool2d(2), c(32, 64), c(64, 64), nn.MaxPool2d(2), c(64, 128))
            self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

        def forward(self, x):
            import torch
            h = self.f(x)
            return self.fc(torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1))[:, 0]
    return Head()


def scene_xy(work, k, seed):
    Z = np.load(Path(work) / 'grids' / f'{k}.npz')
    P = np.load(Path(work) / f'probs_s{seed}' / f'{k}.npy')
    lab, prim, inst = frame_labels(work, k, Z['frames'])
    keep = Z['valid'] & np.array([l is not None for l in lab])
    X = np.concatenate([Z['X'], P[:, None]], 1)[keep]
    labk = [l for l, m in zip(lab, keep) if m]
    return X, np.array([l == 'MISS_near' for l in labk], np.float32), labk, prim[keep], [i for i, m in zip(inst, keep) if m], Z, keep


def train(work, seed, epochs=10):
    import torch
    work = Path(work); seed = int(seed); torch.manual_seed(7 + seed); np.random.seed(7 + seed)
    Xs, Ys = [], []
    for k in TRAIN:
        X, y, *_ = scene_xy(work, k, seed); Xs.append(X.astype(np.float16)); Ys.append(y)
    X = torch.from_numpy(np.concatenate(Xs)); Y = torch.from_numpy(np.concatenate(Ys))
    npos = float(Y.sum()); pw = min((len(Y) - npos) / max(npos, 1), 100.)
    print('train frames', len(Y), 'MISS_near', int(npos), 'pos_weight', round(pw, 1), flush=True)
    net = make_head().cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Y) // 256 + 1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device='cuda'))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Y)); tot = 0.
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]; x, y = X[i].cuda().float(), Y[i].cuda()
            loss = lossf(net(V.pad(x)), y); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach()) * len(i)
        print(f'epoch {ep} loss {tot / len(Y):.4f}', flush=True)
    torch.save(net.state_dict(), work / f'decision_head_s{seed}.pt')


# ---------------------------------------------------------------- evaluation under labels v2
def tally(items):
    """items: (scene, lab, inst, alert) over primary-stratum frames."""
    miss = [x for x in items if x[1] == 'MISS_near']
    return dict(miss_near=len(miss), miss_kept=sum(x[3] for x in miss),
                miss_objects=len({(x[0], x[2]) for x in miss}), miss_objects_kept=len({(x[0], x[2]) for x in miss if x[3]}),
                cane_near=sum(x[1] == 'CANE_near' for x in items), cane_alert=sum(x[3] for x in items if x[1] == 'CANE_near'),
                neg=sum(x[1] == 'NEG' for x in items), neg_alert=sum(x[3] for x in items if x[1] == 'NEG'))


def evaluate(work):
    import torch
    work = Path(work); rep = dict(seeds={}); taus = np.round(np.arange(.05, .96, .05), 2)

    def collect(ids, seed):
        out = []
        for k in ids:
            X, y, labk, prim, inst, Z, keep = scene_xy(work, k, seed)
            out.append((k, X, labk, prim, inst, np.asarray(Z['Y'])[keep], np.asarray(Z['frames'])[keep]))
        return out
    for seed in (0, 1, 2):
        head = make_head().cuda(); head.load_state_dict(torch.load(work / f'decision_head_s{seed}.pt')); head.eval()
        data = {'val': collect(VAL, seed), 'test': collect(TEST, seed)}
        S = {}
        for split, items in data.items():
            S[split] = []
            for k, X, labk, prim, inst, Yg, frs in items:
                with torch.no_grad():
                    s = torch.sigmoid(head(V.pad(torch.from_numpy(X).cuda().float()))).cpu().numpy()
                for j in np.flatnonzero(prim):
                    occ = X[j, 8].astype(np.float32)
                    S[split].append(dict(scene=k, lab=labk[j], inst=inst[j], s=float(s[j]),
                                         n={t: (lambda r: bool(r[0] and not r[2]))(V.decide(occ > t)) for t in taus},
                                         g=(lambda r: bool(r[0] and not r[2]))(V.decide(Yg[j] > 0)),
                                         h=(lambda r: bool(r[0] and not r[2]))(V.decide((X[j, 0] > 0) | (X[j, 1] > 0)))))
        vc = [r for r in S['val'] if r['lab'] == 'CANE_near']
        theta = float(np.quantile([r['s'] for r in vc], .95))           # val CANE_near alert rate <= 5 %
        ok = [t for t in taus if np.mean([r['n'][t] for r in vc]) <= .05]
        tau = max(ok, key=lambda t: np.mean([r['n'][t] for r in vc])) if ok else None
        T = S['test']
        R = dict(theta=round(theta, 4), tau=tau,
                 D=tally([(r['scene'], r['lab'], r['inst'], r['s'] > theta) for r in T]),
                 N=tally([(r['scene'], r['lab'], r['inst'], r['n'][tau]) for r in T]) if tau else None,
                 G=tally([(r['scene'], r['lab'], r['inst'], r['g']) for r in T]),
                 H=tally([(r['scene'], r['lab'], r['inst'], r['h']) for r in T]),
                 curve_D=[tally([(r['scene'], r['lab'], r['inst'], r['s'] > q) for r in T]) | dict(thr=float(q))
                          for q in np.quantile([r['s'] for r in T if r['lab'] == 'CANE_near'], [.80, .90, .95, .98, .99, .995])],
                 curve_N={str(t): tally([(r['scene'], r['lab'], r['inst'], r['n'][t]) for r in T]) for t in taus})
        rep['seeds'][seed] = R
        print('seed', seed, 'theta', R['theta'], 'tau', tau, '\n D', R['D'], '\n N', R['N'], flush=True)
    rep['G'], rep['H'] = rep['seeds'][0]['G'], rep['seeds'][0]['H']
    # point-cloud rule baselines re-tallied under labels v2
    for arm, fn in (('A2s', 'ladder_A2s.json'), ('A2o', 'ladder_A2r_A2o.json'), ('A1', 'ladder_A1_A1i.json')):
        rows = json.loads((work / fn).read_text())['rows']; items = []
        for k in TEST:
            lab_rows = {r['frame']: r for r in json.loads((work / 'labels_v2' / f'{k}.json').read_text())['rows']}
            for r in rows[str(k)][arm]['rows']:
                lr = lab_rows.get(r['frame'])
                if lr is None or not strata(lr):
                    continue
                lab = (({'MISS': 'MISS', 'CANE': 'CANE'}.get(lr['cat'], 'UNOBS') + ('_near' if lr['d_body'] <= A.NEAR else '_far'))
                       if lr['pos'] else 'NEG')
                items.append((k, lab, lr.get('inst'), r['alert']))
        rep[arm] = tally(items)
    print(' G', rep['G'], '\n H', rep['H'], '\n A2s', rep['A2s'], '\n A2o', rep['A2o'], '\n A1', rep['A1'])
    (work / 'decision_eval.json').write_text(json.dumps(rep, indent=1, default=lambda o: o.item()), encoding='utf8')


def crossval(work, epochs=10):
    """PLAN_DECISION appendix: 5-fold scene CV over training scenes with D8 (8-channel grid only), H and G."""
    import torch
    work = Path(work); folds = {f: [k for k in TRAIN if k % 5 == f] for f in range(5)}
    cache = {k: scene_xy(work, k, 0) for k in TRAIN + VAL}             # (X 9ch, y, labk, prim, inst, Z, keep)
    pooled = dict(D8=[], H=[], G=[]); scores = []
    for f, held in folds.items():
        torch.manual_seed(100 + f)
        tr = [k for k in TRAIN if k not in held]
        X = torch.from_numpy(np.concatenate([cache[k][0][:, :8].astype(np.float16) for k in tr]))
        Y = torch.from_numpy(np.concatenate([cache[k][1] for k in tr]))
        pw = min(float((len(Y) - Y.sum()) / max(Y.sum(), 1)), 100.)
        net = make_head(8).cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Y) // 256 + 1))
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device='cuda'))
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Y))
            for b in range(0, len(perm), 256):
                i = perm[b:b + 256]; loss = lossf(net(V.pad(X[i].cuda().float())), Y[i].cuda())
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()

        def score(k):
            with torch.no_grad():
                return torch.sigmoid(net(V.pad(torch.from_numpy(cache[k][0][:, :8]).cuda().float()))).cpu().numpy()
        vc = [s for k in VAL for s, l, p in zip(score(k), cache[k][2], cache[k][3]) if p and l == 'CANE_near']
        theta = float(np.quantile(vc, .95))
        for k in held:
            Xk, y, labk, prim, inst, Z, keep = cache[k]; s = score(k); Yg = np.asarray(Z['Y'])[keep]
            for j in np.flatnonzero(prim):
                pooled['D8'].append((k, labk[j], inst[j], bool(s[j] > theta)))
                r = V.decide(Yg[j] > 0); pooled['G'].append((k, labk[j], inst[j], bool(r[0] and not r[2])))
                r = V.decide((Xk[j, 0] > 0) | (Xk[j, 1] > 0)); pooled['H'].append((k, labk[j], inst[j], bool(r[0] and not r[2])))
                scores.append((k, labk[j], inst[j], float(s[j])))
        print('fold', f, 'theta', round(theta, 4), 'D8', tally([x for x in pooled['D8'] if x[0] in held]), flush=True)
    rep = {a: tally(v) for a, v in pooled.items()}
    cane = np.array([s for k, l, i, s in scores if l == 'CANE_near'])
    rep['curve_D8_pooled_scores'] = [tally([(k, l, i, s > q) for k, l, i, s in scores]) | dict(cane_rate=r)
                                     for r, q in ((r, float(np.quantile(cane, 1 - r))) for r in (.01, .02, .05, .10, .20))]
    (work / 'decision_cv.json').write_text(json.dumps(rep, indent=1, default=lambda o: o.item()), encoding='utf8')
    print(json.dumps({a: rep[a] for a in ('D8', 'H', 'G')}, indent=1))
    for c in rep['curve_D8_pooled_scores']:
        print('curve', c)


# ---------------------------------------------------------------- calibrated folds + event-level metrics (PLAN_EVENT.md)
def scene_records(work, k):
    """All labelled valid frames of scene k in time order: X (8 ch), frame ids, labels, instances, d_body, H/G decisions."""
    Z = np.load(Path(work) / 'grids' / f'{k}.npz')
    rows = {r['frame']: r for r in json.loads((Path(work) / 'labels_v2' / f'{k}.json').read_text())['rows']}
    keep, lab, inst, d = [], [], [], []
    for j, fr in enumerate(Z['frames']):
        r = rows.get(int(fr))
        if not Z['valid'][j] or r is None or 'skip' in r:
            continue
        keep.append(j)
        if r['pos']:
            c = {'MISS': 'MISS', 'CANE': 'CANE'}.get(r['cat'], 'UNOBS')
            lab.append(c + ('_near' if r['d_body'] <= A.NEAR else '_far')); inst.append(r['inst']); d.append(r['d_body'])
        else:
            lab.append('NEG'); inst.append(None); d.append(None)
    keep = np.array(keep, int); X = Z['X'][keep]; Y = Z['Y'][keep]
    dec = lambda occ: (lambda r: bool(r[0] and not r[2]))(V.decide(occ))
    return dict(X=X, frames=np.asarray(Z['frames'])[keep], lab=lab, inst=inst, d=d,
                H=np.array([dec((x[0] > 0) | (x[1] > 0)) for x in X]), G=np.array([dec(y > 0) for y in Y]))


def event_metrics(recs, alerts):
    """recs: {scene: record}; alerts: {scene: bool array}. Events = per-object MISS_near runs (gap <= 1 s)."""
    ev, caught, t07, t10, nuis, eps, nframes = 0, 0, 0, 0, 0, 0, 0
    for k, R in recs.items():
        a = alerts[k]; fr = R['frames']; nframes += len(fr)
        runs = {}
        for j, (l, i) in enumerate(zip(R['lab'], R['inst'])):
            if l == 'MISS_near':
                rl = runs.setdefault(i, [])
                if rl and fr[j] - fr[rl[-1][-1]] <= 10:
                    rl[-1].append(j)
                else:
                    rl.append([j])
        for rl in runs.values():
            for run in rl:
                ev += 1; hit = [j for j in run if a[j]]
                if hit:
                    caught += 1; d0 = R['d'][hit[0]]
                    t07 += d0 >= .7; t10 += d0 >= 1.
        j = 0
        while j < len(fr):                                                # alert episodes: consecutive alerts, gap <= 1 sample
            if not a[j]:
                j += 1; continue
            e = [j]
            while j + 1 < len(fr) and a[j + 1] and fr[j + 1] - fr[e[-1]] <= 4:
                j += 1; e.append(j)
            eps += 1; nuis += not any(R['lab'][q] == 'MISS_near' for q in e); j += 1
    minutes = nframes * .2 / 60
    return dict(events=ev, caught=caught, timely_07=t07, timely_10=t10, episodes=eps, nuisance=nuis,
                minutes=round(minutes, 1), nuisance_per_min=round(nuis / minutes, 3))


def events(work, epochs=10, confirm=None):
    import torch
    confirm = tuple(int(v) for v in confirm.split('of')) if confirm else None; epochs = int(epochs)
    work = Path(work); folds = {f: [k for k in TRAIN if k % 5 == f] for f in range(5)}
    recs = {k: scene_records(work, k) for k in TRAIN}
    ys = {k: np.array([l == 'MISS_near' for l in recs[k]['lab']], np.float32) for k in TRAIN}
    scores, thetas = {}, {}
    for f in range(5):
        cal = folds[(f + 1) % 5]; tr = [k for k in TRAIN if k not in folds[f] and k not in cal]
        torch.manual_seed(100 + f)
        X = torch.from_numpy(np.concatenate([recs[k]['X'] for k in tr])); Y = torch.from_numpy(np.concatenate([ys[k] for k in tr]))
        pw = min(float((len(Y) - Y.sum()) / max(Y.sum(), 1)), 100.)
        net = make_head(8).cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Y) // 256 + 1))
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device='cuda'))
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Y))
            for b in range(0, len(perm), 256):
                i = perm[b:b + 256]; loss = lossf(net(V.pad(X[i].cuda().float())), Y[i].cuda())
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()

        def score(k):
            with torch.no_grad():
                return np.concatenate([torch.sigmoid(net(V.pad(torch.from_numpy(recs[k]['X'][b:b + 1024]).cuda().float()))).cpu().numpy()
                                       for b in range(0, len(recs[k]['X']), 1024)])
        cc = np.concatenate([score(k)[[l == 'CANE_near' for l in recs[k]['lab']]] for k in cal])
        thetas[f] = float(np.quantile(cc, .95))
        for k in folds[f]:
            scores[k] = score(k)
        print('fold', f, 'theta', round(thetas[f], 4), 'train MISS frames', int(Y.sum()), flush=True)
    test = {k: recs[k] for k in TRAIN}
    fold_of = {k: f for f, ks in folds.items() for k in ks}
    np.savez(work / 'decision_event_scores.npz', **{f's{k}': scores[k] for k in TRAIN},
             **{f't{f}': np.array(t) for f, t in thetas.items()})
    if confirm:                                                       # post-hoc (PLAN_EVENT appendix): m-of-n confirmation
        m, n = confirm

        def conf(a, fr):
            out = np.zeros_like(a)
            for j in range(len(a)):
                w = [q for q in range(max(0, j - n + 1), j + 1) if fr[j] - fr[q] <= 2 * (n - 1)]
                out[j] = sum(a[q] for q in w) >= m
            return out
        rep_c = {}
        for name, al in (('D8', {k: scores[k] > thetas[fold_of[k]] for k in TRAIN}),
                         ('H', {k: recs[k]['H'] for k in TRAIN}), ('G', {k: recs[k]['G'] for k in TRAIN})):
            rep_c[name] = event_metrics(test, {k: conf(al[k], recs[k]['frames']) for k in TRAIN})
        allc = np.concatenate([scores[k][[l == 'CANE_near' for l in recs[k]['lab']]] for k in TRAIN])
        rep_c['curve_D8'] = [event_metrics(test, {k: conf(scores[k] > q, recs[k]['frames']) for k in TRAIN}) | dict(cane_rate=r)
                             for r, q in ((r, float(np.quantile(allc, 1 - r))) for r in (.01, .02, .05, .10, .20, .30))]
        print(f'--- {m}-of-{n} confirmation (post-hoc)')
        for a in ('D8', 'H', 'G'):
            print(a, rep_c[a])
        for c in rep_c['curve_D8']:
            print('curve', c)
        (work / f'decision_events_confirm{m}of{n}.json').write_text(json.dumps(rep_c, indent=1, default=lambda o: o.item()), encoding='utf8')
    rep = dict(thetas=thetas,
               D8=event_metrics(test, {k: scores[k] > thetas[fold_of[k]] for k in TRAIN}),
               H=event_metrics(test, {k: recs[k]['H'] for k in TRAIN}),
               G=event_metrics(test, {k: recs[k]['G'] for k in TRAIN}))
    allc = np.concatenate([scores[k][[l == 'CANE_near' for l in recs[k]['lab']]] for k in TRAIN])
    rep['curve_D8'] = [event_metrics(test, {k: scores[k] > q for k in TRAIN}) | dict(cane_rate=r)
                       for r, q in ((r, float(np.quantile(allc, 1 - r))) for r in (.005, .01, .02, .05, .10, .20))]
    (work / ('decision_events_rerun.json' if confirm else 'decision_events.json')).write_text(json.dumps(rep, indent=1, default=lambda o: o.item()), encoding='utf8')
    for a in ('D8', 'H', 'G'):
        print(a, rep[a])
    for c in rep['curve_D8']:
        print('curve', c)


if __name__ == '__main__':
    dict(relabel=relabel, probs=probs, train=train, eval=evaluate, cv=crossval, events=events)[sys.argv[1]](*sys.argv[2:])

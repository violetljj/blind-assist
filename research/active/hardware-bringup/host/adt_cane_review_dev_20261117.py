"""ADT cane-reach label review: bilinear vs nearest-neighbour GT depth, and the ASE v2 rule (EXPLORE, 2026-11-17).
Rules frozen in <work>/PLAN_ADT_REVIEW.md before any new result.
Usage:
  adt_cane_review_dev_20261117.py prep    <adt data dir> <work>      (projectaria-tools venv; writes prep_b/, prep_n/)
  adt_cane_review_dev_20261117.py fmt     <work>                     (format check N vs B away from depth edges)
  adt_cane_review_dev_20261117.py label   <adt data dir> <work>      (L0 = B old rule, L1 = N old rule, L2 = N v2 rule)
  adt_cane_review_dev_20261117.py summary <adt data dir> <work>
  adt_cane_review_dev_20261117.py sheet   <work> <adt data dir>
ADT license: non-commercial research; frames and derived data stay in ignored artifacts and are never redistributed.
"""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

F, W_RAW, H_RAW, STEP = 425.0, 640, 480, 2
# v2 rule (ASE labels v2)
BODY_H, BODY_HALF, CANE_HALF, BAND2, Z_RANGE, MARGIN, MIN_VOX, VOX, WIN_S, MIN_PTS = (.60, 1.90), .25, .35, (.05, .70), (.30, 2.0), .15, 3, .03, 10., 38


# ---------------------------------------------------------------- prep (projectaria-tools venv)
def prep_seq(args):
    seq_dir, work = Path(args[0]), Path(args[1])
    from projectaria_tools.core import calibration as C
    from projectaria_tools.core import image as Im
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider
    name = seq_dir.name; old = work / 'prep' / name
    ob, on = work / 'prep_b' / name, work / 'prep_n' / name
    if (ob / 'frames.npz').exists() and (on / 'frames.npz').exists():
        return name, 'cached'
    dp = AriaDigitalTwinDataProvider(AriaDigitalTwinDataPathsProvider(str(seq_dir)).get_datapaths())
    sid = StreamId('214-1'); raw = dp.get_aria_camera_calibration(sid)
    lin = C.get_linear_camera_calibration(W_RAW, H_RAW, F, 'pinhole', raw.get_transform_device_camera())
    ts = np.load(old / 'poses.npz')['t_ns']
    rgbs, db, dn = [], [], []
    for t in ts:
        img, dep = dp.get_aria_image_by_timestamp_ns(int(t), sid), dp.get_depth_image_by_timestamp_ns(int(t), sid)
        assert img.is_valid() and dep.is_valid(), (name, int(t))
        D = dep.data().to_numpy_array()
        rgbs.append(np.ascontiguousarray(np.rot90(C.distort_by_calibration(img.data().to_numpy_array(), lin, raw), 3)))
        db.append(np.ascontiguousarray(np.rot90(C.distort_depth_by_calibration(D, lin, raw), 3)).astype(np.uint16))
        dn.append(np.ascontiguousarray(np.rot90(C.distort_by_calibration(D, lin, raw, Im.InterpolationMethod.NEAREST_NEIGHBOR), 3)).astype(np.uint16))
    for o in (ob, on):
        o.mkdir(parents=True, exist_ok=True)
        for f in ('meta.json', 'poses.npz'):
            shutil.copy2(old / f, o / f)
    np.savez_compressed(ob / 'frames.npz', rgb=np.array(rgbs), depth=np.array(db), t_ns=ts)
    np.savez_compressed(on / 'frames.npz', depth=np.array(dn), t_ns=ts)
    return name, f'{len(ts)} frames'


def prep(adt, work):
    seqs = sorted(p.name for p in (Path(work) / 'prep').iterdir() if (p / 'poses.npz').exists())
    with ProcessPoolExecutor(3) as ex:
        for name, msg in ex.map(prep_seq, [(str(Path(adt) / s), str(work)) for s in seqs]):
            print('prep', name, msg, flush=True)


# ---------------------------------------------------------------- format check
def fmt(work):
    import cv2
    work = Path(work); out = {}
    for sd in sorted((work / 'prep_n').iterdir()):
        with np.load(work / 'prep_b' / sd.name / 'frames.npz') as Bz, np.load(sd / 'frames.npz') as Nz:
            B, N = Bz['depth'][::20].astype(np.float32), Nz['depth'][::20].astype(np.float32)
        r, mixed = [], []
        for b, n in zip(B, N):
            lo, hi = cv2.erode(n, np.ones((5, 5))), cv2.dilate(n, np.ones((5, 5)))
            flat = (n > 0) & (b > 0) & (hi - lo < 30)                          # away from depth edges (< 3 cm spread)
            r.append(np.median(b[flat] / n[flat]))
            edge = (n > 0) & (b > 0) & (hi - lo > 300)
            mixed.append(float(np.mean(np.abs(b[edge] - n[edge]) > 100)) if edge.any() else 0.)
        out[sd.name] = dict(ratio_median=round(float(np.median(r)), 4), ratio_p5_p95=np.round(np.percentile(r, [5, 95]), 4).tolist(),
                            edge_px_diff_gt_10cm=round(float(np.mean(mixed)), 4))
        print(sd.name, out[sd.name], flush=True)
    (work / 'review_fmt.json').write_text(json.dumps(out, indent=1), encoding='utf8')


# ---------------------------------------------------------------- labels
def label_v2(args):
    import adt_cane_reach_dev_20261015 as M
    seq, floor, adt_seq = Path(args[0]), args[1], Path(args[2])
    boxes = M.dynamic_boxes(adt_seq)
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    vv, uu = np.mgrid[0:h:STEP, 0:w:STEP]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)
    with np.load(seq / 'frames.npz') as Z:
        depth, t_ns = Z['depth'][:, ::STEP, ::STEP], Z['t_ns']
    T = np.load(seq / 'poses.npz')['T_sc']; n = len(t_ns); scene, ground = [], []
    for j in range(n):
        z = depth[j].reshape(-1) / 1000.; ok = z > .1
        P = (T[j, :3, :3] @ (dirs[ok] * z[ok, None]).T).T + T[j, :3, 3]
        scene.append(P.astype(np.float32)); hgt = P[:, 1] - floor
        ground.append(np.unique(np.floor(P[(hgt >= BAND2[0]) & (hgt <= BAND2[1])] / VOX).astype(np.int64), axis=0))
    t = (t_ns - t_ns[0]) / 1e9; rows = []
    pos = T[:, [0, 2], 3]; lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    speed = np.linalg.norm(pos[hi] - pos[lo], axis=1) / np.maximum(t[hi] - t[lo], 1e-3)
    for i in range(n):
        Rc = M.corridor(T[i, :3, :3]); o = T[i, :3, 3]
        c = (Rc @ (scene[i] - o).T).T; hgt = scene[i][:, 1] - floor
        body = (np.abs(c[:, 0]) <= BODY_HALF) & (c[:, 2] >= Z_RANGE[0]) & (c[:, 2] <= Z_RANGE[1]) & (hgt >= BODY_H[0]) & (hgt <= BODY_H[1])
        if not body.any():
            rows.append(dict(neg=True)); continue
        held = M.inside_dynamic(scene[i][body], int(t_ns[i]), boxes)
        if (~held).sum() < MIN_PTS:
            rows.append(dict(held_only=bool(held.any()), few=True)); continue
        d_body = float(np.percentile(c[body][~held, 2], 5))
        win = np.flatnonzero(np.abs(t - t[i]) <= WIN_S)
        keys = np.concatenate([ground[j] for j in win])
        gc = (Rc @ ((keys + .5) * VOX - o).T).T
        sup = (np.abs(gc[:, 0]) <= CANE_HALF) & (gc[:, 2] >= 0) & (gc[:, 2] <= d_body + MARGIN)
        k = keys[sup] + (1 << 20); nvox = len(np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2])) if sup.any() else 0
        rows.append(dict(d_body=round(d_body, 3), support_vox=int(nvox)))
    return seq.name, dict(rows=rows, speed=np.round(speed, 3).tolist())


def label(adt, work):
    import adt_cane_reach_dev_20261015 as M
    work = Path(work); R = json.loads((work / 'adt_eval.json').read_text(encoding='utf8'))
    for tag, fn, src in (('L0', M.label_sequence, 'prep_b'), ('L1', M.label_sequence, 'prep_n'), ('L2', label_v2, 'prep_n')):
        out = work / f'review_{tag}.json'
        if out.exists():
            continue
        jobs = [(work / src / s, R[s]['floor'], Path(adt) / s) for s in R]
        with ProcessPoolExecutor(7) as ex:
            labels = dict(ex.map(fn, jobs))
        out.write_text(json.dumps(labels), encoding='utf8'); print('labelled', tag, flush=True)


# ---------------------------------------------------------------- summary
def cat_old(f, lab):
    """Old categorisation (truth px gate from adt_eval, old rule)."""
    import adt_cane_reach_dev_20261015 as M
    n = [f['truth'][q]['n'] for q in (0, 1)]
    if max(n) >= 150 and lab and not lab.get('held_only'):
        return ('CANE' if lab['support_vox'] >= M.MIN_VOX else 'MISS'), lab['d_body'] <= 1.5
    if max(n) == 0:
        return 'NEG', False
    return None, False


def cat_v2(lab):
    if lab.get('neg'):
        return 'NEG', False
    if 'd_body' not in lab:
        return None, False
    return ('CANE' if lab['support_vox'] >= MIN_VOX else 'MISS'), lab['d_body'] <= 1.5


def summary(adt, work):
    import adt_cane_reach_summary_dev_20261015 as S
    import frozen_fusion_eval_v2 as V2
    work = Path(work); R = json.loads((work / 'adt_eval.json').read_text(encoding='utf8'))
    old = json.loads((work / 'cane_reach_v3.json').read_text(encoding='utf8'))['labels']
    L = {t: json.loads((work / f'review_{t}.json').read_text(encoding='utf8')) for t in ('L0', 'L1', 'L2')}
    nd = {s: S.nearest_dynamic(work / 'prep_b' / s, Path(adt) / s) for s in R}
    rep = {}
    # reproduction: L0 vs stored labels on positive frames
    same = tot = 0
    for s, v in R.items():
        for k, f in enumerate(v['frames']):
            a, _ = cat_old(f, old[s]['rows'][k]); b, _ = cat_old(f, L['L0'][s]['rows'][k])
            if a in ('CANE', 'MISS') or b in ('CANE', 'MISS'):
                tot += 1; same += a == b
    rep['reproduction_L0'] = dict(positive_frames=tot, identical=same, rate=round(same / max(tot, 1), 4))
    print('reproduction', rep['reproduction_L0'], flush=True)
    for sub in ('walk', 'walk_noManip'):
        for tag in ('old', 'L1', 'L2'):
            c = {}; trans = {}
            for s, v in R.items():
                fr = v['frames']; spd = old[s]['speed']
                al = [np.asarray(V2.confirm([f['arms']['chain'][q][1] for f in fr]), bool) for q in (0, 1)]
                for k, f in enumerate(fr):
                    if spd[k] < .30 or (sub == 'walk_noManip' and nd[s][k] < S.MANIP_M):
                        continue
                    if tag == 'old':
                        cat, near = cat_old(f, old[s]['rows'][k])
                    elif tag == 'L1':
                        cat, near = cat_old(f, L['L1'][s]['rows'][k])
                    else:
                        cat, near = cat_v2(L['L2'][s]['rows'][k])
                    if cat is None:
                        continue
                    o, onear = cat_old(f, old[s]['rows'][k])
                    key = f'{o}{"_near" if onear else ""}->{cat}{"_near" if near else ""}'
                    trans[key] = trans.get(key, 0) + 1
                    for arm, a in (('any', bool(al[0][k] or al[1][k])), ('head', bool(al[0][k]))):
                        c[f'{arm}|{cat}'] = c.get(f'{arm}|{cat}', 0) + a
                        if near:
                            c[f'{arm}_near|{cat}'] = c.get(f'{arm}_near|{cat}', 0) + a
                    c[f'n|{cat}'] = c.get(f'n|{cat}', 0) + 1
                    if near:
                        c[f'n_near|{cat}'] = c.get(f'n_near|{cat}', 0) + 1
            alerts = sum(c.get(f'any|{x}', 0) for x in ('CANE', 'MISS', 'NEG'))
            c['redundancy_any'] = round(c.get('any|CANE', 0) / max(alerts, 1), 4)
            ha = sum(c.get(f'head|{x}', 0) for x in ('CANE', 'MISS', 'NEG'))
            c['redundancy_head'] = round(c.get('head|CANE', 0) / max(ha, 1), 4)
            rep[f'{sub}|{tag}'] = dict(counts=c, transitions_from_old=trans if tag != 'old' else None)
            print(sub, tag, json.dumps(c, sort_keys=True), flush=True)
            if tag != 'old':
                print('   transitions', json.dumps(dict(sorted(trans.items())), ensure_ascii=False), flush=True)
    (work / 'review_summary.json').write_text(json.dumps(rep, indent=1), encoding='utf8')


# ---------------------------------------------------------------- audit sheet
def sheet(work):
    import cv2
    import adt_cane_reach_summary_dev_20261015 as S
    work = Path(work); R = json.loads((work / 'adt_eval.json').read_text(encoding='utf8'))
    old = json.loads((work / 'cane_reach_v3.json').read_text(encoding='utf8'))['labels']
    L2 = json.loads((work / 'review_L2.json').read_text(encoding='utf8'))
    adt = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    nd = {s: S.nearest_dynamic(work / 'prep_b' / s, adt / s) for s in R} if adt else None
    cand = []
    for s in R:
        for k, lab in enumerate(L2[s]['rows']):
            cat, near = cat_v2(lab)
            if cat == 'MISS' and near and old[s]['speed'][k] >= .30 and (nd is None or nd[s][k] >= S.MANIP_M):
                cand.append((s, k))
    rng = np.random.default_rng(0); pick = [cand[i] for i in sorted(rng.choice(len(cand), min(16, len(cand)), replace=False))]
    tiles = []
    for s, k in pick:
        with np.load(work / 'prep_b' / s / 'frames.npz') as Z:
            im = cv2.cvtColor(Z['rgb'][k], cv2.COLOR_RGB2BGR) if Z['rgb'][k].ndim == 3 else cv2.cvtColor(Z['rgb'][k], cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (240, 320))
        cv2.putText(im, f'{s.split("_")[2][:5]}{s.split("seq")[1][:3]} f{k} d={L2[s]["rows"][k]["d_body"]:.2f}', (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(work / 'review_L2_miss_sample.jpg'), np.vstack(rows))
    (work / 'review_L2_miss_sample.json').write_text(json.dumps(dict(candidates=len(cand), picked=pick)), encoding='utf8')
    print('candidates', len(cand), 'picked', len(pick))


if __name__ == '__main__':
    cmd = sys.argv[1]
    dict(prep=lambda: prep(sys.argv[2], sys.argv[3]), fmt=lambda: fmt(sys.argv[2]), label=lambda: label(sys.argv[2], sys.argv[3]),
         summary=lambda: summary(sys.argv[2], sys.argv[3]), sheet=lambda: sheet(sys.argv[2]))[cmd]()

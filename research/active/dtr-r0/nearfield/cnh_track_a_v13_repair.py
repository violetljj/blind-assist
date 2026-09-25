"""Sequential cross-unit exact-dedup repair after parallel v1.3 generation.

Restores the single-process semantics lost by per-unit parallel generation: units
and configs are visited in order (0..N-1, 0..31); a config whose nonempty 1 cm
fingerprint equals any already-visited accepted config is regenerated with fresh
candidates (candidate ids from 1000, the same trajectory replans 0..3, the same
acceptance checks) and checked against all visited fingerprints. Gate thresholds
are unchanged; per-unit checks and G0 are recomputed for repaired units.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import cnh_track_a_v13_generate as gen
from cnh_track_a_generate import check_unit, check_coordinates, save

CANDIDATE_BASE = 1000


def key_of(fp):
    return hashlib.sha256(json.dumps(sorted(map(list, fp)), separators=(',', ':')).encode()).hexdigest()


def regenerate(u, ci, split, height, width, seen):
    for replan in range(gen.REPLANS+1):
        path = gen.trajectory(u, ci, height, replan)
        for attempt in range(32):
            candidate = CANDIDATE_BASE+replan*32+attempt
            seeds = dict(layout=gen.purpose_seed(u, -1, 'layout'), trajectory=path['seed'],
                         size_placement=gen.purpose_seed(u, ci, 'size_placement', candidate),
                         material=gen.purpose_seed(u, ci, 'material', candidate))
            srng = np.random.default_rng(seeds['size_placement'])
            mrng = np.random.default_rng(seeds['material'])
            templates = gen.object_templates(split, gen.schedule(u)[ci], srng, mrng)
            if ci < 28:
                centres, planning = gen.plan_centres(templates, path, (.8, 1.6, 2.4)[ci % 3], ci % 14 >= 7)
                boundary_plan = None
            else:
                centres, boundary_plan = gen.boundary_centres(templates, path, ci, srng)
                planning = dict(boundary=boundary_plan)
            if centres is None:
                continue
            c = gen.evaluate_candidate(u, ci, split, height, width, path, templates, centres)
            good, _ = gen.candidate_check(c, ci, boundary_plan)
            if not good:
                continue
            fp = frozenset(map(tuple, gen.fingerprint(c)))
            if fp and fp in seen.get(key_of(fp), set()):
                continue
            c.update(seeds=seeds, candidate=candidate, replan=replan, planning=planning, repaired=True,
                     fingerprint=sorted(map(list, fp)), planned_multi=bool(ci < 28 and ci % 14 >= 7),
                     boundary_plan=boundary_plan, planned_patterns=list(gen.schedule(u)[ci]),
                     speed=path['speed'], angles_10hz=path['angles_10hz'], world_from_Q_10hz=path['world_from_Q_10hz'],
                     size_classes={str(o['id']): o['size_class'] for o in templates},
                     extents={str(o['id']): o['extent'] for o in templates})
            return c, fp
    return None, None


def run(root, n_units):
    root = Path(root)
    seen, log = {}, []
    for u in range(n_units):
        path = root/f'unit{u:02d}'/f'unit{u:02d}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        changed = False
        for i, c in enumerate(data['configs']):
            fp = frozenset(map(tuple, c['fingerprint']))
            if fp and fp in seen.get(key_of(fp), set()):
                new, nfp = regenerate(u, c['config'], data['split'], data['height'], data['width'], seen)
                log.append(dict(unit=u, config=c['config'], repaired=new is not None))
                if new is None:
                    save(root/'repair.json', dict(status='REPAIR_FAIL', log=log))
                    raise SystemExit(f'unit {u} config {c["config"]}: no non-duplicate candidate')
                data['configs'][i], fp, changed = new, nfp, True
            if fp:
                seen.setdefault(key_of(fp), set()).add(fp)
        if changed:
            data['checks'] = check_unit(data['configs'])
            data['G0'] = check_coordinates(data['configs'])
            save(path, data)
    save(root/'repair.json', dict(status='REPAIRED', repaired_configs=len(log), log=log))
    return log


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--n-units', type=int, required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--split-counts', type=int, nargs=3, required=True)
    p.add_argument('--fast-margin', action='store_true')
    a = p.parse_args()
    gen.FAMILY, gen.SPLIT_COUNTS = a.family, tuple(a.split_counts)
    if a.fast_margin:
        import cnh_track_a_v12_generate as v12
        from cnh_track_a_fastmargin import fast_margin
        v12.signed_margin = fast_margin
        gen.signed_margin = fast_margin
    log = run(a.root, a.n_units)
    print(json.dumps(dict(repaired=len(log))))

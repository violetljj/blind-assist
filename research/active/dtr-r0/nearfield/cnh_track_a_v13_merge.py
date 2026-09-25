"""Merge per-unit v1.3 geometry runs; cross-unit exact dedup (G1) and split quotas.

Units were generated in independent processes, so the global exact 1 cm voxel
dedup is re-checked here across all accepted configs; any duplicate is a hard
G1 failure. Near-duplicates (Jaccard >= .98) as in v1.2: candidate pairs from
MinHash banding (128 hashes, 32 bands x 4 rows), each candidate verified by the
exact Jaccard; a unit fails if >= 90% of its nonempty configs have a partner in
another unit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cnh_track_a_generate import check_unit, save

HASHES, BANDS = 128, 32
PRIME = (1 << 61) - 1


def signatures(fps):
    rng = np.random.default_rng(20260926)
    a, b = rng.integers(1, 2**61-2, HASHES), rng.integers(0, 2**61-2, HASHES)
    keys = [np.asarray(sorted(fp), dtype=np.int64) for _, _, fp in fps]
    sig = np.empty((len(fps), HASHES), dtype=np.uint64)
    for i, k in enumerate(keys):
        v = ((k[:, 0]*1_000_003 + k[:, 1])*1_000_033 + k[:, 2]).astype(np.uint64)
        # Vectorized universal hashing modulo 2^64 (numpy wraparound), then min.
        sig[i] = ((v[:, None]*a.astype(np.uint64)[None] + b.astype(np.uint64)[None])).min(0)
    return sig


def run(root, n_units=12, v2_gates=False, split_counts=None):
    root = Path(root)
    units, incomplete, attempts, rejects = [], [], 0, 0
    configs_by_split = {s: [] for s in ('train', 'calib', 'audit')}
    fingerprints = []
    for u in range(n_units):
        folder = root/f'unit{u:02d}'
        result = json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
        attempts += result['attempts']
        rejects += result['rejections']
        incomplete.extend(result['incomplete_units'])
        path = folder/f'unit{u:02d}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        units.append({k: v for k, v in data.items() if k != 'configs'})
        configs_by_split[data['split']].extend(dict(c, config=u*32+c['config']) for c in data['configs'])
        fingerprints.extend((u, c['config'], frozenset(map(tuple, c['fingerprint']))) for c in data['configs'] if c['fingerprint'])
    groups = {}
    for u, c, fp in fingerprints:
        groups.setdefault(hash(fp), []).append((u, c, fp))
    exact = [(a[0], a[1], b[0], b[1]) for g in groups.values() for i, a in enumerate(g) for b in g[i+1:] if a[2] == b[2]]
    sig = signatures(fingerprints)
    rows = HASHES // BANDS
    candidates = set()
    for band in range(BANDS):
        buckets = {}
        for i, row in enumerate(sig[:, band*rows:(band+1)*rows]):
            buckets.setdefault(row.tobytes(), []).append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x+1, len(members)):
                    if fingerprints[members[x]][0] != fingerprints[members[y]][0]:
                        candidates.add((members[x], members[y]))
    near = {}
    verified = 0
    for i, j in candidates:
        (ua, ca, fa), (ub, cb, fb) = fingerprints[i], fingerprints[j]
        if len(fa & fb)/max(1, len(fa | fb)) >= .98:
            verified += 1
            near.setdefault((ua, ub), set()).add(ca)
            near.setdefault((ub, ua), set()).add(cb)
    near_fail = [dict(unit=a, other=b, count=len(cs)) for (a, b), cs in near.items()
                 if len(cs) >= .9*sum(u == a for u, _, _ in fingerprints)]
    split_checks = {s: check_unit(cs) if cs else None for s, cs in configs_by_split.items()}
    for check in split_checks.values():
        if check and check['eligible_combinations'] < 20:
            check['failures'].append('split_eligible_combinations_lt20')
            check['pass_gate'] = False
    g0 = bool(units) and all(u['G0']['pass_gate'] for u in units)
    g1 = not exact and not near_fail
    g2_units = [u['unit'] for u in units if u['checks']['pass_gate']]
    if v2_gates:
        # v2: split-level label diversity is the hard gate; unit shortfalls are reported.
        g2 = all(c and c['pass_gate'] for c in split_checks.values())
        ntr, nca, nau = split_counts
        limits = dict(train=int(.05*ntr), calib=int(.05*nca), audit=int(.05*nau))
        excluded = {s: sum(1 for i in incomplete if i['split'] == s) for s in limits}
        complete = all(excluded[s] <= limits[s] for s in limits)
    else:
        g2 = len(g2_units) == n_units and all(c and c['pass_gate'] for c in split_checks.values())
        complete = len(units) == n_units and not incomplete
        limits = excluded = None
    result = dict(status='GEOMETRY_READY' if g0 and g1 and g2 and complete else 'STOP_GEOMETRY_GATE_FAIL',
                  n_units=n_units, gates='v2' if v2_gates else 'v1', completed_units=[u['unit'] for u in units],
                  incomplete_units=incomplete, exclusion=dict(limits=limits, excluded=excluded),
                  attempts=attempts, rejections=rejects, rejection_rate=rejects/max(1, attempts),
                  G0=dict(pass_gate=g0),
                  G1=dict(pass_gate=g1, exact_duplicates=exact, near_candidates=len(candidates),
                          near_verified_pairs=verified, near_fail=near_fail, nonempty_configs=len(fingerprints)),
                  G2=dict(pass_gate=g2, units_pass=g2_units,
                          unit_failures={u['unit']: u['checks']['failures'] for u in units if u['checks']['failures']},
                          split_checks=split_checks),
                  units=units)
    save(root/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--n-units', type=int, default=12)
    parser.add_argument('--v2-gates', action='store_true')
    parser.add_argument('--split-counts', type=int, nargs=3)
    args = parser.parse_args()
    r = run(args.root, args.n_units, args.v2_gates, args.split_counts)
    print(json.dumps({k: r[k] for k in ('status', 'incomplete_units', 'rejection_rate')}))
    print(json.dumps(dict(G0=r['G0']['pass_gate'], G1=r['G1']['pass_gate'], exact=len(r['G1']['exact_duplicates']),
                          near_candidates=r['G1']['near_candidates'], G2=r['G2']['pass_gate'],
                          G2_fail_units=r['G2']['unit_failures'],
                          split={s: (c['eligible_combinations'], c['failures']) if c else None
                                 for s, c in r['G2']['split_checks'].items()})))

"""Merge per-unit v1.3 geometry runs; cross-unit exact dedup (G1) and split quotas.

Units were generated in independent processes, so the global exact 1 cm voxel
dedup is re-checked here across all accepted configs; any duplicate is a hard
G1 failure. Near-duplicate (Jaccard >= .98 for >= 90% of a unit) as in v1.2.
"""
import argparse
import json
from pathlib import Path
from cnh_track_a_generate import check_unit, save


def run(root):
    root = Path(root)
    units, incomplete, attempts, rejects = [], [], 0, 0
    configs_by_split = {s: [] for s in ('train', 'calib', 'audit')}
    fingerprints = []
    for u in range(12):
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
    exact = [(a[0], a[1], b[0], b[1]) for i, a in enumerate(fingerprints) for b in fingerprints[i+1:] if a[2] == b[2]]
    near = {}
    for i, (ua, ca, fa) in enumerate(fingerprints):
        for ub, cb, fb in fingerprints[i+1:]:
            if ua != ub and len(fa & fb)/max(1, len(fa | fb)) >= .98:
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
    result = dict(status='GEOMETRY_READY' if g0 and g1 else 'STOP_INTEGRITY_FAIL',
                  completed_units=[u['unit'] for u in units], incomplete_units=incomplete,
                  attempts=attempts, rejections=rejects, rejection_rate=rejects/max(1, attempts),
                  G0=dict(pass_gate=g0), G1=dict(pass_gate=g1, exact_duplicates=exact, near_fail=near_fail,
                                                  nonempty_configs=len(fingerprints)),
                  G2=dict(report_only=True, units_pass=[u['unit'] for u in units if u['checks']['pass_gate']],
                          unit_failures={u['unit']: u['checks']['failures'] for u in units},
                          split_checks=split_checks),
                  units=units)
    save(root/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    r = run(parser.parse_args().root)
    print(json.dumps({k: r[k] for k in ('status', 'completed_units', 'incomplete_units', 'rejection_rate')}))
    print(json.dumps(dict(G1=r['G1']['pass_gate'], exact=len(r['G1']['exact_duplicates']),
                          G2_units_pass=r['G2']['units_pass'],
                          split={s: (c['eligible_combinations'], c['failures']) if c else None
                                 for s, c in r['G2']['split_checks'].items()})))

"""Run the minimal evaluation kit: intake self-check per dataset (stop that dataset if it fails), then every method on every
frame; writes report.json and report.md. Usage: run_minimal.py <out dir> [dataset ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core
from methods import METHOD_NAMES, Methods

REGIONS = ['near', 'near_nonground'] + [f'corridor_{b}' for b in core.BANDS]


def fmt(x, nd=3):
    return '—' if x is None else f'{x:.{nd}f}'


def main(out_dir, *names):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117); M = Methods(); report = {}
    for name in names or adapters.ADAPTERS:
        gen = adapters.ADAPTERS[name]
        n_all = sum(1 for _ in gen()); k = max(1, n_all // 20)
        sample = [f for i, f in enumerate(gen()) if i % k == 0][:20]
        sc = core.self_check(sample, rng)
        print(name, 'self-check', json.dumps(sc), flush=True)
        report[name] = dict(protocol=core.PROTOCOL_VERSION, self_check=sc, ref_grade=sample[0].ref_grade, frames_total=n_all)
        if not sc['pass']:
            report[name]['status'] = 'STOPPED_AT_SELF_CHECK'; continue
        rows, t0 = [], time.time()
        for f in gen():
            reg = core.regions(f, rng)
            row = dict(seq=f.seq, t=f.t, block=f.block, truth={b: reg.get(f'truth_{b}') for b in core.BANDS},
                       truth_opened={b: reg.get(f'truth_opened_{b}') for b in core.BANDS},
                       truth_dist={b: reg.get(f'truth_dist_{b}') for b in core.BANDS}, depth={}, alert={})
            for m in METHOD_NAMES:
                pred = getattr(M, m)(f)
                row['depth'][m] = {r: core.depth_metrics(pred, f.ref_depth, reg[r]) for r in REGIONS if r in reg}
                row['alert'][m] = core.band_alert(pred, f, reg)
            rows.append(row)
        report[name].update(frames=len(rows), blocks=len({r['block'] for r in rows}), seconds=round(time.time() - t0),
                            truth_counts={b: {v: sum(r['truth'][b] == v for r in rows) for v in (1, 0, -1, None)} for b in core.BANDS},
                            summary={m: core.summarize(rows, m) for m in METHOD_NAMES})
        (out / f'rows_{name}.json').write_text(json.dumps(rows, default=float), encoding='utf-8')
        print(name, 'done', len(rows), 'frames', flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    lines = [f'# Evaluation kit report (minimal, protocol {core.PROTOCOL_VERSION})', '']
    for name, r in report.items():
        lines += [f'## {name}', '', f"reference grade: {r['ref_grade']}; frames {r.get('frames', 0)}/{r['frames_total']}; "
                  f"blocks {r.get('blocks', 0)}; self-check {'PASS' if r['self_check']['pass'] else 'FAIL'} {r['self_check']['checks']}", '']
        if 'summary' not in r:
            continue
        lines += [f"truth counts per band (1 occupied / 0 empty / -1 ambiguous / None no ground): {r['truth_counts']}", '',
                  '| method | ' + ' | '.join(f'AbsRel {x}' for x in REGIONS) + ' | ' + ' | '.join(f'recall {b}' for b in core.BANDS)
                  + ' | ' + ' | '.join(f'FA frame (strict neg / opened neg) / episodes·min⁻¹ {b}' for b in core.BANDS) + ' |',
                  '|' + '---|' * (1 + len(REGIONS) + 2 * len(core.BANDS))]
        for m, s in r['summary'].items():
            lines.append(f'| {m} | ' + ' | '.join(fmt(s[x]['absrel']) for x in REGIONS) + ' | '
                         + ' | '.join(fmt(s[f'detect_{b}']['recall'], 2) for b in core.BANDS) + ' | '
                         + ' | '.join(f"{fmt(s[f'detect_{b}']['fa_frame_rate'], 2)} ({s[f'detect_{b}']['negatives']}) / {fmt(s[f'detect_{b}']['fa_frame_rate_opened'], 2)} ({s[f'detect_{b}']['negatives_opened']}) / {fmt(s[f'detect_{b}']['fa_episodes_per_min'], 2)}" for b in core.BANDS) + ' |')
        lines.append('')
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(*sys.argv[1:])

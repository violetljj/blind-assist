"""Consumed dorm pilot replay: incremental effect of the v5 contract parts.

Development description only. The pilot is consumed and these corrections were
found by inspecting it, so nothing here is independent confirmation. Tail ratios
are leave-one-recording-out over the target-free indoor donors (3, 4, 4b, 9);
other recordings use all four donors. Frozen models and cuts are unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import real_pilot_calibration as P
from local_signed_bias_adapter import ContractReplay
from real_contract_v5 import adapter_v5, tail_ratios

ROOT = P.ROOT
OUT = ROOT / 'artifacts.local/work/real-contract-v5-dev-20261011'
PREV = ROOT / 'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
DONORS = ('3', '4', '4b', '9')
ORDER = ('1', '2', '3', '4', '4b', '9', '5', '6', '7', '8')
EXPECT = {'1': 'UNKNOWN (railing)', '2': 'alarm', '3': 'alarm', '4': 'quiet*', '4b': 'quiet*', '9': 'quiet*',
          '5': 'BODY alarm', '6': 'BODY alarm', '7': 'no strong', '8': 'HEAD alarm'}
VARIANTS = (  # name, orient, tail, baseline
    ('v4_raw_order', False, False, False),
    ('v4+orient', True, False, False),
    ('v4+orient+baseline', True, False, True),
    ('v5_orient+tail', True, True, False),
    ('v5_orient+tail+baseline', True, True, True),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rates(result):
    ready = result['ready']
    out = {}
    for q, height in enumerate(('HEAD', 'BODY')):
        s = result['S_grade'][ready, q]
        out[height] = dict(S_strong=float((s == 2).mean()), S_light=float((s == 1).mean()),
                           old5=float(result['old5'][ready, q].astype(bool).mean()),
                           M3=float(result['m3'][ready, q].astype(bool).mean()), ready=int(ready.sum()))
    return out


def main():
    began = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = np.load(P.OUT / 'instrument_arrays.npz')
    bias = arrays['bias'].reshape(64, 16)
    noise = P.read('instrument_parameters.json')['noise']
    data = P.data()
    ratio_for = {sid: tail_ratios([data[d]['H'] for d in DONORS if d != sid]) for sid in ORDER}
    donor_maps = {d: tail_ratios([data[d]['H']]) for d in DONORS}
    corr = {f'{a}-{b}': float(np.corrcoef(donor_maps[a][:, 0], donor_maps[b][:, 0])[0, 1])
            for i, a in enumerate(DONORS) for b in DONORS[i+1:]}
    rows, results = [], {}
    for arm, manifest in (('old', None), ('default', PREV / 'main_real_manifest.json')):
        engine = ContractReplay(manifest)
        try:
            for name, orient, tail, baseline in VARIANTS:
                for sid in ORDER:
                    d = data[sid]
                    z = adapter_v5(d['H'], d['A'], bias, noise, ratio_for[sid] if tail else None, orient, baseline)
                    r = rates(engine.infer(z, d['seq'], d['ms']))
                    results[f'{arm}|{name}|{sid}'] = r
                    for h, v in r.items():
                        rows.append(dict(arm=arm, variant=name, segment=sid, expectation=EXPECT[sid], height=h, **v))
        finally:
            engine.close()
    with (OUT / 'replay_table.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    receipt = dict(status='COMPLETE', seconds=round(time.monotonic()-began, 3),
                   role='Consumed dorm pilot; corrections found on it; Development description only',
                   donors=DONORS, loso=True, donor_bin1_ratio_corr=corr,
                   ratio_median_by_bin={sid: np.median(v, 0).round(5).tolist() for sid, v in ratio_for.items()},
                   code_sha256={p: sha(Path(__file__).with_name(p)) for p in
                                ('real_contract_v5.py', 'real_contract_v5_replay.py', 'cnh_zone_orientation.py')},
                   bias_source_sha256=sha(P.OUT / 'instrument_arrays.npz'),
                   default_manifest_sha256=sha(PREV / 'main_real_manifest.json'), results=results)
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf8')
    for arm in ('old', 'default'):
        print(f'\n## {arm}: S strong HEAD/BODY %  [old5 / M3 in brackets for HEAD]')
        for name, *_ in VARIANTS:
            cells = []
            for sid in ORDER:
                r = results[f'{arm}|{name}|{sid}']
                cells.append(f"{sid}:{100*r['HEAD']['S_strong']:.0f}/{100*r['BODY']['S_strong']:.0f}")
            print(f'{name:26s} ' + '  '.join(cells))
    print('donor bin1 ratio-map corr', {k: round(v, 2) for k, v in corr.items()})
    print('seconds', receipt['seconds'])


if __name__ == '__main__':
    main()

"""Consumed dorm pilot replay of the nuisance-DR M3 (Development description only).

Pairs each real input path with the backbone cuts recalibrated under the matching sim
condition: orientation-only input with 'real_like_nuisance' cuts, full v5 input with
'v5_residual' cuts. Ordinary/S weights and S tau are the default candidate's.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import real_pilot_calibration as P
from local_signed_bias_adapter import ContractReplay
from real_contract_v5 import adapter_v5, tail_ratios
from real_contract_v5_replay import DONORS, ORDER, EXPECT, rates

NEW = P.ROOT / 'artifacts.local/work/cnh-nuisance-dr-m3-dev-20261011'
PAIRS = (('orient_only', 'real_like_nuisance', False, False), ('v5_full', 'v5_residual', True, True),
         ('orient_only_m3_only', 'm3_only', False, False))


def main():
    arrays = np.load(P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = P.read('instrument_parameters.json')['noise']; data = P.data()
    ratio_for = {sid: tail_ratios([data[d]['H'] for d in DONORS if d != sid]) for sid in ORDER}
    results = {}
    for path_name, cond, tail, baseline in PAIRS:
        engine = ContractReplay(NEW / f'real_manifest_{cond}.json')
        try:
            for sid in ORDER:
                d = data[sid]
                z = adapter_v5(d['H'], d['A'], bias, noise, ratio_for[sid] if tail else None, True, baseline)
                results[f'{path_name}|{sid}'] = rates(engine.infer(z, d['seq'], d['ms']))
        finally:
            engine.close()
    (NEW / 'real_replay_receipt.json').write_text(json.dumps(dict(results=results, expect=EXPECT,
        role='Consumed dorm pilot, Development description'), indent=1), encoding='utf8')
    for path_name, *_ in PAIRS:
        print(f'\n## nuisance-DR M3, {path_name}: S strong H/B% [M3 H/B%]')
        for sid in ORDER:
            r = results[f'{path_name}|{sid}']
            print(f"{sid:>3} ({EXPECT[sid]:>17s})  S {100*r['HEAD']['S_strong']:5.1f}/{100*r['BODY']['S_strong']:5.1f}"
                  f"   M3 {100*r['HEAD']['M3']:5.1f}/{100*r['BODY']['M3']:5.1f}   light {100*r['HEAD']['S_light']:.1f}/{100*r['BODY']['S_light']:.1f}")


if __name__ == '__main__':
    main()

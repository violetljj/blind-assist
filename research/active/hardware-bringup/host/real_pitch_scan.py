"""Pitch-assumption sensitivity of the nuisance-DR M3-only chain on the consumed dorm pilot.

Head-worn pitch was not measured; the frozen projection assumes -10 deg. Orientation-only
input (cnh_zone_orientation); cuts fixed from real_manifest_m3_only. Development description.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import real_pilot_calibration as P
from local_signed_bias_adapter import ContractReplay
from real_contract_v5 import adapter_v5
from real_contract_v5_replay import ORDER, rates

NEW = P.ROOT / 'artifacts.local/work/cnh-nuisance-dr-m3-dev-20261011'
PITCHES = (-25, -20, -15, -10, -5, 0, 5)

def main(manifest, tag):
    arrays = np.load(P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = P.read('instrument_parameters.json')['noise']; data = P.data()
    z = {sid: adapter_v5(data[sid]['H'], data[sid]['A'], bias, noise, None, True, False) for sid in ORDER}
    out = {}
    for pitch in PITCHES:
        engine = ContractReplay(manifest, pitch_deg=float(pitch))
        try:
            cells = []
            for sid in ORDER:
                r = rates(engine.infer(z[sid], data[sid]['seq'], data[sid]['ms'])); out[f'{pitch}|{sid}'] = r
                cells.append(f"{sid}:{100*r['HEAD']['S_strong']:.0f}/{100*r['BODY']['S_strong']:.0f}")
            print(f'pitch {pitch:+4d}  ' + '  '.join(cells), flush=True)
        finally:
            engine.close()
    (Path(manifest).parent / f'real_pitch_scan{tag}.json').write_text(json.dumps(out, indent=1), encoding='utf8')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', default=str(NEW / 'real_manifest_m3_only.json')); ap.add_argument('--tag', default='')
    a = ap.parse_args(); main(a.manifest, a.tag)

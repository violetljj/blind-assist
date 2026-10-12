"""Paired full-window detection and common-timely timing supplement.

Reads saved Development notifications only; no calibration or model execution.
The frozen metric and its completed independent audit are left untouched.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-sparse-ray-track-dev-20261010'
SOURCE = ROOT / 'artifacts.local/work/cnh-counterfactual-dev-20261009'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first(values):
    positive = values > 0
    return np.where(positive.any(axis=2), positive.argmax(axis=2), -1)


def stats(values):
    return dict(n=int(values.size), min=int(values.min()) if values.size else None,
                median=float(np.median(values)) if values.size else None,
                max=int(values.max()) if values.size else None, sum=int(values.sum()))


def run():
    start = time.monotonic()
    execution = read(OUT / 'evaluation_receipt.json')
    assert execution['status'] == 'COMPLETE'
    audit = read(OUT / 'metrics_audit_receipt.json')
    assert audit['status'] == 'PASS'
    report = read(OUT / 'metrics.json')
    author = read(SOURCE / 'scene_rows.json')
    sealed = read(OUT / 'sealed_calibration.json')
    comparisons, inputs = {}, {}
    for split in ('cal', 'validation'):
        path = OUT / f'{split}_grades_notifications.npz'
        inputs[str(path)] = sha(path)
        assert inputs[str(path)] == execution['outputs_sha256'][path.name]
        with np.load(path, allow_pickle=False) as archive:
            names = archive['keys'].tolist()
            category = archive['category']
            notices = archive['notifications']
        rows = author[split]
        assert notices.shape[3] == 13 and notices.shape[-1] == 2
        contact = category == 'contact'
        head = np.zeros_like(contact); head[:, 0] = contact[:, 0]
        body = np.zeros_like(contact); body[:, 1] = contact[:, 1]
        horizontal = np.array([r['shape_family'] == 'horizontal' for r in rows])[:, None]
        thin = np.array([abs(r['target_box']['hi'][1] - r['target_box']['lo'][1] - .04) < 1e-8
                         for r in rows])[:, None]
        dark = np.array([r['rho'] == .25 for r in rows])[:, None]
        masks = dict(HEAD=head, BODY=body, HEAD_horizontal4cm=head & horizontal & thin,
                     HEAD_horizontal4cm_dark=head & horizontal & thin & dark)
        base = first(notices[names.index('old5')])
        for name in sealed['calibrations']:
            candidate = first(notices[names.index(name)])
            layers = {}
            for layer, mask in masks.items():
                truth = np.broadcast_to(mask[:, None, :], base.shape)
                detected_base = truth & (base >= 0)
                detected_candidate = truth & (candidate >= 0)
                timely_base = detected_base & (base < 11)
                timely_candidate = detected_candidate & (candidate < 11)
                common_timely = timely_base & timely_candidate
                common_full = detected_base & detected_candidate
                delta = candidate - base
                rescue = ~detected_base & detected_candidate
                loss = detected_base & ~detected_candidate
                layers[layer] = dict(
                    denominator=int(truth.sum()), scene_denominator=int(mask.any(axis=1).sum()),
                    fullwindow_baseline_detected=int(detected_base.sum()),
                    fullwindow_candidate_detected=int(detected_candidate.sum()),
                    fullwindow_rescue=int(rescue.sum()), fullwindow_loss=int(loss.sum()),
                    fullwindow_net=int(rescue.sum() - loss.sum()),
                    common_timely=int(common_timely.sum()),
                    earlier_common_timely=int((common_timely & (delta < 0)).sum()),
                    later_common_timely=int((common_timely & (delta > 0)).sum()),
                    unchanged_common_timely=int((common_timely & (delta == 0)).sum()),
                    delta_frames_common_timely=stats(delta[common_timely]),
                    common_fullwindow=int(common_full.sum()),
                    delta_frames_common_fullwindow=stats(delta[common_full]))
                original = report['comparisons'][f'{split}/{name}']['contacts'][layer]
                assert original['common_timely'] == layers[layer]['common_timely']
                assert original['delta_frames_common_timely'] == layers[layer]['delta_frames_common_timely']
                assert int((~timely_base & timely_candidate).sum()) == original['rescue']
                assert int((timely_base & ~timely_candidate).sum()) == original['loss']
            comparisons[f'{split}/{name}'] = layers
    assert len(comparisons) == 8
    for path in (OUT / 'metrics.json', OUT / 'sealed_calibration.json',
                 OUT / 'evaluation_receipt.json', OUT / 'metrics_audit_receipt.json',
                 SOURCE / 'scene_rows.json'):
        inputs[str(path)] = sha(path)
    result = dict(status='PASS', evidence='Consumed controlled Development only',
                  source_sha256=sha(Path(__file__)), inputs_sha256=inputs,
                  executed_metrics_source_sha256=execution['source_sha256'],
                  model_forward=0, training=0, recalibration=0, fresh_e2e_access=0,
                  definitions=dict(fullwindow_frames=list(range(3, 16)), timely_frames=list(range(3, 14)),
                      detection='First nonzero saved gap1 notification at the actual contact query',
                      fullwindow_rescue='Old5 silent throughout frames3..15, candidate notified within those frames',
                      fullwindow_loss='Old5 notified within frames3..15, candidate silent throughout those frames',
                      common_timely='Both first notifications within frames3..13 at the actual contact query',
                      delay='Candidate first frame minus old5 first frame; negative is earlier',
                      units='Scene x K replica x actual contact query; K replicas are correlated',
                      intervals='No additional confidence interval or cost interval calculated'),
                  comparisons=comparisons, seconds=time.monotonic() - start)
    with (OUT / 'evaluation_supplement.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], comparisons=comparisons),
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    run()

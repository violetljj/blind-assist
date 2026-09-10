"""Read saved candidate scores; no inference, calibration or parameter updates."""
import argparse
from pathlib import Path
import numpy as np
from mz5_ensemble_readout import read, write, sha


def main(root, output):
    work = root / 'artifacts.local/work'; result = {}; inputs = {}
    for tag, folder, arm in [('MZ16', work / 'mz16-visual-detail-20260910/run-v1', 'HIGH_DETAIL'),
                             ('MZ18', work / 'mz18-body-objective-20260910/run-v1', 'BODY_WITNESS')]:
        receipt = read(folder / 'receipt.json')
        for name in ['predictions.npz', 'local_samples.npz', arm + '-cutoff.npy']:
            h = sha(folder / name); assert h == receipt['outputs'][name]; inputs[str(folder / name)] = h
        result[tag] = {}
        with np.load(folder / 'predictions.npz') as a, np.load(folder / 'local_samples.npz') as local:
            cut = np.load(folder / (arm + '-cutoff.npy'))
            for cohort in ['clean', 'stress']:
                result[tag][cohort] = {}
                for q in [1, 3]:
                    score, label, frame = [local[f'{arm}/{cohort}/{q}/{k}'] for k in ['score', 'label', 'frame']]
                    ids = np.arange(100, 125); ids = ids[a[f'{arm}/{cohort}/truth'][ids, q]]
                    detected = a[f'{arm}/{cohort}/candidate'][ids, q] >= 0
                    winner = a[f'{arm}/{cohort}/winning_query'][ids, q]
                    best = []
                    for i in ids:
                        values = score[label & (frame == i)]
                        best.append(float(values.max()) if len(values) else None)
                    above = np.array([v is not None and v >= float(cut[q]) for v in best])
                    assert not (above & ~detected).any()
                    result[tag][cohort][str(q)] = dict(opportunities=int(len(ids)), detected=int(detected.sum()),
                        winner_actual_query=int(winner.sum()), detected_actual_winner=int((winner & detected).sum()),
                        actual_candidate_available=sum(v is not None for v in best), actual_candidate_above_alert_cutoff=int(above.sum()),
                        cutoff=float(cut[q]), best_actual_margins=[v-float(cut[q]) if v is not None else None for v in best])
    write(output, dict(result=result, inputs=inputs, source_sha256=sha(Path(__file__)), interpretation='Conditional eligible known actual-query cells; trained pole regression, not independent events'))
    for tag, cohorts in result.items():
        for cohort, qs in cohorts.items():
            for q, row in qs.items(): print(tag, cohort, q, {k:v for k,v in row.items() if k != 'best_actual_margins'})


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); main(a.root,a.output)

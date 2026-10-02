"""Frozen DEPTHOR pre/post-refine localization on the existing 300 frames."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import h5py
import numpy as np
import cnh_zju_scene_calibration as prior

ROOT = prior.ROOT
OUT = ROOT / 'artifacts.local/work/cnh-zju-depthor-stages-20261002'
RUN_ID = 'CNH_ZJU_DEPTHOR_STAGES_20261002'
CASE = 'cafe2/1646382573.618292.h5'
read, save, sha = prior.read, prior.save, prior.sha


def prepare():
    assert not (OUT / 'PLAN.json').exists()
    old = read(prior.OUT / 'PLAN.json')
    seal = read(prior.OUT / 'prediction-seal.json')
    assert seal['plan_sha256'] == sha(prior.OUT / 'PLAN.json')
    rows = [dict(r, prediction_sha256=seal['files'][r['prediction']]) for r in old['cal']] + old['eval']
    assert len(rows) == 300 and len({r['id'] for r in rows}) == 300
    for i, r in enumerate(rows):
        r['stages'] = str(OUT / 'predictions' / f'{i:03d}-stages.npz')
    result = read(prior.OUT / 'result.json')
    thresholds = {'nominal': 2.1}
    for budget, answer in result['budgets'].items():
        for policy, values in answer['policies'].items():
            rule = values['rules']['depthor']
            assert rule['defined'] and not rule['all_finite']
            thresholds[f'{policy}_{budget}'] = rule['threshold']
    dependencies = {str(p): sha(p) for p in (
        Path(__file__), Path(prior.__file__), Path(prior.grid.__file__),
        prior.OUT / 'PLAN.json', prior.OUT / 'prediction-seal.json', prior.OUT / 'result.json',
        Path(__file__).with_name('mz140_depthor.py'), prior.OLD / 'depthor-zju-small.pt')}
    dependencies.update({str(p): sha(p) for p in sorted((prior.OLD / 'upstream').rglob('*.py'))})
    line = (f'| 2026-10-02 | {RUN_ID} | PRE_RUN; same140cal+160eval, all300 consumed Development; unchanged frozen forward returns depth_0/final; no train/download/new threshold | '
            'Same public nodes, sensor domain, nominal2.1 and four previously frozen thresholds; main/closer/far transition tables all frames/scenes/mixed/thin. Reproduce cachedfinal atol1e-4m, rtol0 and zero call changes on all image pixels at all5 thresholds before interpreting | '
            'Stage localization only: depth_0 already RGB+ToF fused, CSPN sparse reinsertion inactive; cafe2 bookcase retained as posthoc-selected case, no causal modality/freshconfirmation/bodyalarm claim; preserve all denominators. No automatic model modification from diagnostic | '
            '`artifacts.local/work/cnh-zju-depthor-stages-20261002/REPORT.md` |\n')
    OUT.mkdir(parents=True, exist_ok=True)
    body = prior.grid.RUNS.read_text(encoding='utf-8')
    assert RUN_ID not in body
    prior.grid.RUNS.write_text(body.rstrip() + '\n' + line, encoding='utf-8')
    (OUT / 'prerun-row.txt').write_text(line, encoding='utf-8')
    save(OUT / 'PLAN.json', dict(run_id=RUN_ID, rows=rows, thresholds=thresholds,
        dependencies=dependencies, case=CASE, preregistration=line,
        rules=dict(parity_atol=1e-4, parity_rtol=0, maximum_call_changes=0,
            outputs='Same unchanged forward depth_0 and final, clip .001..10 as original; no relative-depth metric conversion',
            evaluation='All sensor-valid public grid nodes; repeated pixels retained; GT only after seal; main1.2<=d<2.1, closer.001<d<1.2, far2.1<=d<10; unknown retained',
            attribution='Pre-refine already fused; final-stage differences do not establish RGB/ToF independent causes or benefit of deleting CSPN',
            decision='Descriptive localization, no new pass threshold, no threshold fitting or automatic stage replacement; verify original final before inference; preserve stage and final failures')))
    print('PREPARED_300', flush=True)


def verify(plan):
    for path, digest in plan['dependencies'].items():
        assert sha(path) == digest, path


def predict():
    p = read(OUT / 'PLAN.json'); verify(p)
    assert not (OUT / 'prediction-seal.json').exists()
    import torch
    from mz140_depthor import load_model
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    model, info = load_model(prior.OLD / 'upstream', prior.OLD / 'depthor-zju-small.pt', 'cuda')
    from src.utils.dataloader import dtof_to_sparse_depth
    (OUT / 'predictions').mkdir(exist_ok=True)
    timings = []; start = time.perf_counter()
    try:
        with torch.inference_mode():
            for i, row in enumerate(p['rows']):
                target = Path(row['stages']); assert not target.exists()
                assert sha(row['input']) == row['input_sha256']
                assert sha(row['prediction']) == row['prediction_sha256']
                with h5py.File(row['input'], 'r') as f:
                    rgb, hist, fr, mask = (f[k][:] for k in ('rgb', 'hist_data', 'fr', 'mask'))
                sparse = dtof_to_sparse_depth(torch.tensor(hist).float(), torch.tensor(fr), torch.tensor(mask))
                image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].cuda() / 255.
                torch.cuda.synchronize(); t = time.perf_counter()
                before, final = model(dict(image=image, sparse_depth=sparse[None].cuda()))
                torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t)
                values = [x[0, 0].float().cpu().numpy() for x in (before, final)]
                assert all(x.shape == (480, 640) and np.isfinite(x).all() for x in values)
                clipped = [np.clip(x, .001, 10) for x in values]
                with np.load(row['prediction']) as f: cached = f['depth']
                delta = np.abs(clipped[1] - cached)
                changes = {k: int(((clipped[1] < v) != (cached < v)).sum()) for k, v in p['thresholds'].items()}
                np.savez_compressed(target, before=clipped[0], final=clipped[1])
                timings.append(dict(id=row['id'], role=row['role'], ms=ms,
                    max_abs_difference=float(delta.max()), changed_values=int((delta != 0).sum()),
                    call_changes=changes, clipped_values=[int(((x < .001) | (x > 10)).sum()) for x in values]))
                if (i + 1) % 50 == 0: print(f'PREDICTED {i+1}/300', flush=True)
        passed = all(r['max_abs_difference'] <= p['rules']['parity_atol'] and not any(r['call_changes'].values()) for r in timings)
        save(OUT / 'runtime.json', dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
            model=info, seconds=time.perf_counter()-start, frames=timings, parity_pass=passed,
            backend='Existing CUDA model backend reused; scalar scoring on CPU TASK_NOT_GPU_SUITABLE'))
        save(OUT / 'prediction-seal.json', dict(plan_sha256=sha(OUT / 'PLAN.json'),
            files={r['stages']: sha(r['stages']) for r in p['rows']}))
    finally:
        del model
        torch.cuda.empty_cache()
    print(f'SEALED_300 PARITY={passed}', flush=True)


def transition(mask, before, final):
    return dict(n=int(mask.sum()), both_call=int((mask & before & final).sum()),
        added_call=int((mask & ~before & final).sum()), removed_call=int((mask & before & ~final).sum()),
        neither_call=int((mask & ~before & ~final).sum()))


def score(row, plan, seal):
    assert sha(row['input']) == row['input_sha256']
    assert sha(row['prediction']) == row['prediction_sha256']
    assert sha(row['stages']) == seal['files'][row['stages']]
    with np.load(row['stages']) as f: before, final = f['before'], f['final']
    with h5py.File(row['input'], 'r') as f:
        boxes, mask, returns, gt = f['fr'][:], f['mask'][:], f['hist_data'][:, 0], f['depth'][:]
    chunks = {k: [] for k in ('before', 'final', 'known', 'main', 'closer', 'far', 'sensor', 'mixed', 'thin')}
    zone_counts = dict(total=64, sensor_valid=0, empty=0)
    for k, box in enumerate(boxes):
        n = prior.grid.public_nodes(box, before.shape)[0]
        valid = bool(mask[k] and np.isfinite(returns[k]) and .001 < returns[k] < 10)
        zone_counts['sensor_valid'] += valid
        zone_counts['empty'] += len(n) == 0
        d = gt[n[:, 0], n[:, 1]]
        known = np.isfinite(d) & (d > .001) & (d < 10)
        near = known & (d < 2.1); far = known & (d >= 2.1)
        values = dict(before=before[n[:, 0], n[:, 1]], final=final[n[:, 0], n[:, 1]],
            known=known, main=near & (d >= 1.2), closer=near & (d < 1.2), far=far,
            sensor=np.full(len(n), valid), mixed=np.full(len(n), near.any() and far.any()),
            thin=np.full(len(n), 1 <= near.sum() <= 8 and far.any()))
        for key, v in values.items(): chunks[key].append(v)
    data = {k: np.concatenate(v) for k, v in chunks.items()}
    table = {}
    for name, threshold in plan['thresholds'].items():
        a, b = data['before'] < threshold, data['final'] < threshold
        table[name] = {group: {label: transition(data['sensor'] & subset & data[label], a, b)
            for label in ('main', 'closer', 'far')} for group, subset in (
                ('all', np.ones(len(a), dtype=bool)), ('mixed', data['mixed']), ('thin', data['thin']))}
    return dict(id=row['id'], scene=row['scene'], role=row['role'], zones=zone_counts,
        nodes=len(data['sensor']), sensor_nodes=int(data['sensor'].sum()),
        unknown=int((data['sensor'] & ~data['known']).sum()), table=table)


def total(rows, thresholds):
    return dict(frames=len(rows), nodes=sum(r['nodes'] for r in rows),
        sensor_nodes=sum(r['sensor_nodes'] for r in rows), unknown=sum(r['unknown'] for r in rows),
        zones={k: sum(r['zones'][k] for r in rows) for k in rows[0]['zones']},
        table={t: {g: {label: {k: sum(r['table'][t][g][label][k] for r in rows)
            for k in ('n', 'both_call', 'added_call', 'removed_call', 'neither_call')}
            for label in ('main', 'closer', 'far')} for g in ('all', 'mixed', 'thin')} for t in thresholds})


def evaluate():
    p = read(OUT / 'PLAN.json'); verify(p)
    assert not (OUT / 'result.json').exists()
    runtime = read(OUT / 'runtime.json')
    assert runtime['parity_pass'], 'Cached final reproduction failed; do not interpret stages'
    seal = read(OUT / 'prediction-seal.json')
    assert seal['plan_sha256'] == sha(OUT / 'PLAN.json')
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda row: score(row, p, seal), p['rows']))
    result = dict(verdict='STAGE_LOCALIZATION_DEVELOPMENT',
        roles={role: total([r for r in rows if r['role'] == role], p['thresholds']) for role in ('cal', 'eval')},
        scenes={s: total([r for r in rows if r['scene'] == s], p['thresholds']) for s in sorted({r['scene'] for r in rows})},
        case=next(r for r in rows if r['id'] == CASE), seconds=time.perf_counter()-start)
    save(OUT / 'frame-ledger.json', rows)
    save(OUT / 'result.json', result)
    print(result['case']['table']['nominal']['all'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'predict', 'evaluate'))
    globals()[parser.parse_args().action]()

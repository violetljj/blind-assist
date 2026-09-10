"""Two matched bounded fits, old-DEV cutoffs, and frozen placement replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_shared_support import matched_pair
from mz8_train import metrics
from mz15_evaluator import align_stress_labels


def balanced_local(logits, target, known):
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
    known = known.expand_as(target)
    values = []
    for mask in [known & target, known & ~target]:
        total = mask.flatten(1).sum(1)
        valid = total > 0
        if valid.any():
            values.append(((loss*mask).flatten(1).sum(1)/total.clamp_min(1))[valid].mean())
    return torch.stack(values).mean() if values else logits.sum()*0


def cutoff_zero_added(raw, support, baseline, truth):
    result = []
    for q in range(4):
        eligible = support[:, q] & (baseline[:, q] < 0)
        negative = eligible & ~truth[:, q]
        result.append(np.nextafter(np.float64(raw[negative, q].max()), np.inf)
            if negative.any() else (-np.inf if eligible.any() else np.inf))
    return np.array(result)


def main(root, cache, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    work = root/'artifacts.local/work'
    old = work/'mz8-attribution-20260910/cache-v5'
    labels = work/'mz9-source-supervision-20260910/labels-v1'
    run9, run13 = work/'mz9-source-supervision-20260910/run-v1', work/'mz13-training-coverage-20260910/run-v1'
    inputs = {}
    def bind(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        inputs[str(path)] = digest
    receipt = read(cache/'receipt.json')
    assert receipt['status'] == 'PASS'
    for folder, names, hashes in [(old, ['dense.npy', 'observations.npz', 'evaluator.npz'], read(old/'receipt.json')['files']),
            (cache, ['dense.npy', 'observations.npz', 'evaluator.npz', 'selected.json'], receipt['files'])]:
        for name in names:
            bind(folder/name, hashes[name])
    bind(labels/'evaluator.npz', read(labels/'receipt.json')['labels_sha256'])
    bind(run9/'normalization.npz', read(run9/'receipt.json')['outputs']['normalization.npz'])
    bind(run13/'predictions.npz', read(run13/'receipt.json')['outputs']['predictions.npz'])
    bind(work/'mz14-evidence-trace-20260910/run-v2/traces.npz')
    d, new = load_npz(old/'observations.npz'), load_npz(cache/'observations.npz')
    oldlab, newlab = load_npz(labels/'evaluator.npz'), load_npz(cache/'evaluator.npz')
    oldtruth = load_npz(old/'evaluator.npz')['truth']
    truth = np.concatenate([oldtruth, newlab['truth']])
    ranges, valid = (np.concatenate([d[key], new[key]]) for key in ['ranges', 'valid'])
    source_labels = np.concatenate([oldlab['source_counts'] > 0, newlab['source_presence']])
    query_labels = np.concatenate([oldlab['query_counts'] > 0, newlab['query_presence']])
    known = np.concatenate([oldlab['cell_known_counts'] > 0, newlab['known']])
    del oldlab, newlab
    n = load_npz(run9/'normalization.npz')
    oldmaps, newmaps = (np.load(p/'dense.npy', mmap_mode='r') for p in [old, cache])
    p13 = load_npz(run13/'predictions.npz')
    cohorts = dict(TRAIN=np.flatnonzero(d['role'] == 'TRAIN_ONLY'), DEV=np.flatnonzero(d['role'] == 'DEV_ONLY'),
        clean=np.arange(3500, 3700), train_relation=np.arange(3700, 8700), train_distance=np.arange(8700, 11200),
        relation10000=np.arange(11200, 13200), distance5000=np.arange(13200, 14200))
    for name, ids in cohorts.items():
        np.testing.assert_array_equal(truth[ids], p13[name+'/truth'])
    records = read(cache/'selected.json')
    assert len(records) == 10500
    assert all(r['role'] == 'TRAIN_ONLY' for r in records[:7500])
    assert all(r['role'] == 'DEV_ONLY' for r in records[7500:])
    assert not {r['site'] for r in records[:7500]} & {r['site'] for r in records[7500:]}
    torch.set_num_threads(1); torch.manual_seed(115)
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    shared, query = matched_pair()
    generator = np.random.default_rng(115)
    train_names = ['TRAIN', 'train_relation', 'train_distance', 'clean']
    batches = np.stack([np.concatenate([generator.choice(cohorts[k], 4) for k in train_names]) for _ in range(1200)])
    np.save(output/'batches.npy', batches)
    write(output/'start.json', dict(status='STARTED', steps_per_arm=1200, seed=115,
        training_cohorts={k: len(cohorts[k]) for k in train_names}, inputs=inputs,
        protocol_sha256=sha(Path(__file__).with_name('MZ15_SHARED_SUPPORT_PROTOCOL_20260910.md')),
        code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('mz15_shared_support.py')]},
        batches_sha256=sha(output/'batches.npy')))

    def get(ids, stress=False):
        images = np.empty((len(ids), 64, 18, 32), dtype=np.float32)
        old_mask = ids < 3700
        images[old_mask] = oldmaps[ids[old_mask]]
        images[~old_mask] = newmaps[ids[~old_mask]-3700]
        x = torch.from_numpy((images-n['mean'])/n['std']).cuda()
        r = d['stress_ranges'][ids-3500].astype(np.float32) if stress else ranges[ids]
        v = d['stress_valid'][ids-3500] if stress else valid[ids]
        return x, torch.from_numpy(r).cuda(), torch.from_numpy(v).cuda()

    all_arrays, results, fits, cutoffs = {}, {}, {}, {}
    names = ['DEV', 'clean', 'stress', 'relation10000', 'distance5000']
    for arm, model in [('SHARED', shared), ('QUERY', query)]:
        begin_fit = time.perf_counter()
        model = model.cuda().train()
        torch.save(model.cpu().state_dict(), output/(arm+'-initial.pt')); model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        history = []
        for step, ids in enumerate(batches):
            x, r, v = get(ids)
            out = model.inspect(x, r, v)
            sl = torch.from_numpy(source_labels[ids]).cuda()
            ql = torch.from_numpy(query_labels[ids]).cuda()
            kl = torch.from_numpy(known[ids]).cuda()[:, :, None, :, None] & v[:, :, :, None, None]
            target = sl[..., None] if model.shared else ql
            local = balanced_local(out['field'], target, kl)
            full_truth = torch.from_numpy(truth[ids]).cuda()
            positive_support = (ql & out['eligible']).flatten(1, 3).any(1)
            query_mask = out['support'] & (~full_truth | positive_support)
            qloss = F.binary_cross_entropy_with_logits(out['logits'], full_truth.float(), reduction='none')
            qloss = (qloss*query_mask).sum()/query_mask.sum().clamp_min(1)
            loss = local + .25*qloss
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if step % 200 == 0 or step == 1199:
                row = dict(step=step+1, loss=float(loss.detach()), local=float(local.detach()), query=float(qloss.detach()))
                history.append(row); print(arm, row, flush=True)
                write(output/'progress.json', dict(arm=arm, **row))
        model.eval()
        torch.save(model.cpu().state_dict(), output/(arm+'.pt')); model.cuda()
        fits[arm] = dict(steps=1200, parameters=sum(p.numel() for p in model.parameters()),
            seconds=time.perf_counter()-begin_fit, history=history)

        def infer(name, wrong=False):
            cohort = 'clean' if name == 'stress' else name
            ids = cohorts[cohort]
            outputs = []
            with torch.inference_mode():
                for begin in range(0, len(ids), 16):
                    ii = ids[begin:begin+16]
                    out = model.inspect(*get(ii, name == 'stress'), wrong_zone=wrong)
                    score = out['candidate_logits'].masked_fill(~out['eligible'], -1e6)
                    flat = score.flatten(1, 3)
                    winner = flat.argmax(1)
                    src_np, q_np = source_labels[ii], query_labels[ii]
                    if name == 'stress':
                        src_np, q_np, _ = align_stress_labels(src_np, q_np, ranges[ii], valid[ii],
                            d['stress_ranges'][ii-3500], d['stress_valid'][ii-3500])
                    src = torch.from_numpy(src_np).cuda()[..., None].expand_as(score)
                    qlab = torch.from_numpy(q_np).cuda()
                    outputs.append(dict(raw=out['logits'].cpu().numpy(), support=out['support'].cpu().numpy(),
                        winning_source=src.flatten(1, 3).gather(1, winner[:, None]).squeeze(1).cpu().numpy(),
                        winning_query=qlab.flatten(1, 3).gather(1, winner[:, None]).squeeze(1).cpu().numpy(),
                        best_true=score.masked_fill(~qlab, -1e6).flatten(1, 3).amax(1).cpu().numpy()))
            return {k: np.concatenate([o[k] for o in outputs]) for k in outputs[0]}

        dev = infer('DEV')
        cutoff = cutoff_zero_added(dev['raw'], dev['support'], p13['DEV/baseline'], p13['DEV/truth'])
        cutoffs[arm] = cutoff.tolist()
        np.save(output/(arm+'-cutoff.npy'), cutoff)
        arm_result = {}
        for name in names:
            for wrong in ([False] if name == 'DEV' else [False, True]):
                key = name+('_wrong' if wrong else '')
                a = dev if name == 'DEV' else infer(name, wrong)
                baseline, gt = p13[name+'/baseline'], p13[name+'/truth']
                margin = np.where(a['support'], a['raw'].astype(np.float64)-cutoff, -1e6)
                accepted = (baseline < 0) & (margin >= 0) & a['support']
                candidate = np.where(accepted, margin, baseline)
                a.update(margin=margin, accepted=accepted, candidate=candidate, baseline=baseline, truth=gt)
                arm_result[key] = dict(baseline=metrics(baseline, gt), candidate=metrics(candidate, gt),
                    standalone=metrics(margin, gt), added_tp=(accepted & gt).sum(0).tolist(),
                    added_fp=(accepted & ~gt).sum(0).tolist(),
                    added_positive_wrong_source=(accepted & ~a['winning_source']).sum(0).tolist(),
                    added_tp_correct_evidence=(accepted & gt & (a['best_true'] >= cutoff)).sum(0).tolist(),
                    no_candidate=(~a['support']).sum(0).tolist())
                for k, value in a.items():
                    all_arrays[arm+'/'+key+'/'+k] = value
                print('EVAL', arm, key, arm_result[key], flush=True)
        results[arm] = arm_result
        model.cpu()
        write(output/'fit-progress.json', fits)
    gates, thin, groups, special = {}, {}, {}, {}
    previous_trace = load_npz(work/'mz14-evidence-trace-20260910/run-v2/traces.npz')
    for arm in ['SHARED', 'QUERY']:
        thin[arm] = {}
        for name in ['clean', 'stress', 'clean_wrong', 'stress_wrong']:
            prefix = arm+'/'+name+'/'
            thin[arm][name] = metrics(all_arrays[prefix+'candidate'][100:125], all_arrays[prefix+'truth'][100:125])
        g = dict(no_added_fp=all(sum(results[arm][k]['added_fp']) == 0 for k in names),
            placement_far_gain=all(sum(results[arm][k]['added_tp'][1::2]) > 0 for k in ['relation10000', 'distance5000']),
            thin_clean=sum(thin[arm]['clean']['tp'][1::2]) >= 48,
            thin_stress=sum(thin[arm]['stress']['tp'][1::2]) >= 48,
            baseline_near_retained=all(all(results[arm][k]['candidate']['tp'][q] >= results[arm][k]['baseline']['tp'][q] for q in [0, 2]) for k in names))
        g['useful_effect'] = all(g.values())
        g['wrong_correspondence_reduces_far_gain'] = sum(sum(results[arm][k]['added_tp'][1::2]) for k in ['clean', 'relation10000', 'distance5000']) > sum(sum(results[arm][k+'_wrong']['added_tp'][1::2]) for k in ['clean', 'relation10000', 'distance5000'])
        gates[arm] = g
        groups[arm], special[arm] = {}, {}
        for name in ['relation10000', 'distance5000']:
            rows = [r for r in records if r['dataset'] == name]
            a = {k: all_arrays[arm+'/'+name+'/'+k] for k in ['candidate', 'baseline', 'truth', 'accepted']}
            groups[arm][name] = {}
            for field in ['site', 'group', 'family']:
                units = {}
                for i, row in enumerate(rows): units.setdefault(row[field], []).append(i)
                groups[arm][name][field] = {unit: dict(frames=len(ii), baseline=metrics(a['baseline'][ii], a['truth'][ii]),
                    candidate=metrics(a['candidate'][ii], a['truth'][ii])) for unit, ii in units.items()}
            old_added = (previous_trace[name+'/previous'] >= 0) & (previous_trace[name+'/baseline'] < 0) & previous_trace[name+'/truth']
            lost = old_added & (previous_trace[name+'/oracle_source'] < 0)
            special[arm][name] = dict(oracle_lost_opportunities=lost.sum(0).tolist(),
                recovered=(lost & (a['candidate'] >= 0)).sum(0).tolist(),
                sign_added_fp=(a['accepted'][np.array([r['family'] == 'hanging_sign' for r in rows])] & ~a['truth'][np.array([r['family'] == 'hanging_sign' for r in rows])]).sum(0).tolist())
    np.savez_compressed(output/'predictions.npz', **all_arrays)
    write(output/'result.json', dict(metrics=results, fits=fits, cutoffs=cutoffs, gates=gates,
        thin=thin, groups=groups, six_oracle_lost=special, scope='CONSUMED_DEVELOPMENT_TWO_MATCHED_FITS_NO_CONFIRMATION'))
    torch.cuda.synchronize()
    write(output/'receipt.json', dict(status='PASS', backend='CUDA', device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-started, cache_receipt_sha256=sha(cache/'receipt.json'),
        outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    print('PASS', gates, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['root', 'cache', 'output']: p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); main(a.root, a.cache, a.output)

"""No-fit TRAIN field-gradient diagnosis at the frozen MZ16 HIGH_DETAIL snapshot.

All tied amax winners are retained. Signs describe infinitesimal independent
field-score descent, not a parameter update or the preceding training history.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_train import balanced_local
from mz16_detail_readout import make_model


NAMES = ['BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR']


def losses(field, eligible, support, target, known, truth):
    pooled = field.masked_fill(~eligible, -torch.inf).flatten(1, 3).amax(1)
    logits = torch.where(support, pooled, pooled.new_full(pooled.shape, -20.))
    positive = (target & eligible).flatten(1, 3).any(1)
    mask = support & (~truth | positive)
    local = balanced_local(field, target, known)
    query = .25 * (F.binary_cross_entropy_with_logits(
        logits, truth.float(), reduction='none') * mask).sum() / mask.sum().clamp_min(1)
    return local, query, mask, logits


def analytical_check():
    # Equal maxima have exactly equal query-gradient shares, including UNKNOWN.
    field = torch.tensor([2., 2., 0.], device='cuda').reshape(1, 1, 1, 3, 1).requires_grad_()
    eligible = torch.ones_like(field, dtype=torch.bool)
    target = torch.tensor([False, False, True], device='cuda').reshape_as(field)
    known = torch.tensor([True, False, True], device='cuda').reshape_as(field)
    local, query, mask, _ = losses(field, eligible, torch.ones((1, 1), device='cuda', dtype=torch.bool),
                                  target, known, torch.ones((1, 1), device='cuda', dtype=torch.bool))
    gl = torch.autograd.grad(local, field, retain_graph=True)[0].flatten()
    gq = torch.autograd.grad(query, field)[0].flatten()
    expected = .25 * (torch.sigmoid(field.detach().flatten()[0])-1)/2
    torch.testing.assert_close(gq[:2], expected.expand(2))
    assert gq[2] == 0 and gl[0] > 0 and gl[1] == 0 and gl[2] < 0 and mask.all()
    return dict(status='PASS', query_gradient=gq.tolist(), local_gradient=gl.tolist(),
                tie_count=2, unknown_local_gradient_zero=True)


def main(root, output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    work = root/'artifacts.local/work'
    cache = work/'mz16-visual-detail-20260910/cache-v2'
    run16 = work/'mz16-visual-detail-20260910/run-v1'
    prior = work/'mz15-shared-support-20260910/run-v1'
    old = work/'mz8-attribution-20260910/cache-v5'
    new = work/'mz15-shared-support-20260910/cache-v1'
    labels = work/'mz9-source-supervision-20260910/labels-v1'
    run9 = work/'mz9-source-supervision-20260910/run-v1'
    inputs = {}

    def bind(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        inputs[str(path)] = digest

    bind(Path(__file__).with_name('MZ17_WITNESS_OBJECTIVE_PROTOCOL_20260910.md'))
    for folder in [cache, run16, prior, old, new, labels, run9]:
        bind(folder/'receipt.json')
        assert read(folder/'receipt.json')['status'] == 'PASS'
    cr, rr = read(cache/'receipt.json'), read(run16/'receipt.json')
    assert sha(cache/'receipt.json') == rr['cache_receipt_sha256']
    for name in ['ids.npy', 'dense_HIGH_DETAIL.npy']:
        bind(cache/name, cr['files'][name])
    bind(run16/'HIGH_DETAIL.pt', rr['outputs']['HIGH_DETAIL.pt'])
    for name in ['batches.npy', 'QUERY-initial.pt']:
        bind(prior/name, read(prior/'receipt.json')['outputs'][name])
    assert sha(prior/'batches.npy') == rr['outputs']['batches.npy']
    for folder, names in [(old, ['observations.npz', 'evaluator.npz']),
                          (new, ['observations.npz', 'evaluator.npz', 'selected.json'])]:
        for name in names:
            bind(folder/name, read(folder/'receipt.json')['files'][name])
    bind(labels/'evaluator.npz', read(labels/'receipt.json')['labels_sha256'])
    bind(run9/'normalization.npz', read(run9/'receipt.json')['outputs']['normalization.npz'])
    d, nd = load_npz(old/'observations.npz'), load_npz(new/'observations.npz')
    ol, nl = load_npz(labels/'evaluator.npz'), load_npz(new/'evaluator.npz')
    truth = np.concatenate([load_npz(old/'evaluator.npz')['truth'], nl['truth']])
    ranges, valid = (np.concatenate([d[k], nd[k]]) for k in ['ranges', 'valid'])
    query = np.concatenate([ol['query_counts'] > 0, nl['query_presence']])
    source = np.concatenate([ol['source_counts'] > 0, nl['source_presence']])
    known = np.concatenate([ol['cell_known_counts'] > 0, nl['known']])
    del ol, nl
    batches = np.load(prior/'batches.npy')[:64].copy()
    assert batches.shape == (64, 16)
    selected = read(new/'selected.json')
    for index in np.unique(batches):
        if index < 3500:
            assert d['role'][index] == 'TRAIN_ONLY'
        elif index < 3700:
            pass  # MZ15's disclosed targeted clean training/regression cohort.
        else:
            assert selected[index-3700]['role'] == 'TRAIN_ONLY'
    np.save(output/'batches.npy', batches)
    cache_ids = np.load(cache/'ids.npy')
    lookup = np.full(len(truth), -1, dtype=int)
    lookup[cache_ids] = np.arange(len(cache_ids))
    maps = np.load(cache/'dense_HIGH_DETAIL.npy', mmap_mode='r')
    norm = load_npz(run9/'normalization.npz')
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    sys.path.insert(0, str(root/'tools'))
    from research_backend import DeviceObservation
    initial = torch.load(prior/'QUERY-initial.pt', weights_only=True)
    checkpoint = torch.load(run16/'HIGH_DETAIL.pt', weights_only=True)
    model = make_model(initial)
    model.load_state_dict(checkpoint)
    model = model.cuda().eval().requires_grad_(False)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    check = analytical_check()
    write(output/'start.json', dict(status='STARTED', inputs=inputs, steps=0, optimizer_created=False,
          draws=int(batches.size), unique_frames=int(np.unique(batches).size),
          batch_slice='first64_saved_MZ15_batches_each16', source_sha256=sha(Path(__file__))))
    rows, history = [], []
    torch.cuda.synchronize()
    probe_started = time.perf_counter()
    for batch_index, ids in enumerate(batches):
        indices = lookup[ids]
        assert (indices >= 0).all()
        x = torch.from_numpy((np.array(maps[indices])-norm['mean'])/norm['std']).cuda()
        r, v = torch.from_numpy(ranges[ids]).cuda(), torch.from_numpy(valid[ids]).cuda()
        with torch.no_grad():
            out = model.inspect(x, r, v)
        field = out['field'].detach().requires_grad_(True)
        q = torch.from_numpy(query[ids]).cuda()
        k = torch.from_numpy(known[ids]).cuda()[:, :, None, :, None] & v[:, :, :, None, None]
        gt = torch.from_numpy(truth[ids]).cuda()
        local, query_loss, mask, logits = losses(field, out['eligible'], out['support'], q, k, gt)
        torch.testing.assert_close(logits, out['logits'], rtol=0, atol=0)
        gl = torch.autograd.grad(local, field, retain_graph=True)[0]
        gq = torch.autograd.grad(query_loss, field, retain_graph=True)[0]
        gs = torch.autograd.grad(local+query_loss, field)[0]
        torch.testing.assert_close(gs, gl+gq)
        flat = field.detach().flatten(1, 3)
        winners = out['eligible'].flatten(1, 3) & (flat == logits[:, None, :])
        arrays = [z.detach().cpu().numpy() for z in [flat, winners, q.flatten(1, 3),
                  k.expand_as(q).flatten(1, 3), gl.flatten(1, 3), gq.flatten(1, 3), gs.flatten(1, 3)]]
        ff, ww, qq, kk, ll, qg, ss = arrays
        sup, mm = out['support'].cpu().numpy(), mask.cpu().numpy()
        for i, frame in enumerate(ids):
            for j, name in enumerate(NAMES):
                winner_ids = np.flatnonzero(ww[i, :, j])
                wr = []
                for candidate in winner_ids:
                    zone, echo, cell = np.unravel_index(candidate, (64, 2, 49))
                    wr.append(dict(candidate_index=int(candidate), zone=int(zone), echo=int(echo), cell=int(cell),
                              score=float(ff[i, candidate, j]), known=bool(kk[i, candidate, j]),
                              query_positive=bool(qq[i, candidate, j]), source_positive=bool(source[frame, zone, echo, cell]),
                              local_gradient=float(ll[i, candidate, j]), weighted_query_gradient=float(qg[i, candidate, j]),
                              total_gradient=float(ss[i, candidate, j])))
                bad = [w for w in wr if w['known'] and not w['query_positive']]
                rows.append(dict(batch_index=batch_index, draw_index=batch_index*16+i, frame_id=int(frame),
                            output=name, truth=bool(truth[frame, j]), supervised=bool(mm[i, j]), support=bool(sup[i, j]),
                            tie_count=len(wr), known_incorrect_winners=len(bad),
                            unknown_winners=sum(not w['known'] for w in wr),
                            query_up_wrong=sum(w['weighted_query_gradient'] < 0 for w in bad),
                            local_down_wrong=sum(w['local_gradient'] > 0 for w in bad),
                            total_up_wrong=sum(w['total_gradient'] < 0 for w in bad), winners=wr))
        history.append(dict(batch=batch_index, local=float(local.detach()), weighted_query=float(query_loss.detach())))
    torch.cuda.synchronize()
    probe_seconds = time.perf_counter()-probe_started
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert sha(run16/'HIGH_DETAIL.pt') == rr['outputs']['HIGH_DETAIL.pt']

    def summarize(subset):
        pos = [r for r in subset if r['truth'] and r['supervised']]
        bad = [r for r in pos if r['known_incorrect_winners']]
        total_up = sum(r['total_up_wrong'] > 0 for r in bad)
        return dict(all_outputs=len(subset), gt_positive_supervised=len(pos),
                    known_incorrect_winner_queries=len(bad),
                    exclusively_known_incorrect_winner_queries=sum(r['known_incorrect_winners'] == r['tie_count'] for r in bad),
                    unknown_winner_queries=sum(r['unknown_winners'] > 0 for r in pos),
                    tied_winner_queries=sum(r['tie_count'] > 1 for r in pos),
                    known_incorrect_winner_candidates=sum(r['known_incorrect_winners'] for r in bad),
                    query_raises_wrong_queries=sum(r['query_up_wrong'] > 0 for r in bad),
                    local_suppresses_wrong_queries=sum(r['local_down_wrong'] > 0 for r in bad),
                    total_raises_wrong_queries=total_up,
                    total_raises_all_wrong_queries=sum(r['total_up_wrong'] == r['known_incorrect_winners'] for r in bad),
                    known_wrong_fraction=len(bad)/len(pos) if pos else None,
                    total_up_among_known_wrong=total_up/len(bad) if bad else None)

    summary = summarize(rows)
    gate = (summary['known_wrong_fraction'] is not None and summary['known_wrong_fraction'] >= .10
            and summary['total_up_among_known_wrong'] is not None and summary['total_up_among_known_wrong'] >= .50)
    result = dict(scope='Final MZ16 HIGH_DETAIL snapshot TRAIN field gradients; no parameter-step or historical causality claim',
                  rows=len(rows), draws=int(batches.size), unique_frames=int(np.unique(batches).size),
                  summary=summary, by_output={name: summarize([r for r in rows if r['output'] == name]) for name in NAMES},
                  admission=dict(known_wrong_threshold=.10, total_up_threshold=.50, passed=gate,
                    definition='A query counts when any known incorrect tied max winner has the indicated gradient sign; all ties retained.'),
                  analytical_check=check, loss_history=history)
    write(output/'rows.json', rows)
    write(output/'result.json', result)
    write(output/'admission.json', dict(proceed=gate, summary=summary, thresholds=result['admission'],
          rationale='At least10percent known incorrect winners among positive supervised queries and at least50percent net upward among those queries.'))
    actual = asdict(DeviceObservation(str(field.device), torch.cuda.get_device_name(), 'torch'))
    del model, before, field, out, x, r, v, gl, gq, gs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    write(output/'receipt.json', dict(status='PASS', backend='CUDA', actual_device=actual,
          seconds=time.perf_counter()-started, probe_seconds=probe_seconds, optimizer_created=False,
          training_steps=0, parameter_and_buffer_unchanged=True, checkpoint_unchanged=True,
          inference_cohorts=['saved_TRAIN_batches_only'], outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
          code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('mz15_train.py'),
            Path(__file__).with_name('mz16_detail_readout.py'), Path(__file__).with_name('mz15_shared_support.py')]}))
    print('PASS', summary, 'admission', gate, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)

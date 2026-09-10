"""Frozen evidence tracing. Native-derived masks are evaluator-only oracles."""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from mz5_ensemble_readout import read, write, sha, load_npz
from mz9_source_readout import SourceReadout
from mz9_contributors import reconstruct
from multizone64_observation import native_events, EVENT_ORDER
from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from mz8_train import metrics


@torch.inference_mode()
def trace(source, dense, ranges, valid, source_counts, query_counts, threshold):
    # Model inputs end here. All native-derived operations below are diagnostic.
    out = source.inspect(dense, ranges, valid)
    source_counts, query_counts = source_counts.long(), query_counts.long()
    eligible, evidence = out['eligible'], out['candidate_logits']
    score = (evidence + source.query_bias).double() - threshold
    flat = score.masked_fill(~eligible, -1e6).flatten(1, 3)
    winner = flat.argmax(1)
    has_source = source_counts > 0
    has_query = query_counts > 0
    def pool(mask):
        return score.masked_fill(~mask, -1e6).flatten(1, 3).amax(1)
    def take(array):
        return array.flatten(1, 3).gather(1, winner[:, None]).squeeze(1)
    values = dict(
        source=pool(eligible), available=out['support'],
        oracle_source=pool(eligible & has_source[..., None]),
        oracle_query=pool(eligible & has_query),
        selected_query_pixels=query_counts.sum((1, 2, 3)),
        eligible_query_pixels=(query_counts * eligible).sum((1, 2, 3)),
        winning_index=winner,
        winning_source_pixels=take(source_counts[..., None].expand_as(query_counts)),
        winning_query_pixels=take(query_counts),
        correct_candidate_count=(eligible & has_query).sum((1, 2, 3)),
        candidate_count=eligible.sum((1, 2, 3)),
        no_packet=~valid.flatten(1).any(1),
    )
    return {k: v.cpu().numpy() for k, v in values.items()}


def summarize(a, baseline, previous, truth):
    pos = a['source'] >= 0
    added = (previous >= 0) & (baseline < 0)
    fp = added & ~truth
    win = a['winning_source_pixels'] > 0
    qwin = a['winning_query_pixels'] > 0
    result = dict(frames=len(truth), positive=truth.sum(0).tolist(),
        source=metrics(a['source'], truth),
        evaluator_oracle_source=metrics(a['oracle_source'], truth),
        evaluator_oracle_query=metrics(a['oracle_query'], truth),
        added_tp=(added & truth).sum(0).tolist(), added_fp=fp.sum(0).tolist(),
        added_fp_winner_no_source=(fp & ~win).sum(0).tolist(),
        added_fp_winner_source_outside_query=(fp & win & ~qwin).sum(0).tolist(),
        added_fp_winner_query_under_truth_pixel_budget=(fp & qwin).sum(0).tolist(),
        added_fp_surviving_source_oracle=(fp & (a['oracle_source'] >= 0)).sum(0).tolist(),
        added_fp_surviving_query_oracle=(fp & (a['oracle_query'] >= 0)).sum(0).tolist(),
        true_selected_return_support=(truth & (a['selected_query_pixels'] >= 3)).sum(0).tolist(),
        true_eligible_return_support=(truth & (a['eligible_query_pixels'] >= 3)).sum(0).tolist(),
        true_any_correct_candidate=(truth & (a['correct_candidate_count'] > 0)).sum(0).tolist(),
        true_correct_candidate_above_threshold=(truth & (a['oracle_query'] >= 0)).sum(0).tolist(),
        source_tp_with_wrong_winner=(truth & pos & ~qwin).sum(0).tolist(),
        source_fn_no_selected_pixels=(truth & ~pos & (a['selected_query_pixels'] == 0)).sum(0).tolist(),
        source_fn_selected_but_no_correct_candidate=(truth & ~pos & (a['selected_query_pixels'] > 0) & (a['correct_candidate_count'] == 0)).sum(0).tolist(),
        source_fn_correct_candidate_below_threshold=(truth & ~pos & (a['correct_candidate_count'] > 0)).sum(0).tolist(),
        no_packet=int(a['no_packet'].sum()))
    assert sum(map(sum, [result[k] for k in ['added_fp_winner_no_source',
        'added_fp_winner_source_outside_query', 'added_fp_winner_query_under_truth_pixel_budget']])) == int(fp.sum())
    return result


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    work = root/'artifacts.local/work'
    cache = work/'mz8-attribution-20260910/cache-v5'
    run9 = work/'mz9-source-supervision-20260910/run-v1'
    run13 = work/'mz13-training-coverage-20260910/run-v1'
    inventory = work/'mz12-existing-data-20260910/inventory-v1'
    labels = work/'mz9-source-supervision-20260910/labels-v1'
    inputs = {}
    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        inputs[str(path)] = digest
    cr = read(cache/'receipt.json')
    for name in ['dense.npy', 'observations.npz', 'evaluator.npz']:
        bind(cache/name, cr['files'][name])
    bind(labels/'evaluator.npz', read(labels/'receipt.json')['labels_sha256'])
    r9 = read(run9/'receipt.json')
    for name in ['SOURCE_RGB.pt', 'normalization.npz', 'result.json']:
        bind(run9/name, r9['outputs'][name])
    bind(run13/'predictions.npz', read(run13/'receipt.json')['outputs']['predictions.npz'])
    bind(inventory/'selected.json')
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    source = SourceReadout().cuda().eval()
    source.load_state_dict(torch.load(run9/'SOURCE_RGB.pt', weights_only=True))
    norm = load_npz(run9/'normalization.npz')
    threshold = torch.tensor(read(run9/'result.json')['dev']['SOURCE_RGB']['threshold'], device='cuda', dtype=torch.float64)
    d, lab = load_npz(cache/'observations.npz'), load_npz(labels/'evaluator.npz')
    p13 = load_npz(run13/'predictions.npz')
    maps = np.load(cache/'dense.npy', mmap_mode='r')
    rows, arrays, summaries, parity = [], {}, {}, {}
    write(output/'start.json', dict(status='STARTED', frames=4200, training_steps=0,
        source_sha256=sha(Path(__file__)), protocol_sha256=sha(Path(__file__).with_name('MZ14_EVIDENCE_TRACE_PROTOCOL_20260910.md'))))

    def save_cohort(name, values, records):
        a = {k: np.concatenate([v[k] for v in values]) for k in values[0]}
        expected = p13[name+'/source']
        mismatch = np.argwhere((a['source'] >= 0) != (expected >= 0))
        # Float arithmetic order may differ at a frozen cutoff. Preserve both.
        parity[name] = dict(sign_mismatches=mismatch.tolist(), max_error_supported=float(
            np.abs(a['source'][a['available']]-expected[a['available']]).max(initial=0)))
        assert len(mismatch) == 0, (name, mismatch.tolist())
        baseline, previous, truth = (p13[name+'/'+k] for k in ['baseline', 'candidate', 'truth'])
        a.update(baseline=baseline, previous=previous, truth=truth)
        summaries[name] = summarize(a, baseline, previous, truth)
        for key, val in a.items():
            arrays[name+'/'+key] = val
        rows.extend([dict(cohort=name, row=i, **r) for i, r in enumerate(records)])
        print('TRACE', name, summaries[name], flush=True)

    for name, ids in [('DEV', np.flatnonzero(d['role'] == 'DEV_ONLY')), ('clean', np.arange(3500, 3700))]:
        values = []
        for begin in range(0, len(ids), 32):
            ii = ids[begin:begin+32]
            dense = torch.from_numpy((np.array(maps[ii])-norm['mean'])/norm['std']).cuda()
            values.append(trace(source, dense, torch.from_numpy(d['ranges'][ii]).cuda(),
                torch.from_numpy(d['valid'][ii]).cuda(), torch.from_numpy(lab['source_counts'][ii]).cuda(),
                torch.from_numpy(lab['query_counts'][ii]).cuda(), threshold))
        save_cohort(name, values, [dict(cache_index=int(i), **cr['inputs'][i]) for i in ids])

    base = work/'body-query-10000-b-20260909/run-v1'
    decoder = work/'body-query-context-decoder-20260909/run-v1'
    for name, digest in FROZEN.items():
        bind((base if name == 'NEW-step2000.pt' else decoder)/name, digest)
    context = ContextEvidence(base, decoder, work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
    m = context.base
    selected = read(inventory/'selected.json')
    for name in ['relation10000', 'distance5000']:
        records = [r for r in selected if r['dataset'] == name]
        values = []
        for begin in range(0, len(records), 16):
            batch = records[begin:begin+16]
            images, depths = [], []
            for row in batch:
                assert row['role'] == 'DEV_ONLY'
                bind(row['rgb'], row['rgb_sha']); bind(row['native'], row['native_sha'])
                with Image.open(row['rgb']) as im:
                    images.append(np.array(im.convert('RGB').resize((256, 144), Image.Resampling.BOX)))
                depths.append(np.load(row['native'], allow_pickle=False))
            with torch.inference_mode():
                x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).cuda().float()/255
                deep, shallow = m.extract((x-m.image_mean)/m.image_std)
                deep = F.interpolate(m.deep_projection(deep), size=(18, 32), mode='bilinear', align_corners=False)
                dense = torch.cat([deep, m.detail(shallow)], 1)
                dense = (dense-torch.from_numpy(norm['mean']).cuda())/torch.from_numpy(norm['std']).cuda()
                depth = torch.from_numpy(np.stack(depths)).cuda()
                packet = reconstruct(depth)
                r, v = torch.nan_to_num(packet['range_m']).float(), packet['valid']
                values.append(trace(source, dense, r, v, packet['source_counts'], packet['query_counts'], threshold))
                truth = native_events(depth, crop=False)['events'].cpu().numpy()
                np.testing.assert_array_equal(truth, p13[name+'/truth'][begin:begin+len(batch)])
            if begin % 320 == 0:
                print('EXTRACT', name, begin+len(batch), '/', len(records), flush=True)
        save_cohort(name, values, records)
    # Every classified bit has a trace, including all previous added false alerts.
    cases = []
    for row in rows:
        name, i = row['cohort'], row['row']
        for q in range(4):
            added = arrays[name+'/previous'][i, q] >= 0 and arrays[name+'/baseline'][i, q] < 0
            pole = name == 'clean' and 100 <= i < 150 and q in [1, 3]
            if not (added and not arrays[name+'/truth'][i, q] or pole):
                continue
            idx = int(arrays[name+'/winning_index'][i, q])
            case = dict(**row, query=EVENT_ORDER[q], q=q, zone=idx//98, echo=(idx//49)%2, cell=idx%49)
            for key in ['source', 'baseline', 'previous', 'oracle_source', 'oracle_query', 'truth',
                        'selected_query_pixels', 'eligible_query_pixels', 'winning_source_pixels', 'winning_query_pixels']:
                case[key] = arrays[name+'/'+key][i, q].item()
            cases.append(case)
    thin = slice(100, 125)
    summaries['thin_pole_single_configuration'] = summarize(
        {k.split('/', 1)[1]: v[thin] for k, v in arrays.items() if k.startswith('clean/')},
        arrays['clean/baseline'][thin], arrays['clean/previous'][thin], arrays['clean/truth'][thin])
    write(output/'result.json', dict(scope='CONSUMED_DEVELOPMENT_DIAGNOSTIC_ORACLES_NOT_INFERENCE',
        cohorts=summaries, parity=parity, event_order=EVENT_ORDER))
    write(output/'cases.json', cases)
    write(output/'selected.json', rows)
    np.savez_compressed(output/'traces.npz', **arrays)
    torch.cuda.synchronize()
    write(output/'receipt.json', dict(status='PASS', frames=4200, training_steps=0, cutoff_changes=0,
        backend='CUDA', device=torch.cuda.get_device_name(), seconds=time.perf_counter()-started,
        source_sha256=sha(Path(__file__)), inputs=inputs,
        outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    print('PASS', time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)

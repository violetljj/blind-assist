"""Frozen MZ15 existing-data cache. Native contributor labels stay evaluator-only."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from mz5_ensemble_readout import CompactEnsemble, load_npz, read, write, sha
from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from body_query_range import range_from_counts
from multizone64_observation import native_events, observe
from mz9_contributors import reconstruct
from mz9_source_readout import SourceReadout


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    work = root/'artifacts.local/work'
    def log(*args):
        line = ' '.join(map(str, args))
        print(line, flush=True)
        with (output/'console.log').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    inputs, frozen, rows = {}, {}, []
    train = work/'mz13-training-coverage-20260910/features-v1'
    dev = work/'mz12-existing-data-20260910/inventory-v1'
    prior_dev = work/'mz12-existing-data-20260910/run-v1'
    tr, dr, pr = read(train/'receipt.json'), read(dev/'receipt.json'), read(prior_dev/'receipt.json')
    assert all(r['status'] == 'PASS' for r in [tr, dr, pr])
    for folder, receipt, count, role, origin in [(train, tr, 7500, 'TRAIN_ONLY', 'MZ13'), (dev, dr, 3000, 'DEV_ONLY', 'MZ12')]:
        path = folder/'selected.json'
        digest = sha(path)
        assert digest == (receipt['selected_sha256'] if origin == 'MZ13' else receipt['outputs']['selected.json'])
        inputs[str(path)] = digest
        inputs[str(folder/'receipt.json')] = sha(folder/'receipt.json')
        selected = read(path)
        assert len(selected) == count and all(r['role'] == role for r in selected)
        for i, row in enumerate(selected):
            rows.append(dict(row, cache_index=len(rows), selection_origin=origin, source_row=i))
    assert len({r['rgb_sha'] for r in rows}) == 10500
    assert pr['selection_sha256'] == inputs[str(dev/'selected.json')]
    for path, expected in [(train/'features.npz', tr['files']['features.npz']),
                           (prior_dev/'predictions.npz', pr['outputs']['predictions.npz'])]:
        assert sha(path) == expected
        inputs[str(path)] = expected
    references = [load_npz(train/'features.npz'), load_npz(prior_dev/'predictions.npz')]
    for reference, selected in zip(references, [rows[:7500], rows[7500:]]):
        np.testing.assert_array_equal(reference['dataset'], [r['dataset'] for r in selected])
    base = work/'body-query-10000-b-20260909/run-v1'
    decoder = work/'body-query-context-decoder-20260909/run-v1'
    run9 = work/'mz9-source-supervision-20260910/run-v1'
    checkpoint = work/'mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    for name, digest in FROZEN.items():
        path = (base if name == 'NEW-step2000.pt' else decoder)/name
        assert sha(path) == digest
        frozen[str(path)] = digest
    r9 = read(run9/'receipt.json')
    assert r9['status'] == 'PASS'
    inputs[str(run9/'receipt.json')] = sha(run9/'receipt.json')
    for name in ['SOURCE_RGB.pt', 'normalization.npz', 'result.json']:
        assert sha(run9/name) == r9['outputs'][name]
        frozen[str(run9/name)] = sha(run9/name)
    assert sha(checkpoint) == 'ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    frozen[str(checkpoint)] = sha(checkpoint)
    source_threshold = np.array(read(run9/'result.json')['dev']['SOURCE_RGB']['threshold'])
    normalization = load_npz(run9/'normalization.npz')
    write(output/'selected.json', rows)
    write(output/'start.json', dict(status='STARTED', frames=10500, train=7500, dev=3000,
        batch_size=16, training_steps=0, source_sha256=sha(Path(__file__)),
        selected_sha256=sha(output/'selected.json'), inputs=inputs, frozen_hashes=frozen,
        scope='Existing original TRAIN plus consumed DEV; no EVAL pixels or fitting'))
    log('START', 10500, 'TRAIN7500 DEV3000', 'CUDA batch16')
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    context = ContextEvidence(base, decoder, work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
    m = context.base
    fixed = CompactEnsemble.from_checkpoint(checkpoint).cuda().eval()
    source = SourceReadout().cuda().eval()
    source.load_state_dict(torch.load(run9/'SOURCE_RGB.pt', weights_only=True))
    mean, std = torch.from_numpy(normalization['mean']).cuda(), torch.from_numpy(normalization['std']).cuda()
    dense_cache = np.lib.format.open_memmap(output/'dense.npy', mode='w+', dtype=np.float32, shape=(10500,64,18,32))
    obs = dict(ranges=np.empty((10500,64,2),np.float32), valid=np.empty((10500,64,2),bool))
    evaluator = dict(truth=np.empty((10500,4),bool), source_presence=np.empty((10500,64,2,49),bool),
        query_presence=np.empty((10500,64,2,49,4),bool), known=np.empty((10500,64,49),bool))
    parity = []
    try:
        for cohort, (offset, size) in enumerate([(0,7500), (7500,3000)]):
            for local_begin in range(0,size,16):
                begin = offset+local_begin
                batch = rows[begin:offset+min(local_begin+16,size)]
                images, depths = [], []
                for row in batch:
                    assert sha(row['rgb']) == row['rgb_sha'] and sha(row['native']) == row['native_sha']
                    assert abs(row['camera']['pitch']) < 1e-6 and abs(row['camera']['roll']) < 1e-6
                    with Image.open(row['rgb']) as im:
                        assert im.size == (640,360)
                        images.append(np.array(im.convert('RGB').resize((256,144), Image.Resampling.BOX)))
                    depth = np.load(row['native'], allow_pickle=False)
                    assert depth.shape == (360,640)
                    depths.append(depth)
                x = torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
                depth = torch.from_numpy(np.stack(depths)).cuda()
                with torch.inference_mode():
                    deep, shallow = m.extract((x-m.image_mean)/m.image_std)
                    deep = F.interpolate(m.deep_projection(deep), size=(18,32), mode='bilinear', align_corners=False)
                    dense = torch.cat([deep,m.detail(shallow)],1)
                    packet = reconstruct(depth)
                    ranges, valid = torch.nan_to_num(packet['range_m']).float(), packet['valid']
                    truth = native_events(depth,crop=False)['events'].cpu().numpy()
                    end = begin+len(batch)
                    dense_cache[begin:end] = dense.cpu().numpy()
                    obs['ranges'][begin:end], obs['valid'][begin:end] = ranges.cpu().numpy(), valid.cpu().numpy()
                    evaluator['truth'][begin:end] = truth
                    evaluator['source_presence'][begin:end] = packet['source_counts'].cpu().numpy() > 0
                    evaluator['query_presence'][begin:end] = packet['query_counts'].cpu().numpy() > 0
                    evaluator['known'][begin:end] = packet['cell_known_counts'].cpu().numpy() > 0
                    np.testing.assert_array_equal(truth,references[cohort]['truth'][local_begin:local_begin+len(batch)])
                    if local_begin == 0:
                        sampled = (dense.flatten(2)@m.query_projection.T).transpose(1,2).reshape(len(batch),12,27,64)
                        mask = m.query_valid[None,:,:,None]
                        pooled = (sampled*mask).sum(2)/mask.sum(2).clamp_min(1)
                        normalized = (pooled-context.feature_mean)/context.feature_std
                        visual = torch.cat([normalized.flatten(1),range_from_counts(context.decoder(normalized)).sigmoid().flatten(1)],1)
                        tof = torch.cat([ranges.flatten(1)/4,valid.flatten(1).float()],1)
                        baseline = fixed(visual,tof).cpu().numpy()
                        result = source.inspect((dense-mean)/std,ranges,valid)
                        support = result['support'].cpu().numpy()
                        margin = np.where(support,result['logits'].cpu().numpy()-source_threshold,-1e6)
                        ref = references[cohort]
                        b = ref['baseline' if cohort == 0 else 'MZ5'][:len(batch)]
                        s = ref['source' if cohort == 0 else 'SOURCE'][:len(batch)]
                        np.testing.assert_allclose(baseline,b,atol=1e-4,rtol=0)
                        np.testing.assert_allclose(margin,s,atol=1e-4,rtol=0)
                        np.testing.assert_array_equal(baseline>=0,b>=0)
                        np.testing.assert_array_equal(margin>=0,s>=0)
                        direct = observe(depth[:1],readout='multi_surface')
                        np.testing.assert_array_equal(direct['valid'].cpu().numpy(),valid[:1].cpu().numpy())
                        range_error = float((torch.nan_to_num(direct['range_m']).float()-ranges[:1]).abs().max())
                        assert range_error < 1e-6
                        parity.append(dict(cohort=['TRAIN','DEV'][cohort],frames=len(batch),
                            baseline_max_abs=float(np.abs(baseline-b).max()),source_max_abs=float(np.abs(margin-s).max()),
                            direct_observe_range_max_abs_m=range_error,signs_equal=True))
                        log('PARITY',parity[-1])
                if local_begin%320 == 0 or local_begin+len(batch) == size:
                    dense_cache.flush()
                    write(output/'progress.json',dict(frames=end,total=10500,seconds=time.perf_counter()-started))
                    log('CACHE',end,'/10500')
        dense_cache.flush()
        del dense_cache
        np.savez_compressed(output/'observations.npz',**obs)
        np.savez_compressed(output/'evaluator.npz',**evaluator)
        assert np.all(~evaluator['query_presence'] | evaluator['source_presence'][...,None])
        assert np.all(~evaluator['source_presence'] | obs['valid'][...,None])
        assert np.all(~evaluator['source_presence'] | evaluator['known'][:,:,None,:])
        torch.cuda.synchronize()
        write(output/'parity.json',dict(status='PASS',checks=parity,all_truth_frames_equal=10500))
        log('PASS',10500,'seconds',time.perf_counter()-started)
        files = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
        write(output/'receipt.json',dict(status='PASS',frames=10500,train=7500,dev=3000,training_steps=0,
            backend='CUDA',device=torch.cuda.get_device_name(),batch_size=16,cudnn_allow_tf32=True,matmul_allow_tf32=False,
            seconds=time.perf_counter()-started,source_sha256=sha(Path(__file__)),inputs=inputs,frozen_hashes=frozen,
            selected_sha256=sha(output/'selected.json'),files=files,parity=parity,
            role_counts=dict(Counter(r['role'] for r in rows)),dataset_role_counts=dict(Counter(r['dataset']+'/'+r['role'] for r in rows)),
            schema=dict(dense=[10500,64,18,32],observations={k:list(v.shape) for k,v in obs.items()},evaluator={k:list(v.shape) for k,v in evaluator.items()}),
            scope='Frozen existing-data cache; native truth/contributor masks evaluator-only; no new collection, fitting, EVAL pixels, or independent confirmation'))
    except BaseException:
        log(traceback.format_exc())
        write(output/'failure.json',dict(status='FAILED',seconds=time.perf_counter()-started,source_sha256=sha(Path(__file__))))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    main(args.root,args.output)

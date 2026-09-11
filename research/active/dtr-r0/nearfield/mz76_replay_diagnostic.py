"""Engineering diagnosis of frozen MZ70/MZ76 parity; no fit or cutoff changes.

Probe selects the 16-row MZ76 block containing each source's largest saved raw
discrepancy. This outcome-selected, consumed diagnostic is not fresh evaluation.
Only tensor layout changes between paired head calls. Historical checkpoints,
source packets, RGB members and prediction files are hash-bound and read-only.
"""
import argparse
import gc
import time
import traceback
import shutil
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha
from mz74_return_survival import SOURCES, setup, load_models, DesignBindings, RGBStore, DESIGN_SHA
from mz76_observed_reliability import open_images, full_features, observation, R70_SHA
from mz56_global_anchor import saved_outputs
from mz54_full_rgb import parameters_sha
from mz36_frozen_inference import fixed_batch_dense


def nchw_features(images, models):
    """Explicit encoder layout; pixel values and frozen arithmetic unchanged."""
    rgb = torch.from_numpy(np.stack([np.asarray(im,np.uint8) for im in images])).permute(0,3,1,2).cuda().float()/255
    packed = rgb.contiguous()
    assert torch.equal(rgb, packed)
    dense = fixed_batch_dense(models.context.base, packed)
    return (dense-models.detail_mean)/models.detail_std


def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(max_abs=float(delta.max()), exact=bool(np.array_equal(actual, expected)),
                violations=int((delta > 2e-5 + 1e-6 * np.abs(expected)).sum()),
                values=int(actual.size))


def run(root, out):
    out = out.resolve()
    assert out.is_relative_to((root/'artifacts.local').resolve())
    out.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    design_path = root/'artifacts.local/work/mz69-topology-transfer-20260911/design-inputs.json'
    assert sha(design_path) == DESIGN_SHA
    design = read(design_path); bind = DesignBindings(design)
    bind(design_path, DESIGN_SHA)
    shutil.copyfile(__file__, out/'source.py'); bind(out/'source.py')
    work = root/'artifacts.local/work'
    old = work/'mz70-diverse-learning-20260911/run-v1'
    recent = work/'mz76-observed-reliability-20260911/run-v1'
    r70 = read(bind(old/'receipt.json', R70_SHA))
    r76 = read(bind(recent/'receipt.json'))
    prior_path = bind(old/'predictions.npz', r70['outputs']['predictions.npz'])
    recent_path = bind(recent/'predictions.npz', r76['outputs']['predictions.npz'])
    store = models = spatial = heads = None
    try:
        setup()
        models, spatial, heads, cuts, checkpoints = load_models(root, bind, design, out)
        head = heads['DIVERSE']; before = parameters_sha(head)
        store = RGBStore(); summaries = []; arrays = {}
        schedule_path = bind(old/'schedule.npz', r70['outputs']['schedule.npz'])
        with np.load(prior_path) as prior, np.load(recent_path) as saved, np.load(schedule_path) as schedule, torch.inference_mode():
            for source, (name, digest, loader) in SOURCES.items():
                data = loader(work/name, bind, digest)[0]
                all_ids = saved[source+'/original_indices']
                np.testing.assert_array_equal(data['frame_ids'][all_ids], saved[source+'/frame_ids'])
                key = source+'/IDEAL/MZ70/DIVERSE/raw'
                expected_all = prior[key][all_ids]
                worst = int(np.abs(saved[key]-expected_all).max(1).argmax())
                positions = np.arange(worst//16*16, worst//16*16+16)
                ids = all_ids[positions]
                write(out/'selection.json', dict(rule='largest saved raw discrepancy block per source',
                    selected_so_far=summaries, current_source=source, original_indices=ids.tolist()))
                images = open_images([data['rgb_refs'][int(i)] for i in ids], store)
                try:
                    # This is the unchanged MZ76 on-device feature path.
                    dense = full_features(images, models)
                    encoder_nchw = nchw_features(images, models)
                finally:
                    for image in images: image.close()
                contiguous = dense.contiguous()
                assert torch.equal(dense, contiguous)
                # Rebuild only nonfit encoder batches used by MZ70 FeatureStore.
                # Its 16-row evaluation blocks omit cached TRAIN rows before
                # fixed_batch_dense repeats the final nonfit image to size16.
                historical = torch.empty_like(contiguous)
                groups = []
                for start in sorted(set((ids//16*16).tolist())):
                    original_block = np.arange(start, min(start+16,len(data['ranges'])))
                    missing = original_block[~np.isin(original_block,schedule[source+'_fit_ids'])]
                    images = open_images([data['rgb_refs'][int(i)] for i in missing], store)
                    try: values = full_features(images, models)
                    finally:
                        for image in images: image.close()
                    for j,index in enumerate(ids):
                        if index in missing: historical[j] = values[int(np.flatnonzero(missing==index)[0])]
                    groups.append(dict(original_block=original_block.tolist(), encoded_ids=missing.tolist(),
                                       normalized_stride=list(values.stride())))
                row = dict(source=source, original_indices=ids.tolist(), frame_ids=data['frame_ids'][ids].tolist(),
                    direct_stride=list(dense.stride()), contiguous_stride=list(contiguous.stride()),
                    normalized_values_exact=True, historical_groups=groups,
                    historical_features_vs_direct=compare(historical.cpu().numpy(),dense.cpu().numpy()),profiles={})
                row['encoder_nchw_vs_historical_features'] = compare(encoder_nchw.cpu().numpy(),historical.cpu().numpy())
                for profile, old_profile in [('IDEAL','IDEAL'), ('CLOSEST_REPORTED_PROXY',None),
                                              ('FARTHEST_REPORTED_PROXY',None)]:
                    obs = observation(data['ranges'][ids],data['valid'][ids],profile)
                    rr = torch.from_numpy(obs['ranges']).cuda(); vv = torch.from_numpy(obs['valid']).cuda()
                    direct = saved_outputs(head, (dense,rr,vv))
                    packed = saved_outputs(head, (contiguous,rr,vv))
                    historical_output = saved_outputs(head, (historical,rr,vv))
                    nchw_output = saved_outputs(head, (encoder_nchw,rr,vv))
                    prefix = source+'/'+profile+'/MZ70/DIVERSE/'
                    result = dict(direct_vs_saved76=compare(direct['raw'],saved[prefix+'raw'][positions]),
                        contiguous_vs_direct=compare(packed['raw'],direct['raw']),
                        winner_changes=int((packed['winner']!=direct['winner']).sum()))
                    for mode, values in [('direct',direct),('contiguous',packed),('historical',historical_output),('encoder_nchw',nchw_output)]:
                        for k,v in values.items(): arrays[source+'/'+profile+'/'+mode+'/'+k] = v
                    if old_profile is not None:
                        historic = source+'/'+old_profile+'/MZ70/DIVERSE/'
                        result['direct_vs70'] = compare(direct['raw'],prior[historic+'raw'][ids])
                        result['contiguous_vs70'] = compare(packed['raw'],prior[historic+'raw'][ids])
                        result['contiguous_winner_changes_vs70'] = int((packed['winner']!=prior[historic+'winner'][ids]).sum())
                        result['historical_vs70'] = compare(historical_output['raw'],prior[historic+'raw'][ids])
                        result['historical_winner_changes_vs70'] = int((historical_output['winner']!=prior[historic+'winner'][ids]).sum())
                        result['encoder_nchw_vs70'] = compare(nchw_output['raw'],prior[historic+'raw'][ids])
                    row['profiles'][profile] = result
                summaries.append(row)
                print(source, row, flush=True)
        assert before == parameters_sha(head)
        np.savez_compressed(out/'paired-outputs.npz', **arrays)
        bind.check()
        write(out/'result.json', dict(status='DIAGNOSTIC_COMPLETE', comparison_tolerance=dict(atol=2e-5,rtol=1e-6),
            comparisons=summaries, device=torch.cuda.get_device_name(), torch_version=torch.__version__,
            cudnn_version=torch.backends.cudnn.version(), backend=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark),
            training_steps=0, new_cutoffs=0, decoded_frames=store.loads, parameters_unchanged=True,
            inputs=bind.inputs, seconds=time.perf_counter()-tick,
            outputs={'paired-outputs.npz':sha(out/'paired-outputs.npz')}))
    except BaseException:
        write(out/'failure.json',dict(error=traceback.format_exc(),inputs=bind.inputs)); raise
    finally:
        if store is not None: store.close()
        models = spatial = heads = None
        gc.collect()
        write(out/'release.json',dict(rgb_handles_closed=True, persistent_feature_cache=False,
                                     process_exit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('E:/linnan/linnan'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.root.resolve(),args.output)

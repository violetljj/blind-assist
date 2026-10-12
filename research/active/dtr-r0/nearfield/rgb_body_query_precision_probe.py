"""Fixed public-only FP16/FP32 Depth Pro numerical probe, no evaluator access."""
from __future__ import annotations
import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
from rgb_body_query_input_diagnostic import sha, write

WEIGHT_SHA = '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
PUBLIC = {'environment','scan','split','frame','rgb_path','rgb_sha256','color_K','color_shape','depth_K','depth_shape'}


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def prepare(repo, output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'roster.json').exists():
        raise FileExistsError('Preserve frozen roster')
    work = repo/'artifacts.local/work'
    root = work/'rgb-body-query-query-level-dev-20261009'
    sources = [('arkit16', root/'fixed-grid-sensor/arkit16'),
               ('arkit_40777060', root/'additional-arkit-sensor/40777060'),
               ('arkit_40777065', root/'additional-arkit-sensor/40777065')]
    rows, inputs = [], []
    def choose(cohort, observations, predictions, scan=None):
        obs, pm = load(observations), load(predictions)
        assert pm['status'] == 'COMPLETE' and pm['weight_sha256'] == WEIGHT_SHA
        rr = [r for r in obs['rows'] if scan is None or r['scan'] == scan]
        assert all(set(r) == PUBLIC for r in rr)
        arkit = all(r['scan'].startswith('arkitscenes_') for r in rr)
        if arkit:
            assert len(rr) == 16
            key = lambda r: (float(Path(r['rgb_path']).stem.rsplit('_',1)[1]), r['frame'])
        else:
            assert len(rr) == (24 if cohort == 'original_validation24' else 64)
            key = lambda r: (r['frame'], r['scan'])
        ordered = sorted(rr, key=key)
        lookup = {(p['scan'],p['frame']):p for p in pm['rows']}
        for endpoint, r in [('first', ordered[0]), ('last', ordered[-1])]:
            p = lookup[r['scan'],r['frame']]
            assert p['rgb_sha256'] == r['rgb_sha256']
            rows.append(dict(r, cohort=cohort, endpoint=endpoint, selection_sort_key=list(key(r)),
                cached_fp16_path=p['path'], cached_fp16_sha256=p['sha256'],
                source_observations_path=str(observations), source_predictions_path=str(predictions)))
        inputs.extend(dict(path=str(p), sha256=sha(p)) for p in (observations,predictions))
    for cohort, folder in sources:
        choose(cohort, folder/'observations.json', folder/'depthpro/predictions.json')
    additional = work/'rgb-body-query-arkit-cal-dev-20261010/additional-inference'
    scans = sorted({r['scan'] for r in load(additional/'observations.json')['rows']})
    assert len(scans) == 3
    for scan in scans:
        choose(scan, additional/'observations.json', additional/'predictions.json', scan)
    for cohort in ('original_validation24','new_3rscan64'):
        folder = root/'fixed-grid-sensor'/cohort
        choose(cohort, folder/'observations.json', folder/'depthpro/predictions.json')
    assert len(rows) == len({(r['scan'],r['frame']) for r in rows}) == 16
    for r in rows:
        assert sha(r['rgb_path']) == r['rgb_sha256']
        assert sha(r['cached_fp16_path']) == r['cached_fp16_sha256']
    write(output/'roster.json', dict(status='FROZEN_PUBLIC_ONLY', rows=rows, input_manifests=inputs,
        selection='Six ARKit: timestamp extracted from public RGB stem, frame tie-break; two 3RScan cohorts: (frame,scan); first+last each, before model/label/score access',
        evaluator_read=False, preparation_wall_s=time.perf_counter()-started))
    print(f'ROSTER_FROZEN 16 frames prepare={time.perf_counter()-started:.3f}s', flush=True)


def finite_stats(array):
    finite = np.isfinite(array)
    a = array[finite].astype(np.float64)
    return dict(dtype=str(array.dtype), shape=list(array.shape), pixels=array.size,
        nonfinite=int((~finite).sum()), negative=int((finite & (array < 0)).sum()),
        zero=int((finite & (array == 0)).sum()),
        finite_min=float(a.min()) if a.size else None,
        finite_max=float(a.max()) if a.size else None,
        finite_quantiles=np.quantile(a,[.01,.5,.95,.99]).tolist() if a.size else None)


def run(repo, output, budget_s):
    started = time.perf_counter()
    with (output/'writer.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), script=str(Path(__file__).resolve())),f)
    if (output/'terminal.json').exists():
        (output/'writer.json').unlink()
        raise FileExistsError('No implicit rerun/budget reset')
    result = dict(status='STARTING', pid=os.getpid(), budget_gpu_process_wall_s=budget_s,
                  arms={}, evaluator_read=False, training_calls=0, download_calls=0)
    model = transform = x = pred = torch = None
    capture = {}
    original_clamp = None
    try:
        roster = load(output/'roster.json')
        assert len(roster['rows']) == 16 and roster['status'] == 'FROZEN_PUBLIC_ONLY'
        result.update(roster_sha256=sha(output/'roster.json'), source_sha256=sha(__file__))
        shutil.copyfile(__file__, output/'runner_executed.py')
        base = repo/'artifacts.local/work/ba-nfo-depthpro-20260919'
        weight = base/'depth_pro.pt'
        assert sha(weight) == WEIGHT_SHA
        backend = load(base/'backend.json')
        upstream = base/'upstream/src'
        codes = [dict(path=p.relative_to(upstream).as_posix(),sha256=sha(p))
                 for p in sorted((upstream/'depth_pro').rglob('*.py'))]
        fingerprint = hashlib.sha256(json.dumps(codes,sort_keys=True).encode()).hexdigest()
        write(output/'model_identity.json',dict(weight_sha256=WEIGHT_SHA,
            backend_sha256=sha(base/'backend.json'),upstream_python_files=codes,upstream_code_sha256=fingerprint))
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(4)
        assert torch.cuda.is_available()
        assert backend['selected_framework'] == f'torch-{torch.__version__}'
        assert backend['selected_device_name'] == torch.cuda.get_device_name(0)
        result.update(framework=f'torch-{torch.__version__}',device=torch.cuda.get_device_name(0),
            weight_sha256=WEIGHT_SHA,upstream_code_sha256=fingerprint,
            cuda_version=torch.version.cuda, torch_flags=dict(
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic))
        sys.path.insert(0,str(upstream))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(weight))
        original_clamp = torch.clamp
        def observed_clamp(input, *args, **kwargs):
            lo = kwargs.get('min',args[0] if args else None)
            hi = kwargs.get('max',args[1] if len(args)>1 else None)
            if isinstance(lo,(int,float)) and isinstance(hi,(int,float)) and lo == 1e-4 and hi == 1e4:
                capture['count'] = capture.get('count',0)+1
                capture['inverse'] = input.detach().clone()
            return original_clamp(input,*args,**kwargs)
        for arm, precision in [('fp16',torch.float16),('fp32',torch.float32)]:
            arm_started = time.perf_counter()
            folder = output/arm; folder.mkdir()
            manifest = dict(status='STARTING',arm=arm,rows=[],precision=str(precision),
                roster_sha256=result['roster_sha256'],weight_sha256=WEIGHT_SHA,
                inverse_contract='exact infer tensor after canonical W/fx scaling and native resize, before original torch.clamp(min=1e-4,max=1e4)',
                clamp_min=1e-4,clamp_max=1e4,provided_focal=True,evaluator_read=False)
            result['arms'][arm] = manifest
            try:
                if time.perf_counter()-started > budget_s-10:
                    raise TimeoutError('GPU process wall budget before model load')
                torch.cuda.reset_peak_memory_stats()
                model, transform = depth_pro.create_model_and_transforms(cfg,device=torch.device('cuda'),precision=precision)
                model.eval().requires_grad_(False)
                manifest['startup_s'] = time.perf_counter()-arm_started
                write(folder/'predictions.json',manifest)
                for i,r in enumerate(roster['rows']):
                    if budget_s-(time.perf_counter()-started) < 7:
                        raise TimeoutError('GPU process wall budget before inference')
                    assert sha(r['rgb_path']) == r['rgb_sha256']
                    with Image.open(r['rgb_path']) as image:
                        rgb = image.convert('RGB')
                    assert [rgb.height,rgb.width] == r['color_shape']
                    x = transform(rgb)
                    capture.clear()
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    torch.clamp = observed_clamp
                    try:
                        with torch.inference_mode():
                            pred = model.infer(x,f_px=torch.tensor(r['color_K'][0][0],device='cuda'))
                    finally:
                        torch.clamp = original_clamp
                    torch.cuda.synchronize(); call_s = time.perf_counter()-begin
                    assert capture['count'] == 1 and pred['depth'].device.type == 'cuda'
                    inverse = capture.pop('inverse').cpu().numpy()
                    depth_raw = pred['depth'].cpu().numpy()
                    depth = depth_raw.astype(np.float32)
                    focal = pred['focallength_px'].cpu().numpy()
                    assert list(depth.shape) == r['color_shape']
                    path = folder/f'{i:02d}_{r["scan"]}_{r["frame"]:06d}.npz'
                    np.savez_compressed(path,depth=depth,depth_raw=depth_raw,inverse_preclamp=inverse,focallength_px=focal)
                    lo,hi = [float(np.asarray(v,dtype=inverse.dtype)) for v in (1e-4,1e4)]
                    stats = dict(inverse=finite_stats(inverse),depth_raw=finite_stats(depth_raw),
                        inverse_less_than_clamp_min=int((inverse<lo).sum()),
                        inverse_equal_clamp_min=int((inverse==lo).sum()),
                        inverse_greater_than_clamp_max=int((inverse>hi).sum()),
                        effective_clamp_min=lo,effective_clamp_max=hi,
                        depth_ge_10000=int((depth>=10000).sum()),depth_eq_10000=int((depth==10000).sum()))
                    parity = None
                    if arm == 'fp16':
                        assert sha(r['cached_fp16_path']) == r['cached_fp16_sha256']
                        with np.load(r['cached_fp16_path']) as f: cached = f['depth']
                        assert cached.shape == depth.shape
                        finite = np.isfinite(cached)&np.isfinite(depth)
                        delta = np.abs(cached[finite].astype(np.float64)-depth[finite].astype(np.float64))
                        parity = dict(exact_array_equal=bool(np.array_equal(cached,depth,equal_nan=True)),
                            max_abs_difference=float(delta.max()) if delta.size else None,
                            changed_pixels=int((cached!=depth).sum()),
                            matching_finite_masks=bool(np.array_equal(np.isfinite(cached),np.isfinite(depth))))
                    manifest['rows'].append(dict(r,path=str(path),sha256=sha(path),inference_s=call_s,
                        input_tensor_dtype=str(x.dtype),inverse_tensor_dtype=str(inverse.dtype),
                        raw_depth_dtype=str(depth_raw.dtype),depth_shape=list(depth.shape),
                        focal_px=float(focal),focal_dtype=str(focal.dtype),stats=stats,cached_fp16_parity=parity))
                    manifest.update(status='RUNNING',completed_calls=len(manifest['rows']),elapsed_s=time.perf_counter()-arm_started)
                    write(folder/'predictions.json',manifest)
                    print(f'{arm} {i+1}/16 {r["cohort"]} frame={r["frame"]} call={call_s:.3f}s totalwall={time.perf_counter()-started:.2f}s',flush=True)
                    x = pred = None; capture.clear()
                manifest['status'] = 'COMPLETE'
            except Exception as error:
                manifest.update(status='FAILED_PARTIAL' if manifest['rows'] else 'FAILED',error=repr(error),
                    error_kind='CUDA_OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else type(error).__name__)
                raise
            finally:
                torch.clamp = original_clamp
                manifest.update(inference_s=sum(r['inference_s'] for r in manifest['rows']),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),completed_calls=len(manifest['rows']))
                x = pred = model = transform = None; capture.clear()
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                manifest.update(allocation_wall_s=time.perf_counter()-arm_started,
                    cuda_allocated_after_cleanup=torch.cuda.memory_allocated(),clamp_restored=torch.clamp is original_clamp)
                write(folder/'predictions.json',manifest)
                write(folder/'terminal.json',{k:v for k,v in manifest.items() if k!='rows'})
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PARTIAL' if result['arms'] else 'FAILED',error=repr(error))
    finally:
        if torch is not None and original_clamp is not None:
            torch.clamp = original_clamp
        x = pred = model = transform = None; capture.clear(); gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            result['cuda_allocated_after_cleanup'] = torch.cuda.memory_allocated()
        result['gpu_process_wall_s'] = time.perf_counter()-started
        result['clamp_restored'] = torch is None or original_clamp is None or torch.clamp is original_clamp
        result['resources_release'] = 'model/tensors released, cache cleared, clamp restored; process exits'
        write(output/'predictions.json',result)
        write(output/'terminal.json',dict(result,arms={k:{a:b for a,b in v.items() if a!='rows'} for k,v in result['arms'].items()}))
        (output/'writer.json').unlink()
        print(json.dumps(load(output/'terminal.json')),flush=True)
    return result['status'] == 'COMPLETE'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--budget-s',type=float,default=300.)
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.repo.resolve(),a.output.resolve())
    else:
        raise SystemExit(0 if run(a.repo.resolve(),a.output.resolve(),a.budget_s) else 1)

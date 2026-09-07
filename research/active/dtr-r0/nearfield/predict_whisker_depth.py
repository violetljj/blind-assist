"""NF-G8 fixed RGB predicted-depth baseline with observed distance closing.

No evaluator/native geometry, poses or wearer speed are inputs. The fixed 1.70 m
camera, -5 degree pitch and camera/body heading alignment are assumptions; head
rotation violates nominal alignment by design. Zero means no visible query
obstruction observed, never a free-space certificate. -1 is UNKNOWN.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
from near_field import Camera, surface_support
from ground_anchor import fit_ground
from rgb_replay import _metric_class
from research_backend import torch_observation

TARGETS = ['BODYnear','HEADnear','BODYapproaching','HEADapproaching']
WEIGHT_SHA = 'b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545'
BATCH_ATOL_M = 1e-4
BATCH_RTOL = 1e-5

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.inference_mode()
def frame_geometry(depth, hfov, device='cuda:0'):
    h,w = depth.shape
    camera = Camera(w,h,hfov,1.70,-5.)
    fit = fit_ground(depth,camera)
    d = torch.as_tensor(depth,device=device,dtype=torch.float32)
    yy,xx = torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    focal = w/(2*np.tan(np.deg2rad(hfov/2)))
    up = -(yy-(h-1)/2)/focal
    lateral = (xx-(w-1)/2)/focal
    pitch = np.deg2rad(-5.)
    x = d*(np.cos(pitch)-np.sin(pitch)*up)
    y = d*lateral
    z = d*(np.sin(pitch)+np.cos(pitch)*up)
    heights = [z+1.70]
    if fit.status == 'ACCEPTED':
        a,b,c = fit.plane
        heights.append(z-a*x-b*y-c)
    valid = torch.isfinite(d)&(d>.08)&(d<12.)
    scores,nearest,weak_counts,masks = [],[],[],[]
    for low,high,halfwidth,front in ((.65,1.4,.28,.18),(1.4,1.85,.18,.13)):
        selected = torch.zeros_like(valid)
        weak = torch.zeros_like(valid)
        corridor = (y.abs()<=halfwidth)&(x>=-front)
        for height in heights:
            eligible = valid&(height>=low)&(height<=high)
            supported = surface_support(d,height,eligible)
            selected |= supported&corridor
            weak |= eligible&corridor&~supported&((x-front)<=3.)
        best = max(0.,float(x[selected].min())-front) if selected.any() else None
        near = best is not None and best<=3.
        uncertain = bool(weak.any()) or float(valid.float().mean())<.5 or fit.status!='ACCEPTED'
        scores.append(1 if near else -1 if uncertain else 0)
        nearest.append(best)
        weak_counts.append(int(weak.sum()))
        masks.append((selected&((x-front)<=3.)).float())
    # Raster uses max pooling so a supported thin feature survives the 18x32 map.
    maps = torch.nn.functional.adaptive_max_pool2d(torch.stack(masks), (18,32)).cpu().numpy()
    return dict(near=scores,nearest_m=nearest,weak_counts=weak_counts,ground=fit.summary(),
                valid_fraction=float(valid.float().mean())), maps

def temporal_scores(history, timestamps):
    if len(history)!=3 or len(timestamps)!=3:
        raise ValueError('Exactly three causal observations required')
    t = np.asarray(timestamps,dtype=float)
    if not np.isfinite(t).all() or not np.all(np.diff(t)>0):
        raise ValueError('Strictly increasing finite observed timestamps required')
    near = history[-1]['near']
    approach,closing = [],[]
    for h in range(2):
        distances = [f['nearest_m'][h] for f in history]
        if near[h] == 0:
            approach.append(0); closing.append(None)
        elif near[h] == -1 or any(v is None or not np.isfinite(v) for v in distances):
            approach.append(-1); closing.append(None)
        else:
            dt = t-t.mean()
            speed = -float(np.dot(dt, np.asarray(distances))/np.dot(dt,dt))
            approach.append(int(speed>.1)); closing.append(speed)
    return list(near)+approach, closing

@torch.inference_mode()
def infer_images(model, images, batch_size):
    """Exact cached DAv2 preprocessing and bilinear align_corners=True resize.

    Batch 1 remains the official infer_image reference. No AMP or dtype change.
    """
    depths = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start+batch_size]
        if batch_size == 1:
            depths.append(np.asarray(model.infer_image(chunk[0],518), dtype=np.float32))
            continue
        prepared = [model.image2tensor(image,518) for image in chunk]
        tensors = [item[0] for item in prepared]
        if any(t.dtype != torch.float32 or t.shape[1:] != tensors[0].shape[1:] for t in tensors):
            raise ValueError('Batch preprocessing requires equal tensor shape and original FP32 dtype')
        depth = model.forward(torch.cat(tensors, dim=0))
        for i,(_,size) in enumerate(prepared):
            resized = torch.nn.functional.interpolate(depth[i:i+1,None],size,mode='bilinear',align_corners=True)[0,0]
            depths.append(resized.cpu().numpy())
    return depths

def batch_preflight(model, images, hfov, total_frames):
    """Bounded TRAIN-only hardware choice, never threshold/model selection."""
    started = time.perf_counter()
    report = dict(probe_frames=len(images), split='train', candidate_batches=[1,4],
        dtype='FP32', input_size=518, depth_tolerance=dict(atol_m=BATCH_ATOL_M,rtol=BATCH_RTOL),
        frame_attempts=0, forward_call_attempts=0, measurements_ms={}, selected_batch=1)
    def run(batch, values):
        output = []
        for start in range(0,len(values),batch):
            chunk = values[start:start+batch]
            report['frame_attempts'] += len(chunk)
            report['forward_call_attempts'] += 1
            output.extend(infer_images(model,chunk,batch))
        return output
    reference, candidate = None, None
    for batch in (1,4):
        try:
            run(batch,images[:batch])  # one warmup forward for each candidate
            measured = []
            for _ in range(2):
                torch.cuda.synchronize(); tick = time.perf_counter()
                output = run(batch,images)
                torch.cuda.synchronize(); measured.append((time.perf_counter()-tick)*1000)
            report['measurements_ms'][str(batch)] = measured
            if batch == 1:
                reference = output
            else:
                candidate = output
        except torch.cuda.OutOfMemoryError as exc:
            if batch == 1:
                raise  # no smaller authorized batch exists
            report['fallback_reason'] = 'BATCH4_CUDA_OUT_OF_MEMORY'
            report['oom_error'] = str(exc)
            exc.__traceback__ = None
            torch.cuda.empty_cache()
            break
    tick = time.perf_counter()
    reference_geometry = [frame_geometry(depth,hfov)[0] for depth in reference]
    torch.cuda.synchronize()
    report['reference_geometry_ms_per_frame'] = (time.perf_counter()-tick)*1000/len(images)
    if candidate is not None:
        differences = []
        state_equal = True
        for old,new,geometry in zip(reference,candidate,reference_geometry):
            same_shape = old.shape == new.shape
            finite = same_shape and np.isfinite(old).all() and np.isfinite(new).all()
            difference = float(np.max(np.abs(old-new))) if finite else None
            equal = bool(finite and np.allclose(old,new,atol=BATCH_ATOL_M,rtol=BATCH_RTOL))
            current = frame_geometry(new,hfov)[0]
            states = current['near']==geometry['near'] and current['ground']['status']==geometry['ground']['status']
            state_equal &= states
            differences.append(dict(max_abs_m=difference,depth_equivalent=equal,geometry_states_equal=states))
        report['agreement'] = differences
        equivalent = all(x['depth_equivalent'] for x in differences) and state_equal
        faster = np.median(report['measurements_ms']['4']) < np.median(report['measurements_ms']['1'])
        if equivalent and faster:
            report['selected_batch'] = 4
            report['selection_reason'] = 'BATCH4_FASTER_WITH_DEPTH_AND_GEOMETRY_AGREEMENT'
        else:
            report['fallback_reason'] = 'BATCH4_AGREEMENT_FAILED' if not equivalent else 'BATCH4_NOT_FASTER'
    selected_ms = float(np.median(report['measurements_ms'][str(report['selected_batch'])]))/len(images)
    report['selected_inference_ms_per_frame'] = selected_ms
    report['estimated_remaining_s'] = total_frames*(selected_ms+report['reference_geometry_ms_per_frame'])/1000
    report['estimate_scope'] = 'Inference plus unchanged geometry; excludes filesystem/hash cost; actual progress estimate follows'
    report['elapsed_s'] = time.perf_counter()-started
    return report

def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('capture','metric-source','weights','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    capture,out = args.capture.resolve(),args.output.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()) or out.exists():
        p.error('Use a new output directory inside artifacts.local')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for actual baseline inference')
    if sha(args.weights)!=WEIGHT_SHA:
        raise ValueError('Frozen metric-depth weights hash mismatch')
    data_path = capture/'model/dataset.json'
    data = json.loads(data_path.read_text(encoding='utf-8-sig'))
    frames = {f['sample_index']:f for f in data['frames']}
    required = sorted({i for s in data['samples'] for i in s['frame_indices']})
    out.mkdir(parents=True); (out/'depth').mkdir()
    write = lambda name,value:(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cls,identity = _metric_class(args.metric_source.resolve())
    model = cls(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20.)
    model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
    model.to('cuda:0').eval()
    details, maps, times = {},{},[]
    started = time.perf_counter()
    forward_calls = 0
    fallback_events = []
    try:
        with torch.inference_mode():
            def load_rgb(index):
                path = capture/'model'/frames[index]['rgb_path']
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('RGB image unreadable: '+str(path))
                return image
            train_indices = sorted({i for s in data['samples'] if s['split']=='train' for i in s['frame_indices']})[:8]
            if len(train_indices)!=8:
                raise ValueError('Preflight requires eight TRAIN RGB frames')
            hfov = data['calibration']['horizontal_fov_degrees']
            preflight = batch_preflight(model,[load_rgb(i) for i in train_indices],hfov,len(required))
            preflight['frame_indices'] = train_indices
            write('batch-preflight.json',preflight)
            print(json.dumps(dict(stage='batch_preflight',**preflight)),flush=True)
            batch_size = preflight['selected_batch']
            cursor = 0
            processing_started = time.perf_counter()
            while cursor<len(required):
                indices = required[cursor:cursor+batch_size]
                images = [load_rgb(i) for i in indices]
                torch.cuda.synchronize();tick=time.perf_counter()
                try:
                    forward_calls += 1
                    depths = infer_images(model,images,batch_size)
                except torch.cuda.OutOfMemoryError as exc:
                    if batch_size == 1:
                        raise
                    fallback_events.append(dict(frame_indices=indices,from_batch=batch_size,to_batch=1,error=str(exc)))
                    exc.__traceback__ = None
                    torch.cuda.empty_cache()
                    batch_size = 1
                    forward_calls += len(images)
                    depths = infer_images(model,images,1)
                torch.cuda.synchronize();infer_ms=(time.perf_counter()-tick)*1000/len(indices)
                for index,depth in zip(indices,depths):
                    tick=time.perf_counter()
                    detail,raster = frame_geometry(depth,hfov)
                    torch.cuda.synchronize();post_ms=(time.perf_counter()-tick)*1000
                    depth_path = out/'depth'/f'{index:05d}.npy'
                    np.save(depth_path,depth,allow_pickle=False)
                    path = capture/'model'/frames[index]['rgb_path']
                    details[index] = dict(**detail,rgb_sha256=sha(path),depth_sha256=sha(depth_path))
                    maps[index] = raster
                    times.append(dict(infer_ms=infer_ms,post_ms=post_ms))
                cursor += len(indices)
                if cursor%64==0 or cursor==len(required):
                    elapsed = time.perf_counter()-processing_started
                    progress=dict(completed=cursor,total=len(required),elapsed_s=time.perf_counter()-started,
                                  selected_batch=batch_size,estimated_remaining_s=elapsed/cursor*(len(required)-cursor))
                    write('progress.json',progress); print(json.dumps(progress),flush=True)
        scores,closing,test_ids,test_maps = [],[],[],[]
        for sample in data['samples']:
            indices = sample['frame_indices']
            values, speeds = temporal_scores([details[i] for i in indices],[frames[i]['time_s'] for i in indices])
            scores.append(values);closing.append(dict(sample_id=sample['sample_id'],closing_m_s=speeds))
            if sample['split']=='test':
                test_ids.append(sample['sample_id']);test_maps.append(maps[indices[-1]])
        write('predictions.json',dict(schema='nf-g8-whisker-predictions-v1',complete=True,
            target_order=TARGETS,sample_ids=[s['sample_id'] for s in data['samples']],
            unknown_score=-1,arms={'mde_observed_closing':{'seeds':{'0':{'normal':scores}}}}))
        np.savez_compressed(out/'support_predictions.npz',test_sample_ids=np.asarray(test_ids),
            mde_observed_closing__seed0__normal=np.asarray(test_maps),
            mde_observed_closing__ensemble__normal=np.asarray(test_maps))
        write('receipt.json',dict(status='COMPLETE',frames=details,closing=closing,
            model=dict(weights_sha256=sha(args.weights),source=identity,backend=asdict(torch_observation(model=model))),
            input_dataset_sha256=sha(data_path),code_sha256=sha(Path(__file__)),
            unique_frames_inferred=len(required),production_forward_call_attempts=forward_calls,
            inference_calls=len(required),inference_calls_scope='Unique production frames; preflight frame/forward attempts reported separately',
            batch_preflight=preflight,final_selected_batch=batch_size,production_oom_fallbacks=fallback_events,
            timing_scope='infer_ms is amortized production batch inference per frame including adapter preprocessing; post_ms is sequential geometry; neither is full single-frame camera-to-alert latency',
            input_boundary='RGB plus fixed calibration/timestamps only; no poses, speed, native depth, evaluator labels or phase',
            assumptions='1.70m camera height; -5 degree pitch; camera and body heading aligned nominally, imperfect under head rotation',
            rule='Supported raw/ground union BODY/HEAD <=3m from box front; approaching requires current near and observed 3-frame fitted closing >0.1m/s; unavailable distances are UNKNOWN -1',
            timing={k:dict(p50=float(np.median([t[k] for t in times])),p95=float(np.quantile([t[k] for t in times],.95))) for k in times[0]}))
        print(json.dumps(dict(status='COMPLETE',samples=len(scores),frames=len(required))))
    except BaseException as exc:
        write('failure.json',dict(error=repr(exc),completed_frames=len(details)))
        raise
    finally:
        del model
        torch.cuda.empty_cache()

if __name__=='__main__':
    main()

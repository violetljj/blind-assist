"""Selected observation/bin/query feature lineage; frozen replay is parity only.

Selection truth identifies diagnostic examples, and never enters model inputs.
CPU prepare copies only existing native histories, transforms and valid lengths.
CUDA run reconstructs the unchanged center/compact observation features. Saved
scores and own frozen thresholds remain the response evidence; optional frozen
replay merely checks lineage. No CUDA work is performed on import or prepare.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-pass-mechanism-chain-dev-20261009'
PASS = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
DATA = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009/data/validation'
ARMS = ('control', 'weak_pass')
BRANCHES = ('ideal', 'yaw_plus3')
SEED = 2026100956
KEPT = (0, 1, 2, 3, 4, 5, 12, 13, 14)
COMPACT_NAMES = ('signedlog_current', 'signedlog_history_mean', 'membership_gated_history_mean',
                 'current_membership_mean', 'current_membership_min', 'current_membership_max',
                 'current_outside_distance_min_over_0p6m', 'current_outside_distance_max_over_0p6m', 'radial_center_over_3m')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def smooth(raw):
    raw = np.asarray(raw, np.float64)
    out = np.empty_like(raw)
    for f in range(13):
        start = max(0, f-4)
        weights = 2.**np.arange(f-start+1)
        out[:, :, f] = (raw[:, :, start:f+1]*weights[:, None]).sum(2)/weights.sum()
    return out


class Budget:
    def __init__(self, phase):
        self.phase = phase
        self.start = time.monotonic()
        self.plan = read(OUT/'PLAN.json')
        self.folder = OUT/'stage_receipts'
        self.folder.mkdir(parents=True, exist_ok=True)
        previous = [read(p) for p in self.folder.glob('projection_'+phase+'_*.json')]
        self.previous = sum(p['seconds'] for p in previous)
        self.cap = self.plan['allocations']['projection_cpu'] if phase == 'prepare' else self.plan['gpu_stage_cap_seconds']
        self.identifier = time.time_ns()

    def check(self):
        if self.previous+time.monotonic()-self.start >= self.cap:
            raise TimeoutError('Projection-lineage stage wall cap reached')

    def finish(self, status, **details):
        receipt = dict(stage='projection_'+self.phase, status=status, seconds=time.monotonic()-self.start,
            previous_seconds=self.previous, cap_seconds=self.cap, plan_sha256=sha(OUT/'PLAN.json'),
            source_sha256=sha(__file__), **details)
        save(self.folder/f'projection_{self.phase}_{self.identifier}.json', receipt)
        return receipt


def prepare(selection_path=OUT/'selection.json'):
    budget = Budget('prepare')
    folder = OUT/'projection'
    try:
        selection = read(selection_path)
        events = selection['events']
        if len(events) > 24 or selection.get('seed', SEED) != SEED:
            raise ValueError('Frozen purposeful selection identity differs')
        if folder.exists():
            raise FileExistsError('Preserve prior projection preparation')
        folder.mkdir()
        paths = {k: DATA/(k+'.npy') for k in ('histories', 'transforms', 'transforms_yaw3', 'length')}
        source = {k: np.load(path, mmap_mode='r', allow_pickle=False) for k,path in paths.items()}
        thresholds_path = PASS/'workpoints_continued/thresholds.json'
        thresholds = read(thresholds_path)['arms']
        raw_scores, smoothed, score_sources = {}, {}, []
        for arm in ARMS:
            for branch in BRANCHES:
                suffix = '_yaw_plus3' if branch == 'yaw_plus3' else ''
                path = PASS/'scores'/f'{arm}_seed{SEED}_validation{suffix}.npz'
                with np.load(path, allow_pickle=False) as data:
                    raw = data['raw']
                    if str(data['arm'].item()) != arm or int(data['seed'].item()) != SEED:
                        raise ValueError('Saved response identity differs')
                if raw.shape != (384, 4, 13, 2) or not np.isfinite(raw).all():
                    raise ValueError('Complete finite saved response cohort required')
                raw_scores[arm,branch] = raw
                smoothed[arm,branch] = smooth(raw)
                score_sources.append(dict(arm=arm, branch=branch, path=str(path), sha256=sha(path)))
        native_rows, bindings = [], []
        for event in events:
            scene, replica = int(event['scene_id']), int(event['replica'])
            if not 0 <= scene < 384 or not 0 <= replica < 4 or event['query_index'] not in (0,1):
                raise ValueError('Selection event axes differ')
            for frame in range(3,14):
                native_rows.append((scene*4+replica)*13+frame-3)
        native_rows = np.array(native_rows, np.int64)
        histories = np.array(source['histories'][native_rows])
        lengths = np.array(source['length'][native_rows])
        t_ideal = np.array(source['transforms'][native_rows])
        t_yaw = np.array(source['transforms_yaw3'][native_rows])
        for ei,event in enumerate(events):
            for branch in BRANCHES:
                for j,frame in enumerate(range(3,14)):
                    hi = ei*11+j
                    row = dict(input_index=len(bindings), history_index=hi, selection_id=event['selection_id'],
                        event_index=ei, scene_id=int(event['scene_id']), replica=int(event['replica']),
                        query_index=int(event['query_index']), height=event['height'], branch=branch, frame=frame,
                        native_row=int(native_rows[hi]), valid_length=int(lengths[hi]))
                    bindings.append(row)
        transform_rows = np.stack([t_ideal[r['history_index']] if r['branch']=='ideal' else t_yaw[r['history_index']] for r in bindings]) if bindings else np.empty((0,8,4,4),np.float64)
        geometry_index, geometry_lookup, unique_t, unique_l = [], {}, [], []
        for i,row in enumerate(bindings):
            # Geometry deduplication depends on public transforms/length only.
            length = int(lengths[row['history_index']])
            key = (length, transform_rows[i].astype(np.float32).tobytes())
            if key not in geometry_lookup:
                geometry_lookup[key] = len(unique_t)
                unique_t.append(transform_rows[i]); unique_l.append(length)
            geometry_index.append(geometry_lookup[key])
        geometry_index = np.array(geometry_index, np.int64)
        unique_t = np.array(unique_t, np.float64).reshape(-1,8,4,4)
        unique_l = np.array(unique_l, np.int64)
        np.savez_compressed(folder/'selected_native.npz', histories=histories, lengths=lengths,
            native_rows=native_rows, unique_transforms=unique_t, unique_lengths=unique_l,
            geometry_index=geometry_index, history_index=np.array([r['history_index'] for r in bindings], np.int64))
        responses = []
        for row in bindings:
            for arm in ARMS:
                record = thresholds[arm][str(SEED)]['standalone']['pass_30pct']['threshold']
                theta = np.inf if record['positive_infinity'] else float(record['value'])
                si,k,j,q = row['scene_id'],row['replica'],row['frame']-3,row['query_index']
                raw = float(raw_scores[arm,row['branch']][si,k,j,q])
                sm = float(smoothed[arm,row['branch']][si,k,j,q])
                responses.append(dict(**row, arm=arm, raw_saved=raw, smoothed_saved=sm,
                    own_threshold=theta if np.isfinite(theta) else None,
                    own_margin=sm-theta if np.isfinite(theta) else None, alarm=int(sm>=theta)))
        save(folder/'input_bindings.json', dict(bindings=bindings, selection_sha256=sha(selection_path),
            example_truth_separation='Bindings select examples; only history/transforms/length enter feature construction'))
        save(folder/'responses.json', dict(rows=responses, thresholds_sha256=sha(thresholds_path)))
        saved_queries = np.array([[raw_scores[arm,row['branch']][row['scene_id'],row['replica'],row['frame']-3]
                                   for arm in ARMS] for row in bindings],np.float32).reshape(-1,2,2)
        np.savez_compressed(folder/'saved_response_arrays.npz',raw_saved=saved_queries,
            arm_names=np.array(ARMS),query_names=np.array(['HEAD','BODY']))
        if responses:
            with (folder/'saved_responses.csv').open('x',newline='',encoding='utf8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(responses[0]));writer.writeheader();writer.writerows(responses)
        declaration = dict(status='PREPARED', events=len(events), native_history_rows=len(histories),
            input_rows=len(bindings), unique_public_geometries=len(unique_t), seed=SEED,
            branches=list(BRANCHES), arms=list(ARMS), compact_channels=list(KEPT), compact_channel_names=list(COMPACT_NAMES),
            source_sha256=sha(__file__), model_source_sha256=sha(Path(__file__).with_name('cnh_boundary_token_model.py')),
            inherited_feature_source_sha256=sha(Path(__file__).with_name('cnh_counterfactual_train_dev.py')),
            selection_sha256=sha(selection_path), threshold_sha256=sha(thresholds_path),
            parent_input_bindings_sha256=sha(OUT/'input_bindings.json'),
            native_payload_sha256=sha(folder/'selected_native.npz'),
            selected_input_array_sha256={k:array_sha(v) for k,v in [('histories',histories),('lengths',lengths),('unique_transforms',unique_t)]},
            observations_source_signatures={k:dict(path=str(p.resolve()),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for k,p in paths.items()},
            saved_scores=score_sources,
            model_checkpoint_sha256={arm:sha(PASS/'models'/f'{arm}_seed{SEED}.pt') for arm in ARMS},
            geometry='Public transforms only; center 1 ray/node, soft-box mean/min/max duplicate; fixed original query boxes',
            parity='Optional frozen replay only; original full inference batch256, padding repeated final selected row then discard',
            example_scope='Purposefully selected consumed Development examples, not representative or independent evidence')
        save(folder/'prepare_manifest.json', declaration)
        budget.check()
        return budget.finish('COMPLETE', **{k:v for k,v in declaration.items() if k not in ('status','source_sha256')}, cpu_only=True)
    except Exception as error:
        budget.finish('FAILED',error=repr(error),cpu_only=True)
        raise


def _support_rows(bindings, signedlog, geometry):
    rows=[]
    for record in bindings:
        h,g,q=record['history_index'],record['geometry_index'],record['query_index']
        length=record['valid_length']
        values=signedlog[h,-length:]
        membership=geometry[g,q,-length:,...,0]
        gated=values*membership
        for scope,z,m,e in (('current',values[-1],membership[-1],gated[-1]),
                            ('history_sum',values,membership,gated)):
            rows.append(dict(**record,scope=scope,raw_signedlog_sum=float(z.sum(dtype=np.float64)),
                raw_positive_sum=float(np.maximum(z,0).sum(dtype=np.float64)),
                raw_negative_sum=float(np.minimum(z,0).sum(dtype=np.float64)),
                gated_signed_sum=float(e.sum(dtype=np.float64)),gated_positive_sum=float(np.maximum(e,0).sum(dtype=np.float64)),
                gated_negative_sum=float(np.minimum(e,0).sum(dtype=np.float64)),
                membership_sum=float(m.sum(dtype=np.float64)),membership_nonzero_bins=int((m>0).sum()),
                valid_exposures=1 if scope=='current' else length))
    return rows


def run(replay=True):
    budget=Budget('gpu')
    folder=OUT/'projection'
    torch=None
    nets={}
    gpu_geometry=hist_gpu=compact=features=model=None
    try:
        import torch
        import cnh_boundary_token_model as M
        import cnh_counterfactual_train_dev as T
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for selected original feature reconstruction')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        declaration=read(folder/'prepare_manifest.json')
        if declaration['source_sha256']!=sha(__file__) or declaration['model_source_sha256']!=sha(M.__file__) or declaration['inherited_feature_source_sha256']!=sha(T.__file__):
            raise ValueError('Frozen feature lineage source changed after prepare')
        if (folder/'run_manifest.json').exists() or (folder/'compact_inputs.npz').exists():
            raise FileExistsError('Preserve completed/partial projection GPU outputs')
        if sha(folder/'selected_native.npz')!=declaration['native_payload_sha256']:
            raise ValueError('Prepared native selected inputs changed')
        with np.load(folder/'selected_native.npz',allow_pickle=False) as native:
            histories,lengths,transforms,geometry_lengths,geometry_index,history_index=(native[k] for k in ('histories','lengths','unique_transforms','unique_lengths','geometry_index','history_index'))
        bindings=read(folder/'input_bindings.json')['bindings']
        responses=read(folder/'responses.json')['rows']
        hist_gpu=torch.as_tensor(histories,device='cuda')
        length_gpu=torch.as_tensor(lengths,device='cuda')
        valid=torch.arange(8,device='cuda')[None]>=8-length_gpu[:,None]
        z=torch.where(valid[:,:,None,None,None],hist_gpu.float(),0.)
        signedlog_gpu=z.sign()*z.abs().log1p()
        signedlog=signedlog_gpu.cpu().numpy()
        observed_mean=(signedlog_gpu.sum(1)/length_gpu.float()[:,None,None,None]).cpu().numpy()
        geo=[]
        with torch.no_grad():
            for start in range(0,len(transforms),64):
                budget.check()
                t=torch.as_tensor(transforms[start:start+64],device='cuda',dtype=torch.float32)
                le=torch.as_tensor(geometry_lengths[start:start+64],device='cuda')
                n=len(t)
                if n<64:
                    t=torch.cat((t,t[-1:].expand(64-n,-1,-1,-1)))
                    le=torch.cat((le,le[-1:].expand(64-n)))
                geo.append(M.feature_geometry(t,le,'center')[:n].cpu().numpy())
        geometry=np.concatenate(geo) if geo else np.empty((0,2,8,8,8,16,11),np.float32)
        compact_array=np.empty((len(bindings),2,16,9,8,8),np.float16)
        with torch.no_grad():
            for start in range(0,len(bindings),64):
                budget.check()
                ix=slice(start,min(start+64,len(bindings)))
                hi=torch.as_tensor(history_index[ix],device='cuda')
                gpu_geometry=torch.as_tensor(geometry[geometry_index[ix]],device='cuda')
                compact=M.build_features(hist_gpu[hi],gpu_geometry,length_gpu[hi])[:,:,:,list(KEPT)].half()
                compact_array[ix]=compact.cpu().numpy()
        if not np.isfinite(compact_array).all():
            raise ValueError('Nonfinite selected compact inputs')
        np.savez_compressed(folder/'compact_inputs.npz',compact=compact_array,
            geometry_index=geometry_index,history_index=history_index,native_channels=np.array(KEPT),
            channel_names=np.array(COMPACT_NAMES))
        np.savez_compressed(folder/'observed_signedlog.npz',signedlog=signedlog,lengths=lengths,
            current=signedlog[:,-1],history_mean=observed_mean)
        # Original time/zone/bin axes retained; dedup index is explicit.
        np.savez_compressed(folder/'public_geometry.npz',membership=geometry[...,:3],
            outside_distance=geometry[...,9:11],unique_transforms=transforms,
            unique_lengths=geometry_lengths,geometry_index=geometry_index)
        detailed_bindings=[dict(r,geometry_index=int(geometry_index[i])) for i,r in enumerate(bindings)]
        support=_support_rows(detailed_bindings,signedlog,geometry)
        if support:
            with (folder/'support_sums.csv').open('x',newline='',encoding='utf8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(support[0]));writer.writeheader();writer.writerows(support)
        parity=[]
        if replay and bindings:
            with np.load(folder/'saved_response_arrays.npz',allow_pickle=False) as response_arrays:
                expected=response_arrays['raw_saved']
            for ai,arm in enumerate(ARMS):
                checkpoint=PASS/'models'/f'{arm}_seed{SEED}.pt'
                if sha(checkpoint)!=declaration['model_checkpoint_sha256'][arm]:
                    raise ValueError('Frozen parity checkpoint changed')
                model=M.BoundaryTokenReadout().cuda().eval()
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
                nets[arm]=model
                maxdiff=0.
                actual=np.empty((len(bindings),2),np.float32)
                with torch.inference_mode():
                    for start in range(0,len(bindings),256):
                        budget.check()
                        end=min(start+256,len(bindings));n=end-start
                        x=torch.as_tensor(compact_array[start:end],device='cuda')
                        le=torch.as_tensor(lengths[history_index[start:end]],device='cuda')
                        if n<256:
                            x=torch.cat((x,x[-1:].expand(256-n,-1,-1,-1,-1,-1)))
                            le=torch.cat((le,le[-1:].expand(256-n)))
                        features=T.restore_features(x)
                        if bool(features[:,:,:,6:12].count_nonzero()):
                            raise ValueError('Six signed-face channels must be exact zero')
                        actual[start:end]=model(features,le)[:n].cpu().numpy()
                for i,row in enumerate(bindings):
                    for q in (0,1):
                        value=float(actual[i,q]);before=float(expected[i,ai,q])
                        delta=abs(value-before);maxdiff=max(maxdiff,delta)
                        parity.append(dict(input_index=i,arm=arm,selection_id=row['selection_id'],branch=row['branch'],
                            frame=row['frame'],query_index=q,saved_raw=before,replay_raw=value,abs_difference=delta))
                if maxdiff>1e-5:
                    save(folder/f'parity_failure_{arm}_{time.time_ns()}.json',dict(arm=arm,max_abs_difference=maxdiff,atol=1e-5))
                    raise ValueError(f'Frozen replay lineage mismatch {arm}: {maxdiff}')
                nets.pop(arm);model=None
        if parity:
            with (folder/'replay_parity.csv').open('x',newline='',encoding='utf8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(parity[0]));writer.writeheader();writer.writerows(parity)
        maxdiff=max((r['abs_difference'] for r in parity),default=0.)
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.glob('*.npz')}
        report=dict(status='COMPLETE',input_rows=len(bindings),geometry_rows=len(geometry),
            compact_shape=list(compact_array.shape),compact_dtype='float16',signed_faces_restored_exact_zero=True,
            membership_axes='unique_geometry,query,time,zone_y,zone_x,radial_bin,mean_min_max',
            observed_axes='native_history_row,time,zone_y,zone_x,radial_bin',
            preserved='Original bin/time identities retained in NPZ; scalar sums explicitly distinguish current vs history_sum',
            replay_performed=bool(replay and bindings),replay_pairs=len(parity),replay_atol=1e-5,max_abs_replay_difference=maxdiff,
            saved_scores_remain_authoritative=True,new_performance_evaluation=False,outputs=outputs,
            source_sha256=sha(__file__),model_source_sha256=sha(M.__file__))
        save(folder/'run_manifest.json',report)
        budget.check()
        return budget.finish('COMPLETE',**{k:v for k,v in report.items() if k not in ('status','source_sha256')},CUDA_device=torch.cuda.get_device_name(0))
    except Exception as error:
        budget.finish('FAILED',error=repr(error),outputs_preserved=True)
        raise
    finally:
        nets.clear()
        gpu_geometry=hist_gpu=compact=features=model=None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--selection',type=Path,default=OUT/'selection.json')
    parser.add_argument('--no-replay',action='store_true',help='Reconstruct observation inputs without optional frozen parity replay')
    args=parser.parse_args()
    print(json.dumps(prepare(args.selection) if args.stage=='prepare' else run(not args.no_replay)),flush=True)

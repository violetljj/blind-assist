"""Frozen metric-RGB features and predicted depth, learned query readout.

Reference is supervision/scoring only; full observed RGB/K produces features.
All trained arms share initialization, sampling and fixed final step. No old
checkpoint is continued, and no validation threshold/checkpoint is selected.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval, confusion
from rgb_body_query_sensor_pilot import cutoff_for_fpr, summarize

ARMS = ('metric_rgb', 'depth_only', 'geometry')


def run(sensor, feature_root, output, budget_s=240, steps=200):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'receipt.json').exists():
        raise FileExistsError('No implicit restart/overwrite')
    obs_path, ref_path = sensor/'observations.json', sensor/'dataset_manifest.json'
    feat_path = feature_root/'feature_manifest.json'
    receipt = dict(status='STARTING', budget_s=budget_s, steps=steps, seed=7, batch_size=8192,
                   observations_sha256=sha(obs_path), reference_manifest_sha256=sha(ref_path),
                   features_manifest_sha256=sha(feat_path), script_sha256=sha(Path(__file__)),
                   input_contract='frozen RGB-generated 32-channel metric feature and predicted log-Z, public K/ray/query; no sensor depth input',
                   calibration='raw Depth Pro cal FPR, each new head calibration separately; no validation selection',
                   phase='Consumed Development, final fixed-step checkpoint; not confirmation', stages={})
    torch = heads = None
    def check(reserve=0):
        if time.perf_counter()-start >= budget_s-reserve:
            raise TimeoutError('Metric readout budget reached')
    try:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(4); torch.manual_seed(7)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA needed for bounded readout')
        receipt.update(device=torch.cuda.get_device_name(0), framework=torch.__version__)
        torch.cuda.reset_peak_memory_stats()
        obs = json.loads(obs_path.read_text('utf-8-sig'))
        refs = json.loads(ref_path.read_text('utf-8-sig'))
        fm = json.loads(feat_path.read_text('utf-8-sig'))
        if fm['status'] != 'COMPLETE':
            raise ValueError('Complete fixed feature extraction required')
        source_rows = refs['rows']; features = fm['rows']; rows = obs['rows']
        key = lambda r: (r['scan'], r['frame'])
        if [key(r) for r in rows] != [key(r) for r in features] or [key(r) for r in rows] != [key(r) for r in source_rows]:
            raise ValueError('Source identity/order differs')
        if obs['dataset_manifest_sha256'] != receipt['reference_manifest_sha256']:
            raise ValueError('Reference manifest changed')
        qs = obs['queries']; nq = len(qs)
        groups = {s: {r['environment'] for r in rows if r['split'] == s} for s in ('train','cal','validation')}
        if any(groups[a] & groups[b] for a,b in [('train','cal'),('train','validation'),('cal','validation')]):
            raise ValueError('Environment leakage')
        train_idx = np.array([i for i,r in enumerate(rows) if r['split']=='train'])
        cal_idx = [i for i,r in enumerate(rows) if r['split']=='cal']
        val_idx = [i for i,r in enumerate(rows) if r['split']=='validation']
        values, ray_values, labels, reachable = [], [], [], []
        stage = time.perf_counter()
        for i, (r, f, ref) in enumerate(zip(rows, features, source_rows)):
            check(20)
            if r['rgb_sha256'] != f['rgb_sha256'] or sha(f['feature_path']) != f['feature_sha256']:
                raise ValueError('RGB/feature identity differs')
            k = np.asarray(r['depth_K']); rx,ry = rays(k, r['depth_shape'])
            with np.load(f['feature_path']) as a:
                visual = a['features'].astype(np.float32)
                predicted = a['predicted_depth'].astype(np.float32)
                if visual.shape != tuple(r['depth_shape'])+(32,) or predicted.shape != tuple(r['depth_shape']):
                    raise ValueError('Feature shape differs')
                if not np.isfinite(visual).all() or not np.isfinite(predicted).all() or not (predicted > 0).all():
                    raise ValueError('Invalid observation feature/depth')
                values.append(np.concatenate([visual, np.log(predicted[...,None])],-1).reshape(-1,33))
            ray_values.append(np.stack([rx,ry],-1).reshape(-1,2).astype(np.float32))
            reachable.append(np.stack([ray_interval(rx,ry,q)[2] for q in qs]))
            if sha(ref['reference_path']) != ref['reference_sha256']:
                raise ValueError('Changed reference')
            with np.load(ref['reference_path']) as a:
                labels.append(a['labels'])
        values = np.stack(values); label_array = np.stack(labels)
        nr, npix, _ = values.shape
        mean = values[train_idx].mean(axis=(0,1), dtype=np.float64).astype(np.float32)
        std = np.maximum(values[train_idx].std(axis=(0,1), dtype=np.float64), .001).astype(np.float32)
        values = np.clip((values-mean)/std, -20, 20)
        normalization = output/'observation_normalization.npz'
        np.savez(normalization, mean=mean, std=std)
        receipt['normalization'] = dict(source='train RGB-generated features only, no reference/depth fit', sha256=sha(normalization))
        value_tensor = torch.as_tensor(values, device='cuda')
        ray_tensor = torch.as_tensor(np.stack(ray_values), device='cuda')
        bounds = torch.tensor([q['low']+q['high'] for q in qs], dtype=torch.float32, device='cuda')
        bounds /= torch.tensor([.9,.55,6.,.9,.55,6.], device='cuda')
        receipt['stages']['prepare'] = dict(wall_s=time.perf_counter()-stage, observation_channels=33)
        del values
        known = np.flatnonzero((label_array[train_idx] < 2).reshape(-1))
        global_indices = train_idx[known//(nq*npix)]*(nq*npix)+known%(nq*npix)
        known_tensor = torch.as_tensor(global_indices, device='cuda', dtype=torch.long)
        targets = torch.as_tensor(label_array.reshape(-1), device='cuda', dtype=torch.uint8)
        def make_head():
            return torch.nn.Sequential(torch.nn.Linear(41,64),torch.nn.ReLU(),torch.nn.Linear(64,64),
                                       torch.nn.ReLU(),torch.nn.Linear(64,1)).cuda()
        torch.manual_seed(7)
        initial = make_head().state_dict()
        initial_sha = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in initial.values())).hexdigest()
        heads = {}
        for arm in ARMS:
            head = make_head(); head.load_state_dict(initial); heads[arm]=head
        receipt['initial_state_sha256'] = initial_sha
        def point_input(fi, qi, pi, arm, source=None):
            vi = fi if source is None else source
            observed = value_tensor[vi,pi]
            if arm == 'geometry':
                observed = torch.zeros_like(observed)
            elif arm == 'depth_only':
                observed = torch.cat([torch.zeros_like(observed[:,:32]),observed[:,32:]],1)
            return torch.cat([observed, ray_tensor[fi,pi], bounds[qi]],1)
        stage = time.perf_counter(); trained={}
        for arm, head in heads.items():
            opt = torch.optim.Adam(head.parameters(), lr=.002)
            gen = torch.Generator(device='cuda').manual_seed(7)
            losses=[]
            for step in range(steps):
                check(25)
                selected = known_tensor[torch.randint(known_tensor.numel(),(8192,),generator=gen,device='cuda')]
                fi,qi,pi = selected//(nq*npix),selected//npix%nq,selected%npix
                logits = head(point_input(fi,qi,pi,arm))[:,0]
                loss = F.binary_cross_entropy_with_logits(logits, targets[selected].float())
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if step%50 == 0 or step == steps-1:
                    losses.append(dict(step=step+1,bce=float(loss.detach())))
            torch.cuda.synchronize(); head.eval()
            path=output/f'{arm}_final.pt'
            torch.save(dict(state_dict=head.state_dict(), steps=steps, seed=7, initial_state_sha256=initial_sha),path)
            trained[arm]=dict(checkpoint=str(path),sha256=sha(path),steps=steps,loss_samples=losses)
            del opt
        receipt['stages']['train']=dict(wall_s=time.perf_counter()-stage,arms=trained)
        dp=json.loads((sensor/'evaluation.json').read_text('utf-8-sig'))
        target_fpr=dp['summary']['cal']['fpr']
        receipt.update(target_cal_fpr=target_fpr,depthpro_evaluation_sha256=sha(sensor/'evaluation.json'))
        all_q=torch.arange(nq,device='cuda').repeat_interleave(npix)
        all_p=torch.arange(npix,device='cuda').repeat(nq)
        order=sorted(groups['validation']); shuffle={}
        for i in val_idx:
            env=rows[i]['environment']; dest=order[(order.index(env)+1)%len(order)]
            this=[j for j in val_idx if rows[j]['environment']==env]
            that=[j for j in val_idx if rows[j]['environment']==dest]
            shuffle[i]=that[this.index(i)]
        receipt['shuffle_mapping']=shuffle
        def predict(indices,arm,shuffled=False):
            result=[]
            for i in indices:
                check(3); chunks=[]
                for j in range(0,nq*npix,65536):
                    qi,pi=all_q[j:j+65536],all_p[j:j+65536]
                    fi=torch.full_like(pi,i); src=torch.full_like(pi,shuffle[i]) if shuffled else None
                    with torch.inference_mode():
                        score=heads[arm](point_input(fi,qi,pi,arm,src)).sigmoid()[:,0]
                    chunks.append(score.cpu().numpy())
                result.append((i,np.concatenate(chunks).reshape(label_array.shape[1:])))
            return result
        def score(predictions,cutoff):
            records=[]
            for i,probability in predictions:
                row=source_rows[i]
                for j,q in enumerate(qs):
                    predicted=(probability[j]>=cutoff)&reachable[i][j]
                    c=confusion(label_array[i,j],predicted)
                    records.append(dict(environment=row['environment'],scan=row['scan'],frame=row['frame'],split=row['split'],
                                        query=q['name'],reference_state=row['queries'][j]['state'],
                                        predicted_positive=int(predicted.sum())>=16,predicted_support=int(predicted.sum()),**c))
            summary=summarize(records)
            summary['known_witness_hits']=sum(r['reference_state']=='POSITIVE' and r['tp']>=16 for r in records)
            return dict(summary=summary,records=records)
        def save_predictions(preds,arm):
            saved=[]
            for i,a in preds:
                path=output/f'{arm}_{rows[i]["scan"]}_{rows[i]["frame"]:06d}.npy';np.save(path,a)
                saved.append(dict(row=i,path=str(path),sha256=sha(path)))
            return saved
        stage=time.perf_counter(); result={}
        for arm in ARMS:
            cal=predict(cal_idx,arm)
            free=np.concatenate([s[label_array[i]==0] for i,s in cal])
            cutoff=cutoff_for_fpr(free,target_fpr)
            val=predict(val_idx,arm)
            result[arm]=dict(cutoff=cutoff,cal=score(cal,cutoff),validation=score(val,cutoff),
                             predictions=save_predictions(cal+val,arm))
            print(arm,json.dumps(result[arm]['validation']['summary']),flush=True)
        shuffled=predict(val_idx,'metric_rgb',True)
        result['metric_rgb_shuffled']=dict(cutoff=result['metric_rgb']['cutoff'],
                validation=score(shuffled,result['metric_rgb']['cutoff']),predictions=save_predictions(shuffled,'metric_rgb_shuffled'))
        receipt['stages']['evaluate']=dict(wall_s=time.perf_counter()-stage)
        evaluation=dict(status='COMPLETE',arms=result,depthpro=dp['summary'],
                        reference_manifest_sha256=receipt['reference_manifest_sha256'],
                        feature_manifest_sha256=receipt['features_manifest_sha256'],
                        contract='full RGB metric-pretrained features/predicted depth; reference only train/cal/scoring',
                        limitations='Related native sensor query-rays, no sufficiently covered negative query/body/walking event labels; Development reuse')
        write(output/'evaluation.json',evaluation);receipt['status']='COMPLETE'
    except TimeoutError as e:
        receipt.update(status='PARTIAL_BUDGET',error=repr(e))
    except Exception as e:
        receipt.update(status='FAILED',error=repr(e));raise
    finally:
        if torch is not None:
            receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            heads=None;gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache();torch.cuda.synchronize()
        receipt['allocation_wall_s']=time.perf_counter()-start
        receipt['resource_release']='subprocess termination releases all remaining tensor references'
        write(output/'receipt.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k!='stages'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sensor',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--budget-s',type=float,default=240)
    a=p.parse_args();run(a.sensor.resolve(),a.features.resolve(),a.output.resolve(),a.budget_s)

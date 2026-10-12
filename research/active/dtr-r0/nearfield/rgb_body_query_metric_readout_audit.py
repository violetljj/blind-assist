"""Independent CPU recomputation and frozen-head scene diagnostics.

Preserves fixed cal thresholds and checkpoints. The added shuffle exchanges
RGB-generated depth across environments, retaining destination ray/query.
Canonical scores use exactly common ray/query input; labels/depth observations
are nearest public-K samples, not exact reference ray correspondence.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import confusion, rays, ray_interval
from rgb_body_query_sensor_pilot import summarize
from rgb_body_query_scene_diagnostic import (paired_counts, concordance,
    finish_concordance, correspondence, write_csv, rate)


def run(source, sensor, features, output, budget_s=240.):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'receipt.json').exists():
        raise FileExistsError('Preserve existing audit evidence')
    receipt = dict(status='STARTING', budget_s=budget_s, gpu_s=0, download_bytes=0,
                   stages={}, sources=[], script_sha256=sha(Path(__file__)))
    def check():
        if time.perf_counter()-start >= budget_s:
            raise TimeoutError('CPU audit budget reached')
    def verified_json(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Changed identity: {path}')
        receipt['sources'].append(dict(path=str(path), sha256=digest))
        return json.loads(path.read_text('utf-8-sig'))
    try:
        import torch
        torch.set_num_threads(4)
        ev = verified_json(source/'evaluation.json')
        old = verified_json(source/'receipt.json')
        manifest = verified_json(sensor/'dataset_manifest.json', old['reference_manifest_sha256'])
        obs = verified_json(sensor/'observations.json', old['observations_sha256'])
        fm = verified_json(features/'feature_manifest.json', old['features_manifest_sha256'])
        rows, qs = manifest['rows'], manifest['queries']
        selected = [i for i,r in enumerate(rows) if r['split'] in ('cal','validation')]
        val = [i for i in selected if rows[i]['split']=='validation']
        labels, domain, scores = {}, {}, {}
        stage=time.perf_counter()
        for i in selected:
            check(); r=rows[i]
            if sha(r['reference_path']) != r['reference_sha256']:
                raise ValueError('Changed reference')
            with np.load(r['reference_path']) as a:
                labels[i] = np.array(a['labels'])
            rx,ry=rays(r['depth_K'],r['depth_shape'])
            domain[i]=np.stack([ray_interval(rx,ry,q)[2] for q in qs])
        def score(predictions, cutoff):
            records=[]
            for i,a in predictions.items():
                for j,q in enumerate(qs):
                    p=(a[j]>=cutoff)&domain[i][j]
                    c=confusion(labels[i][j],p)
                    r=rows[i]
                    records.append(dict(environment=r['environment'],scan=r['scan'],frame=r['frame'],
                        split=r['split'],query=q['name'],reference_state=r['queries'][j]['state'],
                        predicted_positive=int(p.sum())>=16,predicted_support=int(p.sum()),**c))
            s=summarize(records)
            s['known_witness_hits']=sum(r['reference_state']=='POSITIVE' and r['tp']>=16 for r in records)
            return dict(summary=s,records=records)
        checks=[]
        for arm,info in ev['arms'].items():
            scores[arm]={}
            for p in info['predictions']:
                check()
                if sha(p['path']) != p['sha256']:
                    raise ValueError('Changed saved prediction')
                scores[arm][p['row']]=np.load(p['path'],allow_pickle=False)
                receipt['sources'].append(dict(path=p['path'],sha256=p['sha256']))
            for split in ('cal','validation'):
                if split not in info:
                    continue
                actual=score({i:a for i,a in scores[arm].items() if rows[i]['split']==split}, info['cutoff'])
                if actual != info[split]:
                    raise ValueError(f'Saved array recomputation differs: {arm}/{split}')
                checks.append(dict(arm=arm,split=split,records=len(actual['records']),exact_match=True,summary=actual['summary']))
        receipt['stages']['saved_prediction_recomputation']=dict(wall_s=time.perf_counter()-stage,checks=checks)
        def head():
            return torch.nn.Sequential(torch.nn.Linear(41,64),torch.nn.ReLU(),torch.nn.Linear(64,64),
                                       torch.nn.ReLU(),torch.nn.Linear(64,1))
        torch.manual_seed(7); initial=head().state_dict()
        initial_sha=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        if initial_sha != old['initial_state_sha256']:
            raise ValueError('Seed-7 initialization mismatch')
        heads={}; updates=[]
        for arm in ('metric_rgb','depth_only','geometry'):
            p=source/f'{arm}_final.pt'; digest=sha(p)
            if digest!=old['stages']['train']['arms'][arm]['sha256']:
                raise ValueError('Changed checkpoint')
            cp=torch.load(p,map_location='cpu',weights_only=True)
            if cp['steps']!=200 or cp['seed']!=7 or cp['initial_state_sha256']!=initial_sha:
                raise ValueError('Final fixed-step checkpoint contract mismatch')
            state=cp['state_dict']; changed={k:int((v!=initial[k]).sum()) for k,v in state.items()}
            nzero=33 if arm=='geometry' else 32 if arm=='depth_only' else 0
            if nzero and not torch.equal(state['0.weight'][:,:nzero],initial['0.weight'][:,:nzero]):
                raise ValueError('Zero-observation columns unexpectedly updated')
            if not any(changed.values()):
                raise ValueError('Checkpoint contains no updates')
            h=head();h.load_state_dict(state);h.eval();heads[arm]=h
            updates.append(dict(arm=arm,checkpoint_sha256=digest,changed_parameters=changed,
                zero_observation_columns=nzero,zero_columns_unchanged=True))
        normpath=source/'observation_normalization.npz'
        if sha(normpath)!=old['normalization']['sha256']:
            raise ValueError('Changed normalization')
        with np.load(normpath) as a:
            mean,std=a['mean'],a['std']
        predicted_depth={}
        for i in val:
            check();f=fm['rows'][i]
            if (f['scan'],f['frame'])!=(rows[i]['scan'],rows[i]['frame']) or f['rgb_sha256']!=obs['rows'][i]['rgb_sha256']:
                raise ValueError('Feature identity/order mismatch')
            if sha(f['feature_path'])!=f['feature_sha256']:
                raise ValueError('Changed RGB-generated depth')
            with np.load(f['feature_path']) as a:
                predicted_depth[i]=np.array(a['predicted_depth'],dtype=np.float32).reshape(-1)
        bounds=torch.tensor([q['low']+q['high'] for q in qs],dtype=torch.float32)
        bounds/=torch.tensor([.9,.55,6.,.9,.55,6.])
        def infer(arm,k,shape,depth):
            rx,ry=rays(k,shape)
            xy=torch.tensor(np.stack([rx,ry],-1).reshape(-1,2),dtype=torch.float32)
            observed=torch.zeros((len(xy),33),dtype=torch.float32)
            if arm=='depth_only':
                observed[:,32]=torch.tensor(np.clip((np.log(depth)-mean[32])/std[32],-20,20))
            result=[]
            with torch.inference_mode():
                for q in bounds:
                    check()
                    x=torch.cat([observed,xy,q[None].expand(len(xy),-1)],1)
                    result.append(heads[arm](x).sigmoid().numpy()[:,0])
            return np.stack(result).reshape((len(qs),)+tuple(shape))
        stage=time.perf_counter(); shuffled={}; saved=[]; cpu_equivalence=[]
        mapping={int(k):v for k,v in old['shuffle_mapping'].items()}
        for i in val:
            r=rows[i]; j=mapping[i]
            if r['environment']==rows[j]['environment']:
                raise ValueError('Shuffle did not exchange environments')
            shuffled[i]=infer('depth_only',r['depth_K'],r['depth_shape'],predicted_depth[j])
            p=output/f'depth_only_shuffled_{r["scan"]}_{r["frame"]:06d}.npy'
            np.save(p,shuffled[i]);saved.append(dict(row=i,source_row=j,path=str(p),sha256=sha(p)))
        for i in val[:2]:
            r=rows[i]; own=infer('depth_only',r['depth_K'],r['depth_shape'],predicted_depth[i])
            delta=float(np.max(np.abs(own-scores['depth_only'][i])))
            if delta>2e-6:
                raise ValueError('CPU/GPU own-source score disagreement')
            cpu_equivalence.append(dict(row=i,max_absolute_score_difference=delta,tolerance=2e-6))
        cutoff=ev['arms']['depth_only']['cutoff']
        extra=dict(cutoff=cutoff,validation=score(shuffled,cutoff),predictions=saved,
                   contract='Frozen depth-only head; cross-environment RGB-generated log-depth; destination public ray/query; first 32 observation channels zero')
        write(output/'depth_only_shuffled_evaluation.json',extra)
        scores['depth_only_shuffled']=shuffled
        receipt['stages']['depth_only_shuffle']=dict(wall_s=time.perf_counter()-stage,cpu_gpu_same_source_checks=cpu_equivalence)
        cutoffs={a:d['cutoff'] for a,d in ev['arms'].items()};cutoffs['depth_only_shuffled']=cutoff
        increments=[]
        for arm,base in [('metric_rgb','geometry'),('metric_rgb','depth_only'),('depth_only','geometry'),
                         ('metric_rgb_shuffled','geometry'),('depth_only_shuffled','geometry')]:
            for i in val:
                r=rows[i]
                for qi,q in enumerate(qs):
                    a=(scores[arm][i][qi]>=cutoffs[arm])&domain[i][qi]
                    b=(scores[base][i][qi]>=cutoffs[base])&domain[i][qi]
                    c=paired_counts(labels[i][qi],a,b)
                    c.update(positive_queries=int(r['queries'][qi]['state']=='POSITIVE'),
                        candidate_known_witness=int(r['queries'][qi]['state']=='POSITIVE' and c['rgb_tp']>=16),
                        baseline_known_witness=int(r['queries'][qi]['state']=='POSITIVE' and c['geometry_tp']>=16))
                    increments.append(dict(arm=arm,baseline=base,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],
                        band=f"{q['low'][2]:g}-{q['high'][2]:g}m",**c))
        numeric=[k for k,v in increments[0].items() if isinstance(v,int)]
        # Frame is an identifier and must not be summed.
        numeric.remove('frame')
        def aggregate(keys):
            buckets={}
            for r in increments:
                key=tuple(r[k] for k in keys)
                dest=buckets.setdefault(key,dict(zip(keys,key),**{k:0 for k in numeric},frame_query_records=0))
                for k in numeric:dest[k]+=r[k]
                dest['frame_query_records']+=1
            for r in buckets.values():
                r.update(candidate_recall=rate(r['rgb_tp'],r['positive']),baseline_recall=rate(r['geometry_tp'],r['positive']),
                    candidate_fpr=rate(r['rgb_fp'],r['free']),baseline_fpr=rate(r['geometry_fp'],r['free']))
            return list(buckets.values())
        aggregates={n:aggregate(k) for n,k in {'all':['arm','baseline'], 'environment':['arm','baseline','environment'],
            'band':['arm','baseline','band'],'environment_band':['arm','baseline','environment','band']}.items()}
        write_csv(output/'frame_query_increments.csv',increments)
        for n,data in aggregates.items():write_csv(output/f'increments_{n}.csv',data)
        stage=time.perf_counter(); envs=sorted({rows[i]['environment'] for i in val})
        canonical=next(i for i in val if rows[i]['environment']==envs[0]);k=rows[canonical]['depth_K'];shape=rows[canonical]['depth_shape']
        maps={};canonical_depth={};grid=[]
        for env in envs:
            i=next(i for i in val if rows[i]['environment']==env)
            idx,valid,g=correspondence(k,rows[i]['depth_K'],shape,True,1e-3)
            maps[env]=(idx,valid);grid.append(dict(environment=env,**g))
            for j in val:
                if rows[j]['environment']==env:
                    canonical_depth[j]=infer('depth_only',k,shape,predicted_depth[j][idx]).reshape(len(qs),-1)
        geometry=infer('geometry',k,shape,np.ones(np.prod(shape),dtype=np.float32)).reshape(len(qs),-1)
        repeated=infer('geometry',k,shape,np.ones(np.prod(shape),dtype=np.float32)).reshape(len(qs),-1)
        if not np.array_equal(geometry,repeated):raise ValueError('Identical geometry inputs differ')
        np.save(output/'geometry_canonical.npy',geometry)
        pair_rows=[]
        for ea,eb in combinations(envs,2):
            ia,va=maps[ea];ib,vb=maps[eb]
            ra=next(i for i in val if rows[i]['environment']==ea); rb=next(i for i in val if rows[i]['environment']==eb)
            ax,ay=rays(rows[ra]['depth_K'],shape);bx,by=rays(rows[rb]['depth_K'],shape)
            direct=np.hypot(ax.reshape(-1)[ia]-bx.reshape(-1)[ib],ay.reshape(-1)[ia]-by.reshape(-1)[ib])
            allowed=va&vb&(direct<=1e-3)
            for qi,q in enumerate(qs):
                local={a:[] for a in ('depth_only_canonical_exact_input','geometry_canonical_exact_input')}
                for a in [i for i in val if rows[i]['environment']==ea]:
                    for b in [i for i in val if rows[i]['environment']==eb]:
                        check();la=labels[a].reshape(len(qs),-1)[qi,ia];lb=labels[b].reshape(len(qs),-1)[qi,ib]
                        mask=allowed&(la<2)&(lb<2)&(la!=lb);sign=np.where(la==1,1.,-1.)
                        local['depth_only_canonical_exact_input'].append(concordance(canonical_depth[a][qi],canonical_depth[b][qi],mask,sign))
                        local['geometry_canonical_exact_input'].append(concordance(geometry[qi],geometry[qi],mask,sign))
                for arm,c in local.items():pair_rows.append(dict(arm=arm,environment_a=ea,environment_b=eb,query=q['name'],frame_cartesian_pairs=64,**finish_concordance(c)))
        write_csv(output/'paired_scene_query.csv',pair_rows)
        pair_summary=[dict(arm=a,**finish_concordance([r for r in pair_rows if r['arm']==a])) for a in local]
        receipt['stages']['canonical_scene']=dict(wall_s=time.perf_counter()-stage,canonical_environment=rows[canonical]['environment'],
            shape=shape,depth_K=k,reference_and_observation_correspondence='public-K nearest approximate <=1e-3, actual paired rays <=1e-3; head ray/query exactly canonical',
            grid=grid,geometry_identical_input_exact_tie=True)
        write(output/'audit.json',dict(status='COMPLETE',recomputation_checks=checks,initial_state_sha256=initial_sha,
            checkpoint_updates=updates,shuffle_summary=extra['validation']['summary'],increments=aggregates,pair_summary=pair_summary,
            cutoffs=cutoffs,thresholds_refitted=False,limitations='Consumed Development; related query-ray pairs; sensor label/depth nearest approximate; no query-empty or walking event claims'))
        receipt['status']='COMPLETE'
        print(json.dumps(dict(status='COMPLETE',shuffle=extra['validation']['summary'],all_increments=aggregates['all'],pair_summary=pair_summary)),flush=True)
    except Exception as e:
        receipt.update(status='PARTIAL_BUDGET' if isinstance(e,TimeoutError) else 'FAILED',error=repr(e));raise
    finally:
        receipt.update(wall_s=time.perf_counter()-start,resource_release='CPU-only subprocess; no services or workers allocated')
        write(output/'receipt.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','sensor','features','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=240.)
    a=p.parse_args();run(a.source.resolve(),a.sensor.resolve(),a.features.resolve(),a.output.resolve(),a.budget_s)

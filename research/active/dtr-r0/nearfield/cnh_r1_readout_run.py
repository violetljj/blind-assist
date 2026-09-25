"""Frozen R1 readout diagnosis. No network fitting, capture, City or test."""
from __future__ import annotations
import argparse
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cnh_route_sensor import SensorParameters, _pulse_matrix, RAW_BIN_M
from cnh_qg1_response_decomposition import raw_stage, COMPONENTS
from cnh_street_development_baseline import sample_depth
from cnh_h3_geometry_diagnostic import query_weights, summarize, train_threshold
from cnh_r1_readout_ops import (train_median_bias, known_xtalk, raw_query_weights,
    matched_correlation, query_scores, temporal_mean)

PARAMS=SensorParameters()
PROTOCOL=Path(__file__).with_name('CNH_R1_READOUT_PROTOCOL_20260925.md')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def h3(raw):return raw.reshape(-1,64,16,8).sum(-1)
def h3score(hist):return np.einsum('nzb,qzb->nq',hist.astype(np.float64),query_weights())

def details(labels,scores,threshold):
    return dict(**summarize(labels,scores,threshold),
                query=[summarize(labels[:,q],scores[:,q],threshold) for q in range(6)])

def evaluate(labels,scores,train,dev,threshold=None,full=None):
    threshold=train_threshold(labels[train],scores[train]) if threshold is None else threshold
    result=dict(threshold=threshold,dev=details(labels[dev],scores[dev],threshold))
    if np.any(train):result['train']=details(labels[train],scores[train],threshold)
    if full is not None:
        sel=dev&full
        result['full_history']=dict(frames=int(sel.sum()),**details(labels[sel],scores[sel],threshold))
    return result

def synthesize(data,load_frame,deadline,params=PARAMS,new_seed=False):
    n=len(data['rows']); raw=np.empty((n,64,128)); quiet=np.empty_like(raw)
    samples=[]; seeds=[]
    for i in range(n):
        if time.monotonic()>deadline:raise TimeoutError('Frozen 1800s budget')
        item=load_frame(data,i)
        radial,area=sample_depth(item['depth'],item['camera'],16)
        seed=int(item['seed'])
        samples.append((radial,area));seeds.append(seed)
        raw[i]=raw_stage(radial,area,COMPONENTS,seed,params).reshape(64,128)
        quiet[i]=raw_stage(radial,area,COMPONENTS[:-1],seed,params).reshape(64,128)
        if not np.array_equal(h3(raw[i:i+1]).astype(np.float32)[0],data['histogram'][i]):
            raise ValueError(f'Original H3 parity failed frame {i}')
        if (i+1)%160==0:print(json.dumps(dict(synthesized=i+1,total=n)),flush=True)
    return raw,quiet,samples,seeds

def temporal(raw,data,k,comp,step=.1):
    return temporal_mean(raw,data['clip_ids'],data['steps'],k,nominal_step_m=step,compensate=comp)

def arms(raw,data,bias):
    matrix=_pulse_matrix(PARAMS); weights=raw_query_weights()
    known=known_xtalk(PARAMS.signal_counts,matrix,int(PARAMS.crosstalk_range_m/RAW_BIN_M))
    scores={'R0':h3score(data['histogram']),
            'R1_MEDIAN':h3score(data['histogram'].astype(float)-h3(bias[None])[0]),
            'R1_KNOWN':h3score(data['histogram'].astype(float)-known.reshape(16,8).sum(-1)),
            'R2_RAW_CONTROL':query_scores(raw,weights),
            'R2_ONLY':query_scores(matched_correlation(raw,matrix),weights)}
    matched=matched_correlation(raw-bias,matrix)
    scores['R2_MEDIAN']=query_scores(matched,weights)
    full={}
    for k in (1,2,4,8):
        for comp in (False,True):
            suffix=f'K{k}_'+('MC' if comp else 'AVG')
            name='R3_'+suffix
            scores[name]=query_scores(temporal(matched,data,k,comp),weights)
            full[name]=np.asarray(data['steps'])>=k-1
            name='R3_ONLY_'+suffix
            scores[name]=h3score(h3(temporal(raw,data,k,comp)).astype(np.float32))
            full[name]=np.asarray(data['steps'])>=k-1
    for name in ('R3_K1_MC','R3_K1_AVG'):
        if not np.array_equal(scores[name],scores['R2_MEDIAN']):raise ValueError('K1 != R2')
    return scores,full,matched

def run(output,worker_capture=None):
    from cnh_r1_readout_data import load_alley,load_street,load_frame
    started=time.monotonic();deadline=started+1800
    output.mkdir(parents=True,exist_ok=False)
    try:
        data=load_alley()
        raw,quiet,samples,seeds=synthesize(data,load_frame,deadline)
        bias=train_median_bias(raw,data['train'])
        np.save(output/'train-bias.npy',bias)
        scores,full,matched=arms(raw,data,bias)
        y,tm,dm=data['labels'],data['train'],data['dev']
        candidates={k:summarize(y[tm],scores[f'R3_K{k}_MC'][tm],0)['auprc'] for k in (1,2,4)}
        best_k=max(candidates,key=lambda k:(candidates[k],-k));best=f'R3_K{best_k}_MC'
        write(output/'selection.json',dict(source='alley_train_only',train_ap=candidates,best_k=best_k,best=best))
        results={name:evaluate(y,s,tm,dm,full=full.get(name)) for name,s in scores.items()}
        if abs(results['R0']['dev']['auprc']-.21011599820183707)>1e-12:raise ValueError('R0 AP parity')
        qb=train_median_bias(quiet,tm)
        known=known_xtalk(PARAMS.signal_counts,_pulse_matrix(PARAMS),int(PARAMS.crosstalk_range_m/RAW_BIN_M))
        calibration={name:evaluate(y,h3score(h3(quiet-b)),tm,dm) for name,b in [('MEDIAN',qb),('KNOWN',known)]}
        errors={}
        for step in (.07,.10,.13):
            s=query_scores(temporal(matched,data,best_k,True,step),raw_query_weights())
            errors[str(step)]=evaluate(y,s,tm,dm,full=np.asarray(data['steps'])>=best_k-1)
            scores['ERROR_'+str(step)]=s
        sensitivity={}; newseeds=[]
        for row,old in zip(data['rows'],seeds):
            seed=int(hashlib.sha256(('cnh-r1-readout-noise-v1|'+row['frame_key']).encode()).hexdigest()[:16],16)
            if seed==old:raise ValueError('New noise seed collision')
            newseeds.append(seed)
        for name,ambient,signal in [('base',1,1),('ambient_half',.5,1),('ambient_double',2,1),('signal_half',1,.5),('signal_double',1,2)]:
            p=replace(PARAMS,ambient_counts=PARAMS.ambient_counts*ambient,signal_counts=PARAMS.signal_counts*signal)
            response=np.empty_like(raw)
            for i,((radial,area),seed) in enumerate(zip(samples,newseeds)):
                if time.monotonic()>deadline:raise TimeoutError('Frozen 1800s budget')
                response[i]=raw_stage(radial,area,COMPONENTS,seed,p).reshape(64,128)
            b=train_median_bias(response,tm)
            r0=h3score(h3(response).astype(np.float32))
            improved=query_scores(temporal(matched_correlation(response-b,_pulse_matrix(p)),data,best_k,True),raw_query_weights())
            sensitivity[name]=dict(parameters=asdict(p),R0=evaluate(y,r0,tm,dm),best=evaluate(y,improved,tm,dm))
            scores['NOISE_'+name+'_R0']=r0;scores['NOISE_'+name+'_best']=improved
            np.save(output/f'bias-{name}.npy',b)
            print(json.dumps(dict(sensitivity=name)),flush=True)
            if time.monotonic()>deadline:raise TimeoutError('Frozen budget')
        np.savez_compressed(output/'alley-scores.npz',labels=y,train=tm,dev=dm,frame_key=[r['frame_key'] for r in data['rows']],**scores)
        write(output/'noise-seeds.json',dict(frame_key=[r['frame_key'] for r in data['rows']],original=seeds,new=newseeds))
        decision=dict(crosstalk=calibration['MEDIAN']['dev']['auprc']>=.85,
                      breakthrough=results[best]['dev']['auprc']>=.42,
                      compensation=results[best]['dev']['auprc']>results[f'R3_K{best_k}_AVG']['dev']['auprc'],
                      robustness=all(sensitivity[n]['best']['dev']['auprc']>sensitivity[n]['R0']['dev']['auprc'] for n in ('ambient_double','signal_half')))
        decision['overall']=all(decision.values())
        report=dict(status='ALLEY_COMPLETE',results=results,calibration=calibration,best=best,best_k=best_k,
                    step_sensitivity=errors,noise_sensitivity=sensitivity,decision=decision,
                    source_hashes={p.name:sha(p) for p in [Path(__file__),PROTOCOL,Path(__file__).with_name('cnh_r1_readout_ops.py'),Path(__file__).with_name('cnh_r1_readout_data.py')]},
                    protocol_sha256=sha(PROTOCOL),alley_identity=data.get('identity'),parity_frames=len(raw))
        write(output/'result.json',report)
        # Street has no fitting or model selection role. No other roots searched.
        street=load_street(require_depth=False,worker_capture=worker_capture)
        sr,sq,_,_=synthesize(street,load_frame,deadline)
        ss,sfull,sm=arms(sr,street,bias)
        sy=street['labels'];allstreet=np.ones(len(sy),dtype=bool);notrain=~allstreet
        street_results={name:evaluate(sy,s,notrain,allstreet,threshold=results[name]['threshold'],full=sfull.get(name)) for name,s in ss.items()}
        np.savez_compressed(output/'street-scores.npz',labels=sy,frame_key=[r['frame_key'] for r in street['rows']],**ss)
        report.update(status='COMPLETE',street=street_results,street_identity=street.get('identity'),wall_s=time.monotonic()-started)
        write(output/'result.json',report)
        return report
    except BaseException as exc:
        write(output/'failure.json',dict(status='STOP',error=repr(exc),wall_s=time.monotonic()-started))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-capture',type=Path)
    a=p.parse_args();r=run(a.output,a.worker_capture);print(json.dumps(dict(status=r['status'],decision=r['decision'])))

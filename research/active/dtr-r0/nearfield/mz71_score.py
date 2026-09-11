"""Independent MZ71 saved-output state-cut audit. No models, fitting or ranking.

--selftest opens no experiment files. Scientific score requires ROOT_SCORE_GO
with run_exit_code0 and exact run_receipt_sha256/scorer_sha256 before output.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import traceback
import numpy as np
from mz58_score import read, write, sha, load, metrics, scalar_metrics, paired
from mz59_score import scalar_cutoff, native_masks, assert_metric_equal
from mz64_score import partitions, compact_native, event_rows, summarize_margin, COHORTS, PROFILES, SOURCE_PROFILES, QUERIES
from mz68_score import reconstruct as reconstruct68
from mz69_score import split, scalar_exchange
from mz63_score import exchange_masks, check_packets

TASK='mz71-missing-state-calibration-20260911'
ARMS=('CONTROL','DIVERSE');OLD_KEYS=tuple('MZ70/'+a for a in ARMS);KEYS=tuple('MZ71/'+a for a in ARMS)
NEW=tuple(k+s for k in KEYS for s in ('/candidate','/UNION'))
BASE='OLD_NEG/UNION';OLD_LOCAL=('OLD_NEG/OPEN','OLD_NEG/GATED')
RUN70='a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'
SCORE70='2fe35ce61155ba80c7197400d4331f60410159cc550270dca623153e2f23f84a'
OLD_ROWS=dict(DEV=1000,relation10000=2000,distance5000=1000,rich=44,mz36=380,mz48=2560,mz55=2560)
LOCAL_FIELDS=('raw','support','winner','anchor_vector','anchor_available')
CLAIM='Frozen MZ70 heads, original-cut versus missing-state calibration on the same original1256 calibration IDs under ALL_INVALID. At application time missing means no valid packet slot, regardless of profile name. Every nonmissing row preserves the original decision bytes; all historical arrays remain untouched. No fit, ranking, selected operating-point sweep or overall promotion gate.'


def exact(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def scalar_candidate(raw,support,base,cut):
    assert raw.shape==support.shape==base.shape and raw.ndim==2 and raw.shape[1]==4
    assert support.dtype==bool and np.isfinite(raw).all() and np.isfinite(base).all()
    assert cut.shape in ((4,),raw.shape) and np.isfinite(cut).all()
    out=np.empty(base.shape,np.float64)
    for i in range(len(base)):
        for q in range(4):
            delta=float(raw[i,q])-float(cut[q] if cut.ndim==1 else cut[i,q])
            out[i,q]=delta if base[i,q]<0 and support[i,q] and delta>=0 else base[i,q]
    return out


def check_state(p,prefix,newcut,oldcut,oldnegcuts):
    null=prefix.endswith('ALL_INVALID/');base=p[prefix+'MZ37'];n=len(base);checked=0
    ranges,valid=p[prefix+'ranges'],p[prefix+'valid']
    assert ranges.shape==valid.shape==(n,64,2) and ranges.dtype==np.float32 and valid.dtype==bool
    assert np.isfinite(ranges).all() and ((ranges[valid]>0)&(ranges[valid]<=4)).all()
    missing=~valid.any((1,2));exact(p[prefix+'MZ71/missing_mask'],missing)
    if null:
        assert not ranges.any() and missing.all()
        for key in OLD_LOCAL:
            cand=scalar_candidate(p[prefix+key+'/raw'],p[prefix+key+'/support'],base,oldnegcuts[key])
            np.testing.assert_array_equal(cand,p[prefix+key+'/candidate']);checked+=cand.size
        np.testing.assert_array_equal(p[prefix+BASE],np.maximum(*(p[prefix+k+'/candidate'] for k in OLD_LOCAL)))
    for old,key in zip(OLD_KEYS,KEYS):
        raw,sup,w,vec,av=(p[prefix+old+'/'+f] for f in LOCAL_FIELDS)
        assert raw.shape==sup.shape==w.shape==(n,4) and sup.dtype==bool
        assert np.issubdtype(w.dtype,np.integer) and ((w>=0)&(w<3600)).all()
        assert av.dtype==bool and av.shape==(n,) and vec.shape==(n,8) and np.isfinite(vec).all() and (vec[~av]==0).all()
        np.testing.assert_array_equal(av,~missing)
        if null:assert not av.any() and not vec.any()
        original=scalar_candidate(raw,sup,base,oldcut[old])
        np.testing.assert_array_equal(original,p[prefix+old+'/candidate'])
        np.testing.assert_array_equal(p[prefix+old+'/UNION'],np.maximum(p[prefix+BASE],original))
        row_cut=np.where(missing[:,None],newcut[key][None,:],oldcut[old][None,:])
        expected=scalar_candidate(raw,sup,base,row_cut)
        np.testing.assert_array_equal(expected,p[prefix+key+'/candidate'])
        np.testing.assert_array_equal(p[prefix+key+'/UNION'],np.maximum(p[prefix+BASE],expected))
        for suffix in ('candidate','UNION'):exact(p[prefix+old+'/'+suffix][~missing],p[prefix+key+'/'+suffix][~missing])
        # Alias only in the scorer's local view: no added model outputs or decoding.
        for field in LOCAL_FIELDS:
            if prefix+key+'/'+field in p:exact(p[prefix+key+'/'+field],p[prefix+old+'/'+field])
            else:p[prefix+key+'/'+field]=p[prefix+old+'/'+field]
        checked+=n*16
    return checked


def effect(a,t,k,ii,new,old,win):
    masks=exchange_masks(a[new][ii],a[old][ii],t[ii],k[ii]);local={name:v[ii] for name,v in win.items()}
    return dict(frames=len(ii),original=metrics(a[old][ii],t[ii],k[ii]),state_cut=metrics(a[new][ii],t[ii],k[ii]),
        exchanges={name:v.sum(0).tolist() for name,v in masks.items()},
        gained_true=split(masks['TP_GAIN'],local),lost_true=split(masks['TP_LOST'],local),
        added_false=split(masks['FP_ADDED'],local),removed_false=split(masks['FP_REMOVED'],local),
        limitation='Same saved winner before/after; no raw learning or localization change. Native attribution is lookup, not dense argmax replay or feature causality.')


def state_margin(raw,cut,t,k,native,ii):
    answer={}
    for q,name in enumerate(QUERIES):
        m=raw[ii,q].astype(float)-cut[ii,q];pos=t[ii,q]&k[ii,q]
        answer[name]={}
        for label,mask in (('positive',pos),('negative',~t[ii,q]&k[ii,q]),('native_positive',pos&native[ii,q])):
            v=m[mask];answer[name][label]=dict(n=len(v),quantiles=np.quantile(v,[0,.25,.5,.75,1]).tolist() if len(v) else [])
    return answer


def run(task):
    task=task.resolve(strict=True);assert task.name==TASK;work=task.parent;rd=task/'run-v1';out=task/'score-v1'
    go=read(task/'score-go-v1.json');assert go['status']=='ROOT_SCORE_GO' and go['run_exit_code']==0
    assert go['run_receipt_sha256']==sha(rd/'receipt.json') and go['scorer_sha256']==sha(__file__)
    rr=read(rd/'receipt.json');assert rr['status']=='PASS'
    assert not out.exists();out.mkdir();inputs={};start=time.perf_counter()
    try:
        def bind(path,expected=None):
            path=Path(path).resolve(strict=True);h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h;return path
        for name in ('mz71_score.py','mz68_score.py','mz66_score.py','mz64_score.py','mz63_score.py','mz62_score.py','mz59_score.py','mz58_score.py','mz69_score.py'):bind(Path(__file__).with_name(name))
        bind(task/'score-go-v1.json');bind(rd/'receipt.json',go['run_receipt_sha256'])
        assert rr['training_steps']==rr['threshold_sweeps']==0 and rr['new_cutoffs']==2
        assert rr['original_calibration_frames']==1256 and rr['calibration_profile']=='ALL_INVALID'
        assert rr['mz55_calibration_rows_used']==rr['mz61_calibration_rows_used']==rr['mz67_calibration_rows_used']==0
        assert rr['new_inference_frames']==9544 and rr['new_inference_cohorts']==list(OLD_ROWS)
        assert rr['initial_parity_unique_train_frames']==32 and rr['source61_67_cohort_reinference_frames']==0
        assert rr['original_mz70_receipt_sha256']==RUN70 and rr['fixed_checkpoints'] and rr['old_negative_cutoffs_unchanged']
        assert not rr['raw_scores_changed'] and rr['nonmissing_rows_byte_exact'] and rr['source_unknown_preserved']
        assert rr['native_depth_reads']==rr['old_dense_cache_reads']==0 and not rr['permanent_dense_cache']
        for path,h in rr['inputs'].items():
            if path.endswith(('.py','.json','.md')):bind(path,h)
        for name,h in rr['outputs'].items():bind(rd/name,h)
        def producer(path):
            path=Path(path).resolve(strict=True);matches=[h for p,h in rr['inputs'].items() if Path(p).resolve()==path]
            assert len(matches)==1,str(path);return bind(path,matches[0])
        prep=task/'source-preparation-v1';prep_receipt=read(producer(prep/'receipt.json'))
        assert prep_receipt['status']=='PASS' and prep_receipt['observed_packet_prefixes']==36 and prep_receipt['training_steps']==0
        observed=load(producer(prep/'observed-packets.npz'));assert sha(prep/'observed-packets.npz')==prep_receipt['outputs']['observed-packets.npz']
        prior=work/'mz70-diverse-learning-20260911';pr=read(bind(prior/'run-v1/receipt.json',RUN70));ps=read(bind(prior/'score-v1/receipt.json',SCORE70))
        assert set(rr['checkpoints'])==set(ARMS)
        for arm in ARMS:
            ref=rr['checkpoints'][arm];assert ref['sha256']==pr['outputs'][arm+'.pt']
            assert Path(ref['path']).resolve()==(prior/'run-v1'/(arm+'.pt')).resolve();bind(ref['path'],ref['sha256'])
        previous=read(bind(prior/'score-v1/result.json',ps['outputs']['result.json']))
        old=load(bind(prior/'run-v1/predictions.npz',pr['outputs']['predictions.npz']));p=load(rd/'predictions.npz')
        for key,value in old.items():exact(p[key],value)
        preserved=len(old);preserved_names=sorted(old);del old
        assert sorted(rr['baseline_arrays_preserved'])==preserved_names
        groups=read(rd/'groups.json');pg=read(bind(prior/'run-v1/groups.json',pr['outputs']['groups.json']));assert groups==pg
        parity=read(rd/'initial-parity.json');assert parity['status']=='PASS' and parity['unique_train_frames']==32
        assert parity['profiles']==['ALL_INVALID'] and parity['atol']==2e-5 and parity['rtol']==1e-6 and not parity['scientific_source_predictions_replaced']
        for c in ('mz61','mz67'):
            chosen=sorted(i for i,r in enumerate(groups[c+'_records']) if r['role']=='TRAIN_CANDIDATE')[:16]
            assert parity['selected'][c]==chosen
            keys={x['key'] for x in parity['comparisons'] if x['cohort']==c}
            assert {'MZ37'}|{key+'/'+f for key in OLD_KEYS+OLD_LOCAL for f in ('raw','support','candidate')}<=keys
        records={'mz48':groups['records'],**{c:groups[c+'_records'] for c in ('mz55','mz61','mz67')}}
        g48={key:np.array(value,int) for key,value in groups['groups'].items()};g48['nonfit']=np.sort(np.r_[g48['heldout_site'],g48['nonfit_family']])
        r55=records['mz55'];g55={role:np.array([i for i,r in enumerate(r55) if r['role']==role]) for role in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_SITE')}
        g55.update(heldout_new_family=np.array([i for i in g55['HELDOUT_SITE'] if r55[i]['family']!='retained_rod']),nonfit=np.sort(np.r_[g55['CALIBRATION'],g55['HELDOUT_SITE']]))
        parts={c:partitions(rec,g48 if c=='mz48' else g55 if c=='mz55' else None) for c,rec in records.items()}
        labels={}
        for c,rec in records.items():
            ref=rr['source_info']['label_refs'][c];assert ref==pr['source_info']['label_refs'][c]
            lab=load(bind(ref['path'],ref['sha256']));labels[c]=lab;counts,vc=lab['fullframe_event_counts'],lab['valid_counts'];n=len(rec)
            assert counts.shape==(n,45,80,4) and vc.shape==(n,45,80) and counts.dtype==vc.dtype==np.uint8
            assert (counts<=vc[...,None]).all() and (vc<=64).all()
            np.testing.assert_array_equal(lab['frame_ids'],p[c+'/frame_ids']);np.testing.assert_array_equal(counts.sum((1,2))>=3,p[c+'/truth'])
        cut={};oldcut={};authority={};cal=g48['calibration'];assert len(cal)==256
        def calibration(name):return np.concatenate([p['DEV/ALL_INVALID/'+name],p['mz48/ALL_INVALID/'+name][cal]])
        truth=np.concatenate([p['DEV/truth'],p['mz48/truth'][cal]]);known=np.concatenate([p['DEV/known'],p['mz48/known'][cal]])
        calibration_ids=load(bind(prep/'calibration-ids.npz',prep_receipt['outputs']['calibration-ids.npz']))
        np.testing.assert_array_equal(calibration_ids['DEV'],np.arange(1000));np.testing.assert_array_equal(calibration_ids['mz48'],cal)
        for arm,key,original in zip(ARMS,KEYS,OLD_KEYS):
            oldcut[original]=np.load(bind(prior/'run-v1'/(arm+'-cutoff.npy'),pr['outputs'][arm+'-cutoff.npy']),allow_pickle=False)
            exact(oldcut[original],np.load(rd/('MZ70-'+arm+'-cutoff.npy'),allow_pickle=False))
            cut[key]=np.load(rd/(arm+'-cutoff.npy'),allow_pickle=False)
            rebuilt,auth=scalar_cutoff(calibration(original+'/raw'),calibration(original+'/support'),calibration('MZ37'),truth,known)
            np.testing.assert_array_equal(cut[key],rebuilt);authority[key]=auth
        oldneg={};r69=read(bind(work/'mz69-topology-transfer-20260911/run-v1/receipt.json','b65fa2e2c45c2014761fb03254267b38307c1ceaf49c846653da562e6ffdf259'))
        for key in OLD_LOCAL:
            name=key.replace('/','-')+'-cutoff.npy';path=bind(work/'mz69-topology-transfer-20260911/run-v1'/name,r69['outputs'][name]);oldneg[key]=np.load(path,allow_pickle=False)
            exact(oldneg[key],np.load(rd/name,allow_pickle=False))
        scores={};changes={};attrs={};margins={};events={};contexts={};effects={};routing={};cache={};scalar=prior_rows=lookups=nonmissing=0
        for c in COHORTS+('mz67',):
            t,k,ids=(p[c+'/'+name] for name in ('truth','known','frame_ids'))
            if c in OLD_ROWS:assert len(t)==OLD_ROWS[c]
            if c=='mz36':
                t,k,ids=(p[c+'/attempted_'+name] for name in ('truth','known','frame_ids'));assert len(t)==400 and int((~k).sum())==80
            partition=parts.get(c,{'all':np.arange(len(t))})
            original_packet={name:p[c+'/IDEAL/'+name] for name in ('ranges','valid')}
            for dic in (scores,changes,attrs,margins,events,contexts,effects,routing):dic[c]={}
            for profile in SOURCE_PROFILES:
                prefix=c+'/'+profile+'/';before=previous['conditions'][c].get(profile)
                for name in ('ranges','valid'):exact(p[prefix+name],observed[prefix+name])
                exact(p[prefix+'MZ71/missing_mask'],observed[prefix+'missing_mask'])
                if before:
                    oldmethods=tuple(before['all']);assert set(before)==set(partition)
                    if c=='mz67':a={m:p[prefix+m] for m in oldmethods}
                    else:
                        older=tuple(m for m in oldmethods if not m.startswith('MZ70/'));a=reconstruct68(p,prefix,older)
                        a.update({m:p[prefix+m] for m in oldmethods if m.startswith('MZ70/')})
                else:
                    assert c in OLD_ROWS and profile=='ALL_INVALID'
                    oldmethods=('MZ37',BASE)+tuple(key+'/candidate' for key in OLD_LOCAL)+tuple(key+s for key in OLD_KEYS for s in ('/candidate','/UNION'))
                    a={m:p[prefix+m] for m in oldmethods}
                scalar+=check_state(p,prefix,cut,oldcut,oldneg)
                check_packets(original_packet,{name:p[prefix+name] for name in ('ranges','valid')},profile)
                missing=p[prefix+'MZ71/missing_mask'];nonmissing+=int((~missing).sum())*16
                full_missing=missing
                for method in NEW:a[method]=p[prefix+method]
                if c=='mz36':
                    full_missing=np.zeros(400,bool);full_missing[p['mz36/admitted_index']]=missing
                    for m,v in list(a.items()):a[m]=np.full((400,4),np.nan);a[m][p['mz36/admitted_index']]=v
                refs=tuple(m for m in (BASE,OLD_KEYS[0]+'/UNION',OLD_KEYS[1]+'/UNION',KEYS[0]+'/UNION',KEYS[1]+'/UNION','MZ64/GEOMETRY/UNION','MZ68/NULL_COVERAGE/UNION','MZ66/PROJECT/UNION') if m in a)
                for dic in (scores,changes,attrs,margins,events,contexts,effects,routing):dic[c][profile]={}
                for group,ii in partition.items():
                    scores[c][profile][group]={m:metrics(v[ii],t[ii],k[ii]) for m,v in a.items()}
                    if before:
                        for m in oldmethods:assert_metric_equal(scores[c][profile][group][m],before[group][m]);prior_rows+=1
                    changes[c][profile][group]={m:{ref:paired(a[m][ii],a[ref][ii],t[ii],k[ii]) for ref in refs if ref!=m} for m in NEW}
                    routing[c][profile][group]=dict(attempted_frames=len(ii),known_frames=int(k[ii].all(1).sum()),missing_admitted_frames=int(full_missing[ii].sum()),
                        nonmissing_admitted_frames=int((k[ii].all(1)&~full_missing[ii]).sum()),effects={})
                    for oldkey,newkey in zip(OLD_KEYS,KEYS):
                        mask=exchange_masks(a[newkey+'/UNION'][ii],a[oldkey+'/UNION'][ii],t[ii],k[ii])
                        assert not np.logical_or.reduce(list(mask.values()))[~full_missing[ii]].any()
                        routing[c][profile][group]['effects'][newkey]={name:v.sum(0).tolist() for name,v in mask.items()}
                for m in NEW:assert_metric_equal(scalar_metrics(a[m],t,k),scores[c][profile]['all'][m])
                cache.setdefault(profile,{})[c]=(a,t,k);details={};winners={}
                if c in labels:
                    lab=labels[c]
                    for original,key in zip(OLD_KEYS,KEYS):
                        w,native,wk=native_masks(p,prefix,original,lab['fullframe_event_counts'],lab['valid_counts']);lookups+=native.size
                        winners[key]=dict(native=native,known=wk);selected_cut=np.where(missing[:,None],cut[key][None,:],oldcut[original][None,:])
                        details[key]=(w,native,wk,p[prefix+original+'/raw'],selected_cut,p[prefix+original+'/support'],p[prefix+'MZ37'])
                        for group,ii in partition.items():
                            attrs[c][profile].setdefault(group,{})[key]={ref:compact_native(p,prefix,key,(w,native,wk),a,t,k,selected_cut,ii,ref) for ref in refs}
                            margins[c][profile].setdefault(group,{})[key]=state_margin(p[prefix+original+'/raw'],selected_cut,t,k,native,ii)
                            effects[c][profile].setdefault(group,{})[key]=effect(a,t,k,ii,key+'/UNION',original+'/UNION',winners[key])
                for m in NEW:
                    key=m.rsplit('/',1)[0];events[c][profile][m]={ref:event_rows(a,t,k,ids,m,ref,records.get(c)) for ref in refs if ref!=m}
                    for rows in events[c][profile][m].values():
                        for row in rows:
                            i,q=row['index'],QUERIES.index(row['query']);row['missing_state']=bool(full_missing[i])
                            if key in details:
                                w,native,wk,raw,selected_cut,sup,base=details[key]
                                row.update(winner=int(w[i,q]),native=bool(native[i,q]),local_known=bool(wk[i,q]),raw=float(raw[i,q]),cutoff=float(selected_cut[i,q]),margin=float(raw[i,q])-float(selected_cut[i,q]),support=bool(sup[i,q]),native_added_credit=bool(row['change']=='TP_GAIN' and native[i,q] and sup[i,q] and base[i,q]<0))
                if c in records:
                    pairs=defaultdict(list)
                    for i,row in enumerate(records[c]):pairs[row['pair_id']].append(i)
                    assert all(len(v)==2 for v in pairs.values());pi=np.array(list(pairs.values()));np.testing.assert_array_equal(t[pi[:,0]],t[pi[:,1]])
                    for m in tuple(oldmethods)+NEW:
                        changed=(a[m][pi[:,0]]>=0)!=(a[m][pi[:,1]]>=0)
                        contexts[c][profile][m]=dict(pairs=len(pi),changed_pairs=int(changed.any(1).sum()),changed_queries=changed.sum(0).tolist(),pair_ids=[key for key,v in zip(pairs,changed) if v.any()])
        aggregates={};aggregate_changes={}
        for profile in SOURCE_PROFILES:
            data=cache[profile];legacy=[(c,np.arange(len(data[c][1]))) for c in ('relation10000','distance5000','rich','mz36')]
            selections=dict(legacy_noncal=legacy,mz48_fit=[('mz48',g48['fit'])],mz48_nonfit=[('mz48',g48['nonfit'])],mz55_held640=[('mz55',parts['mz55']['HELDOUT_SITE'])])
            selections['all_old_noncal']=legacy+selections['mz48_fit']+selections['mz48_nonfit'];aggregates[profile]={};aggregate_changes[profile]={}
            for group,sel in selections.items():
                t=np.concatenate([data[c][1][ii] for c,ii in sel]);k=np.concatenate([data[c][2][ii] for c,ii in sel]);methods=tuple(data[sel[0][0]][0])
                aa={m:np.concatenate([data[c][0][m][ii] for c,ii in sel]) for m in methods};aggregates[profile][group]={m:metrics(v,t,k) for m,v in aa.items()}
                if profile in previous['aggregates']:
                    for m,row in previous['aggregates'][profile][group].items():assert_metric_equal(aggregates[profile][group][m],row);prior_rows+=1
                aggregate_changes[profile][group]={m:{ref:paired(aa[m],aa[ref],t,k) for ref in (BASE,OLD_KEYS[0]+'/UNION',OLD_KEYS[1]+'/UNION',KEYS[0]+'/UNION',KEYS[1]+'/UNION') if ref!=m} for m in NEW}
        result=dict(status='PASS',claim=CLAIM,primary=effects['mz67']['ALL_INVALID']['HELDOUT_GEOMETRY'],conditions=scores,comparisons=changes,
            aggregates=aggregates,aggregate_comparisons=aggregate_changes,attribution=attrs,margins=margins,same_arm_effects=effects,support_context_pairs=contexts,missing_state_routing=routing,
            original_cuts={key:v.tolist() for key,v in oldcut.items()},missing_cuts={key:v.tolist() for key,v in cut.items()},cutoff_authority=authority,
            limitation='All rankings and raw winners are unchanged within each same-arm calibration comparison. No new model training or model selection; no hardware, sensor-fidelity, natural-scene, safe-clearance, dense-argmax or trained-ceiling claim. Missing-state routing uses packet validity, including naturally missing original-profile rows; profiles are not predictor inputs.')
        audit=dict(status='PASS',baseline_arrays_byte_exact=preserved,prior_metric_rows_exact=prior_rows,scalar_candidate_union_values=scalar,native_winner_lookups=lookups,
            nonmissing_copied_scalar_values=nonmissing,old_all_invalid_inference_frames=sum(OLD_ROWS.values()),new_calibration_vectors=2,calibration_ids=1256,
            observed_packet_prefixes_bound=36,source_packet_profiles_recomputed=True,initial_parity_winner_checks=[x for x in parity['comparisons'] if 'winner_equal' in x],
            mz36_attempted_frames=400,mz36_admitted_frames=380,mz36_UNKNOWN_query_bits=80,local_UNKNOWN_cells={c:int((lab['valid_counts']==0).sum()) for c,lab in labels.items()},new_fit=0,new_model_inference=0,new_cutoff_search=0,native_argmax_recomputed=False,ranking_recomputed=False)
        for path,h in inputs.items():assert sha(path)==h,path
        write(out/'result.json',result);write(out/'audit.json',audit);write(out/'paired-events.json',events);report(out/'report.md',result)
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={n:sha(out/n) for n in ('result.json','audit.json','paired-events.json','report.md')},seconds=time.perf_counter()-start,backend='FROZEN_PROTOCOL_CPU_ONLY',new_fit=0,new_inference=0,new_cutoff_search=0))
        print('PASS',result['primary'])
    except BaseException:write(out/'failure.json',dict(status='FAIL',inputs=inputs,error=traceback.format_exc()));raise


def report(path,r):
    lines=['# MZ71 missing-state calibration','',r['claim'],'','| MZ67 HELD ALL_INVALID arm/query | Original TP/FP/FN | State-cut TP/FP/FN | TP gain/loss | FP added/removed | Gained native/wrong/UNKNOWN |','|---|---|---|---|---|---|']
    for key,row in r['primary'].items():
        for q,name in enumerate(QUERIES):
            ex=row['exchanges'];gain=row['gained_true']
            lines.append('| '+key+'/'+name+' | '+'/'.join(str(row['original'][k][q]) for k in ('tp','fp','fn'))+' | '+'/'.join(str(row['state_cut'][k][q]) for k in ('tp','fp','fn'))+f" | {ex['TP_GAIN'][q]}/{ex['TP_LOST'][q]} | {ex['FP_ADDED'][q]}/{ex['FP_REMOVED'][q]} | {gain['native'][q]}/{gain['known_wrong'][q]}/{gain['unknown'][q]} |")
    lines+=['',r['limitation'],'','All original and state-cut candidate/OR methods, source groups, old costs, local UNKNOWN, exact exchanged IDs and pair effects remain in result.json and paired-events.json. No overall gate suppresses reported costs.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def selftest():
    raw=np.zeros((1256,4),np.float32);raw[7]=2.;raw[8]=1.;sup=np.ones_like(raw,bool);base=np.full_like(raw,-1.);t=np.zeros_like(raw,bool);t[7]=True;k=np.ones_like(t)
    cut,auth=scalar_cutoff(raw,sup,base,t,k);np.testing.assert_array_equal(cut,np.full(4,np.nextafter(1.,np.inf)));assert all(a['maximum_rows']==[8] for a in auth)
    cand=scalar_candidate(raw[:9],sup[:9],base[:9],cut);assert (cand[7]>=0).all() and (cand[8]<0).all()
    pre='DEV/ALL_INVALID/';p={pre+'MZ37':np.full((3,4),-1.),pre+'ranges':np.zeros((3,64,2),np.float32),pre+'valid':np.zeros((3,64,2),bool),pre+'MZ71/missing_mask':np.ones(3,bool)}
    oldneg={key:np.zeros(4) for key in OLD_LOCAL}
    for key in OLD_LOCAL:p[pre+key+'/raw']=np.full((3,4),-1.);p[pre+key+'/support']=np.ones((3,4),bool);p[pre+key+'/candidate']=np.full((3,4),-1.)
    p[pre+BASE]=np.full((3,4),-1.);oldcut={key:np.ones(4) for key in OLD_KEYS};newcut={key:np.zeros(4) for key in KEYS}
    for old,new in zip(OLD_KEYS,KEYS):
        fields=dict(raw=np.full((3,4),.5),support=np.ones((3,4),bool),winner=np.zeros((3,4),int),anchor_vector=np.zeros((3,8)),anchor_available=np.zeros(3,bool))
        for f,v in fields.items():p[pre+old+'/'+f]=v
        p[pre+old+'/candidate']=p[pre+old+'/UNION']=np.full((3,4),-1.)
        p[pre+new+'/candidate']=p[pre+new+'/UNION']=np.full((3,4),.5)
    check_state(p,pre,newcut,oldcut,oldneg)
    for old,new in zip(OLD_KEYS,KEYS):assert p[pre+old+'/raw'] is p[pre+new+'/raw']
    after=p[pre+KEYS[0]+'/UNION'];before=p[pre+OLD_KEYS[0]+'/UNION'];truth=np.zeros((3,4),bool);truth[0]=True;known=np.ones_like(truth);known[2]=False
    masks=exchange_masks(after,before,truth,known);assert {n:v.sum(0).tolist() for n,v in masks.items()}==scalar_exchange(after,before,truth,known)
    assert masks['TP_GAIN'].sum()==4 and masks['FP_ADDED'].sum()==4 and masks['UNKNOWN_DECISION_ADDED'].sum()==4
    p[pre+'ranges'][0,0,0]=1
    try:check_state(p,pre,newcut,oldcut,oldneg)
    except AssertionError:pass
    else:raise AssertionError('Residual range accepted')
    # Actual availability, not the profile label, selects the cutoff.
    other='DEV/DROP_CLOSE/';v={other+key[len(pre):]:value.copy() for key,value in p.items() if key.startswith(pre)}
    v[other+'valid'][0,0,0]=True;v[other+'MZ71/missing_mask'][0]=False
    for old,new in zip(OLD_KEYS,KEYS):
        v[other+old+'/anchor_available'][0]=True;v[other+new+'/anchor_available'][0]=True
        for suffix in ('candidate','UNION'):v[other+new+'/'+suffix][0]=-1.
    check_state(v,other,newcut,oldcut,oldneg)
    v[other+KEYS[0]+'/candidate'][0,0]=.5
    try:check_state(v,other,newcut,oldcut,oldneg)
    except AssertionError:pass
    else:raise AssertionError('Nonmissing recut accepted')
    print('PASS: independent cutoff maximum/nextafter; same raw recut; UNKNOWN/FP/TP exchanges; no residual range; mixed DROP availability routing; nonmissing mutation rejected')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--task',type=Path);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:selftest()
    else:
        assert args.task;run(args.task if args.task.is_absolute() else args.root/args.task)

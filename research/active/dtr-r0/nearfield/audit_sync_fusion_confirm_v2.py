"""Independent v2 protocol, source, old-training, calibration and cache audits.

New eval references and fusion model inference are never opened by this script.
Post-eval metrics use the once-opened immutable query cache only. Public input
observations and predeclared reference admission counts are allowed pre-eval.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np


def protocol_review(root,protocol):
    """Only registered text and inherited contracts; no new candidate access."""
    p=load(protocol)
    prior=load(Path(__file__).with_name('SYNC_FUSION_V1_PLAN_DEV_20261011.json'))
    assert p['features']==prior['features']
    for model in ('HGB','Logistic'):
        assert p['models'][model]==prior['models'][model]
    assert p['models']['seeds']==[955,956,957]
    assert p['calibration']==prior['calibration']
    assert p['training']['roles_of_v1_features']==['train','eval']
    assert (p['new_data']['target_visits'],p['new_data']['cal_visits'],p['new_data']['eval_visits'])==(18,6,12)
    assert p['new_data']['gate']['near_POS_min']==16
    assert p['new_data']['gate']['midfar_strict_FREE_min']==32
    assert p['new_data']['frames_per_visit']==32
    assert p['eval']['bootstrap']['replicates']==2000 and p['eval']['bootstrap']['seed']==20261011
    assert p['confirmation']['primary_method']=='logit'
    assert p['confirmation']['primary_source']=='native_perturbed'
    assert p['confirmation']['primary_band']=='0.8-1.5m'
    assert p['budget']['CPU_command_wall_s']==2400 and p['budget']['GPU_wall_s']==600 and p['budget']['download_bytes']==4000000000
    assert sha(root/'PLAN.json')==sha(protocol)
    return dict(status='PASS',protocol_sha256=sha(protocol),inherited_features_and_model_recipes='byte-equivalent decoded contracts',checks=['Commit before new candidate data','train old6+consumed eval4 only; oldcal excluded','18 fresh visits; fixed6cal/12eval','reference-only unchanged admission; frozen chronological windows','fullties calibration and missing-denominator rules','logit native middle sole primary; CI deltaW/POS positive and FREE<=.075','12visit paired bootstrap2000; fixed best-single identity','FARO descriptive only; RGB saturation .95 declared','incomplete cohort NOT_EVALUABLE; no gate relaxation','one eval open; no post-open model/cutpoint changes'],new_candidate_reads=0,new_eval_reference_reads=0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def synthetic():
    from sync_fusion_v1_train_cal import enumerate_ties, select
    from sync_fusion_features_v1 import features_for_depth
    rng=np.random.default_rng(73411)
    cases=0
    for n in (0,1,17,100):
        for _ in range(15):
            scores=rng.choice(np.array([-np.inf,-1.,0.,.1,.5,1.]),n)
            y=rng.choice(np.array([-1,0,1]),n)
            got=enumerate_ties(scores,y)
            expected=[]
            for threshold in [np.inf]+sorted(set(scores[np.isfinite(scores)]),reverse=True):
                p=scores>=threshold
                expected.append((threshold,int(sum(p&(y==1))),int(sum(p&(y==0))),int(sum(p&(y==-1)))))
            assert len(got)==len(expected)
            for row,(threshold,w,f,u) in zip(got,expected):
                actual=np.inf if row['threshold_kind']=='positive_infinity' else row['threshold']
                assert actual==threshold and (row['W'],row['F'],row['U'])==(w,f,u)
            for target in (.02,.05,.10):
                selected=select(got,y,target)
                if not sum(y==0):
                    assert selected['status']=='NOT_CALIBRATABLE'
                    assert selected['threshold_kind']=='positive_infinity'
                elif not sum(y==1):
                    assert selected['status']=='NO_CAL_POS'
                    assert selected['threshold_kind']=='positive_infinity'
                else:
                    legal=[r for r in expected if r[2]<=target*sum(y==0)+1e-12]
                    want=max(legal,key=lambda r:(r[1],-r[2],r[0]))
                    actual=np.inf if selected['threshold_kind']=='positive_infinity' else selected['threshold']
                    assert (actual,selected['W'],selected['F'],selected['U'])==want
            cases+=1
    z=np.full((4,4),np.inf);entry=np.ones_like(z);exit=entry+1;domain=np.ones_like(z,dtype=bool)
    score,x=features_for_depth(z,entry,exit,domain)
    assert np.isneginf(score) and x[0]==-10 and x[1]==1 and x[3]==1
    z[:]=1.5
    score,x=features_for_depth(z,entry,exit,domain)
    assert score==.5 and x[1]==0 and x[4]==16
    z.flat[0]=np.inf
    score,x=features_for_depth(z,entry,exit,domain)
    assert np.isneginf(score) and x[1]==1 and x[4]==15
    return dict(status='PASS',tie_cases=cases,targets_per_case=3,
                missing_tests=3,eval_reference_reads=0,eval_model_score_reads=0)


def v2_data_integrity(root):
    """Public source metadata/admission receipts only, including eval metadata."""
    import csv
    manifest=load(root/'dataset_manifest.json');records=load(root/'candidate_gate_records.json')
    with (root/'official/metadata.csv').open(newline='') as stream:metadata=list(csv.DictReader(stream))
    entries=manifest['entries'];inventory=set(map(str,load(root/'consumed_inventory.json')['visit_ids']))
    assert len(entries)<=18 and len({e['visit_id'] for e in entries})==len(entries)
    assert not set(str(e['visit_id']) for e in entries)&inventory
    recs={r['capture']:r for r in records};positions=[recs[e['capture']]['metadata_index'] for e in entries]
    assert positions==sorted(positions)
    rgb_dt=[];fov=[];nframes=0
    for ordinal,e in enumerate(entries):
        expected_role='cal' if ordinal<6 else 'eval';rec=recs[e['capture']]
        official=metadata[rec['metadata_index']]
        assert official['video_id']==e['capture'] and official['visit_id']==e['visit_id']
        assert official['is_in_upsampling']=='True' and official['visit_id']!='NA'
        assert e['role']==rec['role']==expected_role and e['accepted_ordinal']==ordinal
        assert rec['status']=='ACCEPTED' and rec['near_POS']>=16 and rec['midfar_strict_FREE']>=32
        selection=root/'source'/e['capture']/'selection.json';s=load(selection)
        for earlier in records:
            if earlier['visit_id']!=e['visit_id'] or earlier['metadata_index']>=rec['metadata_index']:continue
            earlier_selection=root/'source'/earlier['capture']/'selection.json'
            assert not earlier_selection.exists() or len(load(earlier_selection)['rows'])<32, 'Admitted a later capture after time-complete capture consumed this visit'
        assert sha(selection)==rec['selection_sha256'] and s['selected_before_labels']
        assert s['plan_sha256']==sha(root/'PLAN.json') and s['FARO_not_selection_input'] is True
        times=np.array(s['native_common_timestamps_s']);grid=times[0]+np.arange(int((times[-1]-times[0])/.2)+1)*.2
        nearest=np.searchsorted(times,grid).clip(0,len(times)-1);left=np.maximum(nearest-1,0)
        nearest=np.where(abs(times[left]-grid)<=abs(times[nearest]-grid),left,nearest)
        good=abs(times[nearest]-grid)<=.10000001;expected=[];start=0;used=set()
        while start+16<=len(grid) and len(expected)<2:
            ix=nearest[start:start+16]
            if good[start:start+16].all() and len(set(ix))==16 and not set(ix)&used:
                expected.append([(float(g),float(times[j])) for g,j in zip(grid[start:start+16],ix)]);used.update(ix);start+=16
            else:start+=1
        assert len(expected)==2 and len(e['frames'])==len(s['rows'])==32
        for frame,selected in zip(e['frames'],s['rows']):
            w,j=frame['window_id'],frame['window_frame'];gt,rt=expected[w][j]
            assert frame['role']==expected_role and frame['grid_timestamp_s']==gt and frame['rgb_timestamp_s']==rt
            assert (selected['grid_timestamp_s'],selected['rgb_timestamp_s'])==(gt,rt)
            rgb_dt.append(rt-gt);K=np.asarray(frame['depth_K']);h,width=frame['depth_shape']
            fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
            sides=np.degrees(np.arctan([cx/fx,(width-1-cx)/fx,cy/fy,(h-1-cy)/fy]))
            fov.append(sides);nframes+=1
    pose_dt=[];pose_span=[];pose_missing=0
    pose_frames=[]
    if (root/'synthesis_manifest.json').exists():pose_frames=load(root/'synthesis_manifest.json')['frames']
    else:
        for e in entries:
            times=np.loadtxt(e['trajectory_path'])[:,0]
            for fr in e['frames']:
                f={k:fr[k] for k in ('grid_timestamp_s','rgb_timestamp_s')}
                for key,stamp in (('grid_pose_bracket',f['grid_timestamp_s']),('rgb_pose_bracket',f['rgb_timestamp_s'])):
                    right=int(np.searchsorted(times,stamp));left=right-1
                    if right<len(times) and abs(times[right]-stamp)<1e-8:left=right
                    f[key]=dict(unavailable=True) if left<0 or right>=len(times) else dict(left_s=float(times[left]),right_s=float(times[right]))
                pose_frames.append(f)
    for f in pose_frames:
        for key,stamp in (('grid_pose_bracket',f['grid_timestamp_s']),('rgb_pose_bracket',f['rgb_timestamp_s'])):
            b=f[key]
            if b.get('unavailable'):pose_missing+=1;continue
            assert b['left_s']<=stamp<=b['right_s'];pose_dt.append(min(stamp-b['left_s'],b['right_s']-stamp));pose_span.append(b['right_s']-b['left_s'])
    def distribution(values):
        a=np.asarray(values,float)
        return dict(n=len(a),median=float(np.median(a)) if len(a) else None,p90=float(np.percentile(a,90)) if len(a) else None,max=float(np.max(a)) if len(a) else None)
    fov=np.asarray(fov);return dict(status='PASS',cohort_complete=len(entries)==18,admitted_visits=len(entries),historically_excluded_visits=len(inventory),frames=nframes,window_preselection_recomputed=2*len(entries),split_counts={r:sum(e['role']==r for e in entries) for r in ('cal','eval')},FOV_45_fully_covered_frames=int((fov>=22.5).all(1).sum()) if len(fov) else 0,FOV_45_incomplete_frames=int((~(fov>=22.5).all(1)).sum()) if len(fov) else 0,minimum_side_half_FOV_deg=float(fov.min()) if len(fov) else None,RGB_grid_abs_residual_s=distribution(abs(np.array(rgb_dt))),trajectory_nearest_residual_s=distribution(pose_dt),trajectory_bracket_span_s=distribution(pose_span),pose_missing_brackets=pose_missing,new_eval_reference_reads=0,new_eval_model_score_reads=0)


def v2_resource_amendment(root):
    import subprocess
    receipt=load(root/'resource_amendment_receipt.json');repo=Path(__file__).resolve().parents[4]
    path=repo/receipt['path'];committed=subprocess.check_output(['git','show',receipt['commit']+':'+receipt['path']],cwd=repo)
    assert sha(path)==receipt['sha256']==hashlib.sha256(committed).hexdigest()
    assert receipt['original_plan_unchanged'] and not receipt['eval_opened']
    assert sha(root/'PLAN.json')=='e9387389ebd8f842975d665db9fd454ee7d43a07bc173c488fe74f274200be55'
    assert not (root/'eval/eval_open.json').exists()
    text=path.read_text(encoding='utf8')
    for term in ('资源','科学协议完全冻结','一次 eval','18 visit','576 帧','半循环','保护480/test'):assert term in text
    return dict(status='PASS',resource_amendment_commit=receipt['commit'],resource_amendment_sha256=receipt['sha256'],resource_receipt_sha256=sha(root/'resource_amendment_receipt.json'),original_protocol_sha256=sha(root/'PLAN.json'),scope='Resource-only; originally selected cohort/windows/features/fit/cal/bootstrap/primary criterion and once-only eval preserved',old_budget_stop_lineage_retained=True,new_eval_open_exists=False)


def v2_source_extension(root):
    from datetime import datetime
    from email.utils import parsedate_to_datetime
    ext=load(root/'faro_source_extension_receipt.json');resource=load(root/'resource_amendment_receipt.json')
    assert ext['resource_amendment_commit']==resource['commit'] and ext['resource_amendment_document_sha256']==resource['sha256']
    assert ext['resource_amendment_receipt_sha256']==sha(root/'resource_amendment_receipt.json')
    assert parsedate_to_datetime(ext['first_extension_HEAD_server_date'])>datetime.fromisoformat(resource['utc'].replace('Z','+00:00'))
    for snapshot in ext['old_snapshots']:assert sha(snapshot['path'])==snapshot['sha256']
    old=load(root/'faro_source_manifest_final_pre_extension.json')['entries'];new=load(root/'faro_source_manifest_final.json')['entries']
    assert len(old)==len(new)==18 and old[:17]==new[:17]
    assert old[17]['capture']==new[17]['capture']=='42446467' and str(new[17]['visit_id'])=='422523'
    assert new[17]['frames'][:len(old[17]['frames'])]==old[17]['frames']
    additional=new[17]['frames'][len(old[17]['frames']):]
    caps={f['source_id'].rsplit('_',1)[0] for f in additional}
    assert caps=={'42446468','42446478'} and ext['new_source_components']==['42446468','42446478']
    allcaps={f['source_id'].rsplit('_',1)[0] for e in new for f in e['frames']};nframes=sum(len(e['frames']) for e in new)
    assert len(allcaps)==ext['capture_count']==35 and nframes==ext['source_frames']==7554
    assert ext['download_bytes']==load(root/'network_progress.json')['received_bytes']==4109932729
    assert ext['scientific_protocol_unchanged'] and not ext['eval_opened'] and not (root/'eval/eval_open.json').exists()
    return dict(status='PASS',unchanged_visit_entries=17,changed_visit_id='422523',added_registered_captures=sorted(caps),source_captures=35,source_frames=7554,download_bytes=ext['download_bytes'],first_extension_HEAD_server_date=ext['first_extension_HEAD_server_date'],additional_source_after_resource_commit=True,start_time_is_estimate=True,extension_receipt_sha256=sha(root/'faro_source_extension_receipt.json'),source_final_sha256=sha(root/'faro_source_manifest_final.json'),old_budget_stop_snapshots_verified=3,new_eval_reference_reads=0,new_eval_method_score_reads=0)


def v2_source_binding(root):
    """Verify completed surface/source membership and input CNH, without truth."""
    import csv
    source=load(root/'faro_source_manifest_final.json');surface=load(root/'surface_manifest.json')
    synth=load(root/'synthesis_manifest.json');binding=load(root/'source_render_binding_receipt.json')
    assert binding['status']=='PASS' and binding['final_source_sha256']==sha(root/'faro_source_manifest_final.json')
    assert binding['surface_manifest_sha256']==sha(root/'surface_manifest.json')
    assert binding['synthesis_manifest_sha256']==sha(root/'synthesis_manifest.json')
    byvisit={str(e['visit_id']):e for e in surface['entries']};checked=0
    for entry in source['entries']:
        s=byvisit[str(entry['visit_id'])]
        assert set(f['source_id'] for f in entry['frames'])==set(f['source_id'] for f in s['frames'])
        assert len(s['frames'])==s['frames_total']==len(entry['frames'])
        assert sum(f['status']=='USED' for f in s['frames'])==s['frames_used']
        assert (s['capture'],s['role'])==(entry['capture'],entry['role']) and sha(s['path'])==s['sha256']
    assert len(synth['frames'])==576 and len(byvisit)==18
    with (root/'input_coverage_gate.csv').open(newline='',encoding='utf8') as f:
        gate={(r['frame_id'],r['arm'],int(r['K'])):int(r['joint_pass_zones']) for r in csv.DictReader(f)}
    passframes=[]
    for f in synth['frames']:
        for arm in f['arms']:
            assert sha(arm['path'])==arm['sha256'];checked+=1
            if arm['arm']!='faro_rho030_ambient1' or arm['repeat']!=0:continue
            with np.load(arm['path'],allow_pickle=False) as d:
                hist=d['hist'];bg=d['background'];cov=d['coverage'];pk=hist.argmax(-1)
                height=np.take_along_axis(hist,pk[...,None],-1)[...,0];back=np.take_along_axis(bg,pk[...,None],-1)[...,0]
                snr=height/np.sqrt(np.maximum(height,0)+2*back+1)
                count=int(((cov>=.75)&(snr>=3)).sum())
                assert count==gate[f['frame_id'],arm['arm'],0]
                if not f['faro_source_available']:assert not np.any(cov>0)
                if count>=52:passframes.append(f)
    assert checked==2304
    return dict(status='PASS',source_visits=18,source_frames=sum(len(e['frames']) for e in source['entries']),surface_hashes_verified=18,CNH_hashes_verified=checked,FARO_K0_gate_recomputed_frames=576,FARO_K0_zone52_passed_frames=len(passframes),FARO_pass_by_role={r:sum(f['role']==r for f in passframes) for r in ('cal','eval')},source_final_sha256=sha(root/'faro_source_manifest_final.json'),surface_manifest_sha256=sha(root/'surface_manifest.json'),synthesis_manifest_sha256=sha(root/'synthesis_manifest.json'),new_eval_reference_reads=0,new_eval_model_score_reads=0)


def v2_faro_prefix(root):
    prefix=load(root/'faro_source_prefix_receipt.json');terminal=load(root/'faro_source_terminal.json')
    source=load(root/'faro_source_manifest_final.json');raw=load(root/'faro_source_manifest.json')
    assert source['raw_execution_manifest_sha256']==sha(root/'faro_source_manifest.json')
    records=terminal['records'];stop=next(i for i,r in enumerate(records) if r['capture']==prefix['first_byte_stop_component'])
    assert all(not r['status'].startswith('COMPLETE') for r in records[stop:])
    assert prefix['successful_geometry_after_first_byte_stop']==0
    assert prefix['all_received_bytes']<=4000000000
    assert load(root/'network_progress.json')['received_bytes']==prefix['all_received_bytes']
    assert len(source['entries'])==len(raw['entries'])
    visitroles={str(e['visit_id']):e['role'] for e in load(root/'dataset_manifest.json')['entries']}
    paths={};nframes=0
    for original,entry in zip(raw['entries'],source['entries']):
        assert (original['capture'],original['visit_id'],original['role'])==(entry['capture'],entry['visit_id'],entry['role'])
        assert entry['role']==visitroles[str(entry['visit_id'])]
        assert len(original['frames'])==len(entry['frames'])
        for a,b in zip(original['frames'],entry['frames']):
            assert all(b[k]==v for k,v in a.items())
            paths[b['trajectory_path']]=b['trajectory_sha256'];nframes+=1
    for p,checksum in paths.items():assert sha(p)==checksum
    return dict(status='PASS',source_components=len(source['entries']),source_frames=nframes,trajectory_hashes_verified=len(paths),received_bytes=prefix['all_received_bytes'],first_byte_stop_component=prefix['first_byte_stop_component'],geometry_after_stop=0,deviation='256-byte sibling transform plus body-free HEAD after internal byte-stop; geometry prefix unchanged; raw lineage retained',prefix_receipt_sha256=sha(root/'faro_source_prefix_receipt.json'),source_manifest_final_sha256=sha(root/'faro_source_manifest_final.json'),new_eval_reference_reads=0,new_eval_method_score_reads=0)


def v2_coverage(root):
    """Recompute input-only zone coverage/SNR gate; never a method metric."""
    synth=load(root/'synthesis_manifest.json')['frames'];unions=grid_missing=rgb_missing=both_missing=0
    zones=[];source_valid=[];checked=0
    for f in synth:
        gm=bool(f['grid_pose_bracket'].get('unavailable'));rm=bool(f['rgb_pose_bracket'].get('unavailable'))
        unions+=gm or rm;grid_missing+=gm;rgb_missing+=rm;both_missing+=gm and rm
        arm=next(a for a in f['arms'] if a['arm']=='native_perturbed' and a['repeat']==0)
        assert sha(arm['path'])==arm['sha256']
        with np.load(arm['path'],allow_pickle=False) as d:
            hist=d['hist'];bg=d['background'];cov=d['coverage'];pk=hist.argmax(-1)
            height=np.take_along_axis(hist,pk[...,None],-1)[...,0];back=np.take_along_axis(bg,pk[...,None],-1)[...,0]
            snr=height/np.sqrt(np.maximum(height,0)+2*back+1);zones.append(int(((cov>=.75)&(snr>=3)).sum()))
            source_valid.append(bool(np.any(cov>0)))
            if gm or rm:assert not np.any(cov>0)
        checked+=1
    return dict(status='PASS',grid_frames=len(synth),frame_union_grid_or_RGB_pose_missing=int(unions),grid_pose_missing_frames=int(grid_missing),RGB_pose_missing_frames=int(rgb_missing),both_pose_missing_frames=int(both_missing),native_K0_gate_recomputed_frames=checked,native_K0_zone52_passed_frames=int(sum(n>=52 for n in zones)),native_K0_some_coverage_frames=int(sum(source_valid)),native_K0_joint_zone_counts=dict(median=float(np.median(zones)),p10=float(np.percentile(zones,10)),p90=float(np.percentile(zones,90)),minimum=int(min(zones)),maximum=int(max(zones))),scope='Input coverage only; native frames retained regardless of gate',new_eval_reference_reads=0,new_eval_model_score_reads=0)


def v2_trainreview(root):
    import joblib
    plan=load(root/'PLAN.json');seal=load(root/'train/train_seal.json')
    assert seal['plan_sha256']==sha(root/'PLAN.json') and seal['seeds']==[955,956,957]
    refs=load(seal['reference_path'])['rows'];assert sha(seal['reference_path'])==seal['reference_sha256']
    yi={};visits=set();oldcal=set()
    for r in refs:
        if r['role']=='cal':oldcal.add(str(r['visit_id']));continue
        assert r['role'] in ('train','eval');visits.add(str(r['visit_id']))
        for q in r['queries']:yi[r['source_id'],q['name']]={'POSITIVE':1,'FREE_ON_SAMPLED_RAYS':0,'UNKNOWN':-1}[q['state']]
    original=load(root.parent/'sync-rgb-tof-dataset-v1-dev-20261011/public_roster.json')['rows']
    original_training={str(r['visit_id']) for r in original if r['role'] in ('train','eval')}
    oldcal={str(r['visit_id']) for r in original if r['role']=='cal'}
    assert visits==original_training and len(oldcal)==2
    assert len(visits)==10 and set(seal['visit_ids'])==visits and not visits&oldcal
    nmodels=0;counts=[]
    for source in ('native_perturbed','faro_rho030_ambient1'):
        entries=[e for e in seal['feature_entries'] if e['source']==source]
        assert [e['old_role'] for e in entries]==['train','eval'];parts=[]
        for e in entries:
            assert sha(e['path'])==e['sha256']
            with np.load(e['path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
            assert d['feature_names'].tolist()==plan['features']['names']
            assert set(d['visit_id'])<=visits;parts.append(d)
        X=np.concatenate([p['X'] for p in parts]);ids=[(s,q) for p in parts for s,q in zip(p['frame_id'],p['query_id'])]
        assert len(ids)==len(set(ids));y=np.array([yi[key] for key in ids]);known=y>=0
        counts.append(dict(source=source,rows=len(y),POS=int(sum(y==1)),FREE=int(sum(y==0)),UNKNOWN=int(sum(y==-1))))
        for method in ('hgb','logit'):
            models=[e for e in seal['model_entries'] if e['source']==source and e['method']==method]
            assert sorted(e['seed'] for e in models)==[955,956,957]
            for e in models:
                assert sha(e['path'])==e['sha256'];m=joblib.load(e['path']);nmodels+=1
                assert e['train_known_rows']==sum(known)
                if method=='hgb':
                    for k,v in plan['models']['HGB'].items():
                        if k!='class_weight':assert m.get_params()[k]==v
                else:
                    scaler,logit=list(m.named_steps.values())
                    assert np.allclose(scaler.mean_,X[known].mean(0),rtol=1e-12,atol=1e-12)
                    assert scaler.n_samples_seen_==sum(known)
                    assert logit.C==1 and logit.penalty=='l2' and logit.solver=='lbfgs' and logit.max_iter==1000 and logit.class_weight is None
                assert m.classes_.tolist()==[0,1]
    assert counts==seal['counts']
    recovery=load(root/'train/mechanical_recovery_receipt.json')
    for e in recovery['native_models_reused_without_fit']:assert sha(e['path'])==e['sha256']
    return dict(status='PASS',models_verified=nmodels,train_visit_ids=sorted(visits),oldcal_visit_ids_excluded=sorted(oldcal),source_counts=counts,train_seal_sha256=sha(root/'train/train_seal.json'),mechanical_recovery_sha256=sha(root/'train/mechanical_recovery_receipt.json'),new_eval_reference_reads=0,new_eval_model_score_reads=0,training_model_refits=0)


def v2_posteval(root):
    """Independent counts/paired intervals from the immutable cache only."""
    opened=load(root/'eval/eval_open.json');terminal=load(root/'eval/eval_terminal.json')
    assert terminal['status']=='COMPLETE'
    assert opened['source_sha256']==sha(Path(__file__).with_name('sync_fusion_confirm_v2_models.py'))
    path=root/'eval/per_query.jsonl';assert sha(path)==terminal['cache_sha256']
    rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    seal=load(root/'cal/cal_seal.json');summary=load(root/'eval/summary.json')
    assert sha(root/'cal/cal_seal.json')==opened['cal_seal_sha256']
    assert opened['resource_amendment_receipt_sha256']==seal['resource_amendment_receipt_sha256']==sha(root/'resource_amendment_receipt.json')
    assert sha(root/'feature_manifest.json')==opened['feature_manifest_sha256']
    for model in seal['model_entries']:assert sha(model['path'])==model['sha256']
    assert len({(r['source'],r['frame_id'],r['query_id']) for r in rows})==len(rows)
    expected_ids=set()
    for entry in load(root/'feature_manifest.json')['outputs']:
        if entry['role']!='eval':continue
        assert sha(entry['path'])==entry['sha256']
        with np.load(entry['path'],allow_pickle=False) as data:
            expected_ids.update((entry['source'],s,q) for s,q in zip(data['frame_id'],data['query_id']))
    assert {(r['source'],r['frame_id'],r['query_id']) for r in rows}==expected_ids
    common_labels={}
    for r in rows:
        key=(r['frame_id'],r['query_id'])
        if key in common_labels:assert common_labels[key]==(r['band'],r['visit_id'],r['label'])
        common_labels[key]=(r['band'],r['visit_id'],r['label'])
    visits=opened['visit_ids'];assert len(visits)==len(set(visits))==12
    idx={v:i for i,v in enumerate(visits)};draw=np.random.default_rng(20261011).integers(0,12,(2000,12))
    weights=np.stack([(draw==j).sum(1) for j in range(12)],axis=1)
    cells={c['cell_id']:c for c in seal['cells']};checked=0
    for r in rows:
        assert r['label'] in (1,0,-1) and r['visit_id'] in visits
        assert set(r['predictions'])=={cid for cid,c in cells.items() if c['source']==r['source'] and c['band']==r['band']}
        for cid,pred in r['predictions'].items():
            c=cells[cid];assert c['source']==r['source'] and c['band']==r['band']
            s=r['scores'][c['method']];s=-np.inf if s is None else s
            t=np.inf if c['threshold_kind']=='positive_infinity' else c['threshold']
            assert pred==bool(c['status']=='ADOPTED' and s>=t);checked+=1
    def ci(values):
        valid=np.isfinite(values);x=values[valid]
        return dict(lower=float(np.percentile(x,2.5)) if len(x) else None,upper=float(np.percentile(x,97.5)) if len(x) else None,valid_replicates=int(sum(valid)),total_replicates=2000)
    def intervals(numer,denom):
        n=weights@numer;d=weights@denom
        return ci(np.divide(n,d,out=np.full(n.shape,np.nan,float),where=d>0))
    def clusters(rr,values):
        result=np.zeros(12,dtype=int)
        for row,value in zip(rr,values):result[idx[row['visit_id']]]+=int(value)
        return result
    def same(a,b):
        assert set(a)==set(b)
        for k in a:
            assert a[k] is None and b[k] is None or a[k] is not None and b[k] is not None and np.isclose(a[k],b[k],rtol=1e-12,atol=1e-12),(k,a[k],b[k])
    metrics={};predictions={};groups={}
    for m in summary['metrics']:
        key=(m['source'],m['band'],m['method']);rr=[r for r in rows if (r['source'],r['band'])==key[:2]];y=np.array([r['label'] for r in rr]);den=[int(sum(y==s)) for s in (1,0,-1)]
        c=next(c for c in seal['cells'] if (c['source'],c['band'],c['method'])==key)
        p=np.array([r['predictions'][c['cell_id']] for r in rr],bool);num=[int(sum(p&(y==s))) for s in (1,0,-1)]
        assert den==[m['POS'],m['FREE'],m['UNKNOWN']] and num==[m['W'],m['F'],m['U']]
        assert m['cal_status']==c['status']
        for s,k in ((1,'W'),(0,'F')):
            expected=num[0 if s==1 else 1]/den[0 if s==1 else 1] if den[0 if s==1 else 1] else None
            assert expected==m[k+'_rate']
            same(intervals(clusters(rr,p&(y==s)),clusters(rr,y==s)),m[k+'_rate_CI'])
        metrics[key]=m;predictions[key]=p;groups[key[:2]]=rr
    best={}
    for b in summary['best_single']:
        source,band=b['source'],b['band'];chosen=max(('tof','rgb'),key=lambda method:(metrics[source,band,method]['W'],-metrics[source,band,method]['F'],method=='rgb'))
        assert b['method']==chosen;best[source,band]=chosen
    for pair in summary['paired']:
        source,band,method,comp=pair['source'],pair['band'],pair['method'],pair['comparator']
        assert comp==(best[source,band] if pair['comparator_role']=='best_single' else 'or')
        rr=groups[source,band];y=np.array([r['label'] for r in rr]);p=predictions[source,band,method];q=predictions[source,band,comp];added=p&~q;removed=~p&q
        rescue=int(sum(added&(y==1)));loss=int(sum(removed&(y==1)))
        assert (pair['rescue'],pair['loss'],pair['delta_W'])==(rescue,loss,rescue-loss)
        for label,prefix in ((0,'FREE'),(-1,'UNKNOWN')):
            assert pair[prefix+'_added']==sum(added&(y==label)) and pair[prefix+'_removed']==sum(removed&(y==label))
        delta=clusters(rr,added&(y==1))-clusters(rr,removed&(y==1));same(ci(weights@delta),pair['delta_W_CI'])
        same(intervals(delta,clusters(rr,y==1)),pair['delta_W_rate_CI'])
    for check in summary['band_criteria']:
        band,method=check['band'],check['method'];m=metrics['native_perturbed',band,method]
        pair=next(p for p in summary['paired'] if p['source']=='native_perturbed' and p['band']==band and p['method']==method and p['comparator_role']=='best_single')
        c=next(c for c in seal['cells'] if c['source']=='native_perturbed' and c['band']==band and c['method']==method)
        lim=c['target']*1.5;lo=pair['delta_W_rate_CI']['lower'];f=m['F_rate']
        expected=dict(calibrated=c['status']=='ADOPTED',positive_difference=pair['delta_W']>0,interval_lower_positive=lo is not None and lo>0,FREE_tolerance_pass=f is not None and f<=lim+1e-12)
        assert check['criteria']==expected and check['pass_all']==all(expected.values())
        assert check['confirmatory_primary']==(method=='logit' and band=='0.8-1.5m')
        assert check['delta_W']==pair['delta_W'] and check['FREE_rate']==f and check['FREE_tolerance']==lim
        same(check['delta_W_CI'],pair['delta_W_CI']);same(check['delta_W_rate_CI'],pair['delta_W_rate_CI'])
    primary=next(c for c in summary['band_criteria'] if c['confirmatory_primary']);assert summary['primary']==primary
    assert summary['result']==('query级融合在中带得到确认（真实RGB+半合成ToF）' if primary['pass_all'] else '主中带确认未通过')
    for saturation in summary['faro_RGB_saturation']:
        m=metrics['faro_rho030_ambient1',saturation['band'],'rgb']
        assert (saturation['POS'],saturation['RGB_W'],saturation['RGB_W_rate'],saturation['RGB_unwitnessed_POS'])==(m['POS'],m['W'],m['W_rate'],m['POS']-m['W'])
        expected=None if not m['POS'] or m['cal_status']!='ADOPTED' else m['W']/m['POS']>=.95
        assert saturation['saturated_descriptor']==expected
    return dict(status='PASS',cached_rows=len(rows),query_decisions_checked=checked,metrics_checked=len(summary['metrics']),paired_rows_checked=len(summary['paired']),bootstrap_replicates=2000,visit_clusters=12,cache_sha256=sha(path),summary_sha256=sha(root/'eval/summary.json'),new_eval_reference_reopens=0,model_reruns=0,primary=primary,result=summary['result'])


def v2_preeval(root):
    import csv, joblib
    from threadpoolctl import threadpool_limits
    plan=load(root/'PLAN.json');fm=load(root/'feature_manifest.json')
    train=load(root/'train/train_seal.json');seal=load(root/'cal/cal_seal.json')
    assert not (root/'eval/eval_open.json').exists()
    assert train['plan_sha256']==seal['plan_sha256']==sha(root/'PLAN.json')
    assert seal['train_seal_sha256']==sha(root/'train/train_seal.json')
    assert seal['feature_manifest_sha256']==sha(root/'feature_manifest.json')
    assert seal['resource_amendment_receipt_sha256']==sha(root/'resource_amendment_receipt.json')
    resource=load(root/'resource_amendment_receipt.json')
    assert seal['resource_amendment_commit']==resource['commit'] and seal['resource_amendment_sha256']==resource['sha256']
    assert seal['source_sha256']==sha(Path(__file__).with_name('sync_fusion_confirm_v2_models.py'))
    assert seal['source_sha256']==sha(root/'cal/executed_cal.py')
    assert seal['frozen_train_recipe_sha256']==sha(Path(__file__).with_name('sync_fusion_v1_train_cal.py'))
    assert sha(train['reference_path'])==train['reference_sha256']
    assert sha(seal['cal_reference_path'])==seal['cal_reference_sha256']
    feature_inputs=load(root/'feature_inputs.json')
    for entry in feature_inputs:assert sha(entry['path'])==entry['sha256']
    source_binding=load(root/'independent_source_binding_audit.json')
    for filename,key in [('faro_source_manifest_final.json','source_final_sha256'),('surface_manifest.json','surface_manifest_sha256'),('synthesis_manifest.json','synthesis_manifest_sha256')]:
        assert sha(root/filename)==source_binding[key]
    observer=load(root/'observation_code_seal.json')
    assert observer['protocol_sha256']==sha(root/'PLAN.json')
    for entry in observer['files']:assert sha(entry['path'])==entry['sha256']
    original=load(root/'public_roster.json');projected=load(root/'rgb_public_roster.json')
    assert len(original['rows'])==len(projected['rows'])==576
    for a,b in zip(original['rows'],projected['rows']):assert all(a[k]==v for k,v in b.items())
    for model in ('dav2','unidepth'):
        current=load(root/'rgb_inference'/model/'model_identity.json')
        inherited=load(root.parent/'sync-rgb-tof-dataset-v1-dev-20261011/rgb_inference'/model/'model_identity.json')
        assert current['meta']['weight_sha256']==inherited['meta']['weight_sha256']
        assert current['roster_sha256']==sha(root/'rgb_public_roster.json')
    assert fm['observation_only'] and not fm['labels_read'] and not fm['reference_paths_read']
    trainrefs=load(train['reference_path'])['rows'];calrefs=load(seal['cal_reference_path'])['rows']
    assert all(r['role']=='cal' for r in calrefs)
    yi={};rolevis={'train':set(), 'cal':set(), 'eval':set()}
    state={'POSITIVE':1,'FREE_ON_SAMPLED_RAYS':0,'UNKNOWN':-1}
    for refs,role in ((trainrefs,'train'),(calrefs,'cal')):
        for r in refs:
            if role=='train' and r['role'] not in ('train','eval'):continue
            rolevis[role].add(str(r['visit_id']))
            for q in r['queries']:
                if q['state']=='FREE_ON_SAMPLED_RAYS':
                    assert q['unknown_pixels']==q['positive_pixels']==0 and q['free_ray_pixels']==q['domain_pixels']
                yi[r['source_id'],q['name']]=state[q['state']]
    public=load(root/'public_roster.json')['rows'];pub={r['source_id']:r for r in public}
    assert all(r['role'] in ('cal','eval') for r in public)
    sensor={r['source_id']:r['frame_id'] for r in load(root/'synthesis_manifest.json')['frames']}
    with (root/'input_coverage_gate.csv').open(newline='',encoding='utf8') as f:
        allowed={r['frame_id'] for r in csv.DictReader(f) if r['arm']=='faro_rho030_ambient1' and r['K']=='0' and int(r['joint_pass_zones'])>=52}
    data={};counts=[]
    for e in fm['outputs']:
        assert sha(e['path'])==e['sha256']
        with np.load(e['path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
        assert not set(d)&{'y','state','label','labels','reference'}
        assert d['feature_names'].tolist()==plan['features']['names'] and np.isfinite(d['X']).all()
        assert d['X'].shape==(len(d['frame_id']),20)
        ids=list(zip(d['frame_id'].tolist(),d['query_id'].tolist()));assert len(ids)==len(set(ids))
        assert np.array_equal(d['or_score'],np.maximum(d['tof_score'],d['rgb_score']))
        assert np.array_equal(d['and_score'],np.minimum(d['tof_score'],d['rgb_score']))
        assert np.array_equal(d['X'][:,0],np.clip(d['tof_score'],-10,10))
        assert np.array_equal(d['X'][:,1],~np.isfinite(d['tof_score']))
        assert np.array_equal(d['X'][:,1]==0,d['X'][:,9]==2)
        if e['role']!='train':
            expect={s for s,r in pub.items() if r['role']==e['role'] and (e['source']=='native_perturbed' or sensor[s] in allowed)}
            assert set(d['frame_id'])==expect and len(ids)==len(expect)*27
            for s in expect:assert sum(d['frame_id']==s)==27
            for i,s in enumerate(d['frame_id']):assert str(pub[s]['visit_id'])==d['visit_id'][i]
        else:
            inputs=[x for x in train['feature_entries'] if x['source']==e['source']]
            parts=[]
            for x in inputs:
                assert sha(x['path'])==x['sha256']
                with np.load(x['path'],allow_pickle=False) as z:parts.append({k:z[k] for k in z.files})
            assert [x['old_role'] for x in inputs]==['train','eval']
            for key in d:
                if key=='feature_names':continue
                assert np.array_equal(d[key],np.concatenate([x[key] for x in parts]))
        if e['source']=='native_perturbed':rolevis[e['role']]=set(d['visit_id'])
        data[e['source'],e['role']]=d;counts.append({k:e[k] for k in ('source','role','rows','frames')})
    assert [len(rolevis[r]) for r in ('train','cal','eval')]==[10,6,12]
    assert not(rolevis['train']&rolevis['cal'] or rolevis['train']&rolevis['eval'] or rolevis['cal']&rolevis['eval'])
    assert [len(set(data['native_perturbed',r]['frame_id'])) for r in ('train','cal','eval')]==[320,192,384]
    inventory=set(map(str,load(root/'consumed_inventory.json')['visit_ids']))
    assert not((rolevis['cal']|rolevis['eval'])&inventory)
    scores={};model_count=0
    with threadpool_limits(limits=1):
        for source in ('native_perturbed','faro_rho030_ambient1'):
            td=data[source,'train'];cd=data[source,'cal'];yt=np.array([yi[s,q] for s,q in zip(td['frame_id'],td['query_id'])]);known=yt>=0
            scores[source]={m:cd[m+'_score'] for m in ('tof','rgb','or','and')}
            for method in ('hgb','logit'):
                entries=[e for e in seal['model_entries'] if e['source']==source and e['method']==method]
                assert sorted(e['seed'] for e in entries)==[955,956,957];pred=[]
                for e in entries:
                    assert sha(e['path'])==e['sha256'];m=joblib.load(e['path']);model_count+=1
                    if method=='hgb':
                        for k,v in plan['models']['HGB'].items():
                            if k!='class_weight':assert m.get_params()[k]==v
                    else:
                        scaler,logit=list(m.named_steps.values())
                        assert np.allclose(scaler.mean_,td['X'][known].mean(0),atol=1e-12)
                        assert scaler.n_samples_seen_==sum(known)
                        assert logit.C==1 and logit.solver=='lbfgs' and logit.max_iter==1000 and logit.class_weight is None
                    pred.append(m.predict_proba(cd['X'])[:,1] if len(cd['X']) else np.empty(0))
                scores[source][method]=np.mean(pred,axis=0)
    assert len(seal['cells'])==36
    checked_ties=0
    for c in seal['cells']:
        d=data[c['source'],'cal'];take=d['band']==c['band'];y=np.array([yi[s,q] for s,q in zip(d['frame_id'],d['query_id'])])[take]
        sc=scores[c['source']][c['method']][take];target=plan['calibration']['band_targets'][c['band']]
        assert c['target']==target
        assert (sum(y==1),sum(y==0),sum(y==-1))==(c['POS_denominator'],c['FREE_denominator'],c['UNKNOWN_denominator'])
        ties=load(c['all_ties_path'])['rows'];assert sha(c['all_ties_path'])==c['all_ties_sha256']
        thresholds=[np.inf]+sorted(set(sc[np.isfinite(sc)]),reverse=True);assert len(ties)==len(thresholds)
        legal=[];positive=y==1;free=y==0;unknown=y==-1;nfree=np.count_nonzero(free)
        for row,t in zip(ties,thresholds):
            pp=sc>=t;ww=int(np.count_nonzero(pp&positive));ff=int(np.count_nonzero(pp&free));uu=int(np.count_nonzero(pp&unknown))
            rt=np.inf if row['threshold_kind']=='positive_infinity' else row['threshold']
            assert (rt,row['W'],row['F'],row['U'])==(t,ww,ff,uu);checked_ties+=1
            if ff<=np.floor(target*nfree):legal.append((ww,-ff,t,uu))
        want=max(legal) if sum(y==0) and sum(y==1) else (0,0,np.inf,0)
        actual=np.inf if c['threshold_kind']=='positive_infinity' else c['threshold']
        assert (c['W'],-c['F'],actual,c['U'])==want
        assert c['status']==('NOT_CALIBRATABLE' if not sum(y==0) else 'NO_CAL_POS' if not sum(y==1) else 'ADOPTED')
    return dict(status='PASS',feature_tables=counts,feature_input_hashes_verified=len(feature_inputs),models_verified=model_count,cal_cells_recomputed=36,cal_ties_recomputed=checked_ties,visits={r:sorted(v) for r,v in rolevis.items()},cal_seal_sha256=sha(root/'cal/cal_seal.json'),feature_manifest_sha256=sha(root/'feature_manifest.json'),source_binding_audit_sha256=sha(root/'independent_source_binding_audit.json'),resource_amendment_receipt_sha256=sha(root/'resource_amendment_receipt.json'),plan_sha256=sha(root/'PLAN.json'),eval_observation_features_read=True,new_eval_reference_reads=0,new_eval_model_score_reads=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--receipt',default='independent_contract_audit.json');p.add_argument('--cal-seal',type=Path)
    p.add_argument('--preeval',action='store_true');p.add_argument('--posteval',action='store_true');p.add_argument('--train-only',action='store_true');p.add_argument('--data-integrity',action='store_true');p.add_argument('--resource-amendment',action='store_true');p.add_argument('--source-extension',action='store_true');p.add_argument('--source-binding',action='store_true');p.add_argument('--faro-prefix',action='store_true');p.add_argument('--coverage',action='store_true');p.add_argument('--protocol',type=Path);a=p.parse_args()
    start=time.perf_counter()
    result=protocol_review(a.root,a.protocol) if a.protocol else v2_resource_amendment(a.root) if a.resource_amendment else v2_source_extension(a.root) if a.source_extension else v2_source_binding(a.root) if a.source_binding else v2_data_integrity(a.root) if a.data_integrity else v2_faro_prefix(a.root) if a.faro_prefix else v2_coverage(a.root) if a.coverage else v2_trainreview(a.root) if a.train_only else v2_posteval(a.root) if a.posteval else v2_preeval(a.root) if a.preeval or a.cal_seal else synthetic()
    result['seconds']=time.perf_counter()-start
    a.root.mkdir(parents=True,exist_ok=True)
    result['audit_source_sha256']=sha(__file__)
    with (a.root/a.receipt).open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()

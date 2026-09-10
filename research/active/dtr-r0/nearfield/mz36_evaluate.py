"""MZ36 evaluator-only native/group admission and frozen-output scoring.

admit --root ROOT --captures region-map.json --review empty-review.json --output NEW
      [--specs full-specs-manifest.json]
score --root ROOT --admission ADMIT_DIR --predictions INFERENCE_DIR --output NEW
      [--frozen frozen-models.json]

Only complete source/floor/intent/observation-valid groups are scoreable. No
missing event, unavailable native value, or failed group becomes safety CLEAR.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from body_query_collection_labels import (BASIC, EXTRA, confined, expected_near,
    floor_acceptance, read, require, sha, verify_capture, write)
from body_query_labels import labels
from multizone64_observation import native_events, EVENT_ORDER

REGIONS = ('dense_candidate_05', 'dense_candidate_06')
FAMILIES = ('crossbar', 'cabinet', 'oblique_rod', 'hanging_sign')
MODELS = ('MZ5', 'MZ28', 'MZ30', 'MZ35')
CAMERA = dict(width=640, height=360, hfov_deg=100, eye_height_m=1.7, pitch_deg=0, roll_deg=0)


def fresh(root, output):
    artifact = (root/'artifacts.local').resolve(strict=True)
    require(not output.exists() and output.resolve().is_relative_to(artifact) and output.resolve()!=artifact,
            'Fresh canonical artifacts.local output required')
    output.mkdir(parents=True)


def bind(inputs, path, digest=None):
    path = Path(path).resolve(strict=True); actual = sha(path)
    require(digest is None or actual==digest, 'Hash mismatch: '+str(path))
    inputs[str(path)] = actual
    return path


def code_hashes():
    return {n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,
        'body_query_collection_labels.py', 'body_query_labels.py', 'body_query_model.py',
        'multizone64_observation.py', 'worlds_verify.py', 'contact_retina_spec.py']}


def native_frame(native, case, world_row, saved_support):
    """No model call. Revalidate inherited source labels and compute MZ events."""
    require(native.shape==(360,640) and native.dtype==np.float32, 'Native float32 axial640x360 required')
    tensor = torch.from_numpy(native).cuda()
    with torch.inference_mode():
        original = labels(tensor,case['camera'],case['floor_z_m'])
        np.testing.assert_array_equal(original['support'].cpu().numpy(),saved_support)
        near = original['near'].cpu().tolist()
        require(near==world_row['body_head_visible_targets'], 'World BODY/HEAD target mismatch')
        raw = original['raw_counts'].cpu().numpy()
        require(raw.reshape(2,6).sum(1).tolist()==world_row['visible_pixels_per_height'], 'World native count mismatch')
        event = native_events(tensor[None],crop=False)
        truth = event['events'][0].cpu().numpy()
        # The four MZ labels are observed native range bins, never declared_range.
        np.testing.assert_array_equal(truth,(raw.reshape(2,2,3).sum(2)>=3).reshape(4))
        valid = torch.isfinite(tensor)&(tensor>0)&(tensor<100)
        return dict(raw_truth=truth.tolist(),event_counts=event['counts'][0].cpu().tolist(),
            observation_valid=bool(event['observation_valid'][0]),native_valid_pixels=int(valid.sum()),
            unknown_native_pixels=int((~valid).sum()),mz_domain_valid_pixels=int(event['valid_pixel_count'][0]),
            body_head=near,raw_query_counts=raw.tolist(),expected_body_head=expected_near(case['variant_id']),
            intent_matches=near==expected_near(case['variant_id']))


def group_admission(records):
    grouped = defaultdict(list)
    for i,r in enumerate(records): grouped[(r['region_id'],r['group_id'])].append(i)
    groups=[]
    for (region,gid),indices in grouped.items():
        rows=[records[i] for i in indices];family=rows[0]['family']
        required=set(BASIC)|(set(EXTRA) if family=='crossbar' else set())
        conditions=Counter(r['condition'] for r in rows)
        complete=set(conditions)==required and all(v==1 for v in conditions.values())
        consistent=all(len({r[k] for r in rows})==1 for k in ['family','site_id','source_role','declared_range'])
        reasons=[]
        if not complete:reasons.append('INCOMPLETE_OR_DUPLICATE_CONDITIONS')
        if not consistent:reasons.append('INCONSISTENT_GROUP_METADATA')
        if any(r['native_status']!='PASS' for r in rows):reasons.append('NATIVE_OR_CAPTURE_FAILURE')
        if any(not r['source_accepted'] for r in rows):reasons.append('SOURCE_NOT_ADMITTED')
        if any(not r.get('floor_probe',{}).get('accepted',False) for r in rows):reasons.append('FLOOR_REJECTED')
        if any(not r.get('intent_matches',False) for r in rows):reasons.append('NATIVE_FIXTURE_INTENT_MISMATCH')
        if any(not r.get('observation_valid',False) for r in rows):reasons.append('NO_VALID_MZ_NATIVE_OBSERVATION')
        accepted=not reasons
        for r in rows:r['group_accepted']=accepted;r['known']=[accepted]*4;r['exclusion_reasons']=list(reasons)
        groups.append(dict(region_id=region,group_id=gid,site_id=rows[0]['site_id'],family=family,
            frame_indices=indices,frame_ids=[r['frame_id'] for r in rows],attempted_frames=len(rows),
            expected_conditions=sorted(required),actual_conditions=dict(conditions),complete=complete,
            metadata_consistent=consistent,accepted=accepted,reasons=reasons))
    return groups


def admit(root, captures_path, review_path, output, specs_path=None):
    fresh(root,output);started=time.perf_counter();inputs={};records=[];captures=[]
    try:
        protocol=Path(__file__).with_name('MZ36_NEW_SOURCE_PROTOCOL_20260910.md');bind(inputs,protocol)
        maps=read(bind(inputs,captures_path));review=read(bind(inputs,review_path))
        require(set(maps)==set(REGIONS),'Exactly two protocol regions required')
        require(review['status'] in ['PASS','NOT_EVALUABLE'],'Unexpected source review status')
        reviewed={r['region_id']:r for r in review['regions']}
        require(set(reviewed)==set(REGIONS),'Source review region mismatch')
        for path,digest in review['inputs'].items():bind(inputs,path,digest)
        planned={}
        if specs_path:
            document=read(bind(inputs,specs_path))
            for entry in document.get('regions',document.get('outputs',[])):
                path=Path(entry.get('spec',entry.get('spec_file','')))
                if not path.is_absolute():path=specs_path.parent/path
                spec=read(bind(inputs,path,entry.get('sha256',entry.get('spec_sha256'))))
                planned[entry['region_id']]=spec
            require(set(planned)==set(REGIONS),'Full spec manifest region mismatch')
        require(torch.cuda.is_available(),'CUDA required for native geometry')
        torch.set_num_threads(1)
        write(output/'start.json',dict(status='STARTED',mode='admit',inputs=inputs,code_sha256=code_hashes(),
            attempted_budget=400,training_steps=0,model_inference_frames=0,backend='CUDA native geometry'))
        for region in REGIONS:
            capture=Path(maps[region]).resolve(strict=True)
            require(Path(maps[region]).is_absolute(),'Absolute capture directory required')
            spec=read(bind(inputs,capture/'source/spec.json'))
            require(len(spec['cases'])==200,(region,'Expected200 attempted cases'))
            if region in planned:
                # Worker may rewrite map_file only; all other frozen spec bytes as data must match.
                a={k:v for k,v in spec.items() if k!='map_file'};b={k:v for k,v in planned[region].items() if k!='map_file'}
                require(a==b,'Worker changed frozen spec fields beyond map_file')
            selected=reviewed[region]['selected_site_ids'];source_rows=reviewed[region]['rows']
            require(len(set(selected))==len(selected),'Duplicate selected sites')
            if review['status']=='PASS':require(len(selected)==10,'Expected10 source-admitted sites')
            site_review={r['site_id']:r for r in source_rows}
            require(len(site_review)==len(source_rows),'Duplicate source review rows')
            ranked=[r['site_id'] for r in sorted(source_rows,key=lambda r:r['rank']) if r['accepted'] and r['floor_accepted']][:10]
            if review['status']=='PASS':require(selected==ranked,'Selection is not first10 source-valid ranked sites')
            error=None;world=None;probes=[]
            try:
                spec,world,bound,probes=verify_capture(capture);inputs.update(bound)
            except Exception:
                error=traceback.format_exc()
            captures.append(dict(region_id=region,capture=str(capture),status='PASS' if error is None else 'FAIL',
                attempted_frames=len(spec['cases']),error=error))
            require(len({c['site_id'] for c in spec['cases']})==10,'Capture does not contain10 sites')
            require(all(n==20 for n in Counter(c['site_id'] for c in spec['cases']).values()),'Each site must have20 cases')
            for i,case in enumerate(spec['cases']):
                family=case['condition']['family'];site=case['site_id'];row_review=site_review.get(site,{})
                require(case['source_role']=='DEV_ONLY' and case['region_id']==region,'New frame role/region mismatch')
                require(family in FAMILIES and case['variant_id'] in set(BASIC)|set(EXTRA),'Unsupported fixture')
                require(case['declared_range'] in ['near','far'],'Missing declared fixture range')
                require(bool(case.get('original_fixture_group_id')),'Missing original TRAIN fixture identity')
                record=dict(frame_id=region+'/'+case['name'],region_id=region,site_id=site,group_id=case['group_id'],
                    condition=case['variant_id'],family=family,source_role=case['source_role'],declared_range=case['declared_range'],
                    capture=str(capture),sample_index=i,name=case['name'],camera=case['camera'],floor_z_m=case['floor_z_m'],
                    original_fixture_group_id=case['original_fixture_group_id'],native_status='FAIL',
                    source_accepted=bool(review['status']=='PASS' and site in selected and row_review.get('accepted') and row_review.get('floor_accepted')),
                    source_visual_note=row_review.get('visual_note'),raw_truth=None,event_counts=None,observation_valid=False)
                try:
                    require(error is None,'Full capture verification failed; see capture record')
                    world_row=world['rows'][i]
                    rgb=confined(capture,f'model/sample/{i:04d}.png');native=confined(capture,f'evaluator/native/{i:04d}.npy')
                    support=confined(capture,world_row['mask_path'])
                    for path,key in [(rgb,'rgb_sha256'),(native,'native_sha256'),(support,'mask_sha256')]:bind(inputs,path,world_row[key])
                    record.update(rgb_path=str(rgb),depth_path=str(native),rgb_sha256=world_row['rgb_sha256'],native_sha256=world_row['native_sha256'],
                        floor_probe=floor_acceptance(case,probes))
                    record.update(native_frame(np.load(native,allow_pickle=False),case,world_row,np.load(support,allow_pickle=False)))
                    record['native_status']='PASS'
                except Exception:record['error']=traceback.format_exc()
                records.append(record)
            write(output/'progress.json',dict(processed_frames=len(records),region_id=region))
        require(len(records)==400 and len({r['frame_id'] for r in records})==400,'Attempted frame identity/count mismatch')
        groups=group_admission(records)
        require(len(groups)==80,'Expected80 site/family groups')
        for region in REGIONS:
            for site in {r['site_id'] for r in records if r['region_id']==region}:
                require({r['family'] for r in records if r['region_id']==region and r['site_id']==site}==set(FAMILIES),'Site family coverage mismatch')
        known=np.array([r['known'] for r in records],bool)
        raw_available=np.array([r['raw_truth'] is not None for r in records],bool)
        truth=np.array([r['raw_truth'] if r['raw_truth'] is not None else [False]*4 for r in records],bool)
        counts=np.array([r['event_counts'] if r['event_counts'] is not None else [0]*4 for r in records],np.int64)
        labels_array=np.where(known,truth.astype(np.int8),-1)
        np.savez_compressed(output/'evaluator.npz',frame_ids=np.array([r['frame_id'] for r in records]),truth=truth,
            known=known,labels=labels_array,raw_available=raw_available,event_counts=counts,
            observation_valid=np.array([r['observation_valid'] for r in records],bool))
        manifest=dict(camera=CAMERA,frames=[{k:r[k] for k in ['frame_id','rgb_path','depth_path']} for r in records if r['group_accepted']])
        write(output/'predictor-manifest.json',manifest)
        admitted=int(known.all(1).sum());status='PASS' if admitted>0 and all(c['status']=='PASS' for c in captures) else 'NOT_EVALUABLE'
        result=dict(status=status,attempted_frames=400,native_labeled_frames=int(raw_available.sum()),admitted_frames=admitted,
            excluded_frames=400-admitted,unknown_by_query=(~known).sum(0).tolist(),attempted_groups=len(groups),
            admitted_groups=sum(g['accepted'] for g in groups),records=records,groups=groups,captures=captures,
            training_steps=0,model_inference_frames=0,event_order=EVENT_ORDER,
            label_contract='native_events(crop=False),>=3 real pixels/event,radial<=4m; all four queries known only after complete source/floor/intent/observation-valid group admission; not CLEAR',
            exclusion_groups_by_reason=dict(Counter(reason for g in groups for reason in g['reasons'])))
        write(output/'result.json',result)
        for path,digest in inputs.items():require(sha(path)==digest,'Input changed: '+path)
        torch.cuda.synchronize()
        write(output/'receipt.json',dict(status=status,mode='admit',inputs=inputs,code_sha256=code_hashes(),
            outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},seconds=time.perf_counter()-started,
            backend='CUDA native evaluator; CPU integrity/group admission',training_steps=0,model_inference_frames=0))
        print(status,'admitted',admitted,'/400; groups',result['admitted_groups'],'/80',flush=True)
        return result
    except Exception:
        write(output/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=inputs,records=records,captures=captures,code_sha256=code_hashes()))
        raise
    finally:
        if torch.cuda.is_available():torch.cuda.empty_cache()


def metrics(logits, truth, known):
    pred=logits>=0;correct=pred==truth;complete=known.all(1)
    return dict(attempted_frames=len(truth),known=known.sum(0).tolist(),unknown=(~known).sum(0).tolist(),
        tp=(pred&truth&known).sum(0).tolist(),fp=(pred&~truth&known).sum(0).tolist(),
        fn=(~pred&truth&known).sum(0).tolist(),tn=(~pred&~truth&known).sum(0).tolist(),
        complete_frames=int(complete.sum()),exact_frames=int((correct.all(1)&complete).sum()))


def paired(after, before, truth, known):
    a=after>=0;b=before>=0
    return dict(tp_gained=(a&~b&truth&known).sum(0).tolist(),tp_lost=(~a&b&truth&known).sum(0).tolist(),
        fp_added=(a&~b&~truth&known).sum(0).tolist(),fp_removed=(~a&b&~truth&known).sum(0).tolist())


def score(root, admission, prediction_dir, output, frozen_path):
    fresh(root,output);started=time.perf_counter();inputs={}
    try:
        ar=read(bind(inputs,admission/'receipt.json'));require(ar['status']=='PASS' and ar['mode']=='admit','Admitted native cohort required')
        for name,digest in ar['outputs'].items():bind(inputs,admission/name,digest)
        result_admit=read(admission/'result.json');records=result_admit['records'];groups=result_admit['groups']
        with np.load(admission/'evaluator.npz',allow_pickle=False) as z:
            ids=z['frame_ids'];truth=z['truth'];known=z['known']
            np.testing.assert_array_equal(z['labels'],np.where(known,truth.astype(np.int8),-1))
        np.testing.assert_array_equal(ids,[r['frame_id'] for r in records])
        expected=[r['frame_id'] for r in records if r['group_accepted']]
        require([r['frame_id'] for r in read(admission/'predictor-manifest.json')['frames']]==expected,'Admission manifest order changed')
        pr=read(bind(inputs,prediction_dir/'receipt.json'));require(pr['status']=='PASS' and pr['training_steps']==0,'Completed frozen inference required')
        for name,digest in pr['outputs'].items():bind(inputs,prediction_dir/name,digest)
        bind(inputs,Path(__file__).with_name('MZ36_NEW_SOURCE_PROTOCOL_20260910.md'))
        frozen=read(bind(inputs,frozen_path));require(frozen['status']=='FROZEN_BEFORE_NEW_CAPTURE','Pre-capture model seal required')
        require(frozen['source_protocol_sha256']==sha(Path(__file__).with_name('MZ36_NEW_SOURCE_PROTOCOL_20260910.md')),'Protocol changed after model seal')
        declared={str(Path(k).resolve()):v for k,v in pr['frozen'].items()}
        for path,digest in frozen['frozen'].items():
            require(declared.get(str(Path(path).resolve()))==digest,'Inference does not use sealed input: '+path)
        inference_inputs={str(Path(k).resolve()):v for k,v in pr['inputs'].items()}
        require(inference_inputs.get(str((admission/'predictor-manifest.json').resolve()))==sha(admission/'predictor-manifest.json'),'Predictor used a different manifest')
        for r in records:
            if r['group_accepted']:
                for key,hkey in [('rgb_path','rgb_sha256'),('depth_path','native_sha256')]:
                    require(inference_inputs.get(str(Path(r[key]).resolve()))==r[hkey],'Predictor raw-frame binding mismatch')
        with np.load(prediction_dir/'predictions.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['frame_ids'],expected)
            ix=np.flatnonzero(known.all(1));require(len(ix)==len(expected),'Known/predictor frame count mismatch')
            logits={}
            for name in MODELS:
                require(z[name].shape==(len(ix),4) and np.isfinite(z[name]).all(),'Invalid prediction shape/value')
                logits[name]=np.full((len(ids),4),np.nan);logits[name][ix]=z[name]
        overall={n:metrics(logits[n],truth,known) for n in MODELS};by={}
        for field in ['region_id','family','site_id']:
            by[field]={}
            for value in sorted({r[field] for r in records}):
                take=np.array([i for i,r in enumerate(records) if r[field]==value])
                by[field][value]={n:metrics(logits[n][take],truth[take],known[take]) for n in MODELS}
                by[field][value]['paired']={n:{b:paired(logits[n][take],logits[b][take],truth[take],known[take])
                    for b in ['MZ5','MZ28'] if b!=n} for n in MODELS if n!='MZ5'}
        group_results=[]
        for g in groups:
            take=np.array(g['frame_indices']);complete=bool(g['accepted'] and known[take].all())
            group_results.append(dict(region_id=g['region_id'],group_id=g['group_id'],family=g['family'],site_id=g['site_id'],
                attempted_frames=len(take),admitted=complete,exclusion_reasons=g['reasons'],
                methods={n:dict(metrics=metrics(logits[n][take],truth[take],known[take]),
                    exact_group=bool(complete and ((logits[n][take]>=0)==truth[take]).all())) for n in MODELS}))
        group_counts={n:dict(attempted=len(groups),admitted=sum(g['admitted'] for g in group_results),
            exact=sum(g['methods'][n]['exact_group'] for g in group_results)) for n in MODELS}
        pairs={n:{b:paired(logits[n],logits[b],truth,known) for b in ['MZ5','MZ28'] if b!=n} for n in MODELS if n!='MZ5'}
        additions=(logits['MZ28']>=0)&(logits['MZ5']<0)&known
        gates={}
        for name in ['MZ30','MZ35']:
            p=pairs[name]['MZ28'];removed=sum(p['fp_removed']);lost=sum(p['tp_lost'])
            preserved=bool(np.all((logits[name]>=0)[additions]))
            fp_opportunity=sum(overall['MZ28']['fp'])
            useful=bool(removed>0 and lost==0 and preserved)
            gates[name]=dict(useful_transfer=useful,removed_fp=removed,lost_mz28_tp=lost,additions_preserved=preserved,
                additions_score_values_unchanged=bool(np.array_equal(logits[name][additions],logits['MZ28'][additions])),
                no_new_positive_bits=bool(not ((logits[name]>=0)&(logits['MZ28']<0)&known).any()),
                mz28_fp_opportunity=fp_opportunity,status='PASS' if useful else ('ZERO_FP_OPPORTUNITY' if fp_opportunity==0 else 'FAIL'),
                rule='Removes FP, loses zero admitted MZ28 TP, preserves MZ28 additions')
        p=pairs['MZ28']['MZ5']
        mz28_test=dict(added_tp=sum(p['tp_gained']),added_fp=sum(p['fp_added']),lost_mz5_tp=sum(p['tp_lost']),
            status='ZERO_ADDED_TP' if sum(p['tp_gained'])==0 else 'OBSERVED_ADDITIONS',
            rule='Protocol requests paired addedTP/FP/lostTP reporting; no new composite promotion threshold')
        result=dict(status='PASS',attempted_frames=len(records),admitted_frames=int(known.all(1).sum()),
            excluded_frames=int((~known.all(1)).sum()),unknown_by_query=(~known).sum(0).tolist(),event_order=EVENT_ORDER,
            methods=overall,by=by,groups=group_results,complete_groups=group_counts,paired=pairs,mz28_test=mz28_test,gates=gates,
            training_steps=0,model_inference_frames=0,scope='Frozen outputs on admitted new XY/fixture Development; shared prior regions/assets; no refit, independent-region, natural-source, device or safety claim')
        write(output/'result.json',result)
        np.savez_compressed(output/'scored.npz',frame_ids=ids,truth=truth,known=known,**logits)
        write(output/'receipt.json',dict(status='PASS',mode='score',inputs=inputs,code_sha256=code_hashes(),
            outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},seconds=time.perf_counter()-started,
            backend='CPU saved-scalar scoring',training_steps=0,model_inference_frames=0))
        print('PASS',result['admitted_frames'],'admitted frames',gates,flush=True)
        return result
    except Exception:
        write(output/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=inputs,code_sha256=code_hashes()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['admit','score'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--captures',type=Path);parser.add_argument('--review',type=Path);parser.add_argument('--specs',type=Path)
    parser.add_argument('--admission',type=Path);parser.add_argument('--predictions',type=Path);parser.add_argument('--frozen',type=Path)
    args=parser.parse_args()
    if args.mode=='admit':
        require(args.captures is not None and args.review is not None,'admit requires --captures and --review')
        result=admit(args.root,args.captures,args.review,args.output,args.specs)
        raise SystemExit(0 if result['status']=='PASS' else 2)
    require(args.admission is not None and args.predictions is not None,'score requires --admission and --predictions')
    frozen=args.frozen or args.root/'artifacts.local/work/mz36-new-source-20260910/frozen-models.json'
    score(args.root,args.admission,args.predictions,args.output,frozen)

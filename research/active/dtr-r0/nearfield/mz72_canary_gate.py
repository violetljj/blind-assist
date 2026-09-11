"""MZ72 saved-native source admission; no model prediction or threshold selection."""
import argparse,hashlib,io,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from data_lightweight import CompactSource
from mz72_source_contract import admit_spec,geometry_metadata

ROOT=Path(__file__).resolve().parents[4]
TASK='mz72-physical-source-20260911'
CORE='5fb19e543e1b6c841a2c61c8a4e4ee4c918985a3883d1c061a2ce673423aeaa3'
RUN70='a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sig(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def reference(task):
    out=task/'reference-v1';assert not out.exists();start=time.perf_counter();inputs={}
    def bind(p,h):assert sha(p)==h;inputs[str(p)]=h;return p
    old=task.parent/'mz70-diverse-learning-20260911/run-v1'
    rr=read(bind(old/'receipt.json',RUN70));groups=read(bind(old/'groups.json',rr['outputs']['groups.json']))
    rows=[]
    for cohort in ('mz48','mz55','mz61','mz67'):
        ref=rr['source_info']['label_refs'][cohort];a=load(bind(Path(ref['path']),ref['sha256']))
        records=groups['records' if cohort=='mz48' else cohort+'_records']
        ids=groups['groups']['fit'] if cohort=='mz48' else [i for i,r in enumerate(records) if r['role']=='TRAIN_CANDIDATE']
        for i in ids:
            counts=a['fullframe_event_counts'][i];assert counts.shape==(45,80,4)
            rows.append(dict(source=cohort,frame_id=records[i]['frame_id'],counts=sig(counts),presence=sig(counts>0),positive=bool((counts.sum((0,1))>=3).any())))
    write(out/'reference.json',dict(status='PASS',rows=rows,frames=len(rows),source_frames=dict(Counter(r['source'] for r in rows)),inputs=inputs,role_policy='Original MZ48 fit IDs; literal TRAIN_CANDIDATE in MZ55/61/67. Descriptive tensor novelty only, no model result.'))
    write(out/'receipt.json',dict(status='PASS',inputs=inputs,code_sha256=sha(__file__),outputs={'reference.json':sha(out/'reference.json')},seconds=time.perf_counter()-start,new_inference=0,new_capture=0))
    print('PASS reference',len(rows))

def gate(task,packages,review):
    start=time.perf_counter();manifest=task/'spec-v1/manifest.json';m=read(manifest);mh=sha(manifest);inputs={}
    def bind(p,h=None):
        p=Path(p).resolve(strict=True);v=sha(p);assert h is None or v==h,str(p);inputs[str(p)]=v;return p
    bind(manifest);bind(__file__);code={}
    for n in ('mz72_source_contract.py','mz72_dataset.py','mz72_fullframe.py','data_lightweight.py'):
        p=task/'deployment-v1'/n;code[n]=sha(bind(p))
        assert sha(Path(__file__).with_name(n))==code[n],n
    refs=read(bind(task/'reference-v1/receipt.json'));ref=read(bind(task/'reference-v1/reference.json',refs['outputs']['reference.json']))
    assert refs['status']==ref['status']=='PASS'
    for p,h in ref['inputs'].items():bind(p,h)
    old_counts={r['counts'] for r in ref['rows'] if r['positive']};old_presence={r['presence'] for r in ref['rows'] if r['positive']}
    visual=read(bind(review));assert visual['manifest_sha256']==mh and len(visual['rows'])==64
    requested=read(bind(packages))['shards'];canaries={s['shard_id']:s for s in m['shards'] if s['canary']}
    assert {s['shard_id'] for s in requested}==set(canaries)
    records=[];pairs=[];health=True
    for entry in requested:
        paths={k:bind(v['path'],v['sha256']) for k,v in entry.items() if isinstance(v,dict) and 'path' in v}
        spec=read(paths['capture_spec']);contract=admit_spec(manifest,manifest.parent/'shards',spec)
        assert contract['shard_id']==entry['shard_id']
        ds,fr=read(paths['dataset_receipt']),read(paths['fullframe_receipt'])
        assert ds['status']==fr['status']=='PASS' and ds['code_sha256']==code['mz72_dataset.py'] and fr['code_sha256']==code['mz72_fullframe.py']
        assert fr['frozen_derivation_sha256']==CORE
        assert fr['manifest_sha256']==ds['manifest_sha256']==mh
        assert fr['captured_spec_sha256']==ds['captured_spec_sha256']==sha(paths['capture_spec'])
        assert fr['original_spec_sha256']==ds['original_spec_sha256']==canaries[entry['shard_id']]['sha256']
        assert fr['result']['native_hashes_matched']==len(spec['cases']) and fr['result']['full_event_counts_matched']==len(spec['cases'])*4
        assert sha(paths['fullframe'])=={k.replace('\\','/'):v for k,v in fr['outputs'].items()}['derive-v1/fullframe-cells.npz']
        full=load(paths['fullframe']);counts=full['fullframe_event_counts'];valid=full['valid_counts'];assert counts.shape==(len(spec['cases']),45,80,4)
        assert counts.dtype==valid.dtype==np.uint8 and (counts<=valid[...,None]).all() and (valid<=64).all()
        with CompactSource(paths['archive']) as source:
            package_hashes={k.replace('\\','/'):v for k,v in ds['package_hashes'].items()}
            def member(name,array=False):
                data=source.read_bytes(name);assert hashlib.sha256(data).hexdigest()==source.entries[name]['sha256']
                if name in package_hashes:assert hashlib.sha256(data).hexdigest()==package_hashes[name]
                return load(io.BytesIO(data)) if array else json.loads(data)
            meta=member('evaluator/metadata.json');labels=member('evaluator/labels.npz',True)
            assert meta['schema']=='mz72-training-source-v1'
            for name in ('completion','receipt','world-verification'):
                health &= member('evaluator/capture-'+name+'.json')['status']=='PASS'
            integrity=member('evaluator/capture-source-integrity.json');release=member('evaluator/capture-process-release.json');render=member('evaluator/capture-render-resource-health.json')
            health &= integrity['unchanged'] and integrity['before']==integrity['after'] and release['released'] and not release['survivors']
            health &= render['ready_data_eligible'] and not any(render['counts'].values())
            np.testing.assert_array_equal(full['frame_ids'],[r['frame_id'] for r in meta['records']])
            for i,(row,case) in enumerate(zip(meta['records'],spec['cases'])):
                assert row['frame_id']==case['name'] and row['index']==i
                for key,v in geometry_metadata(case).items():assert row[key]==v
                truth=counts[i].sum((0,1))>=3
                np.testing.assert_array_equal(truth,labels['truth'][i]);np.testing.assert_array_equal(truth,row['event_truth'])
                required=np.array(case['required_positive_bits'],bool)
                intent=not truth.any() if case['off_path_expected_zero'] else bool(truth[required].all())
                assert row['intent_matches']==intent
                np.testing.assert_array_equal(counts[i].sum((0,1)),row['event_counts'])
                np.testing.assert_array_equal(labels['known'][i],[row['source_valid']]*4)
                assert row['rgb_sha256']==source.entries[row['rgb']]['sha256']
                cs,ps=sig(counts[i]),sig(counts[i]>0)
                records.append(dict(row,counts_sha256=cs,presence_sha256=ps,positive=bool(truth.any()),novel_positive_counts=bool(truth.any() and cs not in old_counts),novel_positive_presence=bool(truth.any() and ps not in old_presence)))
            for pair in meta['pairs']:
                a,b=pair['indices'];np.testing.assert_array_equal(counts[a],counts[b]);pairs.append(pair)
    assert len(records)==len({r['frame_id'] for r in records})==64 and len(pairs)==32
    byid={r['frame_id']:r for r in records};assert set(byid)=={r['frame_id'] for r in visual['rows']}
    for row in visual['rows']:
        actual=byid[row['frame_id']];assert row['rgb_sha256']==actual['rgb_sha256'] and row['native_sha256']==actual['native_sha256']
    checks=dict(source_health=bool(health and all(r['source_valid'] and r['floor']['accepted'] for r in records)),
        all_required_intents=all(r['intent_matches'] for r in records),
        native_pairs=all(p['native_pair_invariant'] and not any(p['changed_event_mask_pixels']) and p['event_depth_max_abs_m']==0 for p in pairs),
        visual_physical_support=visual['status']=='PASS' and all(r['accepted'] and r['rgb_native_agreement'] and r['physical_support_checked'] and r['visual_note'].strip() for r in visual['rows']))
    novel={f:{k:sum(r[k] for r in records if r['family']==f) for k in ('novel_positive_counts','novel_positive_presence')} for f in m['families']}
    result=dict(status='PASS' if all(checks.values()) else 'STOP_NO_EXPANSION',checks=checks,manifest_sha256=mh,frames=64,allowed_next_frames=4032 if all(checks.values()) else 0,
        source_valid_frames=sum(r['source_valid'] for r in records),intent_matching_frames=sum(r['intent_matches'] for r in records),actual_positive_frames_by_query=np.array([r['event_truth'] for r in records]).sum(0).tolist(),
        actual_unknown_query_bits=sum(not r['source_valid'] for r in records)*4,reference_source_frames=ref['source_frames'],novelty_descriptive_only=novel,
        records=records,all_actual_extra_bits_retained=True,model_fits=0,model_predictions=0,root_main_go_issued=False,seconds=time.perf_counter()-start)
    for p,h in inputs.items():assert sha(p)==h
    out=task/'canary-audit-v1';write(out/'result.json',result);write(out/'receipt.json',dict(status='PASS_AUDIT_EXECUTED',inputs=inputs,outputs={'result.json':sha(out/'result.json')},code_sha256=sha(__file__),seconds=time.perf_counter()-start))
    print(result['status'],checks)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,required=True);p.add_argument('--reference',action='store_true');p.add_argument('--packages',type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
    assert a.task.resolve()==(ROOT/'artifacts.local/work'/TASK).resolve()
    if a.reference:reference(a.task.resolve())
    else:gate(a.task.resolve(),a.packages,a.review)

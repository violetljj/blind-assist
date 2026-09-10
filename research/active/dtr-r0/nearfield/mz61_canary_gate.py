"""MZ61 CPU source gate. Counts equality is the registered novelty gate.

Gate: --manifest M --packages P --review V --reference reference.json --output O
Reference: --prepare-reference MZ55/source-index.json --output fresh-directory
Boundary tests: --self-check --output fresh-directory
Pointers are {path,sha256}; relative package paths resolve against packages.json.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from data_lightweight import CompactSource
import data_lightweight
from mz61_source_contract import MANIFEST_SHA, MAP_SHA, admit_spec, geometry_metadata, read, sha

INDEX_SHA = '84848bc92ea077b9bbc651f6d080eedb9cdc97205c5ddc9f1033a1961032e176'
SOURCE_RECEIPT_SHA = '8ef314c51bf96569d1a9e20f45ed5441d4f1d2de3404cc2738efb32e925a17fc'
DATASET_SHA = '3bd7cd7c1bf9331eb89daf8d1474fc8427d5d4239cfedd5f9256df04ffc53fd8'
FULLFRAME_SHA = 'c2993a1184c59015dfb75499d7ff3ff0c8cd62aab0f08f0389e60d7d7c583353'
CORE_SHA = '5fb19e543e1b6c841a2c61c8a4e4ee4c918985a3883d1c061a2ce673423aeaa3'
RELATIONS = ('HEAD_ONLY','BODY_ONLY','BOTH','VISIBLE_NONINTRUDING')
FAMILIES = ('shallow_awning','sign_panel','square_grille','variable_rod')


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def digest_tensor(tensor):
    return hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()


def signatures(counts):
    assert counts.shape == (45,80,4) and counts.dtype == np.uint8 and counts.max() <= 64
    total = counts.sum((0,1))
    return dict(counts_sha256=digest_tensor(counts),presence_sha256=digest_tensor(counts>0),
                nonempty=bool(counts.any()),positive=bool((total>=3).any()),event_counts=total.tolist())


def bound(ref, base, inputs):
    path=Path(ref['path']);path=(base/path).resolve() if not path.is_absolute() else path.resolve()
    digest=sha(path);assert digest==ref['sha256'],str(path);inputs[str(path)]=digest
    return path


def arrays(path, keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}


def finish(output, result, inputs, name='result.json'):
    write(output/name,result)
    write(output/'receipt.json',dict(status='PASS_CHECK_EXECUTED',admission=result.get('status'),inputs=inputs,
        code_sha256=sha(__file__),outputs={name:sha(output/name)},new_captures=0,training_steps=0,model_inference_frames=0))


def prepare_reference(index_path, output):
    assert not output.exists();tick=time.perf_counter()
    assert sha(index_path)==INDEX_SHA
    index=read(index_path);assert index['status']=='COMPLETE' and index['frames']==2560
    inputs={str(index_path):INDEX_SHA,str(Path(__file__)):sha(__file__)}
    paths={name:bound(index['combined'][name],index_path.parent,inputs) for name in
           ('receipt.json','metadata.json','fullframe-cells.npz','evaluator.npz')}
    assert sha(paths['receipt.json'])==SOURCE_RECEIPT_SHA
    receipt=read(paths['receipt.json']);assert receipt['status']=='PASS'
    for name,path in paths.items():
        if name!='receipt.json':assert sha(path)==receipt['outputs'][name]
    records=read(paths['metadata.json'])['records']
    full=arrays(paths['fullframe-cells.npz'],('frame_ids','fullframe_event_counts'))
    labels=arrays(paths['evaluator.npz'],('frame_ids','truth','known'))
    assert len(records)==2560 and full['fullframe_event_counts'].shape==(2560,45,80,4)
    np.testing.assert_array_equal(full['frame_ids'],[r['frame_id'] for r in records])
    np.testing.assert_array_equal(labels['frame_ids'],full['frame_ids'])
    selected=[]
    for i,row in enumerate(records):
        if row['role']!='TRAIN_CANDIDATE':continue
        assert labels['known'][i].all() and row['source_valid']
        sig=signatures(full['fullframe_event_counts'][i])
        np.testing.assert_array_equal(np.array(sig['event_counts'])>=3,labels['truth'][i])
        np.testing.assert_array_equal(sig['event_counts'],row['event_counts'])
        selected.append(dict(frame_id=row['frame_id'],family=row['family'],site_id=row['site_id'],
                             pair_id=row['pair_id'],role=row['role'],known=[True]*4,**sig))
    assert len(selected)==len({r['frame_id'] for r in selected})==1600
    result=dict(status='PASS_REFERENCE',schema='mz61-mz55-train-tensor-reference-v1',rows=selected,frames=1600,
        positive_frames=sum(r['positive'] for r in selected),nonempty_frames=sum(r['nonempty'] for r in selected),
        unique_positive_counts=len({r['counts_sha256'] for r in selected if r['positive']}),
        unique_positive_presence=len({r['presence_sha256'] for r in selected if r['positive']}),
        source_index_sha256=INDEX_SHA,inputs=inputs,
        semantics='uint8 counts[45,80,4] C-order main gate; bool(counts>0) C-order additional spatial-coverage diagnostic; actual>=3 event positivity',
        seconds=time.perf_counter()-tick,rgb_or_native_files_read=0,training_steps=0,model_inference_frames=0)
    output.mkdir(parents=True,exist_ok=False);finish(output,result,inputs,'reference.json')
    print(json.dumps({k:result[k] for k in ('status','frames','positive_frames','unique_positive_counts','unique_positive_presence','seconds')}))


def novelty(records, reference):
    report={}
    for kind in ('counts','presence'):
        key=kind+'_sha256';known={r[key] for r in reference if r['positive']}
        nonempty=defaultdict(set);by_config=defaultdict(list)
        for r in records:
            by_config[r['geometry_id']].append(r)
            if r['nonempty']:nonempty[r[key]].add(r['geometry_id'])
        configs=[]
        for gid,rows in by_config.items():
            assert len({r['family'] for r in rows})==len({r['role'] for r in rows})==1
            positive={r[key] for r in rows if r['positive']}
            configs.append(dict(geometry_id=gid,family=rows[0]['family'],role=rows[0]['role'],
                frame_ids=[r['frame_id'] for r in rows],nonempty_tensor_variants=len({r[key] for r in rows if r['nonempty']}),
                positive_tensor_variants=len(positive),mz55_train_equal_positive_tensors=len(positive&known),
                novel_positive_tensors=sorted(positive-known)))
        held={r[key] for r in records if r['role']=='HELDOUT_GEOMETRY' and r['nonempty']}
        trained={r[key] for r in records if r['role']=='TRAIN_CANDIDATE' and r['nonempty']}
        report[kind]=dict(configurations=configs,
            novel_positive_configs_per_family={f:sum(c['family']==f and bool(c['novel_positive_tensors']) for c in configs) for f in FAMILIES},
            cross_distinct_geometry_collisions=[dict(tensor_sha256=h,geometry_ids=sorted(ids)) for h,ids in nonempty.items() if len(ids)>1],
            held_to_canary_train=dict(held_nonempty_unique_tensors=len(held),train_nonempty_unique_tensors=len(trained),matching_hashes=sorted(held&trained)),
            all_zero_frames=sum(not r['nonempty'] for r in records),
            all_zero_configurations=sum(all(not r['nonempty'] for r in rows) for rows in by_config.values()))
    report['counts_only_novel_families']=[f for f in FAMILIES if report['counts']['novel_positive_configs_per_family'][f]>0 and report['presence']['novel_positive_configs_per_family'][f]==0]
    report['interpretation']='Counts is the registered admission gate. Counts-only novelty does not establish new positive spatial coverage. Canary held/TRAIN collisions are descriptive, not full-source generalization.'
    return report


def criteria(records,pairs,visual,novel,health):
    represented={f:{rel:sum(r['family']==f and r['relation']==rel and r['intent_matches'] for r in records) for rel in RELATIONS} for f in FAMILIES}
    coverage={f:{d:sum(r['family']==f and r['range']==d and any(t and e for t,e in zip(r['event_truth'],r['expected_events'])) for r in records) for d in ('near','far')} for f in FAMILIES}
    checks=dict(source_health=health and all(r['source_valid'] and r['floor']['accepted'] for r in records),
        native_pairs=len(pairs)==32 and all(p['target_dictionary_equal'] and p['actual_target_receipt_equal'] and p['native_pair_invariant'] and not any(p['changed_event_mask_pixels']) and p['event_depth_max_abs_m']==0 for p in pairs),
        intent_min=sum(r['intent_matches'] for r in records)>=56,
        every_family_relation=all(v>0 for counts in represented.values() for v in counts.values()),
        both_intended_ranges_every_family=all(v>0 for counts in coverage.values() for v in counts.values()),
        all64_visual=visual['status']=='PASS' and len(visual['rows'])==64 and all(r['accepted'] and r['rgb_native_agreement'] and r['openings_backface_material_checked'] and isinstance(r['visual_note'],str) and bool(r['visual_note'].strip()) for r in visual['rows']),
        counts_novel_positive_every_family=all(novel['counts']['novel_positive_configs_per_family'][f]>0 for f in FAMILIES))
    return checks,represented,coverage


def gate(manifest,packages,review,reference,output):
    assert not output.exists();m=read(manifest);assert sha(manifest)==MANIFEST_SHA
    inputs={str(p):sha(p) for p in (manifest,packages,review,reference,Path(__file__),Path(data_lightweight.__file__),Path(__file__).with_name('mz61_source_contract.py'),Path(__file__).with_name('MZ61_GEOMETRY_SOURCE_20260911.md'))}
    ref=read(reference);rr=read(reference.parent/'receipt.json')
    assert rr['outputs']['reference.json']==sha(reference) and ref['status']=='PASS_REFERENCE' and ref['source_index_sha256']==INDEX_SHA
    assert len(ref['rows'])==1600 and all(r['role']=='TRAIN_CANDIDATE' and all(r['known']) for r in ref['rows'])
    inputs[str(reference.parent/'receipt.json')]=sha(reference.parent/'receipt.json')
    for path,digest in ref['inputs'].items():
        historical=Path(path)
        if historical.name==Path(__file__).name and sha(historical)!=digest:
            # The reference was sealed by the pre-fix gate. Bind its preserved
            # producer bytes rather than rewriting the reference or its receipt.
            historical=reference.parent.parent/'windows-path-fix-v1'/historical.name
            inputs[str(historical)]=digest
        assert sha(historical)==digest
    originals={s['shard_id']:s for s in m['shards'] if s['canary']}
    requested=read(packages)['shards'];assert len(requested)==2 and {r['shard_id'] for r in requested}==set(originals)
    records=[];pairs=[];health=True;all_cases={};offsets={};off=0
    for shard in m['shards']:offsets[shard['shard_id']]=off;off+=shard['frames']
    for entry in requested:
        paths={key:bound(entry[key],packages.parent,inputs) for key in ('archive','dataset_receipt','fullframe','fullframe_receipt','capture_spec')}
        captured=read(paths['capture_spec']);contract=admit_spec(manifest,manifest.parent/'shards',captured)
        assert contract['shard_id']==entry['shard_id'];inputs.update(contract['inputs'])
        ds,fr=read(paths['dataset_receipt']),read(paths['fullframe_receipt'])
        package_hashes={k.replace('\\','/'):v for k,v in ds['package_hashes'].items()}
        assert len(package_hashes)==len(ds['package_hashes'])
        assert ds['status']==fr['status']=='PASS' and ds['code_sha256']==DATASET_SHA and fr['code_sha256']==FULLFRAME_SHA
        assert fr['frozen_derivation_sha256']==CORE_SHA
        for r in (ds,fr):
            assert r['original_spec_sha256']==originals[entry['shard_id']]['sha256']
            assert r['captured_spec_sha256']==sha(paths['capture_spec']) and r['manifest_sha256']==MANIFEST_SHA
            assert r['shard_id']==entry['shard_id'] and r['map_sha256']==MAP_SHA
        assert {k.replace('\\','/'):v for k,v in fr['outputs'].items()}['derive-v1/fullframe-cells.npz']==sha(paths['fullframe'])
        n=len(captured['cases']);assert ds['native_audit_frames']==fr['frames']==n
        assert fr['result']['native_hashes_matched']==n and fr['result']['full_event_counts_matched']==n*4 and fr['result']['event_labels_matched']==n*4
        full=arrays(paths['fullframe'],('frame_ids','global_indices','fullframe_event_counts','valid_counts'))
        assert full['fullframe_event_counts'].shape==(n,45,80,4) and full['valid_counts'].shape==(n,45,80)
        assert full['valid_counts'].dtype==np.uint8 and full['valid_counts'].max()<=64
        assert (full['fullframe_event_counts']<=full['valid_counts'][...,None]).all()
        np.testing.assert_array_equal(full['global_indices'],np.arange(offsets[entry['shard_id']],offsets[entry['shard_id']]+n))
        with CompactSource(paths['archive']) as source:
            def member(name,decode=True):
                data=source.read_bytes(name);digest=hashlib.sha256(data).hexdigest()
                assert digest==source.entries[name]['sha256']
                if name in package_hashes:assert digest==package_hashes[name]
                return json.loads(data) if decode else data
            meta=member('evaluator/metadata.json');binding=member('evaluator/source-bindings.json')
            assert meta['schema']=='mz61-training-source-v1' and len(meta['records'])==n
            for key in ('original_spec_sha256','captured_spec_sha256','manifest_sha256','shard_id','map_sha256'):assert binding[key]==ds[key]
            assert binding['source_spec_sha256']==binding['captured_spec_sha256'] and binding['code_sha256']==DATASET_SHA
            completion=member('evaluator/capture-completion.json');release=member('evaluator/capture-process-release.json')
            integrity=member('evaluator/capture-source-integrity.json');render=member('evaluator/capture-render-resource-health.json')
            capture=member('evaluator/capture-receipt.json');world=member('evaluator/capture-world-verification.json')
            assert capture['spec_sha256']==world['source_spec_sha256']==binding['captured_spec_sha256']
            health &= completion['status']==capture['status']==world['status']=='PASS' and capture['source_unchanged'] and integrity['unchanged'] and integrity['before']==integrity['after']
            health &= release['released'] and not release['survivors'] and release.get('tracking_complete',True)
            health &= render['schema']=='city-render-resource-health-v1' and render['ready_data_eligible'] and not any(render['counts'].values())
            label=arrays(io.BytesIO(member('evaluator/labels.npz',False)),('truth','known'))
            np.testing.assert_array_equal(full['frame_ids'],[r['frame_id'] for r in meta['records']])
            for i,(row,case) in enumerate(zip(meta['records'],captured['cases'])):
                assert row['index']==i and row['frame_id']==case['name']
                for key,value in geometry_metadata(case).items():assert row[key]==value
                for key,value in dict(role=case['training_role'],pair_id=case['pair_id'],family=case['condition']['family'],relation=case['variant_id'],range=case['declared_range'],support_context=case['support_context']).items():assert row[key]==value
                sig=signatures(full['fullframe_event_counts'][i]);np.testing.assert_array_equal(sig['event_counts'],row['event_counts'])
                truth=np.array(sig['event_counts'])>=3;np.testing.assert_array_equal(truth,label['truth'][i]);np.testing.assert_array_equal(truth,row['event_truth'])
                np.testing.assert_array_equal(label['known'][i],[row['source_valid']]*4)
                collapsed=[bool(truth[:2].any()),bool(truth[2:].any())]
                assert row['expected_events']==case['expected_events'] and row['intent_matches']==(collapsed==[bool(x) for x in case['expected_collapsed_relation']])
                assert source.entries[row['rgb']]['sha256']==row['rgb_sha256']
                assert binding['native_files'][i]['sha256']==row['native_sha256']
                all_cases[row['frame_id']]=case;records.append(dict(row,**sig))
            assert len(meta['pairs'])*2==n
            for p in meta['pairs']:
                a,b=p['indices'];assert [a,b]==[a,a+1] and a%2==0
                ra,rb=meta['records'][a],meta['records'][b]
                assert ra['pair_id']==rb['pair_id']==p['pair_id']
                assert ra['target_actors']==rb['target_actors']
                np.testing.assert_array_equal(full['fullframe_event_counts'][a],full['fullframe_event_counts'][b])
                pairs.append(p)
    assert len(records)==len({r['frame_id'] for r in records})==64 and len(pairs)==32
    visual=read(review);assert visual['manifest_sha256']==MANIFEST_SHA
    assert len(visual['rows'])==64 and len({r['frame_id'] for r in visual['rows']})==64
    by_id={r['frame_id']:r for r in records};assert set(by_id)=={r['frame_id'] for r in visual['rows']}
    for r in visual['rows']:
        assert r['rgb_sha256']==by_id[r['frame_id']]['rgb_sha256'] and r['native_sha256']==by_id[r['frame_id']]['native_sha256']
    novel=novelty(records,ref['rows']);checks,represented,coverage=criteria(records,pairs,visual,novel,bool(health))
    result=dict(status='PASS' if all(checks.values()) else 'STOP_NO_EXPANSION',checks=checks,frames=64,
        manifest_sha256=MANIFEST_SHA,allowed_next_frames=4032 if all(checks.values()) else 0,
        source_valid_frames=sum(r['source_valid'] for r in records),intent_matching_frames=sum(r['intent_matches'] for r in records),
        family_relation_matches=represented,intended_range_coverage=coverage,novelty=novel,
        actual_positive_frames_by_query=np.array([r['event_truth'] for r in records]).sum(0).tolist(),
        actual_labels_and_signatures=[{k:r[k] for k in ('frame_id','geometry_id','family','role','event_truth','event_counts','counts_sha256','presence_sha256','nonempty','positive')} for r in records],
        all_actual_labels_retained=True,new_captures=0,training_steps=0,model_inference_frames=0)
    output.mkdir(parents=True,exist_ok=False);finish(output,result,inputs);print(result['status'])


def self_check(output):
    assert not output.exists()
    base=np.zeros((45,80,4),np.uint8);base[0,0,0]=3
    changed=base.copy();changed[0,0,0]=4
    oldsig=signatures(base);newsig=signatures(changed)
    assert oldsig['counts_sha256']!=newsig['counts_sha256'] and oldsig['presence_sha256']==newsig['presence_sha256']
    rows=[dict(frame_id=f,geometry_id=f,family=f,role='HELDOUT_GEOMETRY',**newsig) for f in FAMILIES]
    report=novelty(rows,[oldsig]);assert all(report['counts']['novel_positive_configs_per_family'].values())
    assert not any(report['presence']['novel_positive_configs_per_family'].values())
    assert report['counts_only_novel_families']==list(FAMILIES)
    rows[0].update(oldsig);assert novelty(rows,[oldsig])['counts']['novel_positive_configs_per_family'][FAMILIES[0]]==0
    one=base.copy();one[0,0,0]=1;assert signatures(one)['nonempty'] and not signatures(one)['positive']
    assert not signatures(np.zeros_like(base))['nonempty']
    synthetic=[]
    for family in FAMILIES:
        for relation in RELATIONS:
            for distance in ('near','far'):
                expected=[False]*4;half=int(distance=='far')
                if relation in ('BODY_ONLY','BOTH'):expected[half]=True
                if relation in ('HEAD_ONLY','BOTH'):expected[2+half]=True
                for support in range(2):
                    synthetic.append(dict(frame_id=str(len(synthetic)),family=family,relation=relation,range=distance,
                        event_truth=expected.copy(),expected_events=expected.copy(),intent_matches=True,
                        source_valid=True,floor=dict(accepted=True)))
    synthetic_pairs=[dict(target_dictionary_equal=True,actual_target_receipt_equal=True,native_pair_invariant=True,
                         changed_event_mask_pixels=[0]*4,event_depth_max_abs_m=0) for _ in range(32)]
    synthetic_visual=dict(status='PASS',rows=[dict(accepted=True,rgb_native_agreement=True,
                          openings_backface_material_checked=True,visual_note='SYNTHETIC CHECK ONLY') for _ in range(64)])
    novel_stub=dict(counts=dict(novel_positive_configs_per_family={f:1 for f in FAMILIES}))
    assert all(criteria(synthetic,synthetic_pairs,synthetic_visual,novel_stub,True)[0].values())
    for r in synthetic[:8]:r['intent_matches']=False
    assert criteria(synthetic,synthetic_pairs,synthetic_visual,novel_stub,True)[0]['intent_min']
    synthetic[8]['intent_matches']=False
    assert not criteria(synthetic,synthetic_pairs,synthetic_visual,novel_stub,True)[0]['intent_min']
    synthetic_visual['rows'][0]['visual_note']=' '
    assert not criteria(synthetic,synthetic_pairs,synthetic_visual,novel_stub,True)[0]['all64_visual']
    result=dict(status='PASS_SYNTHETIC_BOUNDARY_CHECK',cases=['counts gate differs from presence diagnostic','one family without novel positive fails novelty','1pixel nonempty is not actual>=3 positive','allzero separated','all criteria positive synthetic fixture','56 intent accepted,55 rejected','empty visual note rejected'],scope='Synthetic functions only; no canary evidence generated')
    output.mkdir(parents=True,exist_ok=False);finish(output,result,{str(Path(__file__)):sha(__file__)});print(result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','packages','review','reference','prepare-reference'):p.add_argument('--'+key,type=Path)
    p.add_argument('--self-check',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.self_check:self_check(a.output.resolve())
    elif a.prepare_reference:prepare_reference(a.prepare_reference.resolve(),a.output.resolve())
    else:
        assert all(getattr(a,k) for k in ('manifest','packages','review','reference'))
        gate(a.manifest.resolve(),a.packages.resolve(),a.review.resolve(),a.reference.resolve(),a.output.resolve())

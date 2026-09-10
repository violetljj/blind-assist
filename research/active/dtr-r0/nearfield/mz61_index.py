"""Merge the frozen4096 MZ61 source; owner-local80 native audits, no raw transfer.

parts.json: shards[10] with bound archive,dataset_receipt,dataset_result,
auxiliary,auxiliary_receipt,fullframe,fullframe_receipt,capture_spec,release;
native_audits[2] with result/receipt; visual(final80 rows); releases[2].
Every pointer is {path: relative_to_task, sha256: hex}. --self-test does not
read source artifacts. Real merge requires all sealed parts and root GO.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import time
import numpy as np
from data_lightweight import CompactSource
import data_lightweight
import mz61_source_contract as contract

DATASET_SHA='3bd7cd7c1bf9331eb89daf8d1474fc8427d5d4239cfedd5f9256df04ffc53fd8'
FULLFRAME_SHA='c2993a1184c59015dfb75499d7ff3ff0c8cd62aab0f08f0389e60d7d7c583353'
CORE_SHA='5fb19e543e1b6c841a2c61c8a4e4ee4c918985a3883d1c061a2ce673423aeaa3'
NATIVE_SHA='0708c2507a9ac8aec69026116391dbd7c896216312fa93346068a2914b225322'
GATE_SHA='5f5f6a6bd03375d28e861b82f3c2bc2af0d5d726d0f9c82f33f9dfc15d5edc3f'
GATE_RECEIPT_SHA='d78ab4f36f8cd8c4b350a47d19f5438d94bb6aa16320e28cb2a27dbe64a2ff8b'
GEOMETRY_CONFIG_SHA='aeb365a052d4c73d0376f494a4317db5b0e2946b743bb01c6828fd169e66c82a'
GEOMETRY_POSITION_ATOL_M=1e-9
GEOMETRY_YAW_ATOL_DEG=1e-10
EVENTS=('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')
REQUIRED=('archive','dataset_receipt','dataset_result','auxiliary','auxiliary_receipt',
          'fullframe','fullframe_receipt','capture_spec','release')
SCHEMA={'index':'mz61-geometry-source-v1','frames':4096,'shards':10,
        'roles':{'TRAIN_CANDIDATE':2048,'CALIBRATION':1024,'HELDOUT_GEOMETRY':1024},
        'fullframe_event_counts':'uint8[N,45,80,4]','valid_counts':'uint8[N,45,80]',
        'native_audit_frames':{'primary':32,'worker':48},'visual_rows':80,
        'required_shard_pointers':list(REQUIRED),'required_top_pointers':['visual','releases','native_audits']}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def geometry_configurations(path):
    """Frozen identity is hashed before site yaw/translation in the generator."""
    assert sha(path)==GEOMETRY_CONFIG_SHA
    rows=read(path)['configurations']
    by_id={r['geometry_id']:r for r in rows}
    assert len(rows)==len(by_id)==1024
    for row in rows:
        targets=row['targets_camera_coordinates']
        actual=fingerprint([{k:v for k,v in t.items() if k not in ('name','target_part')} for t in targets])
        assert actual==row['geometry_id']
    return by_id


def validate_geometry_case(case,configurations):
    """Inverse only the frozen world() transform; never rehash rounded coordinates.

    Canonical JSON remains the byte-exact identity. Numeric tolerances apply
    only to float64 site-translation/yaw inversion, not meshes/scales/materials,
    dimensions, role, relation, range, recipe or IDs. This checks spec geometry,
    not subnanometre accuracy of the Unreal renderer.
    """
    config=configurations[case['geometry_id']]
    for key,actual in (('family',case['condition']['family']),('relation',case['variant_id']),
                      ('range',case['declared_range']),('recipe',case['profile_id']),
                      ('joint_factors',case['geometry_joint_factors']),('role',case['training_role'])):
        assert config[key]==actual,(case['name'],key)
    assert case['geometry_recipe_id']==config['family']+f"-g{config['recipe']:02d}"
    camera=case['camera'];assert camera['pitch']==camera['roll']==0
    yaw=math.radians(camera['yaw']);c,s=math.cos(yaw),math.sin(yaw)
    targets=[o for o in case['objects'] if o['target_part']]
    canonical=config['targets_camera_coordinates']
    assert len(targets)==len(canonical)
    max_position=max_yaw=0.
    for actual,expected in zip(targets,canonical):
        # All untransformed fields, including scale/material and object order,
        # retain exact equality to the bound camera-relative configuration.
        assert {k:v for k,v in actual.items() if k not in ('center_m','rotation_deg')}=={
            k:v for k,v in expected.items() if k not in ('center_m','rotation_deg')}
        dx,dy=actual['center_m'][0]-camera['x'],actual['center_m'][1]-camera['y']
        local=np.array([c*dx+s*dy,-s*dx+c*dy,actual['center_m'][2]-case['floor_z_m']])
        wanted=np.array(expected['center_m'])
        assert np.isfinite(local).all() and np.isfinite(wanted).all()
        np.testing.assert_allclose(local,wanted,rtol=0,atol=GEOMETRY_POSITION_ATOL_M)
        max_position=max(max_position,float(np.abs(local-wanted).max()))
        ar,er=actual['rotation_deg'],expected['rotation_deg']
        assert set(ar)==set(er)=={'pitch','yaw','roll'}
        assert ar['pitch']==er['pitch'] and ar['roll']==er['roll']
        local_yaw=ar['yaw']-camera['yaw']
        assert math.isfinite(local_yaw)
        error=abs(local_yaw-er['yaw']);assert error<=GEOMETRY_YAW_ATOL_DEG
        max_yaw=max(max_yaw,error)
    return dict(geometry_id=config['geometry_id'],max_position_error_m=max_position,max_yaw_error_deg=max_yaw)


def tensor_collisions(records,counts,truth,known):
    """Exact label equality; collapse replicas but retain variant ambiguity.

    Hashes are checked against array bytes on collision. All-zero tensors
    never count as positive novelty/collision. Geometry IDs are not labels.
    """
    assert len(records)==len(counts)==len(truth)==len(known)
    configs=defaultdict(list)
    for i,r in enumerate(records):configs[r['geometry_id']].append(i)
    assert all(len({records[i]['role'] for i in ids})==1 for ids in configs.values())
    output={}
    for mode in ('counts','presence'):
        tables={}
        for query in (None,0,1,2,3):
            arrays=counts if query is None else counts[...,query]
            signatures=[];representatives={};train=defaultdict(set)
            positive=truth.any(1)&known.all(1) if query is None else truth[:,query]&known[:,query]
            for i,a in enumerate(arrays):
                b=(a>0).astype(np.uint8) if mode=='presence' else a
                payload=np.ascontiguousarray(b).tobytes()
                sig=hashlib.sha256(payload).hexdigest()
                if sig in representatives:assert payload==representatives[sig]
                else:representatives[sig]=payload
                signatures.append((sig,bool(b.any())))
                if records[i]['role']=='TRAIN_CANDIDATE' and known[i].all() and b.any():
                    train[sig].add(records[i]['geometry_id'])
            rows=[]
            for geometry,ii in sorted(configs.items()):
                positive_ids=[i for i in ii if positive[i]]
                variants={signatures[i][0] for i in positive_ids if signatures[i][1]}
                hits={sig:sorted(train.get(sig,set())) for sig in sorted(variants)}
                r=records[ii[0]]
                rows.append(dict(geometry_id=geometry,family=r['family'],role=r['role'],frames=len(ii),
                    known_positive_frames=len(positive_ids),positive_configuration=bool(positive_ids),
                    zero_tensor_frames=sum(not signatures[i][1] for i in ii),
                    all_zero_configuration=all(not signatures[i][1] for i in ii),
                    distinct_positive_tensor_variants=len(variants),
                    any_positive_variant_seen_in_train=any(hits.values()),
                    all_positive_variants_seen_in_train=bool(variants) and all(hits.values()),
                    matching_train_geometries_by_tensor=hits,
                    frame_ids=[records[i]['frame_id'] for i in ii]))
            summaries={}
            for role in contract.ROLES:
                for family in ('ALL',)+tuple(sorted({r['family'] for r in records})):
                    selected=[r for r in rows if r['role']==role and (family=='ALL' or r['family']==family)]
                    positive_rows=[r for r in selected if r['positive_configuration']]
                    summaries[role+'/'+family]=dict(configurations=len(selected),positive_configurations=len(positive_rows),
                        positive_all_variants_seen_in_train=sum(r['all_positive_variants_seen_in_train'] for r in positive_rows),
                        positive_any_variant_seen_in_train=sum(r['any_positive_variant_seen_in_train'] for r in positive_rows),
                        all_zero_configurations=sum(r['all_zero_configuration'] for r in selected),
                        zero_tensor_frames=sum(r['zero_tensor_frames'] for r in selected))
            tables['whole' if query is None else EVENTS[query]]=dict(configurations=rows,summary=summaries)
        output[mode]=tables
    held=output['counts']['whole']['summary']['HELDOUT_GEOMETRY/ALL']
    most=held['positive_all_variants_seen_in_train']>held['positive_configurations']/2
    return dict(schema='mz61-native-tensor-collision-v1',tables=output,
        held_positive_counts_most_repeat_train=most,
        geometry_generalization_claim='REJECT_MAJORITY_LABEL_COLLISION' if most else 'NOT_ESTABLISHED_BY_SOURCE_AUDIT',
        scope='Exact counts and boolean presence separately; whole and per query. All-zero separate. Cross-site/support replicas collapse by geometry; all variants must match for full repeated-config count. TRAIN self-matches are descriptive only. Geometry split disjointness is not label disjointness; no rows are removed.')


def build(task,parts,output):
    task=Path(task).resolve();parts=Path(parts).resolve();output=Path(output).resolve()
    assert task.name=='mz61-geometry-source-20260911' and output.is_relative_to(task)
    assert not (task/'source-index.json').exists() and not output.exists()
    output.mkdir(parents=True)
    started=time.perf_counter();inputs={}
    def bind(path,expected=None):
        path=Path(path).resolve(strict=True);digest=sha(path)
        assert expected is None or digest==expected,str(path)
        inputs[str(path)]=digest;return path
    def bound(ref):
        assert set(ref)=={'path','sha256'}
        path=(task/ref['path']).resolve();assert path.is_relative_to(task)
        return bind(path,ref['sha256'])
    def status(path):
        value=read(path);assert value['status']=='PASS',str(path)
        return value
    try:
        for p in (Path(__file__),Path(contract.__file__),Path(data_lightweight.__file__)):bind(p)
        manifest_path=bind(task/'spec-v2/manifest.json',contract.MANIFEST_SHA)
        manifest=read(manifest_path);assert manifest['frames']==4096 and manifest['roles']==SCHEMA['roles']
        configurations=geometry_configurations(bind(task/'spec-v2/geometry-configurations.json',GEOMETRY_CONFIG_SHA))
        geometry_validation=dict(cases=0,max_position_error_m=0.,max_yaw_error_deg=0.,position_atol_m=GEOMETRY_POSITION_ATOL_M,yaw_atol_deg=GEOMETRY_YAW_ATOL_DEG,canonical_config_sha256=GEOMETRY_CONFIG_SHA)
        admission=read(bind(task/'canary-gate-v1/receipt.json',GATE_RECEIPT_SHA))
        gate=read(bind(task/'canary-gate-v1/result.json',GATE_SHA))
        assert admission['admission']=='PASS' and admission['outputs']['result.json']==GATE_SHA
        assert gate['status']=='PASS' and len(gate['checks'])==7 and all(gate['checks'].values())
        spec_dir=task/'spec-v2/shards'
        bundle=read(bind(parts));entries=bundle['shards'];selected={e['shard_id']:e for e in entries}
        assert len(entries)==len(selected)==10 and set(selected)=={s['shard_id'] for s in manifest['shards']}
        assert len(bundle['releases'])==2
        for ref in bundle['releases']:status(bound(ref))
        visual=status(bound(bundle['visual']));assert visual['manifest_sha256']==contract.MANIFEST_SHA
        reviewed={r['frame_id']:r for r in visual['rows']}
        assert len(reviewed)==len(visual['rows'])==80
        assert all(r['accepted'] and r['rgb_native_agreement'] and r['openings_backface_material_checked']
                   and isinstance(r['visual_note'],str) and bool(r['visual_note'].strip()) for r in reviewed.values())
        assert visual['inputs'], 'Actual preview images must be bound'
        for name,digest in visual['inputs'].items():
            path=Path(name);path=path if path.is_absolute() else task/path
            bind(path,digest)
        native_rows={};audit_pairs=set();hosts=set();native_receipts=[];native_inputs={}
        assert len(bundle['native_audits'])==2
        for item in bundle['native_audits']:
            receipt=status(bound(item['receipt']));result=status(bound(item['result']))
            assert receipt['code_sha256']==NATIVE_SHA and receipt['outputs']['result.json']==item['result']['sha256']
            host=result['host'];assert host not in hosts and host in ('primary','worker');hosts.add(host)
            assert result['frames']==len(result['rows'])==SCHEMA['native_audit_frames'][host]
            assert result['event_cells_checked']==result['frames']*14400 and result['known_cells_checked']==result['frames']*3600
            assert len(result['pairs'])*2==result['frames'] and not (audit_pairs&set(result['pairs']))
            audit_pairs.update(result['pairs'])
            assert not result['raw_native_transferred'] and result['training_steps']==result['model_inference_frames']==result['new_captures']==0
            receipt_inputs=list(receipt['inputs'].values())
            assert contract.MANIFEST_SHA in receipt_inputs and NATIVE_SHA in receipt_inputs
            assert sha(Path(contract.__file__)) in receipt_inputs
            native_inputs[host]=set(receipt_inputs)
            for row in result['rows']:
                assert row['status']=='PASS' and row['frame_id'] not in native_rows
                assert row['event_cells_checked']==14400 and row['known_cells_checked']==3600
                native_rows[row['frame_id']]=dict(row,host=host)
            native_receipts.append(item)
        assert len(native_rows)==80 and len(audit_pairs)==40
        records=[];refs=[];frame_ids=[];pair_ids=set();expected_audits=set();geometries={}
        packet_parts=[];truth_parts=[];known_parts=[];aux_parts=[];aux_known=[];contributor_parts=[];source_parts=[];cell_parts=[]
        counts=np.empty((4096,45,80,4),np.uint8);valid_counts=np.empty((4096,45,80),np.uint8)
        archive_bytes=rgb_bytes=member_bytes=raw_bytes_reported=0;member_count=0;host_seconds=Counter()
        for frozen in manifest['shards']:
            entry=selected[frozen['shard_id']];paths={k:bound(entry[k]) for k in REQUIRED}
            for key,ref in entry.items():
                if key not in REQUIRED and isinstance(ref,dict) and set(ref)=={'path','sha256'}:bound(ref)
            status(paths['release'])
            frozen_path=bind(spec_dir/(frozen['shard_id']+'.json'),frozen['sha256'])
            spec=read(frozen_path);captured=read(paths['capture_spec'])
            admitted=contract.admit_spec(manifest_path,spec_dir,captured)
            for p,h in admitted['inputs'].items():bind(p,h)
            assert admitted['shard_id']==frozen['shard_id']
            ds=status(paths['dataset_receipt']);dr=read(paths['dataset_result']);ar=status(paths['auxiliary_receipt']);fr=status(paths['fullframe_receipt'])
            assert ds['code_sha256']==DATASET_SHA and fr['code_sha256']==FULLFRAME_SHA and fr['frozen_derivation_sha256']==CORE_SHA
            captured_sha=sha(paths['capture_spec']);n=frozen['frames'];offset=len(records)
            for r in (ds,fr):
                assert r['original_spec_sha256']==frozen['sha256'] and r['captured_spec_sha256']==captured_sha
                assert r['manifest_sha256']==contract.MANIFEST_SHA and r['shard_id']==frozen['shard_id'] and r['map_sha256']==contract.MAP_SHA
            assert ds['source_spec_sha256']==ar['source_spec_sha256']==captured_sha
            assert ds['outputs']['result.json']==entry['dataset_result']['sha256'] and ar['output_sha256']==entry['auxiliary']['sha256']
            assert {k.replace('\\','/'):v for k,v in fr['outputs'].items()}['derive-v1/fullframe-cells.npz']==entry['fullframe']['sha256']
            assert dr['frames']==ar['frames']==fr['frames']==n and dr['invariant_pairs']*2==n
            raw_bytes_reported+=dr['raw_bytes']
            fresult=fr['result'];assert fresult['status']=='PASS' and fresult['native_hashes_matched']==n
            assert fresult['full_event_counts_matched']==fresult['event_labels_matched']==n*4 and fresult['event_cell_invariant_pairs']*2==n
            expected_host='primary' if 'candidate_05' in frozen['shard_id'] else 'worker'
            assert fresult['host']==expected_host;host_seconds[expected_host]+=fresult['seconds']
            for key in ('dataset_receipt','fullframe','fullframe_receipt'):
                assert entry[key]['sha256'] in native_inputs[expected_host], 'Owner audit does not bind this exact shard derivation'
            full=npz(paths['fullframe']);aux=npz(paths['auxiliary'])
            assert full['fullframe_event_counts'].shape==(n,45,80,4) and full['fullframe_event_counts'].dtype==np.uint8
            assert full['valid_counts'].shape==(n,45,80) and full['valid_counts'].dtype==np.uint8
            assert aux['cell_event_presence'].shape==(n,64,49,4) and aux['cell_known'].shape==(n,64,49)
            assert aux['cell_event_presence'].dtype==aux['cell_known'].dtype==bool
            np.testing.assert_array_equal(full['global_indices'],np.arange(offset,offset+n))
            counts[offset:offset+n]=full['fullframe_event_counts'];valid_counts[offset:offset+n]=full['valid_counts']
            archive_bytes+=paths['archive'].stat().st_size
            with CompactSource(paths['archive']) as source:
                assert not any(p.startswith('evaluator/native/') for p in source.paths()), 'Return packages must exclude raw native'
                package={k.replace('\\','/'):v for k,v in ds['package_hashes'].items()}
                seen=set()
                def member(name):
                    nonlocal member_count,member_bytes
                    data=source.read_bytes(name);digest=hashlib.sha256(data).hexdigest()
                    assert digest==source.entries[name]['sha256']
                    if name in package:assert digest==package[name]
                    if name not in seen:member_count+=1;member_bytes+=len(data);seen.add(name)
                    return data
                def js(name):return json.loads(member(name))
                meta=js('evaluator/metadata.json');binding=js('evaluator/source-bindings.json');predictor=js('model/predictor.json')
                assert meta['schema']=='mz61-training-source-v1' and predictor['schema']=='mz61-observable-input-v1'
                assert len(meta['records'])==len(spec['cases'])==n and len(binding['native_files'])==n
                for key in ('original_spec_sha256','captured_spec_sha256','manifest_sha256','shard_id','map_sha256'):assert binding[key]==ds[key]
                assert binding['code_sha256']==DATASET_SHA and binding['source_spec_sha256']==captured_sha
                completion=js('evaluator/capture-completion.json');release=js('evaluator/capture-process-release.json')
                integrity=js('evaluator/capture-source-integrity.json');render=js('evaluator/capture-render-resource-health.json')
                capture=js('evaluator/capture-receipt.json');world=js('evaluator/capture-world-verification.json')
                assert completion['status']==capture['status']==world['status']=='PASS'
                assert capture['spec_sha256']==world['source_spec_sha256']==captured_sha and capture['source_unchanged']
                assert integrity['unchanged'] and integrity['before']==integrity['after']
                assert release['released'] and not release['survivors'] and release.get('tracking_complete',True)
                assert render['ready_data_eligible'] and not any(render['counts'].values())
                packet=npz(io.BytesIO(member('model/packets.npz')));labels=npz(io.BytesIO(member('evaluator/labels.npz')))
                assert packet['ranges'].shape==packet['valid'].shape==(n,64,2)
                assert packet['ranges'].dtype==np.float32 and packet['valid'].dtype==bool and np.isfinite(packet['ranges']).all()
                assert ((packet['ranges'][packet['valid']]>0)&(packet['ranges'][packet['valid']]<=4)).all()
                assert predictor['calibration']==dict(native_size=[640,360],horizontal_fov_degrees=100,eye_height_m=1.7,zone_crop_degrees=[45,45])
                assert predictor['rgb_files']==[r['rgb'] for r in meta['records']] and predictor['packets']=='model/packets.npz'
                assert labels['truth'].shape==labels['known'].shape==(n,4) and labels['truth'].dtype==labels['known'].dtype==bool
                np.testing.assert_array_equal(labels['cell_known'],aux['cell_known'])
                assert not (labels['query_presence'].any(2)&~aux['cell_event_presence']).any()
                assert not (aux['cell_event_presence']&~aux['cell_known'][...,None]).any()
                np.testing.assert_array_equal(full['frame_ids'],[r['frame_id'] for r in meta['records']])
                for i,(row,case,native) in enumerate(zip(meta['records'],spec['cases'],binding['native_files'])):
                    assert row['index']==native['index']==i and row['frame_id']==case['name'] and row['native_sha256']==native['sha256']
                    for key,val in contract.geometry_metadata(case).items():assert row[key]==val
                    checked=validate_geometry_case(case,configurations)
                    geo=checked['geometry_id'];assert geo==row['geometry_id']
                    geometries.setdefault(geo,set()).add(row['role'])
                    geometry_validation['cases']+=1
                    for key in ('max_position_error_m','max_yaw_error_deg'):
                        geometry_validation[key]=max(geometry_validation[key],checked[key])
                    for key,val in [('role',case['training_role']),('pair_id',case['pair_id']),('family',case['condition']['family']),
                        ('site_id',case['site_id']),('relation',case['variant_id']),('range',case['declared_range']),
                        ('setting',case['profile_id']),('support_context',case['support_context']),('camera',case['camera']),
                        ('wearer',case['wearer']),('floor_z_m',case['floor_z_m']),('native_audit_sample',case['native_audit_sample'])]:assert row[key]==val
                    assert not row['source_valid'] or row['floor']['accepted']
                    actual=full['fullframe_event_counts'][i].sum((0,1))
                    np.testing.assert_array_equal(actual,row['event_counts']);np.testing.assert_array_equal(actual>=3,labels['truth'][i])
                    np.testing.assert_array_equal(labels['truth'][i],row['event_truth'])
                    np.testing.assert_array_equal(labels['known'][i],[row['source_valid']]*4)
                    expected=np.array(case['expected_events'],bool)
                    np.testing.assert_array_equal(row['extra_range_bits'],labels['truth'][i]&~expected)
                    collapsed=[bool(labels['truth'][i,:2].any()),bool(labels['truth'][i,2:].any())]
                    assert row['intent_matches']==(collapsed==[bool(v) for v in case['expected_collapsed_relation']])
                    data=member(row['rgb']);rgb_bytes+=len(data);assert hashlib.sha256(data).hexdigest()==row['rgb_sha256']
                    assert ar['native_inputs'][i]['native_sha256']==row['native_sha256'] and ar['native_inputs'][i]['frame_id']==row['frame_id']
                    if case['native_audit_sample']:
                        expected_audits.add(row['frame_id']);aud=native_rows[row['frame_id']];vis=reviewed[row['frame_id']]
                        assert aud['host']==expected_host and aud['shard_id']==frozen['shard_id'] and aud['index']==i
                        for key in ('rgb_sha256','native_sha256'):assert aud[key]==vis[key]==row[key]
                    records.append(dict(row,index=offset+i,source_index=i,shard_id=frozen['shard_id']))
                    frame_ids.append(row['frame_id']);refs.append(dict(frame_id=row['frame_id'],archive=entry['archive'],member=row['rgb'],sha256=row['rgb_sha256']))
                assert len(meta['pairs'])*2==n
                for pair in meta['pairs']:
                    a,b=pair['indices'];assert b==a+1 and a%2==0
                    ra,rb=meta['records'][a],meta['records'][b]
                    assert ra['pair_id']==rb['pair_id']==pair['pair_id'] and pair['pair_id'] not in pair_ids
                    assert ra['role']==rb['role'] and ra['target_actors']==rb['target_actors']
                    assert pair['native_pair_invariant'] and pair['target_dictionary_equal'] and pair['actual_target_receipt_equal']
                    np.testing.assert_array_equal(full['fullframe_event_counts'][a],full['fullframe_event_counts'][b]);pair_ids.add(pair['pair_id'])
                for name in source.paths():
                    if name not in seen:member(name)
                packet_parts.append(packet);truth_parts.append(labels['truth']);known_parts.append(labels['known'])
                aux_parts.append(aux['cell_event_presence']);aux_known.append(aux['cell_known'])
                contributor_parts.append(labels['query_presence']);source_parts.append(labels['source_presence']);cell_parts.append(labels['cell_known'])
        assert len(records)==len(set(frame_ids))==4096 and len(pair_ids)==2048
        assert expected_audits==set(native_rows)==set(reviewed) and len(expected_audits)==80
        assert audit_pairs=={r['pair_id'] for r in records if r['native_audit_sample']}
        assert len(geometries)==1024 and all(len(roles)==1 for roles in geometries.values())
        assert Counter(r['role'] for r in records)==SCHEMA['roles']
        assert counts.max()<=64 and valid_counts.max()<=64 and (counts<=valid_counts[...,None]).all()
        truth=np.concatenate(truth_parts);known=np.concatenate(known_parts)
        np.testing.assert_array_equal(counts.sum((1,2))>=3,truth)
        pairs=defaultdict(list)
        for i,r in enumerate(records):pairs[r['pair_id']].append(i)
        valid_changes=0
        for ii in pairs.values():
            assert len(ii)==2;np.testing.assert_array_equal(counts[ii[0]],counts[ii[1]])
            valid_changes+=int(not np.array_equal(valid_counts[ii[0]],valid_counts[ii[1]]))
        np.savez_compressed(output/'packets.npz',frame_ids=np.array(frame_ids),ranges=np.concatenate([p['ranges'] for p in packet_parts]),valid=np.concatenate([p['valid'] for p in packet_parts]))
        np.savez_compressed(output/'evaluator.npz',frame_ids=np.array(frame_ids),truth=truth,known=known)
        np.savez_compressed(output/'fullframe-cells.npz',frame_ids=np.array(frame_ids),global_indices=np.arange(4096,dtype=np.int32),fullframe_event_counts=counts,valid_counts=valid_counts)
        np.savez_compressed(output/'angular-auxiliary.npz',frame_ids=np.array(frame_ids),cell_event_presence=np.concatenate(aux_parts),cell_known=np.concatenate(aux_known))
        np.savez_compressed(output/'angular-contributors.npz',frame_ids=np.array(frame_ids),query_presence=np.concatenate(contributor_parts),source_presence=np.concatenate(source_parts),cell_known=np.concatenate(cell_parts))
        write(output/'metadata.json',dict(records=records,pairs=dict(pairs),roles=manifest['roles'],source_role='CONSUMED_DEVELOPMENT'))
        write(output/'rgb-refs.json',dict(frames=refs,authority='Observable original RGB only; archive paths relative to task'))
        collision=tensor_collisions(records,counts,truth,known);write(output/'tensor-collision-audit.json',collision)
        write(output/'native-audit.json',dict(status='PASS',frames=80,rows=list(native_rows.values()),pairs=sorted(audit_pairs),owner_receipts=native_receipts,raw_native_transferred=False,source='Independent owner-local80 native audits; controller does not decode raw native'))
        coverage={}
        for role in contract.ROLES:
            for family in sorted({r['family'] for r in records}):
                ii=[i for i,r in enumerate(records) if r['role']==role and r['family']==family]
                coverage[role+'/'+family]=dict(frames=len(ii),known_queries=known[ii].sum(0).tolist(),actual_positive_queries=(truth[ii]&known[ii]).sum(0).tolist(),geometry_configurations=len({records[i]['geometry_id'] for i in ii}))
        result=dict(status='PASS',frames=4096,unique_frame_ids=4096,source_valid_frames=int(known.all(1).sum()),roles=manifest['roles'],
            intent_matching_frames=sum(r['intent_matches'] for r in records),actual_event_positives=truth.sum(0).tolist(),
            extra_range_bits=np.array([r['extra_range_bits'] for r in records]).sum(0).tolist(),unknown_query_bits=int((~known).sum()),
            unknown_fullframe_cells=int((valid_counts==0).sum()),invariant_event_cell_pairs=2048,pairs_changed_valid_background=valid_changes,
            independent_native_frames=80,independent_raw_pairs=40,visual_reviewed_frames=80,geometry_ids=1024,role_geometry_intersections=0,
            geometry_identity_validation=geometry_validation,
            coverage=coverage,geometry_generalization_claim=collision['geometry_generalization_claim'],
            tensor_collision_summary={m:{q:t['summary'] for q,t in tables.items()} for m,tables in collision['tables'].items()},
            archive_bytes=archive_bytes,raw_capture_bytes_reported=raw_bytes_reported,allocated_capture_bytes=None,
            rgb_bytes=rgb_bytes,rgb_files_hash_verified=4096,all_package_members_hash_verified=member_count,
            all_package_member_bytes=member_bytes,owner_fullframe_seconds=dict(host_seconds),seconds=time.perf_counter()-started,
            training_steps=0,model_inference_frames=0,new_captures=0,controller_raw_native_reads=0,
            source_health_all_frames=bool(known.all()),
            source_status_boundary='PASS means complete bound source/derivation, not a generalization or hardware result; UNKNOWN/intended-label mismatches retained. source_health_all_frames is explicit.')
        write(output/'result.json',result)
        for p,h in inputs.items():assert sha(p)==h,p
        write(output/'receipt.json',dict(status='PASS',inputs=inputs,outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},original_sources_unchanged=True))
        index=dict(status='COMPLETE',schema=SCHEMA['index'],frames=4096,source_role='CONSUMED_DEVELOPMENT',manifest=dict(path='spec-v2/manifest.json',sha256=contract.MANIFEST_SHA),
            shards=[selected[s['shard_id']] for s in manifest['shards']],native_audits=bundle['native_audits'],visual=bundle['visual'],releases=bundle['releases'],
            combined={p.name:dict(path=p.relative_to(task).as_posix(),sha256=sha(p)) for p in output.iterdir() if p.is_file()},
            predictor_inputs=['packets.npz: original45degree ranges/valid','rgb-refs.json: original RGB only'],
            training_evaluator_only=['evaluator.npz','fullframe-cells.npz','angular-auxiliary.npz','angular-contributors.npz','metadata.json'],
            local_presence='fullframe_event_counts>0',local_known='valid_counts>0',global_event='fullframe_event_counts.sum((1,2))>=3',
            contributor_semantics='IDEAL original echo bins only; DROP mask slots; never inherit to MERGE',geometry_generalization_claim=collision['geometry_generalization_claim'])
        write(task/'source-index.json',index)
        print(json.dumps({k:v for k,v in result.items() if k not in ('coverage','tensor_collision_summary')}),flush=True)
    except BaseException:
        import traceback
        write(output/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=inputs));raise


def self_test():
    # Two TRAIN configurations, three HELD and one CAL. No source provenance.
    roles=['TRAIN_CANDIDATE']*2+['HELDOUT_GEOMETRY']*3+['CALIBRATION']
    records=[dict(frame_id=str(i),geometry_id=str(i),family='fixture',role=r) for i,r in enumerate(roles)]
    c=np.zeros((6,1,2,4),np.uint8)
    c[0,0,0,0]=3;c[0,0,1,1]=3
    c[2]=c[0];c[2,0,0,0]=4 # Same presence; different whole counts.
    c[3,0,0,0]=3 # Same per-BN counts; different whole presence/counts.
    c[5]=c[0] # Calibration also collides but cannot be training authority.
    t=c.sum((1,2))>=3;k=np.ones((6,4),bool)
    audit=tensor_collisions(records,c,t,k)
    def summary(mode,query,role='HELDOUT_GEOMETRY'):
        return audit['tables'][mode][query]['summary'][role+'/ALL']
    assert summary('counts','whole')['positive_all_variants_seen_in_train']==0
    assert summary('presence','whole')['positive_all_variants_seen_in_train']==1
    assert summary('counts','BODY_NEAR')['positive_all_variants_seen_in_train']==1
    assert summary('counts','whole')['all_zero_configurations']==1
    assert summary('counts','whole','CALIBRATION')['positive_all_variants_seen_in_train']==1
    bad=[dict(r) for r in records];bad[2]['geometry_id']='0'
    try:tensor_collisions(bad,c,t,k)
    except AssertionError:pass
    else:raise AssertionError('Cross-role geometry accepted')
    # One configuration may have multiple observed label variants. Require all.
    replicas=[records[0],records[2],dict(records[2],frame_id='replica')]
    a=tensor_collisions(replicas,np.stack([c[0],c[0],c[3]]),np.stack([t[0],t[0],t[3]]),k[:3])
    row=a['tables']['counts']['whole']['summary']['HELDOUT_GEOMETRY/ALL']
    assert row['positive_any_variant_seen_in_train']==1 and row['positive_all_variants_seen_in_train']==0
    repeated=c.copy();repeated[2]=c[0];repeated[3]=c[0]
    majority=tensor_collisions(records,repeated,repeated.sum((1,2))>=3,k)
    assert majority['geometry_generalization_claim']=='REJECT_MAJORITY_LABEL_COLLISION'
    assert not audit['held_positive_counts_most_repeat_train']
    print('PASS synthetic collision boundary: whole/per-query, counts/presence, zero, roles, multi-variant configurations')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('task','parts','output'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        assert args.task is args.parts is args.output is None;self_test()
    else:
        assert args.task is not None and args.parts is not None and args.output is not None
        build(args.task,args.parts,args.output)

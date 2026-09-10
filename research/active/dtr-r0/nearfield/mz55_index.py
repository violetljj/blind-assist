"""Seal complete MZ55 source and independently verify the reserved80 native grids.

Input parts.json: {shards:[{shard_id, archive, dataset_receipt, dataset_result,
auxiliary, auxiliary_receipt, fullframe, fullframe_receipt, capture_spec?,
release}]}. Each pointer is {path: relative_to_task, sha256: hex}. Fullframe
points to fullframe-v1/derive-v1/fullframe-cells.npz; its receipt is the outer
fullframe-v1/receipt.json. Exactly the original ten shards are required.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import data_lightweight
import mz55_fullframe
from data_lightweight import CompactSource
from mz55_fullframe import CORE_SHA


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def npz(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}


def build(task,parts,output):
    assert not (task/'source-index.json').exists(), 'Refuse to replace a sealed source index'
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    inputs={str(p):sha(p) for p in (parts,Path(__file__),Path(data_lightweight.__file__),Path(mz55_fullframe.__file__),task/'spec-v1/manifest.json',task/'root-admission.json',task/'allocation.json',task/'extra-worker-handoff.json',task/'worker-final-release.json',task/'primary-release-final.json')}
    try:
        manifest=read(task/'spec-v1/manifest.json');admission=read(task/'root-admission.json')
        assert admission['status']=='PASS_ROOT_ADMITTED' and admission['manifest_sha256']==sha(task/'spec-v1/manifest.json')
        assert manifest['total_frames']==2560
        allocation=read(task/'allocation.json');assert allocation['manifest_sha256']==sha(task/'spec-v1/manifest.json')
        worker_release=read(task/'worker-final-release.json');primary_release=read(task/'primary-release-final.json')
        assert worker_release['status']==primary_release['status']=='PASS'
        assert worker_release['frames']==1624 and primary_release['frames_completed']==936
        selected={r['shard_id']:r for r in read(parts)['shards']}
        assert len(selected)==10 and set(selected)=={r['shard_id'] for r in manifest['shards']}
        def bound(ref):
            p=(task/ref['path']).resolve();assert p.is_relative_to(task.resolve())
            digest=sha(p);assert digest==ref['sha256'],p;inputs[str(p)]=digest;return p
        for key in ('gate','visual','audit'):bound(admission[key])
        visual=read(bound(admission['visual']));assert visual['status']=='PASS'
        reviewed={r['frame_id']:r for r in visual['rows'] if r['accepted'] and r['rgb_native_agreement']}
        assert len(reviewed)==64
        records=[];refs=[];ranges=[];valids=[];truths=[];knowns=[];auxs=[];auxknowns=[]
        counts=np.empty((2560,45,80,4),np.uint8);known_counts=np.empty((2560,45,80),np.uint8)
        audits=[];raw_pairs=[];frame_ids=[];source_bytes=0;rgb_bytes=0;host_seconds=Counter();fullframe_bytes=0
        y,x=np.indices((360,640),dtype=np.float64);f=320/np.tan(np.deg2rad(50.))
        right=(x-319.5)/f;up=(179.5-y)/f;factor=np.sqrt(1+right**2+up**2)
        for frozen in manifest['shards']:
            entry=selected[frozen['shard_id']]
            paths={k:bound(entry[k]) for k in ('archive','dataset_receipt','dataset_result','auxiliary','auxiliary_receipt','fullframe','fullframe_receipt','release')}
            if not frozen['canary']:
                review=read(bound(entry['visual']));assert review['status']=='PASS'
                bound(review['actually_viewed'])
                assert len(review['frames'])==2
                for row in review['frames']:
                    assert row['frame_id'] not in reviewed;reviewed[row['frame_id']]=row
            if 'host_receipt' in entry:assert read(bound(entry['host_receipt']))['status']=='PASS'
            spec_path=task/'spec-v1/shards'/(frozen['shard_id']+'.json');assert sha(spec_path)==frozen['sha256']
            inputs[str(spec_path)]=frozen['sha256'];spec=read(spec_path)
            capture_sha=frozen['sha256']
            if 'capture_spec' in entry:
                actual=bound(entry['capture_spec']);captured=read(actual)
                assert {k:v for k,v in captured.items() if k!='map_file'}=={k:v for k,v in spec.items() if k!='map_file'}
                capture_sha=entry['capture_spec']['sha256']
            ds=read(paths['dataset_receipt']);dr=read(paths['dataset_result']);ar=read(paths['auxiliary_receipt']);fr=read(paths['fullframe_receipt'])
            assert ds['status']==ar['status']==fr['status']=='PASS'
            assert read(paths['release'])['status']=='PASS'
            assert ds['source_spec_sha256']==ar['source_spec_sha256']==capture_sha
            assert ds['outputs']['result.json']==entry['dataset_result']['sha256']
            assert ar['output_sha256']==entry['auxiliary']['sha256']
            assert fr['frozen_derivation_sha256']==CORE_SHA
            assert fr['result']['host']==('worker' if frozen['canary'] else allocation['owners'][frozen['shard_id']])
            output_hashes={key.replace('\\','/'):value for key,value in fr['outputs'].items()}
            assert output_hashes['derive-v1/fullframe-cells.npz']==entry['fullframe']['sha256']
            n=frozen['frames'];assert dr['frames']==dr['source_valid_frames']==ar['frames']==fr['frames']==n
            assert dr['invariant_pairs']*2==n and fr['result']['event_cell_invariant_pairs']*2==n
            assert fr['result']['native_hashes_matched']==n and fr['result']['full_event_counts_matched']==n*4
            assert fr['result']['event_labels_matched']==n*4
            host_seconds[fr['result']['host']]+=fr['result']['seconds'];fullframe_bytes+=paths['fullframe'].stat().st_size
            source_bytes+=paths['archive'].stat().st_size
            full=npz(paths['fullframe']);aux=npz(paths['auxiliary'])
            assert full['fullframe_event_counts'].shape==(n,45,80,4) and full['fullframe_event_counts'].dtype==np.uint8
            assert full['valid_counts'].shape==(n,45,80) and full['valid_counts'].dtype==np.uint8
            assert aux['cell_event_presence'].shape==(n,64,49,4) and aux['cell_known'].shape==(n,64,49)
            offset=len(records);np.testing.assert_array_equal(full['global_indices'],np.arange(offset,offset+n))
            counts[offset:offset+n]=full['fullframe_event_counts'];known_counts[offset:offset+n]=full['valid_counts']
            with CompactSource(paths['archive']) as source:
                def member(name):
                    data=source.read_bytes(name);assert hashlib.sha256(data).hexdigest()==source.entries[name]['sha256'];return data
                meta=json.loads(member('evaluator/metadata.json'));binding=json.loads(member('evaluator/source-bindings.json'))
                assert meta['schema']=='mz55-training-source-v1' and len(meta['records'])==n
                assert binding['source_spec_sha256']==capture_sha and binding['code_sha256']==ds['code_sha256']
                completion=json.loads(member('evaluator/capture-completion.json'));assert completion['status']=='PASS'
                release=json.loads(member('evaluator/capture-process-release.json'));assert release['released'] and not release['survivors']
                integrity=json.loads(member('evaluator/capture-source-integrity.json'));assert integrity['unchanged']
                packet=npz(io.BytesIO(member('model/packets.npz')))
                labels=npz(io.BytesIO(member('evaluator/labels.npz')),('truth','known','query_presence','cell_known'))
                np.testing.assert_array_equal(aux['cell_known'],labels['cell_known'])
                assert not (labels['query_presence'].any(2)&~aux['cell_event_presence']).any()
                assert not (aux['cell_event_presence']&~aux['cell_known'][...,None]).any()
                np.testing.assert_array_equal(full['frame_ids'],[r['frame_id'] for r in meta['records']])
                previous_audit=None
                for i,(row,case,native) in enumerate(zip(meta['records'],spec['cases'],binding['native_files'])):
                    assert row['index']==native['index']==i and row['frame_id']==case['name']
                    assert row['native_sha256']==native['sha256'] and row['role']==case['training_role']
                    for key,expected in [('pair_id',case['pair_id']),('family',case['condition']['family']),('site_id',case['site_id']),
                                         ('relation',case['variant_id']),('range',case['declared_range']),('setting',case['profile_id']),('support_context',case['support_context'])]:
                        assert row[key]==expected
                    actual=full['fullframe_event_counts'][i].sum((0,1))
                    np.testing.assert_array_equal(actual,row['event_counts']);np.testing.assert_array_equal(actual>=3,labels['truth'][i])
                    np.testing.assert_array_equal(labels['known'][i],[row['source_valid']]*4)
                    collapsed=[bool(labels['truth'][i,:2].any()),bool(labels['truth'][i,2:].any())]
                    assert row['intent_matches']==(collapsed==[bool(v) for v in case['expected_collapsed_relation']])
                    rgb=member(row['rgb']);rgb_bytes+=len(rgb);assert hashlib.sha256(rgb).hexdigest()==row['rgb_sha256']
                    if case['native_audit_sample']:
                        reviewed_row=reviewed[row['frame_id']]
                        assert reviewed_row['rgb_sha256']==row['rgb_sha256'] and reviewed_row['native_sha256']==row['native_sha256']
                        raw=member(f'evaluator/native/{i:04d}.npy');assert hashlib.sha256(raw).hexdigest()==row['native_sha256']
                        depth=np.load(io.BytesIO(raw),allow_pickle=False);assert depth.shape==(360,640) and depth.dtype==np.float32
                        d=depth.astype(np.float64);good=np.isfinite(d)&(d>0)&(d<100)&(d*factor<=4)
                        yy=d*right;zz=1.7+d*up;masks=[]
                        for front,width,lo,hi in ((.18,.28,.65,1.4),(.13,.18,1.4,1.85)):
                            for half in range(2):
                                end=(d<front+1.5) if half==0 else (d<=front+3.)
                                masks.append(good&(d>=front+half*1.5)&end&(abs(yy)<=width)&(zz>=lo)&(zz<=hi))
                        masks=np.stack(masks,-1)
                        # Deliberately independent explicit block loops, not
                        # the producer's reshape reduction or imported helper.
                        independently=np.empty((45,80,4),np.uint8);known_ind=np.empty((45,80),np.uint8)
                        for a in range(45):
                            for b in range(80):
                                independently[a,b]=masks[a*8:a*8+8,b*8:b*8+8].sum((0,1))
                                known_ind[a,b]=good[a*8:a*8+8,b*8:b*8+8].sum()
                        np.testing.assert_array_equal(independently,full['fullframe_event_counts'][i])
                        np.testing.assert_array_equal(known_ind,full['valid_counts'][i])
                        with Image.open(io.BytesIO(rgb)) as image:assert image.size==(640,360);image.load()
                        if previous_audit is None:
                            assert row['support_context']=='unsupported';previous_audit=(row['pair_id'],depth,masks)
                        else:
                            pair,olddepth,oldmasks=previous_audit;assert pair==row['pair_id']
                            np.testing.assert_array_equal(oldmasks,masks);pix=masks.any(-1);np.testing.assert_array_equal(olddepth[pix],depth[pix])
                            raw_pairs.append(pair);previous_audit=None
                        audits.append(dict(frame_id=row['frame_id'],global_index=offset+i,native_sha256=row['native_sha256'],rgb_sha256=row['rgb_sha256'],event_cells=14400,known_cells=3600,status='PASS'))
                    record=dict(row,index=offset+i,source_index=i,shard_id=frozen['shard_id'])
                    records.append(record);frame_ids.append(row['frame_id'])
                    refs.append(dict(frame_id=row['frame_id'],archive=entry['archive'],member=row['rgb'],sha256=row['rgb_sha256']))
                assert previous_audit is None
                for pair in meta['pairs']:
                    assert pair['native_pair_invariant'] and pair['target_dictionary_equal'] and pair['actual_target_receipt_equal']
                    a,b=pair['indices'];np.testing.assert_array_equal(full['fullframe_event_counts'][a],full['fullframe_event_counts'][b])
                ranges.append(packet['ranges']);valids.append(packet['valid']);truths.append(labels['truth']);knowns.append(labels['known'])
                auxs.append(aux['cell_event_presence']);auxknowns.append(aux['cell_known'])
        assert len(records)==len(set(frame_ids))==2560 and len(audits)==len(reviewed)==80 and len(raw_pairs)==40
        assert Counter(r['role'] for r in records)==Counter(manifest['role_counts'])
        assert Counter(r['family'] for r in records)==Counter({f:640 for f in manifest['families']})
        assert Counter(r['site_id'] for r in records)==Counter({s['site_id']:320 for s in manifest['sites']})
        combinations=[tuple(r[k] for k in ('site_id','family','relation','range','setting','support_context')) for r in records]
        assert len(set(combinations))==2560
        pairs={}
        for i,r in enumerate(records):pairs.setdefault(r['pair_id'],[]).append(i)
        assert len(pairs)==1280
        valid_changes=0
        for pair,ii in pairs.items():
            assert len(ii)==2; a,b=ii;assert records[a]['role']==records[b]['role']
            np.testing.assert_array_equal(counts[a],counts[b]);valid_changes+=int(not np.array_equal(known_counts[a],known_counts[b]))
        truth=np.concatenate(truths);known=np.concatenate(knowns);assert known.all()
        np.testing.assert_array_equal(counts.sum((1,2))>=3,truth)
        assert counts.max()<=64 and known_counts.max()<=64 and (counts<=known_counts[...,None]).all()
        np.savez_compressed(output/'packets.npz',frame_ids=np.array(frame_ids),ranges=np.concatenate(ranges),valid=np.concatenate(valids))
        np.savez_compressed(output/'evaluator.npz',frame_ids=np.array(frame_ids),truth=truth,known=known)
        np.savez_compressed(output/'fullframe-cells.npz',frame_ids=np.array(frame_ids),global_indices=np.arange(2560,dtype=np.int32),fullframe_event_counts=counts,valid_counts=known_counts)
        np.savez_compressed(output/'angular-auxiliary.npz',frame_ids=np.array(frame_ids),cell_event_presence=np.concatenate(auxs),cell_known=np.concatenate(auxknowns))
        write(output/'metadata.json',dict(records=records,pairs=pairs,roles=manifest['role_counts'],source_role='CONSUMED_DEVELOPMENT'))
        write(output/'rgb-refs.json',dict(frames=refs,authority='Observable RGB only; archive path relative to task'))
        write(output/'native-audit.json',dict(status='PASS',frames=80,rows=audits,raw_pairs=raw_pairs,event_values=1152000,known_values=288000))
        result=dict(status='PASS',frames=2560,source_valid_frames=int(known.all(1).sum()),unique_frame_ids=2560,roles=dict(Counter(r['role'] for r in records)),
            intent_matching_frames=sum(r['intent_matches'] for r in records),intent_mismatches=[r['frame_id'] for r in records if not r['intent_matches']],
            actual_event_positives=truth.sum(0).tolist(),extra_range_bits=np.array([r['extra_range_bits'] for r in records]).sum(0).tolist(),
            invariant_event_cell_pairs=1280,pairs_with_changed_valid_background=valid_changes,independent_native_frames=80,independent_raw_pairs=40,
            unknown_query_bits=int((~known).sum()),unknown_fullframe_cells=int((known_counts==0).sum()),rgb_files_hash_verified=2560,rgb_bytes=rgb_bytes,
            source_archive_bytes=source_bytes,fullframe_shard_bytes=fullframe_bytes,merged_fullframe_bytes=(output/'fullframe-cells.npz').stat().st_size,
            owner_fullframe_seconds=dict(host_seconds),seconds=time.perf_counter()-started,backend='NumPy CPU; source audit only',
            training_steps=0,model_inference_frames=0,new_captures=0,through_hole_geometry_claim=False)
        write(output/'result.json',result)
        for path,digest in inputs.items():assert sha(Path(path))==digest, path
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}
        write(output/'receipt.json',dict(status='PASS',inputs=inputs,outputs=outputs,original_sources_unchanged=True))
        index=dict(status='COMPLETE',schema='mz55-diverse-source-v1',frames=2560,source_role='CONSUMED_DEVELOPMENT',
            manifest=dict(path='spec-v1/manifest.json',sha256=sha(task/'spec-v1/manifest.json')),shards=[selected[r['shard_id']] for r in manifest['shards']],
            combined={p.name:dict(path=str(p.relative_to(task)),sha256=sha(p)) for p in output.iterdir() if p.is_file()},
            predictor_inputs=['packets.npz: ranges/valid','rgb-refs.json: original RGB only'],
            training_evaluator_only=['evaluator.npz','fullframe-cells.npz','angular-auxiliary.npz','metadata.json'],
            cell_event_presence='fullframe_event_counts>0',cell_known='valid_counts>0',global_event='fullframe_event_counts.sum((1,2))>=3',
            sensor_coverage='Original45degree ToF unchanged; no fullframe depth input to model',through_hole_geometry_claim=False)
        write(task/'source-index.json',index)
        print(json.dumps(result),flush=True)
    except BaseException:
        import traceback
        write(output/'failure.json',dict(status='FAIL',error=traceback.format_exc(),inputs=inputs));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('task','parts','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();build(a.task.resolve(),a.parts.resolve(),a.output.resolve())

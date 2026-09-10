"""Seal the ten source shards for loading only after full identity checks."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from data_lightweight import CompactSource


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(task):
    def bind(path):
        return dict(path=path.relative_to(task).as_posix(),sha256=sha(path))
    manifest=task/'spec-v1/manifest.json'
    shards=[];records=[];pairs=[];raw_bytes=0;zip_bytes=0;native=0
    audit_bindings=[];previews=[];aux_bytes=0;unknown_cells=0;missing_witness=0
    for declared in read(manifest)['shards']:
        name=declared['shard_id']
        candidates=[task/'returned-v1'/name,task/'primary-warm-v1'/('primary-'+name)]
        found=[p for p in candidates if (p/'training.source.zip').exists()]
        assert len(found)==1,(name,'Missing or duplicate archive',found)
        folder=found[0]
        frozen_spec=task/'spec-v1/shards'/(name+'.json')
        capture_spec=frozen_spec
        if folder.parent.name=='primary-warm-v1':
            capture_spec=task/'primary-spec-v1'/(name+'.json')
            frozen=read(frozen_spec);adapted=read(capture_spec)
            assert adapted['map_file']!=frozen['map_file']
            adapted['map_file']=frozen['map_file']
            assert adapted==frozen,'A primary change exceeds the mechanical map_file adaptation'
            restored=(json.dumps(adapted,indent=2)+'\n').encode('utf-8')
            assert hashlib.sha256(restored).hexdigest()==declared['sha256']
        assert sha(frozen_spec)==declared['sha256']
        paths=dict(archive=folder/'training.source.zip',dataset_receipt=folder/'dataset-v1/receipt.json',
            dataset_result=folder/'dataset-v1/result.json',auxiliary=folder/'auxiliary-v1/native-cell-supervision.npz',
            auxiliary_receipt=folder/'auxiliary-v1/receipt.json')
        result=read(paths['dataset_result']);aux_receipt=read(paths['auxiliary_receipt'])
        for rel in ('verify.json','primary-verify.json','release.json','auxiliary-v1/primary-verify.json'):
            proof=folder/rel
            assert read(proof)['status']=='PASS',(name,rel)
            audit_bindings.append(bind(proof))
        assert read(folder/'verify.json')['source_byte_comparison']
        missing_witness+=read(folder/'auxiliary-v1/primary-verify.json')['positive_event_bits_without_native_crop_witness']
        previews.extend(bind(p) for p in sorted((folder/'dataset-v1').glob('preview-*.jpg')))
        assert result['frames']==declared['frames']==result['source_valid_frames']
        assert read(paths['dataset_receipt'])['source_spec_sha256']==sha(capture_spec)
        assert aux_receipt['source_spec_sha256']==sha(capture_spec)
        assert sha(paths['auxiliary'])==aux_receipt['output_sha256']
        assert result['invariant_pairs']==result['pairs']
        with CompactSource(paths['archive']) as source:
            metadata=source.read_json('evaluator/metadata.json')
            rows=metadata['records'];assert len(rows)==declared['frames']
            labels=dict(np.load(io.BytesIO(source.read_bytes('evaluator/labels.npz')),allow_pickle=False))
            with np.load(paths['auxiliary'],allow_pickle=False) as aux:
                assert aux['cell_event_presence'].shape==(len(rows),64,49,4)
                np.testing.assert_array_equal(aux['cell_known'],labels['cell_known'])
                assert not (labels['query_presence'].any(2)&~aux['cell_event_presence']).any()
                unknown_cells+=int((~aux['cell_known']).sum())
            for i,row in enumerate(rows):
                assert row['frame_id']==aux_receipt['native_inputs'][i]['frame_id']
                assert row['native_sha256']==aux_receipt['native_inputs'][i]['native_sha256']
                assert source.entries[row['rgb']]['sha256']==row['rgb_sha256']
                np.testing.assert_array_equal(row['event_truth'],labels['truth'][i])
                if row['native_audit_sample']:
                    assert source.entries[f'evaluator/native/{i:04d}.npy']['sha256']==row['native_sha256']
                    native+=1
            records.extend(rows);pairs.extend(metadata['pairs'])
        shards.append(dict(shard_id=name,frames=declared['frames'],capture_spec=bind(capture_spec),**{key:bind(path) for key,path in paths.items()}))
        raw_bytes+=result['raw_bytes'];zip_bytes+=paths['archive'].stat().st_size
        aux_bytes+=paths['auxiliary'].stat().st_size
    assert len(shards)==10 and len(records)==2560 and len({r['frame_id'] for r in records})==2560
    sites=Counter(r['site_id'] for r in records);roles=Counter(r['role'] for r in records)
    assert len(sites)==8 and set(sites.values())=={320}
    assert roles==dict(TRAIN_CANDIDATE=1920,HELDOUT_SITE=640)
    assert len(pairs)==1280 and len({p['pair_id'] for p in pairs})==1280 and native==80
    assert all(p['native_pair_invariant'] and p['actual_target_receipt_equal'] for p in pairs)
    index=dict(status='COMPLETE',schema='mz48-source-index-v1',spec_manifest=bind(manifest),shards=shards,
        frames=2560,sites=dict(sites),roles=dict(roles),native_audit_frames=80,
        predictor_contract='Only model RGB/packets and declared calibration; evaluator data is supervision/audit only.')
    target=task/'source-index.json'
    assert not target.exists(),'Do not overwrite a sealed index'
    target.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
    receipt=dict(status='PASS',source_index=bind(target),code_sha256=sha(Path(__file__)),
        frames=2560,unique_frame_ids=2560,pairs=1280,native_invariant_pairs=1280,native_audit_frames=80,
        intent_matching_frames=sum(r['intent_matches'] for r in records),source_valid_frames=2560,
        native_event_positives=np.array([r['event_truth'] for r in records]).sum(0).tolist(),
        raw_bytes=raw_bytes,training_zip_bytes=zip_bytes,roles=dict(roles),
        auxiliary_bytes=aux_bytes,angular_unknown_cells=unknown_cells,
        positive_event_bits_without_45deg_native_cell_witness=missing_witness,
        global_unknown_bits=0,unknown_scope='Frozen frame/floor native contract; not full physical-volume visibility',
        audit_bindings=audit_bindings,visual_review=dict(reviewer='MZ48 source agent',frames=80,preview_pages=previews,
            limitations='Controlled scaled/floating forms; small low contrast far targets; leafless opaque birch; no natural support or device-clearance claim'),
        mechanical_map_path_receipt=bind(task/'primary-spec-v1/receipt.json'),
        primary_adaptation='Only map_file path differs; restoring it reproduces the frozen complete spec SHA256.',
        model_inference_frames=0,training_steps=0,original_zip_or_receipt_mutations=0)
    (task/'source-total-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,required=True)
    build(p.parse_args().task.absolute())

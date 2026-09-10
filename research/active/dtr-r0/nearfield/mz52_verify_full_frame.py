"""Independent CPU block-loop audit of80 frozen natives and all2560 saved MZ52 labels."""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from data_lightweight import CompactSource

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def independent_blocks(native):
    """Loop explicit8x8 slices, without the producer's full-image reshape reduction."""
    event=np.zeros((45,80,4),np.uint8);known=np.zeros((45,80),np.uint8)
    focal=320./np.tan(50.*np.pi/180.)
    # Derive directions from original pixel indices, independently of producer arrays.
    for by in range(45):
        vertical=(179.5-np.arange(by*8,by*8+8,dtype=np.float64))/focal
        for bx in range(80):
            horizontal=(np.arange(bx*8,bx*8+8,dtype=np.float64)-319.5)/focal
            z=native[by*8:by*8+8,bx*8:bx*8+8].astype(np.float64)
            radial=z*np.sqrt(1.+vertical[:,None]**2+horizontal[None,:]**2)
            valid=np.isfinite(z)&(z>0)&(z<100)&(radial<=4.)
            known[by,bx]=np.count_nonzero(valid)
            y=z*horizontal[None,:];height=1.7+z*vertical[:,None]
            for q,(start,end,width,bottom,top) in enumerate(((.18,1.68,.28,.65,1.4),
                       (1.68,3.18,.28,.65,1.4),(.13,1.63,.18,1.4,1.85),(1.63,3.13,.18,1.4,1.85))):
                upper=(z<end) if q%2==0 else (z<=end)
                membership=valid&(z>=start)&upper&(y>=-width)&(y<=width)&(height>=bottom)&(height<=top)
                event[by,bx,q]=np.count_nonzero(membership)
    return event,known

def run(task):
    tick=time.perf_counter();manifest=read(task/'manifest.json');rows=manifest['rows']
    out=task/'verified-v1';out.mkdir(exist_ok=False)
    write(out/'start.json',dict(status='RUNNING',backend='NumPy CPU block-loop',native_budget=80,code_sha256=sha(__file__)))
    n=len(rows);assert n==2560
    event=np.empty((n,45,80,4),np.uint8);valid=np.empty((n,45,80),np.uint8);assigned=np.zeros(n,bool)
    inputs={str(task/'manifest.json'):sha(task/'manifest.json'),str(Path(__file__)):sha(__file__)}
    host_results={}
    for host in ('primary','worker'):
        folder=task/(host+'-v1');receipt=read(folder/'receipt.json');request=read(task/(host+'-request.json'))
        assert receipt['status']=='PASS' and receipt['request_sha256']==sha(task/(host+'-request.json'))
        assert receipt['manifest_sha256']==sha(task/'manifest.json') and receipt['code_sha256']==manifest['derivation_code_sha256']
        assert receipt['native_inputs']==request['rows']
        for name,digest in receipt['outputs'].items():
            path=folder/name; assert sha(path)==digest;inputs[str(path)]=digest
        inputs[str(folder/'receipt.json')]=sha(folder/'receipt.json')
        with np.load(folder/'fullframe-cells.npz',allow_pickle=False) as arrays:
            ids=arrays['global_indices'];assert not assigned[ids].any();assigned[ids]=True
            np.testing.assert_array_equal(ids,[r['global_index'] for r in request['rows']])
            np.testing.assert_array_equal(arrays['frame_ids'],[rows[i]['frame_id'] for i in ids])
            assert arrays['fullframe_event_counts'].dtype==arrays['valid_counts'].dtype==np.uint8
            assert arrays['fullframe_event_counts'].shape==(len(ids),45,80,4)
            assert arrays['valid_counts'].shape==(len(ids),45,80)
            event[ids]=arrays['fullframe_event_counts'];valid[ids]=arrays['valid_counts']
        host_results[host]=receipt['result']
    assert assigned.all() and event.max()<=64 and valid.max()<=64
    assert (event<=valid[:,:,:,None]).all()
    total_counts=event.sum((1,2));truth=np.array([r['event_truth'] for r in rows],bool)
    np.testing.assert_array_equal(total_counts,np.array([r['event_counts'] for r in rows]))
    np.testing.assert_array_equal(total_counts>=3,truth)
    pair_map={}
    for i,row in enumerate(rows):pair_map.setdefault(row['pair_id'],[]).append(i)
    assert len(pair_map)==1280
    changed_valid=0
    for pair in pair_map.values():
        assert len(pair)==2;a,b=pair
        np.testing.assert_array_equal(event[a],event[b])
        assert rows[a]['site_id']==rows[b]['site_id'] and rows[a]['role']==rows[b]['role']
        changed_valid+=int(not np.array_equal(valid[a],valid[b]))
    native_ids=[i for i,r in enumerate(rows) if r['native_audit_sample']];assert len(native_ids)==80
    checked=[]
    with ExitStack() as stack:
        archives={name:stack.enter_context(CompactSource(name)) for name in sorted({rows[i]['archive'] for i in native_ids})}
        for position,i in enumerate(native_ids):
            row=rows[i];raw=archives[row['archive']].read_bytes(row['native_member'])
            assert hashlib.sha256(raw).hexdigest()==row['native_sha256']
            depth=np.load(io.BytesIO(raw),allow_pickle=False)
            assert depth.dtype==np.float32 and depth.shape==(360,640)
            expected_event,expected_valid=independent_blocks(depth)
            np.testing.assert_array_equal(event[i],expected_event,err_msg=row['frame_id'])
            np.testing.assert_array_equal(valid[i],expected_valid,err_msg=row['frame_id'])
            checked.append(dict(global_index=i,frame_id=row['frame_id'],native_sha256=row['native_sha256'],
                                event_values=45*80*4,valid_values=45*80,exact=True))
            if position%20==0:print('INDEPENDENT_NATIVE',position+1,80,flush=True)
    known=valid>0;presence=event>0;assert not (presence&~known[:,:,:,None]).any()
    old_witness=np.array([r['old45_witness'] for r in rows],bool);full_witness=presence.any((1,2))
    assert not (truth&~full_witness).any();assert int((truth&~old_witness).sum())==352
    partitions={
        'heldout_site':[i for i,r in enumerate(rows) if r['role']=='HELDOUT_SITE'],
        'nonfit_family':[i for i,r in enumerate(rows) if r['role']!='HELDOUT_SITE' and r['family']=='birch'],
        'calibration':[i for i,r in enumerate(rows) if r['role']!='HELDOUT_SITE' and r['family']!='birch' and r['site_id']=='mz36_dense_candidate_06_site_002'],
        'fit':[i for i,r in enumerate(rows) if r['role']!='HELDOUT_SITE' and r['family']!='birch' and r['site_id']!='mz36_dense_candidate_06_site_002']}
    assert [len(partitions[k]) for k in ('fit','calibration','heldout_site','nonfit_family')]==[1280,256,640,384]
    target=out/'fullframe-cells.npz'
    np.savez_compressed(target,fullframe_event_counts=event,valid_counts=valid,frame_ids=np.array([r['frame_id'] for r in rows]),
                        global_indices=np.arange(2560,dtype=np.int32))
    write(out/'schema.json',dict(**manifest['schema'],event_order=manifest['event_order'],shape=[2560,45,80],
          full_native_event='fullframe_event_counts.sum((1,2))>=3',records_authority='../manifest.json',
          partitions=partitions,metadata_fields_forbidden_at_inference=['native counts','site','family','pair','relation','source split']))
    for path,digest in manifest['source_inputs'].items():assert sha(path)==digest;inputs[path]=digest
    result=dict(status='PASS',frames=n,unique_frame_ids=len({r['frame_id'] for r in rows}),
        all_native_hashes_matched=sum(r['native_hashes_matched'] for r in host_results.values()),
        full_event_counts_exact=10240,full_event_labels_exact=10240,invariant_event_cell_pairs=1280,
        pairs_with_changed_valid_counts=changed_valid,independent_native_frames=80,
        independent_event_values=80*45*80*4,independent_valid_values=80*45*80,
        native_event_positives=truth.sum(0).tolist(),positive_events=int(truth.sum()),
        old45_positive_events_without_witness=352,those_with_new_fullframe_witness=int((truth&~old_witness&full_witness).sum()),
        all_positive_events_with_fullframe_witness=int((truth&full_witness).sum()),
        weak_global_queries_1_or_2_pixels=int(((total_counts>0)&(total_counts<3)).sum()),
        unknown_cells=int((~known).sum()),max_count=int(max(event.max(),valid.max())),
        partition_counts={k:len(v) for k,v in partitions.items()},host_results=host_results,
        compact_merged_bytes=target.stat().st_size,derivation_npz_bytes=sum(r['output_bytes'] for r in host_results.values()),
        backend='NumPy CPU',seconds=time.perf_counter()-tick,training_steps=0,model_inference_frames=0,new_captures=0,
        original_source_hashes_unchanged=True,limit='Supervision coverage only; not RGB model success or sensor observability expansion')
    write(out/'result.json',result);write(out/'independent-native-audit.json',dict(status='PASS',rows=checked))
    write(out/'receipt.json',dict(status='PASS',inputs=inputs,code_sha256=sha(__file__),result=result,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,required=True)
    run(p.parse_args().task.resolve())

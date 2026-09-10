"""Derive native-only full-RGB8x8-cell labels on each original depth owner's CPU."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

EVENTS = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def prepare(root, task):
    from data_lightweight import CompactSource
    from mz50_source import load_source
    task.mkdir(parents=True, exist_ok=False)
    source = root/'artifacts.local/work/mz48-rich-kilotier-20260911'
    index = read(source/'source-index.json')
    total = read(source/'source-total-receipt.json')
    assert total['status'] == 'PASS' and total['source_valid_frames'] == 2560
    loaded = load_source(source, index)
    by_id = {r['frame_id']:i for i,r in enumerate(loaded['records'])}
    rows, inputs = [], {str(source/p):sha(source/p) for p in ('source-index.json','source-total-receipt.json')}
    brief = Path(__file__).with_name('MZ52_FULL_FRAME_SUPERVISION_20260911.md')
    inputs[str(brief)] = sha(brief)
    for shard in index['shards']:
        archive_path = source/shard['archive']['path']; inputs[str(archive_path)] = sha(archive_path)
        assert inputs[str(archive_path)] == shard['archive']['sha256']
        with CompactSource(archive_path) as archive:
            meta = archive.read_json('evaluator/metadata.json')
            bindings = archive.read_json('evaluator/source-bindings.json')
            for row, native in zip(meta['records'], bindings['native_files']):
                i = by_id[row['frame_id']]
                assert native['index'] == row['index'] and native['sha256'] == row['native_sha256']
                path = native['worker_path']
                owner = 'primary' if path.replace('\\','/').lower().startswith('f:/ba-data/blindassist-artifacts-20260805/') else 'worker'
                if owner == 'worker':
                    assert path.replace('\\','/').lower().startswith('g:/devworkspace/blindassist/artifacts/')
                rows.append(dict(global_index=i, source_index=row['index'], frame_id=row['frame_id'],
                    native_path=path, native_sha256=native['sha256'], host=owner,
                    event_counts=row['event_counts'], event_truth=row['event_truth'],
                    known=loaded['known'][i].tolist(), old45_witness=loaded['cell_event_presence'][i].any((0,1)).tolist(),
                    **{k:row[k] for k in ('pair_id','site_id','region_id','family','relation','range','setting','support_context','role','native_audit_sample')},
                    archive=str(archive_path.resolve()), rgb=row['rgb'],
                    native_member=f'evaluator/native/{row["index"]:04d}.npy', shard_id=shard['shard_id']))
    rows.sort(key=lambda r:r['global_index'])
    assert len(rows) == len({r['frame_id'] for r in rows}) == 2560
    assert [r['global_index'] for r in rows] == list(range(2560))
    assert sum(r['native_audit_sample'] for r in rows) == 80
    manifest = dict(status='FROZEN_DERIVATION_INPUTS', frames=2560, event_order=EVENTS, rows=rows,
        source_inputs=inputs, derivation_code_sha256=sha(__file__), captures=0, model_inference_frames=0,
        training_steps=0, native_validity='finite & 0<axial<100 & radial<=4m; HFoV100 eye1.7m level camera',
        schema=dict(fullframe_event_counts='uint8[N,45,80,4]',valid_counts='uint8[N,45,80]',
                    cell_event_presence='DERIVED: fullframe_event_counts>0',cell_known='DERIVED: valid_counts>0',
                    block='Nonoverlapping8x8 pixels from original640x360; row-major', max_count=64,
                    authority='Training/evaluator-only; original ToF45 packet and RGB bytes unchanged'))
    write(task/'manifest.json',manifest)
    for owner, expected in [('primary',936),('worker',1624)]:
        selected = [r for r in rows if r['host'] == owner]; assert len(selected)==expected
        write(task/(owner+'-request.json'),dict(status='FROZEN_DERIVATION_REQUEST', host=owner,
            frames=expected, rows=selected, manifest_sha256=sha(task/'manifest.json'),
            derivation_code_sha256=sha(__file__), schema=manifest['schema']))
    write(task/'prepare-receipt.json',dict(status='PASS',frames=2560,host_frames=dict(primary=936,worker=1624),
        inputs=inputs,code_sha256=sha(__file__),outputs={p.name:sha(p) for p in task.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='PASS',frames=2560,host_frames=dict(primary=936,worker=1624))))

def derive(request, output):
    req=read(request); assert req['status']=='FROZEN_DERIVATION_REQUEST'
    assert sha(__file__)==req['derivation_code_sha256']
    output.mkdir(parents=True,exist_ok=False); tick=time.perf_counter()
    write(output/'start.json',dict(status='RUNNING',pid=os.getpid(),request_sha256=sha(request),
        code_sha256=sha(__file__),frames=req['frames'],host=req['host'],backend='NumPy CPU; no torch or CUDA'))
    try:
        n=req['frames']; counts=np.empty((n,45,80,4),np.uint8); valid_counts=np.empty((n,45,80),np.uint8)
        yy,xx=np.indices((360,640),dtype=np.float64); focal=320/np.tan(np.deg2rad(50.))
        right=(xx-319.5)/focal; up=(179.5-yy)/focal; factor=np.sqrt(1+right**2+up**2)
        native_bytes=0
        for i,row in enumerate(req['rows']):
            path=Path(row['native_path']); raw=path.read_bytes(); native_bytes+=len(raw)
            assert hashlib.sha256(raw).hexdigest()==row['native_sha256'],str(path)
            depth=np.load(io.BytesIO(raw),allow_pickle=False)
            assert depth.dtype==np.float32 and depth.shape==(360,640)
            d=depth.astype(np.float64); valid=np.isfinite(d)&(d>0)&(d<100)&(d*factor<=4)
            lateral=d*right; height=1.7+d*up
            vc=valid.reshape(45,8,80,8).sum((1,3),dtype=np.uint16)
            assert vc.max()<=64; valid_counts[i]=vc
            q=0
            for front,width,low,high in ((.18,.28,.65,1.4),(.13,.18,1.4,1.85)):
                for half in range(2):
                    endpoint=(d<front+1.5) if half==0 else (d<=front+3.)
                    inside=valid&(d>=front+half*1.5)&endpoint&(abs(lateral)<=width)&(height>=low)&(height<=high)
                    reduced=inside.reshape(45,8,80,8).sum((1,3),dtype=np.uint16)
                    assert reduced.max()<=64 and (reduced<=vc).all()
                    assert int(reduced.sum())==row['event_counts'][q]
                    assert bool(reduced.sum()>=3)==row['event_truth'][q]
                    counts[i,:,:,q]=reduced;q+=1
            if i%200==0 or i==n-1:
                write(output/'progress.json',dict(status='RUNNING',processed=i+1,total=n,pid=os.getpid(),seconds=time.perf_counter()-tick))
                print('DERIVE',req['host'],i+1,n,flush=True)
        pairs={}
        for i,row in enumerate(req['rows']): pairs.setdefault(row['pair_id'],[]).append(i)
        changed_valid=0
        for a,b in pairs.values():
            np.testing.assert_array_equal(counts[a],counts[b])
            changed_valid+=int(not np.array_equal(valid_counts[a],valid_counts[b]))
        np.savez_compressed(output/'fullframe-cells.npz',fullframe_event_counts=counts,valid_counts=valid_counts,
            global_indices=np.array([r['global_index'] for r in req['rows']],np.int32),frame_ids=np.array([r['frame_id'] for r in req['rows']]))
        result=dict(status='PASS',host=req['host'],frames=n,native_hashes_matched=n,native_bytes_read=native_bytes,
            full_event_counts_matched=n*4,event_labels_matched=n*4,event_cell_invariant_pairs=len(pairs),
            pairs_with_changed_valid_counts=changed_valid,max_event_count=int(counts.max()),max_valid_count=int(valid_counts.max()),
            event_positives=(counts.sum((1,2))>=3).sum(0).tolist(),array_bytes=int(counts.nbytes+valid_counts.nbytes),
            output_bytes=(output/'fullframe-cells.npz').stat().st_size,seconds=time.perf_counter()-tick,
            backend='NumPy CPU',training_steps=0,model_inference_frames=0,new_captures=0,original_source_mutations=0)
        write(output/'result.json',result)
        write(output/'progress.json',dict(status='PASS',processed=n,total=n,pid=os.getpid()))
        write(output/'receipt.json',dict(status='PASS',host=req['host'],pid=os.getpid(),request_sha256=sha(request),
            manifest_sha256=req['manifest_sha256'],code_sha256=sha(__file__),schema=req['schema'],native_inputs=req['rows'],
            outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},result=result))
        print(json.dumps(result),flush=True)
    except BaseException:
        write(output/'failure.json',dict(status='FAIL',pid=os.getpid(),error=traceback.format_exc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('prepare');s.add_argument('--root',type=Path,required=True);s.add_argument('--task',type=Path,required=True)
    s=sub.add_parser('derive');s.add_argument('--request',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root.resolve(),a.task.resolve())
    else:derive(a.request.resolve(),a.output.resolve())

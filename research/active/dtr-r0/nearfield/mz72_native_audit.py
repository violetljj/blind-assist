"""Independent owner-local audit of the predeclared MZ72 native cases.

parts.json = {shards:[{shard_id,capture,dataset,fullframe,fullframe_receipt}]}.
Paths resolve relative to parts.json. No native arrays enter the return receipt.
--stage canary checks one frozen owner shard/32 selected cases before expansion;
--stage complete (default) checks five owner shards/40 selected cases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from mz72_source_contract import MANIFEST_SHA, admit_spec, read, sha

FROZEN_MANIFEST_SHA='6687a5434a07c0afcdbb8e99a9afdbdc6718e1a298d207d5d150f2a6e2a09233'


def write(p, value):
    p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def selected_shards(frozen,host,stage):
    assert host in ('primary','worker') and stage in ('canary','complete')
    selected={k for k,v in frozen.items() if ('candidate_05' in k)==(host=='primary')
              and (stage=='complete' or v['canary'])}
    assert len(selected)==(1 if stage=='canary' else 5)
    return selected


def run(manifest,specs,parts,output,host,stage='complete'):
    assert not output.exists()
    started=time.perf_counter();assert sha(manifest)==MANIFEST_SHA==FROZEN_MANIFEST_SHA
    m=read(manifest);assert (m['frames'],m['canary_frames'],m['main_frames'],m['native_audit_frames'])==(4096,64,4032,80)
    inputs={str(p):sha(p) for p in (manifest,parts,Path(__file__),Path(__file__).with_name('mz72_source_contract.py'))}
    for name in ('mz72_dataset.py','mz72_fullframe.py'):
        p=Path(__file__).with_name(name);inputs[str(p)]=sha(p)
    frozen={s['shard_id']:s for s in read(manifest)['shards']}
    selected=read(parts)['shards'];assert len(selected)==(1 if stage=='canary' else 5)
    expected_shards=selected_shards(frozen,host,stage)
    assert {s['shard_id'] for s in selected}==expected_shards
    rows=[];pairs=[];native_bytes=rgb_bytes=0;expected_audit_ids=set()
    yy,xx=np.indices((360,640),dtype=np.float64);focal=320/np.tan(np.deg2rad(50.))
    horizontal=(xx-319.5)/focal;vertical=(179.5-yy)/focal
    factor=np.sqrt(1+horizontal**2+vertical**2)
    for entry in selected:
        paths={k:(parts.parent/entry[k]).resolve() for k in ('capture','dataset','fullframe','fullframe_receipt')}
        captured=read(paths['capture']/'source/spec.json');contract=admit_spec(manifest,specs,captured)
        assert contract['shard_id']==entry['shard_id'];inputs.update(contract['inputs'])
        ds=read(paths['dataset']/'receipt.json');fr=read(paths['fullframe_receipt'])
        assert ds['status']==fr['status']=='PASS'
        assert ds['code_sha256']==sha(Path(__file__).with_name('mz72_dataset.py'))
        assert fr['frozen_derivation_sha256']=='5fb19e543e1b6c841a2c61c8a4e4ee4c918985a3883d1c061a2ce673423aeaa3'
        assert ds['manifest_sha256']==fr['manifest_sha256']==MANIFEST_SHA
        assert fr['captured_spec_sha256']==ds['captured_spec_sha256']==sha(paths['capture']/'source/spec.json')
        assert fr['code_sha256']==sha(Path(__file__).with_name('mz72_fullframe.py'))
        hashes={k.replace('\\','/'):v for k,v in ds['package_hashes'].items()}
        metadata=paths['dataset']/'training-source/evaluator/metadata.json'
        assert sha(metadata)==hashes['evaluator/metadata.json']
        assert sha(paths['fullframe'])=={k.replace('\\','/'):v for k,v in fr['outputs'].items()}['derive-v1/fullframe-cells.npz']
        for p in (metadata,paths['fullframe'],paths['fullframe_receipt'],paths['dataset']/'receipt.json'):
            inputs[str(p)]=sha(p)
        records=read(metadata)['records'];assert len(records)==len(captured['cases'])==frozen[entry['shard_id']]['frames']
        previous=None
        with np.load(paths['fullframe'],allow_pickle=False) as full:
            counts=full['fullframe_event_counts'];valid_counts=full['valid_counts']
            np.testing.assert_array_equal(full['frame_ids'],[r['frame_id'] for r in records])
        for i,(row,case) in enumerate(zip(records,captured['cases'])):
            assert row['frame_id']==case['name'] and row['native_audit_sample']==case['native_audit_sample']
            from mz72_source_contract import geometry_metadata
            for key,value in geometry_metadata(case).items():assert row[key]==value
            if not case['native_audit_sample']:continue
            expected_audit_ids.add(case['name'])
            native=paths['capture']/f'evaluator/native/{i:04d}.npy'
            rgb=paths['dataset']/'training-source'/row['rgb']
            assert sha(native)==row['native_sha256'] and sha(rgb)==row['rgb_sha256']
            inputs[str(native)]=row['native_sha256'];inputs[str(rgb)]=row['rgb_sha256']
            native_bytes+=native.stat().st_size;rgb_bytes+=rgb.stat().st_size
            depth=np.load(native,allow_pickle=False);assert depth.shape==(360,640) and depth.dtype==np.float32
            d=depth.astype(np.float64);good=np.isfinite(d)&(d>0)&(d<100)&(d*factor<=4.)
            lateral=d*horizontal;height=1.7+d*vertical;masks=[]
            for front,width,lower,upper in ((.18,.28,.65,1.4),(.13,.18,1.4,1.85)):
                for half in range(2):
                    end=(d<front+1.5) if half==0 else (d<=front+3.)
                    masks.append(good&(d>=front+half*1.5)&end&(abs(lateral)<=width)&(height>=lower)&(height<=upper))
            masks=np.stack(masks,-1)
            # Independent explicit block loops, not the producer's reshape kernel.
            cells=np.empty((45,80,4),np.uint8);known=np.empty((45,80),np.uint8)
            for a in range(45):
                for b in range(80):
                    cells[a,b]=masks[8*a:8*a+8,8*b:8*b+8].sum((0,1))
                    known[a,b]=good[8*a:8*a+8,8*b:8*b+8].sum()
            np.testing.assert_array_equal(cells,counts[i]);np.testing.assert_array_equal(known,valid_counts[i])
            np.testing.assert_array_equal(cells.sum((0,1)),row['event_counts'])
            np.testing.assert_array_equal(cells.sum((0,1))>=3,row['event_truth'])
            with Image.open(rgb) as image:assert image.size==(640,360);image.load()
            if previous is None:
                assert row['support_context']=='unsupported';previous=(row,depth,masks)
            else:
                old,olddepth,oldmasks=previous
                assert old['pair_id']==row['pair_id'] and row['support_context']=='supported'
                np.testing.assert_array_equal(oldmasks,masks);pix=masks.any(-1)
                np.testing.assert_array_equal(olddepth[pix],depth[pix])
                pairs.append(row['pair_id']);previous=None
            rows.append(dict(frame_id=row['frame_id'],shard_id=entry['shard_id'],index=i,
                native_sha256=row['native_sha256'],rgb_sha256=row['rgb_sha256'],status='PASS',
                event_cells_checked=14400,known_cells_checked=3600))
        assert previous is None
    expected=32 if stage=='canary' else 40
    assert len(rows)==len({r['frame_id'] for r in rows})==expected and len(pairs)*2==expected
    assert {r['frame_id'] for r in rows}==expected_audit_ids
    for path,digest in inputs.items():assert sha(path)==digest
    result=dict(status='PASS',host=host,stage=stage,frames=expected,rows=rows,pairs=pairs,
        manifest_sha256=MANIFEST_SHA,fixed_source_budget=4096,fixed_source_audit_frames=80,
        audit_frame_ids_sha256=hashlib.sha256(json.dumps(sorted(expected_audit_ids),separators=(',',':')).encode()).hexdigest(),
        event_cells_checked=expected*14400,known_cells_checked=expected*3600,
        native_bytes_read=native_bytes,rgb_bytes_read=rgb_bytes,seconds=time.perf_counter()-started,
        backend='NumPy CPU independent owner-local native audit',raw_native_transferred=False,
        new_captures=0,model_inference_frames=0,training_steps=0)
    output.mkdir(parents=True);write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',code_sha256=sha(__file__),inputs=inputs,outputs={'result.json':sha(output/'result.json')}))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pairs')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','specs','parts','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--host',choices=('primary','worker'),required=True)
    p.add_argument('--stage',choices=('canary','complete'),default='complete')
    a=p.parse_args();run(a.manifest.resolve(),a.specs.resolve(),a.parts.resolve(),a.output.resolve(),a.host,a.stage)

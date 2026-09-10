"""Independent CPU audit of the64 predeclared MZ55 native/RGB canary rows."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from data_lightweight import CompactSource


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(manifest,packages,output):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    m=read(manifest);expected={}
    for shard in m['shards']:
        if not shard['canary']:continue
        spec=Path(shard['path']);assert sha(spec)==shard['sha256']
        expected.update({c['name']:c for c in read(spec)['cases']})
    assert len(expected)==64
    inputs={str(p):sha(p) for p in (manifest,packages,Path(__file__))}
    rows=[];pair_checks=[];all_ids=[];rgb_bytes=native_bytes=0
    yy,xx=np.indices((360,640),dtype=np.float64);focal=320./np.tan(np.deg2rad(50.))
    horizontal=(xx-319.5)/focal;vertical=(179.5-yy)/focal
    factor=np.sqrt(1+horizontal**2+vertical**2)
    for reference in read(packages)['archives']:
        archive=Path(reference['path']);assert sha(archive)==reference['sha256'];inputs[str(archive)]=reference['sha256']
        with CompactSource(archive) as source:
            def member(path):
                data=source.read_bytes(path)
                assert hashlib.sha256(data).hexdigest()==source.entries[path]['sha256']
                return data
            meta=json.loads(member('evaluator/metadata.json'))
            with np.load(io.BytesIO(member('evaluator/labels.npz')),allow_pickle=False) as z:
                labels,known=z['truth'],z['known']
            previous=None
            for i,row in enumerate(meta['records']):
                frame=row['frame_id'];all_ids.append(frame);case=expected[frame]
                assert row['index']==i and row['native_audit_sample'] and row['canary']
                native=member(f'evaluator/native/{i:04d}.npy');native_bytes+=len(native)
                assert hashlib.sha256(native).hexdigest()==row['native_sha256']
                depth=np.load(io.BytesIO(native),allow_pickle=False)
                assert depth.dtype==np.float32 and depth.shape==(360,640)
                d=depth.astype(np.float64);valid=np.isfinite(d)&(d>0)&(d<100)&(d*factor<=4.)
                lateral=d*horizontal;height=1.7+d*vertical
                masks=[]
                for front,width,lower,upper in ((.18,.28,.65,1.4),(.13,.18,1.4,1.85)):
                    for half in range(2):
                        end=(d<front+1.5) if half==0 else (d<=front+3.)
                        masks.append(valid&(d>=front+half*1.5)&end&(abs(lateral)<=width)&(height>=lower)&(height<=upper))
                masks=np.stack(masks);counts=masks.sum((1,2));truth=counts>=3
                np.testing.assert_array_equal(counts,row['event_counts'])
                np.testing.assert_array_equal(truth,labels[i]);np.testing.assert_array_equal(truth,row['event_truth'])
                np.testing.assert_array_equal(known[i],[row['source_valid']]*4)
                collapsed=[bool(truth[:2].any()),bool(truth[2:].any())]
                assert row['intent_matches']==(collapsed==[bool(x) for x in case['expected_collapsed_relation']])
                rgb=member(row['rgb']);rgb_bytes+=len(rgb)
                assert hashlib.sha256(rgb).hexdigest()==row['rgb_sha256']
                with Image.open(io.BytesIO(rgb)) as image:assert image.size==(640,360);image.load()
                if previous is None:
                    assert row['support_context']=='unsupported';previous=(row,case,depth,masks)
                else:
                    before,bcase,bdepth,bmasks=previous
                    assert before['pair_id']==row['pair_id'] and row['support_context']=='supported'
                    assert bcase['objects']==[x for x in case['objects'] if x['target_part']]
                    np.testing.assert_array_equal(masks,bmasks)
                    pixels=masks.any(0);np.testing.assert_array_equal(depth[pixels],bdepth[pixels])
                    pair_checks.append(dict(pair_id=row['pair_id'],query_mask_pixels=int(masks.size),
                        query_depth_pixels=int(pixels.sum()),masks_exact=True,depth_exact=True,target_dictionary_exact=True))
                    previous=None
                rows.append(dict(frame_id=frame,rgb_sha256=row['rgb_sha256'],native_sha256=row['native_sha256'],
                    actual_counts=counts.tolist(),truth=truth.tolist(),known=known[i].tolist(),intent_matches=row['intent_matches']))
            assert previous is None
    assert len(set(all_ids))==len(all_ids)==64 and set(all_ids)==set(expected) and len(pair_checks)==32
    result=dict(status='PASS',frames=64,native_hashes=64,rgb_hashes=64,rgb_decodes=64,query_counts_exact=256,
        query_labels_exact=256,invariant_pairs=32,rows=rows,pairs=pair_checks,native_bytes_read=native_bytes,rgb_bytes_read=rgb_bytes,
        seconds=time.perf_counter()-started,backend='NumPy CPU; native audit only',model_inference_frames=0,training_steps=0,
        visual_review='SEPARATE_REQUIRED; decoding is not visual admission')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (output/'receipt.json').write_text(json.dumps(dict(status='PASS',inputs=inputs,outputs={'result.json':sha(output/'result.json')}),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pairs')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','packages','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.packages,a.output)

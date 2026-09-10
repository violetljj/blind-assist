"""Controlled input-detail cache using an unchanged frozen visual extractor."""
import argparse
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read, write, sha, load_npz
from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN

CROP = (208,68,432,292)


def dynamic_extract(model, rgb):
    h,w = rgb.shape[-2:]
    shallow = None
    for layer in model.backbone:
        rgb = layer(rgb)
        if tuple(rgb.shape[-2:]) == (h//4,w//4):
            shallow = rgb
    assert shallow is not None and tuple(rgb.shape[-2:]) == ((h+31)//32,(w+31)//32)
    return rgb,shallow


def dense_features(model, rgb, legacy=False):
    normalized = (rgb-model.image_mean)/model.image_std
    deep,shallow = model.extract(normalized) if legacy else dynamic_extract(model,normalized)
    detail = model.detail(shallow)
    deep = F.interpolate(model.deep_projection(deep),size=detail.shape[-2:],mode='bilinear',align_corners=False)
    return torch.cat([deep,detail],1)


def main(root,output):
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    def log(*args):
        text = ' '.join(map(str,args))
        print(text,flush=True)
        with (output/'console.log').open('a',encoding='utf-8') as f: f.write(text+'\n')
    work = root/'artifacts.local/work'
    old,new = work/'mz8-attribution-20260910/cache-v5',work/'mz15-shared-support-20260910/cache-v1'
    run15 = work/'mz15-shared-support-20260910/run-v1'
    inputs,frozen = {},{}
    def bind(path,expected=None):
        digest = sha(path)
        assert expected is None or digest == expected,str(path)
        inputs[str(path)] = digest
        return digest
    cr,nr,rr = (read(p/'receipt.json') for p in [old,new,run15])
    assert all(r['status'] == 'PASS' for r in [cr,nr,rr])
    for p in [old,new,run15]: bind(p/'receipt.json')
    bind(old/'observations.npz',cr['files']['observations.npz'])
    bind(new/'selected.json',nr['files']['selected.json'])
    bind(run15/'batches.npy',rr['outputs']['batches.npy'])
    observations = load_npz(old/'observations.npz')
    batches = np.load(run15/'batches.npy',allow_pickle=False)
    assert batches.shape == (1200,16)
    ids = np.unique(np.concatenate([batches.flatten(),np.flatnonzero(observations['role']=='DEV_ONLY'),np.arange(3500,3700),np.arange(11200,14200)]))
    assert len(ids) == 11562 and ids.min() >= 0 and ids.max() < 14200
    index_path = work/'body-query-5000-20260909/dataset-v1/index.json'
    index = read(index_path)['frames']
    bind(index_path,read(work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json')['source_index_sha256'])
    spec_path = work/'mz6-short-sequence-20260910/capture-v3/evaluator/spec.json'
    spec = read(spec_path); bind(spec_path)
    allrows = []
    for i,row in enumerate(cr['inputs']):
        if i < 3500:
            original = index[int(observations['old_index'][i])]
            assert row['rgb_sha'] == original['rgb_sha256']
            metadata = dict(camera=original['camera'],dataset='old5000')
        else:
            metadata = dict(camera=spec['cases'][i-3500]['camera'],dataset='MZ6',clip=str(observations['clip'][i-3500]))
        allrows.append(dict(row,**metadata,role=str(observations['role'][i]),global_id=i))
    allrows.extend(dict(r,global_id=3700+i) for i,r in enumerate(read(new/'selected.json')))
    rows = [allrows[int(i)] for i in ids]
    assert not any(r['role']=='EVAL_ONLY' for r in rows)
    np.save(output/'ids.npy',ids)
    write(output/'selected.json',rows)
    free = shutil.disk_usage(output).free
    assert free > 10*1024**3,free
    base,decoder = work/'body-query-10000-b-20260909/run-v1',work/'body-query-context-decoder-20260909/run-v1'
    for name,digest in FROZEN.items():
        path = (base if name == 'NEW-step2000.pt' else decoder)/name
        bind(path,digest); frozen[str(path)] = digest
    norm = work/'mz9-source-supervision-20260910/run-v1/normalization.npz'
    bind(norm,read(norm.parent/'receipt.json')['outputs']['normalization.npz'])
    write(output/'start.json',dict(status='STARTED',frames=len(ids),training_steps=0,batch_size=16,
        source_sha256=sha(Path(__file__)),ids_sha256=sha(output/'ids.npy'),inputs=inputs,frozen_hashes=frozen,
        free_bytes=free,crop_xyxy=CROP,input_size=[224,224],output_shape=[len(ids),64,28,28],
        LOW_DETAIL='PIL BOX640x360->256x144; BILINEAR->640x360; crop224',HIGH_DETAIL='Original640x360 crop224',
        scope='Actual frozen MZ15 batch-ID union with consumed DEV and sequence; no fitting or native pixel access'))
    log('START',len(ids),'free_GiB',free/1024**3)
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    maps = {}
    try:
        context = ContextEvidence(base,decoder,work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
        m = context.base
        for name in ['LOW_DETAIL','HIGH_DETAIL']:
            maps[name] = np.lib.format.open_memmap(output/f'dense_{name}.npy',mode='w+',dtype=np.float32,shape=(len(ids),64,28,28))
        elapsed = dict(LOW_DETAIL=0.,HIGH_DETAIL=0.)
        timed_frames = 0
        for begin in range(0,len(rows),16):
            batch = rows[begin:begin+16]
            arms = dict(LOW_DETAIL=[],HIGH_DETAIL=[])
            original_low = []
            for row in batch:
                assert sha(row['rgb']) == row['rgb_sha']
                with Image.open(row['rgb']) as image:
                    assert image.size == (640,360)
                    image = image.convert('RGB')
                    low = image.resize((256,144),Image.Resampling.BOX)
                    original_low.append(np.array(low))
                    arms['LOW_DETAIL'].append(np.array(low.resize((640,360),Image.Resampling.BILINEAR).crop(CROP)))
                    arms['HIGH_DETAIL'].append(np.array(image.crop(CROP)))
            with torch.inference_mode():
                if begin == 0:
                    x = torch.from_numpy(np.stack(original_low)).permute(0,3,1,2).cuda().float()/255
                    dynamic,legacy = dense_features(m,x),dense_features(m,x,legacy=True)
                    error = float((dynamic-legacy).abs().max())
                    assert error <= 1e-4
                    write(output/'parity.json',dict(status='PASS',frames=len(batch),global_ids=ids[:len(batch)].tolist(),
                        max_abs_error=error,within_1e6=error<=1e-6,tolerance=1e-4,shape=list(dynamic.shape)))
                    log('LEGACY_PARITY',error)
                # Alternate arm execution order to reduce systematic timing-order bias.
                order = ['LOW_DETAIL','HIGH_DETAIL'] if (begin//16)%2 == 0 else ['HIGH_DETAIL','LOW_DETAIL']
                for name in order:
                    x = torch.from_numpy(np.stack(arms[name])).permute(0,3,1,2).cuda().float()/255
                    torch.cuda.synchronize(); tick=time.perf_counter()
                    dense = dense_features(m,x)
                    torch.cuda.synchronize(); dt=time.perf_counter()-tick
                    assert tuple(dense.shape) == (len(batch),64,28,28)
                    assert torch.isfinite(dense).all()
                    maps[name][begin:begin+len(batch)] = dense.cpu().numpy()
                    if begin > 0: elapsed[name] += dt
                if begin > 0: timed_frames += len(batch)
            if begin%320 == 0 or begin+len(batch) == len(rows):
                for a in maps.values(): a.flush()
                log('CACHE',begin+len(batch),'/',len(rows))
                write(output/'progress.json',dict(frames=begin+len(batch),total=len(rows),seconds=time.perf_counter()-started))
        for a in maps.values(): a.flush()
        maps.clear()
        torch.cuda.synchronize()
        timing = dict(encoder_seconds=elapsed,timed_frames_per_arm=timed_frames,
            milliseconds_per_frame={k:v/timed_frames*1000 for k,v in elapsed.items()},
            scope='Frozen backbone plus deep_projection/detail, synchronized CUDA; excludes transfer/PIL/cache writes; firstbatch excluded; alternating order')
        write(output/'timing.json',timing)
        log('PASS',len(rows),time.perf_counter()-started)
        write(output/'receipt.json',dict(status='PASS',frames=len(rows),training_steps=0,backend='CUDA',device=torch.cuda.get_device_name(),
            seconds=time.perf_counter()-started,source_sha256=sha(Path(__file__)),inputs=inputs,frozen_hashes=frozen,timing=timing,
            ids_sha256=sha(output/'ids.npy'),selected_sha256=sha(output/'selected.json'),
            files={p.name:sha(p) for p in output.iterdir() if p.is_file()},
            scope='Controlled same FoV/224input two-arm frozen detail cache; no fitting, labels, EVAL pixels, or new capture'))
    except BaseException:
        log(traceback.format_exc())
        write(output/'failure.json',dict(status='FAILED',seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); main(args.root,args.output)

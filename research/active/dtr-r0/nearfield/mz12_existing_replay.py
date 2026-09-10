"""Frozen different-placement Development replay; no fitting or selection."""
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from mz5_ensemble_readout import read, write, sha, load_npz, CompactEnsemble
from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from body_query_range import range_from_counts
from mz9_source_readout import SourceReadout
from mz9_contributors import reconstruct
from multizone64_observation import native_events
from mz11_selective_addition import gate_features, compose
from mz8_train import metrics


def group_exact(pred, truth, rows, key):
    groups={}
    correct=(pred>=0)==truth
    for i,row in enumerate(rows):groups.setdefault(row[key],[]).append(i)
    return dict(correct=sum(bool(correct[ids].all()) for ids in groups.values()),total=len(groups))


def main(root, inventory, output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    work=root/'artifacts.local/work';rows=read(inventory/'selected.json')
    assert len(rows)==3000
    ir=read(inventory/'receipt.json');assert ir['status']=='PASS'
    # Bind the exact preselected input rather than selecting by model outcomes.
    selection_sha=sha(inventory/'selected.json')
    write(output/'start.json',dict(status='STARTED',frames=3000,selection_sha256=selection_sha,
        inventory_receipt_sha256=sha(inventory/'receipt.json'),protocol_sha256=sha(Path(__file__).with_name('MZ12_EXISTING_DATA_PROTOCOL_20260910.md')),
        source_sha256=sha(Path(__file__)),training_steps=0))
    torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    base=work/'body-query-10000-b-20260909/run-v1';decoder=work/'body-query-context-decoder-20260909/run-v1'
    for name,digest in FROZEN.items():assert sha((base if name=='NEW-step2000.pt' else decoder)/name)==digest
    context=ContextEvidence(base,decoder,work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
    m=context.base
    checkpoint=work/'mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint)=='ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    fixed=CompactEnsemble.from_checkpoint(checkpoint).cuda()
    run9=work/'mz9-source-supervision-20260910/run-v1';run11=work/'mz11-selective-addition-20260910/run-v1'
    hashes={}
    for run,names in [(run9,['SOURCE_RGB.pt','MZ5_ADAPT.pt','normalization.npz','result.json']),
                      (run11,['gate.pt','threshold.npy'])]:
        receipt=read(run/'receipt.json')
        for name in names:
            assert sha(run/name)==receipt['outputs'][name];hashes[str(run/name)]=sha(run/name)
    source=SourceReadout().cuda().eval();source.load_state_dict(torch.load(run9/'SOURCE_RGB.pt',weights_only=True))
    adapted=CompactEnsemble.from_checkpoint(run9/'MZ5_ADAPT.pt').cuda()
    n=load_npz(run9/'normalization.npz');prior=read(run9/'result.json')
    st=np.array(prior['dev']['SOURCE_RGB']['threshold']);at=np.array(prior['dev']['MZ5_ADAPT']['threshold'])
    gate=torch.load(run11/'gate.pt',weights_only=True);weight=gate['weight'].cuda();bias=gate['bias'].cuda()
    threshold=np.load(run11/'threshold.npy',allow_pickle=False)

    def extract(batch):
        images=[];depths=[]
        for row in batch:
            assert sha(row['rgb'])==row['rgb_sha'] and sha(row['native'])==row['native_sha']
            assert abs(row['camera']['pitch'])<1e-6 and abs(row['camera']['roll'])<1e-6
            with Image.open(row['rgb']) as im:
                assert im.size==(640,360)
                images.append(np.array(im.convert('RGB').resize((256,144),Image.Resampling.BOX)))
            depths.append(np.load(row['native'],allow_pickle=False))
        x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
        depth=torch.from_numpy(np.stack(depths)).cuda()
        with torch.inference_mode():
            deep,shallow=m.extract((x-m.image_mean)/m.image_std)
            deep=F.interpolate(m.deep_projection(deep),size=(18,32),mode='bilinear',align_corners=False)
            dense=torch.cat([deep,m.detail(shallow)],1)
            sampled=(dense.flatten(2)@m.query_projection.T).transpose(1,2).reshape(len(x),12,27,64)
            mask=m.query_valid[None,:,:,None]
            pooled=(sampled*mask).sum(2)/mask.sum(2).clamp_min(1)
            normalized=(pooled-context.feature_mean)/context.feature_std
            visual=torch.cat([normalized.flatten(1),range_from_counts(context.decoder(normalized)).sigmoid().flatten(1)],1)
            packet=reconstruct(depth)
            ranges=torch.nan_to_num(packet['range_m']).float();valid=packet['valid']
            tof=torch.cat([ranges.flatten(1)/4,valid.flatten(1).float()],1)
            truth=native_events(depth,crop=False)
            unknown=(~(torch.isfinite(depth)&(depth>0)&(depth<100))).sum((1,2))
        return dense,visual,ranges,valid,tof,truth,unknown

    oldrows=read(work/'body-query-5000-20260909/dataset-v1/index.json')['frames'][:16]
    parityrows=[dict(rgb=r['rgb_file'],rgb_sha=r['rgb_sha256'],native=r['native_file'],native_sha=r['native_sha256'],camera=r['camera']) for r in oldrows]
    dense,visual,ranges,valid,tof,truth,unknown=extract(parityrows)
    cache=load_npz(work/'mz1-tiny-fusion-20260910/cache-v1/features.npz')
    visual_error=float(np.abs(visual.cpu().numpy()-cache['visual'][:16]).max())
    range_error=float(np.abs(tof.cpu().numpy()[:,:128]*4-cache['tof'][:16,:128]*4).max())
    assert visual_error<1e-4 and range_error<1e-6
    np.testing.assert_array_equal(valid.cpu().numpy().reshape(16,128),cache['tof'][:16,128:]>0)
    np.testing.assert_array_equal(truth['events'].cpu().numpy(),cache['truth'][:16])
    with torch.inference_mode():
        expected=fixed(torch.from_numpy(cache['visual'][:16]).cuda(),torch.from_numpy(cache['tof'][:16]).cuda()).cpu().numpy()
        observed=fixed(visual,tof).cpu().numpy()
    np.testing.assert_array_equal(observed>=0,expected>=0)
    write(output/'parity.json',dict(status='PASS',frames=16,visual_max_error=visual_error,range_max_error_m=range_error))
    arrays={}
    for begin in range(0,len(rows),16):
        batch=rows[begin:begin+16]
        dense,visual,ranges,valid,tof,truth,unknown=extract(batch)
        with torch.inference_mode():
            b=fixed(visual,tof).cpu().numpy();ad=adapted(visual,tof).cpu().numpy()-at
            normalized=(dense-torch.from_numpy(n['mean']).cuda())/torch.from_numpy(n['std']).cuda()
            out=source.inspect(normalized,ranges,valid);a=out['support'].cpu().numpy()
            s=np.where(a,out['logits'].cpu().numpy()-st,-1e6)
            e=out['eligible'];count=e.sum((1,2,3)).cpu().numpy()
            zones=e.any(3).any(2).sum(1).cpu().numpy();returns=e.any(3).sum((1,2)).cpu().numpy()
            features=gate_features(np.where(a,s,0),count,zones,returns)
            score=((torch.from_numpy(features).cuda()*weight).sum(-1)+bias).cpu().numpy()
            candidate,accepted=compose(b,s,a,score,threshold)
        values=dict(MZ5=b,ADAPTED=ad,SOURCE=s,MZ11=candidate,truth=truth['events'].cpu().numpy(),
            native_counts=truth['counts'].cpu().numpy(),unknown_pixels=unknown.cpu().numpy(),
            no_packet=(~valid.flatten(1).any(1)).cpu().numpy(),available=a,accepted=accepted,gate_score=score,
            candidate_count=count,candidate_zones=zones,candidate_returns=returns)
        for key,value in values.items():arrays.setdefault(key,[]).append(value)
        if begin%320==0:
            print('REPLAY',begin+len(batch),'/3000',flush=True)
            write(output/'progress.json',dict(frames=begin+len(batch),total=3000))
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    arrays['dataset']=np.array([r['dataset'] for r in rows])
    result={}
    for dataset in dict.fromkeys(arrays['dataset'].tolist()):
        ids=np.flatnonzero(arrays['dataset']==dataset);selected=[rows[i] for i in ids];t=arrays['truth'][ids]
        cohort={}
        for arm in ['MZ5','ADAPTED','SOURCE','MZ11']:
            z=arrays[arm][ids];p=z>=0
            cohort[arm]=dict(metrics=metrics(z,t),groups=group_exact(z,t,selected,'group'),sites=group_exact(z,t,selected,'site'),
                wrong_far_near_only=[int((p[:,q+1]&t[:,q]&~t[:,q+1]).sum()) for q in [0,2]],
                wrong_near_far_only=[int((p[:,q]&t[:,q+1]&~t[:,q]).sum()) for q in [0,2]],
                families={f:metrics(z[[r['family']==f for r in selected]],t[[r['family']==f for r in selected]]) for f in sorted({r['family'] for r in selected})})
        b=arrays['MZ5'][ids]>=0;c=arrays['MZ11'][ids]>=0
        cohort['changes']=dict(added_tp=(c&~b&t).sum(0).tolist(),added_fp=(c&~b&~t).sum(0).tolist(),lost_positive_bits=int((b&~c).sum()))
        cohort['observation']=dict(no_packet=int(arrays['no_packet'][ids].sum()),unknown_pixel_total=int(arrays['unknown_pixels'][ids].sum()))
        result[dataset]=cohort
    result['favorable_transfer']=all(not any(c['changes']['added_fp']) and c['changes']['lost_positive_bits']==0 for c in result.values()) and sum(sum(c['changes']['added_tp'][1::2]) for c in result.values())>0
    np.savez_compressed(output/'predictions.npz',**arrays)
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',frames=3000,training_steps=0,threshold_changes=0,
        backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
        frozen_hashes=hashes,context_hashes=FROZEN,selection_sha256=selection_sha,
        outputs={p.name:sha(p) for p in output.iterdir()}))
    print('RESULT', {k:({a:v[a]['metrics'] for a in ['MZ5','ADAPTED','SOURCE','MZ11']} if isinstance(v,dict) else v) for k,v in result.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','inventory','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.root,a.inventory,a.output)

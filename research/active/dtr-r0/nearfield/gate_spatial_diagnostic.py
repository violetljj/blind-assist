"""Consumed G10 TEST-only spatial perturbations; no fitting or fresh claims."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
from diversity_model import DiversityModel
from diversity_evaluate import evaluation_scopes,score_scopes
from factorial_evaluate import primary_thresholds
from grounding_evaluate import read,sha
from evaluate_whisker import normalize_predictions
from whisker_model import IMAGE_SIZE

MODES=('original_map','constant_mean','shuffle_101','shuffle_103','shuffle_107')
SEEDS=(17,29,43)


def shuffle_indices(sample_ids,permutation_seed):
    indices=[]
    for sid in sample_ids:
        row=[]
        for head in range(2):
            seed=int.from_bytes(hashlib.sha256(f'{sid}/{head}/{permutation_seed}'.encode()).digest()[:8],'little')
            row.append(np.random.default_rng(seed).permutation(18*32))
        indices.append(row)
    return np.asarray(indices,np.int64)


def transform_gate(gate,mode,indices=None):
    if mode=='original_map':return gate
    if mode=='constant_mean':return gate.mean((-2,-1),keepdim=True).expand_as(gate)
    if mode.startswith('shuffle_'):
        return gate.flatten(2).gather(2,indices).reshape_as(gate)
    raise ValueError(mode)


def gated_scores(model,features,gate):
    value=features[:,None]*gate[:,:,None];b,heads,c,h,w=value.shape
    pooled=model.near[0](value.reshape(b*heads,c,h,w)).reshape(b,heads,-1)
    return ((pooled*model.near[2].weight[None]).sum(-1)+model.near[2].bias).sigmoid()


@torch.inference_mode()
def run(capture,learned,threshold_source,output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1);start=time.perf_counter()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    dataset=read(capture/'model/dataset.json');spec=read(capture/'evaluator/spec.json')
    _,scopes=evaluation_scopes(dataset,spec)
    # Preserve the original G10 inference batch shapes, including its final32.
    # Only TEST rows enter metrics and perturbation summaries below.
    samples=dataset['samples']
    ids=[s['sample_id'] for s in samples];frames=dataset['frames'];arrays=[];rgb_hashes={}
    for sample in samples:
        frame=frames[sample['frame_indices'][0]];path=capture/'model'/frame['rgb_path']
        rgb_hashes[sample['sample_id']]=sha(path)
        with Image.open(path) as image:arrays.append(np.asarray(image.convert('RGB').resize((IMAGE_SIZE[1],IMAGE_SIZE[0]),Image.Resampling.BOX)).copy())
    rgb=torch.from_numpy(np.stack(arrays)).cuda().permute(0,3,1,2).float().div_(255.)
    permutations={mode:torch.as_tensor(shuffle_indices(ids,int(mode.split('_')[1])),device='cuda') for mode in MODES if mode.startswith('shuffle_')}
    receipt=read(learned/'receipt.json');records={(r['arm'],r['seed']):r for r in receipt['records']}
    rows={};hashes={};gate_checks=[]
    for seed in SEEDS:
        checkpoint=learned/f'expanded_region_seed{seed}.pt';hashes[str(seed)]=sha(checkpoint)
        assert hashes[str(seed)]==records['expanded_region',seed]['checkpoint_sha256']
        model=DiversityModel(True).cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.eval()
        chunks={mode:[] for mode in MODES}
        for first in range(0,len(rgb),64):
            features=model.spatial(rgb[first:first+64]);gate=model.support(features).sigmoid()
            for mode in MODES:
                changed=transform_gate(gate,mode,permutations[mode][first:first+64] if mode in permutations else None)
                torch.testing.assert_close(changed.mean((-2,-1)),gate.mean((-2,-1)),atol=1e-6,rtol=1e-6)
                if mode in permutations:torch.testing.assert_close(changed.flatten(2).sort(-1).values,gate.flatten(2).sort(-1).values,atol=0,rtol=0)
                near=gated_scores(model,features,changed).cpu().numpy();chunks[mode].append(np.concatenate((near,-np.ones_like(near)),1))
        for mode in MODES:rows[mode,str(seed),'normal']={sid:value for sid,value in zip(ids,np.concatenate(chunks[mode]))}
        gate_checks.append(dict(seed=seed,mean_preserved=True,permutation_values_exact=True))
    for mode in MODES:rows[mode,'ensemble','normal']={sid:np.mean([rows[mode,str(seed),'normal'][sid] for seed in SEEDS],axis=0) for sid in ids}
    selected=primary_thresholds(read(threshold_source),rows)
    targets=read(capture/'evaluator/labels.json')['targets']
    evaluations=score_scopes(rows,selected,scopes,targets)
    reference=normalize_predictions(read(learned/'predictions.json'),dataset['samples'])
    reproduction={};changes={}
    test_ids=[s['sample_id'] for s in scopes['all_TEST'][0]]
    for seed in (*map(str,SEEDS),'ensemble'):
        original=np.asarray([rows['original_map',seed,'normal'][sid][:2] for sid in ids])
        previous=np.asarray([reference['expanded_region',seed,'normal'][sid][:2] for sid in ids])
        limits=np.asarray([t['value'] for t in selected['original_map',seed,'normal'][:2]])
        reproduction[seed]=dict(max_abs=float(np.abs(original-previous).max()),alert_flips=int(((original>=limits)!=(previous>=limits)).sum()))
        if not np.allclose(original,previous,atol=1e-6,rtol=1e-5) or reproduction[seed]['alert_flips']:
            output.parent.mkdir(parents=True,exist_ok=True)
            (output.parent/'diagnostic-reproduction-failure.json').write_text(json.dumps(reproduction,indent=2),encoding='utf-8')
        assert np.allclose(original,previous,atol=1e-6,rtol=1e-5) and reproduction[seed]['alert_flips']==0
        for mode in MODES[1:]:
            original_test=np.asarray([rows['original_map',seed,'normal'][sid][:2] for sid in test_ids])
            changed=np.asarray([rows[mode,seed,'normal'][sid][:2] for sid in test_ids]);delta=changed-original_test
            changes[mode+'/'+seed]=dict(mean_absolute_probability_delta=np.abs(delta).mean(0).tolist(),
                alert_flips_per_head=((original_test>=limits)!=(changed>=limits)).sum(0).tolist(),
                samples=[dict(sample_id=sid,probability_delta=delta[i].tolist(),original=original_test[i].tolist(),changed=changed[i].tolist()) for i,sid in enumerate(test_ids)])
    torch.cuda.synchronize();output.mkdir(parents=True)
    result=dict(schema='nf-g11-consumed-spatial-diagnostic-v1',evaluations=evaluations,changes=changes,reproduction=reproduction,
        gate_checks=gate_checks,rgb_sha256=rgb_hashes,checkpoint_sha256=hashes,source_sha256=sha(Path(__file__)),
        input_sha256={n:sha(capture/n) for n in ('model/dataset.json','evaluator/spec.json','evaluator/labels.json')},threshold_sha256=sha(threshold_source),
        seconds=time.perf_counter()-start,backend='CUDA',device=torch.cuda.get_device_name(),
        scope='Consumed G10 TEST diagnostic, no fits. Constant preserves per-image/head mean; permutations preserve values, not spatial correlation. Sensitivity is not faithful localization or fresh generalization.')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(output=str(output),seconds=result['seconds'],reproduction=reproduction)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('capture','learned','threshold-source','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.capture,a.learned,a.threshold_source,a.output)

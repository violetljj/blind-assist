"""Native acceptance and frozen RGB inference for strict forward-distance pairs."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
from body_query_data import read,write,sha,fresh_output
from body_query_collection_labels import verify_capture,floor_acceptance
from body_query_distance_pairs_spec import validate_pair
from body_query_labels import labels
from body_query_model import BodyQueryModel
from body_query_range import range_from_counts
from city_dev_selection import apply_thresholds
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'tools'))
from research_backend import torch_observation

REGIONS=('big01','big03','big05','big06')
CHECKPOINT='c7aef143bcc7ac464097e95219dc5239523df7ce5716fc95926f37592f94e776'


def prepare(root):
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    out=fresh_output(root/'prepared-v1');records=[];images=[];counts=[];near=[];raw=[]
    inputs={};source_manifest=read(root/'specs-v1/manifest.json')
    for region in REGIONS:
        capture=root/'capture-v1'/region
        spec,world,binding,probes=verify_capture(capture);inputs.update(binding)
        expected=next(v for v in source_manifest['outputs'] if v['region_id']==region)
        assert sha(capture/'source/spec.json')==expected['sha256']
        canvas=Image.new('RGB',(640,8*204),'white');draw=ImageDraw.Draw(canvas)
        for i,c in enumerate(spec['cases']):
            if i%2==0:validate_pair(c,spec['cases'][i+1])
            depth=capture/f'evaluator/native/{i:04d}.npy';rgb=capture/f'model/sample/{i:04d}.png';w=world['rows'][i]
            assert sha(depth)==w['native_sha256'] and sha(rgb)==w['rgb_sha256']
            gt=labels(torch.from_numpy(np.load(depth,allow_pickle=False)).cuda(),c['camera'],c['floor_z_m'])
            assert gt['near'].tolist()==w['body_head_visible_targets']
            support=capture/w['mask_path'];assert sha(support)==w['mask_sha256']
            assert np.array_equal(gt['support'].cpu().numpy(),np.load(support,allow_pickle=False))
            count=gt['counts'].cpu().numpy();events=count.reshape(2,2,3).sum(-1)>=3
            desired=np.array([[False,False],[c['declared_range']=='near',c['declared_range']=='far']])
            floor=floor_acceptance(c,probes);accepted=bool(np.array_equal(events,desired) and floor['accepted'])
            target_dims=[o['size_m'] for o in c['objects'] if o.get('target_part')]
            rec=dict(name=c['name'],pair_id=c['pair_id'],region=region,site=c['site_id'],family=c['condition']['family'],
                endpoint=c['declared_range'],distance_m=c['condition']['distance_m'],target_dimensions_m=target_dims,
                source_case=c['source_case_name'],native_events=events.tolist(),accepted=accepted,floor=floor,
                rgb_path=str(rgb.resolve()),rgb_sha256=sha(rgb),native_sha256=sha(depth),unknown_pixels=int((gt['support'][0]<0).sum()))
            records.append(rec);near.append(gt['near'].cpu().numpy());counts.append(count);raw.append(gt['raw_counts'].cpu().numpy())
            with Image.open(rgb) as im:
                images.append(np.asarray(im.convert('RGB').resize((256,144),Image.Resampling.BOX)))
                preview=im.convert('RGB').resize((320,180),Image.Resampling.BOX)
                x=(i%2)*320;y=(i//2)*204;canvas.paste(preview,(x,y+24))
                draw.text((x+4,y+4),f"{i//2+1} {rec['family']} {rec['endpoint']} {rec['distance_m']}m",fill='black')
        canvas.save(out/f'{region}-all-frames.jpg',quality=95)
    assert len(records)==64
    np.save(out/'rgb.npy',np.asarray(images),allow_pickle=False)
    np.savez_compressed(out/'truth.npz',near=near,counts=counts,raw_counts=raw)
    write(out/'records.json',records)
    write(out/'acceptance.json',dict(status='PASS' if all(r['accepted'] for r in records) else 'PARTIAL',
        frames=len(records),accepted_frames=sum(r['accepted'] for r in records),records_sha256=sha(out/'records.json'),
        rgb_sha256=sha(out/'rgb.npy'),truth_sha256=sha(out/'truth.npz'),source_bindings=inputs,
        generator_sha256=sha(root/'specs-v1/manifest.json'),code_sha256=sha(Path(__file__)),
        backend='CUDA native geometry only; no learned inference',model_inference_frames=0))
    print('NATIVE PREPARED',sum(r['accepted'] for r in records),'/64',flush=True)


def infer(root):
    torch.set_num_threads(1)
    prepared=root/'prepared-v1';admit=read(prepared/'acceptance.json');review=read(prepared/'visual-review.json')
    assert review['reviewed_frames']==64 and review['capture_defects']==[]
    for key,name in (('records_sha256','records.json'),('rgb_sha256','rgb.npy'),('truth_sha256','truth.npz')):
        assert sha(prepared/name)==admit[key]
    records=read(prepared/'records.json');out=fresh_output(root/'inference-v1')
    baseline=Path('artifacts.local/work/body-query-expanded-b-20260909/run-v1')
    checkpoint=baseline/'NEW-step2000.pt';assert sha(checkpoint)==CHECKPOINT
    dependencies=read(baseline/'protocol.json')['dependency_sha256']
    for name in ('body_query_model.py','decoupled_model.py','representation_model.py'):
        assert sha(Path(__file__).with_name(name))==dependencies[name]
    selection=read(baseline/'selection.json');thresholds=selection['NEW']['thresholds']
    assert selection['NEW']['checkpoint_sha256']==CHECKPOINT
    model=BodyQueryModel(Path('artifacts.local/work/body-query-v1-20260908/model-inputs/pretrained'),'B').cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    rgb=np.load(prepared/'rgb.npy',allow_pickle=False);pred=[];distributions=[]
    with torch.inference_mode():
        for begin in range(0,len(rgb),32):
            x=torch.from_numpy(rgb[begin:begin+32].copy()).cuda().permute(0,3,1,2).float()/255.
            n,s,c=model(x);pred.append(n.sigmoid().cpu().numpy());distributions.append(c.softmax(-1).cpu().numpy())
        device=asdict(torch_observation(model=model,output=(n,s,c)))
    n=np.concatenate(pred);c=np.concatenate(distributions)
    score=range_from_counts(torch.from_numpy(np.log(np.clip(c.astype(np.float64),1e-300,1)))).sigmoid().numpy()
    alert=apply_thresholds(n,thresholds);pairs=[]
    for pair_id in dict.fromkeys(r['pair_id'] for r in records):
        ids=[i for i,r in enumerate(records) if r['pair_id']==pair_id]
        assert len(ids)==2 and [records[i]['endpoint'] for i in ids]==['near','far']
        i,j=ids;delta=float(score[j,1,1]-score[i,1,1]);r=records[i]
        pairs.append(dict(pair_id=pair_id,accepted=all(records[k]['accepted'] for k in ids),region=r['region'],site=r['site'],
            family=r['family'],dimensions=r['target_dimensions_m'],distances=[records[k]['distance_m'] for k in ids],
            HEAD_scores=n[ids,1].tolist(),BODY_scores=n[ids,0].tolist(),HEAD_alerts=alert[ids,1].tolist(),
            BODY_alerts=alert[ids,0].tolist(),both_HEAD=bool(alert[ids,1].all()),far_scores=score[ids,1,1].tolist(),
            near_scores=score[ids,1,0].tolist(),delta=delta,positive_raw=delta>0,
            direction='far_higher' if delta>1e-6 else ('near_higher' if delta < -1e-6 else 'tie')))
    def summarize(rows):
        good=[r for r in rows if r['accepted']]
        return dict(declared=len(rows),accepted=len(good),far_higher=sum(r['direction']=='far_higher' for r in good),
            near_higher=sum(r['direction']=='near_higher' for r in good),ties=sum(r['direction']=='tie' for r in good),
            raw_positive=sum(r['positive_raw'] for r in good),both_HEAD=sum(r['both_HEAD'] for r in good),
            HEAD_TP=sum(sum(r['HEAD_alerts']) for r in good),BODY_FP=sum(sum(r['BODY_alerts']) for r in good),
            mean_delta=float(np.mean([r['delta'] for r in good])) if good else None,
            median_delta=float(np.median([r['delta'] for r in good])) if good else None)
    summary=summarize(pairs);slices={key:{v:summarize([r for r in pairs if r[key]==v]) for v in dict.fromkeys(r[key] for r in pairs)} for key in ('family','region','site')}
    sizes={}
    for p in pairs:
        key=p['family']+':'+str(p['dimensions']);sizes.setdefault(key,[]).append(p)
    slices['dimensions']={key:summarize(rows) for key,rows in sizes.items()}
    np.savez_compressed(out/'predictions.npz',near=n,counts=c,range_scores=score)
    write(out/'result.json',dict(summary=summary,slices=slices,pairs=pairs,thresholds=thresholds,
        scope='Controlled existing assemblies and sites, rigid forward translation, no HEAD-negative denominator'))
    write(out/'receipt.json',dict(status='PASS',training_steps=0,frames=64,checkpoint_sha256=CHECKPOINT,
        baseline_selection_sha256=sha(baseline/'selection.json'),result_sha256=sha(out/'result.json'),
        predictions_sha256=sha(out/'predictions.npz'),acceptance_sha256=sha(prepared/'acceptance.json'),
        review_sha256=sha(prepared/'visual-review.json'),code_sha256=sha(Path(__file__)),device=device))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','infer']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();(prepare if a.mode=='prepare' else infer)(a.root)

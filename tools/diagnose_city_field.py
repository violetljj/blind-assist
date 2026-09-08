"""One frozen RGB inference pass on an immutable field batch; no training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import evaluate_city_native_route as frozen


def confusion(probability,truth,threshold):
    alert=np.asarray(probability)>=threshold;truth=np.asarray(truth)
    return dict(TP=int((alert&(truth==1)).sum()),FN=int((~alert&(truth==1)).sum()),
        FP=int((alert&(truth==0)).sum()),TN=int((~alert&(truth==0)).sum()),UNKNOWN=int((truth==-1).sum()))


def target_decision(probability,support,positive,threshold):
    positive=np.asarray(positive)==1
    if not positive.any():return None
    alarm=bool(probability>=threshold);prediction=np.asarray(support)>=.5
    overlap=bool((prediction&positive).any())
    return dict(alert=alarm,overlap=overlap,joint=alarm and overlap,
        peak_hit=bool(positive.flat[int(np.asarray(support).argmax())]),
        target_grid_cells=int(positive.sum()),predicted_grid_cells=int(prediction.sum()),
        outcome='ALERT_MISS' if not alarm else 'LOCALIZATION_MISS' if not overlap else 'JOINT_HIT')


def truly_clear(variant,truth):
    return variant=='clear' and list(truth)==[0,0]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def gallery(out,rows,maps,masks):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['method'],row['category'],row['head'])].append(row)
    files=[]
    for (method,category,head),items in grouped.items():
        sheet=Image.new('RGB',(960,240*((len(items)+2)//3)),'#161616');draw=ImageDraw.Draw(sheet)
        for n,row in enumerate(items):
            x=n%3*320;y=n//3*240
            with Image.open(row['rgb_path']) as source:im=source.convert('RGB').resize((320,180))
            key=(row['region'],row['sample_index'],row['target_id'],row['head'])
            gt=masks[key];pred=maps[(row['method'],row['global_index'],row['head'])]
            layer=np.zeros((18,32,4),np.uint8);layer[pred>=.5]=[255,40,40,70];layer[gt==1]=[20,255,40,140]
            overlay=Image.fromarray(layer).resize((320,180),Image.Resampling.NEAREST)
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB');sheet.paste(im,(x,y+60))
            draw.text((x+4,y+3),row['route_id'],fill='white')
            draw.text((x+4,y+19),f"p={row['probability']:.3f} cutoff={row['threshold']:.3f} {row['outcome']}",fill='white')
            draw.text((x+4,y+35),f"overlap={int(row['overlap'])} peak={int(row['peak_hit'])} native-risk={int(row['native_competing_alert'])}",fill='white')
        path=out/'gallery'/f'{method}-{category}-{head}.jpg';path.parent.mkdir(exist_ok=True);sheet.save(path,quality=92)
        files.append(str(path))
    return files


def run(collection,out):
    collection=collection.resolve(strict=True);out=out.resolve()
    if out.exists() or not out.is_relative_to(frozen.ARTIFACTS.resolve()) or out==frozen.ARTIFACTS.resolve():
        raise ValueError('Require fresh output inside canonical artifacts')
    receipt=frozen.read(collection/'receipt.json')
    if not receipt.get('completed') or receipt['status'] in ('FAIL','INCOMPLETE'):raise ValueError('Collection not accepted')
    plan=frozen.read(collection/'plan.json');config=frozen.configuration()
    paths=[];index=[];regions={}
    for region in plan['regions']:
        name=region['region_id'];capture=collection/name/'capture'
        dataset=frozen.read(capture/'model/dataset.json');start=len(paths)
        for frame in dataset['frames']:
            path=frozen.within(capture/'model',frame['rgb_path']);paths.append(path)
            index.append(dict(region=name,sample_index=frame['sample_index'],rgb_path=str(path),sha256=frozen.sha(path)))
        regions[name]=list(range(start,len(paths)))
    if len(paths)!=108:raise ValueError('This fixed diagnostic requires exactly the admitted108 frames')
    out.mkdir(parents=True)
    write(out/'protocol.json',dict(schema='city-field-diagnostic-v1',collection=str(collection),
        plan_sha256=frozen.sha(collection/'plan.json'),collection_receipt_sha256=frozen.sha(collection/'receipt.json'),
        methods=config,inputs=index,source_sha256={p.name:frozen.sha(p) for p in
            [Path(__file__),Path(frozen.__file__)]+[frozen.SOURCE/n for n in ('diversity_model.py','decoupled_model.py','whisker_model.py','representation_model.py')]},
        scope='108 static curated Development frames; same City assets; zero fitting/threshold selection; no timing or depth-causality claim'))
    predictions,timing=frozen.infer(paths,out,config)
    # Freeze predictions before opening any evaluator masks or per-case labels.
    write(out/'prediction-receipt.json',dict(status='PASS',sha256=frozen.sha(out/'predictions.npz'),execution=timing))
    truths=[];metadata=[];target_rows=[];overlay_maps={};overlay_masks={};label_hashes={}
    for region in plan['regions']:
        name=region['region_id'];positions=regions[name];capture=collection/name/'capture'
        spec=frozen.read(capture/'source/spec.json');lp=collection/name/'labels/native-route-labels.json'
        ids=[index[i]['sample_index'] for i in positions]
        near,_,label_info=frozen.labels(lp,ids);label_hashes[name]=label_info
        truths.extend(near.tolist())
        bundles=[frozen.read(p) for p in (collection/name).glob('*-bundle.json')]
        frames={f['sample_index']:f for b in bundles for f in b['frames']}
        clear_by_pair={c['pair_id']:near[j] for j,c in enumerate(spec['cases']) if c.get('variant')=='clear'}
        for j,g in enumerate(positions):
            case=spec['cases'][j];frame=frames[ids[j]]
            metadata.append(dict(**index[g],route_id=case['route_id'],split=case['split'],variant=case['variant'],
                floor_review=frame['floor_probe']['status']!='CONSISTENT',truth=near[j].tolist()))
        for target in frozen.read(lp)['targets']:
            mask=frozen.pooled(np.load(frozen.within(lp.parent,target['masks']),allow_pickle=False))
            for j,g in enumerate(positions):
                binding=next((t for t in frames[ids[j]]['targets'] if t['target_id']==target['target_id']),None)
                if not binding or binding['label_status']!='EVALUABLE':continue
                case=spec['cases'][j]
                for h,head in enumerate(('BODY','HEAD')):
                    if not (mask[j,h]==1).any():continue
                    for method in frozen.METHODS:
                        threshold=config[method]['near_thresholds'][h]['value'];prob=float(predictions[method+'_near'][g,h])
                        support=predictions[method+'_support'][g,h];decision=target_decision(prob,support,mask[j,h],threshold)
                        row=dict(method=method,region=name,split=region['split'],route_id=case['route_id'],sample_index=ids[j],global_index=g,
                            target_id=target['target_id'],category=target['category'],head=head,rgb_path=str(paths[g]),
                            probability=prob,threshold=threshold,native_competing_alert=bool(clear_by_pair[case['pair_id']][h]==1),**decision)
                        target_rows.append(row);overlay_maps[(method,g,head)]=support
                    overlay_masks[(name,ids[j],target['target_id'],head)]=mask[j,h]
    truth=np.asarray(truths);summary={};per_frame=[]
    for method in frozen.METHODS:
        cut=np.array([t['value'] for t in config[method]['near_thresholds']]);prob=predictions[method+'_near'];alerts=prob>=cut
        groups={}
        for selector in ('all','floor_consistent','train','dev','test'):
            selected=np.array([selector=='all' or selector=='floor_consistent' and not m['floor_review'] or m['split']==selector for m in metadata])
            groups[selector]={head:confusion(prob[selected,h],truth[selected,h],cut[h]) for h,head in enumerate(('BODY','HEAD'))}
        clear=np.array([truly_clear(m['variant'],m['truth']) for m in metadata]);fixture_clear=np.array([m['variant']=='clear' for m in metadata])
        counts={}
        for row in [r for r in target_rows if r['method']==method]:
            key=row['category']+'/'+row['head'];c=counts.setdefault(key,Counter())
            c.update(opportunities=1,alert_misses=int(not row['alert']),overlap_misses=int(not row['overlap']),joint_hits=int(row['joint']),peak_hits=int(row['peak_hit']),native_competing=int(row['native_competing_alert']))
        summary[method]=dict(scene=groups,targets={k:dict(v) for k,v in counts.items()},
            clear_controls=dict(frames=int(clear.sum()),false_alert_frames=int(alerts[clear].any(axis=1).sum()),
                BODY_FP=int(alerts[clear,0].sum()),HEAD_FP=int(alerts[clear,1].sum()),native_obstructed_fixture_controls=int((fixture_clear&~clear).sum())))
        per_frame.extend(dict(method=method,**m,probabilities=prob[i].tolist(),alerts=alerts[i].tolist()) for i,m in enumerate(metadata))
    galleries=gallery(out,target_rows,overlay_maps,overlay_masks)
    result=dict(schema='city-field-diagnostic-result-v1',frames=108,methods=summary,inference=timing,
        floor_review_frames=[m for m in metadata if m['floor_review']],label_provenance=label_hashes,
        gallery=galleries,gallery_legend='Green: verified target in20x20 pooled cells; red: predicted support>=0.5. Overlap alone does not prove selective localization.',
        diagnosis_limits=['Direct RGB models: monocular-depth error is not identifiable from this experiment.',
            'All target rows require independently EVALUABLE visible support; no unlabeled/occluded target is counted as a miss.',
            'A query alert may arise from native competing obstacles; joint overlap and peak hits are separate diagnostics.'])
    write(out/'per-frame.json',per_frame);write(out/'target-cases.json',target_rows);write(out/'result.json',result)
    write(out/'receipt.json',dict(status='PASS',result_sha256=frozen.sha(out/'result.json'),predictions_sha256=frozen.sha(out/'predictions.npz'),frames=108,optimizer_steps=0))
    print(json.dumps(dict(status='PASS',methods=summary,output=str(out)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.collection,args.output)

"""Rescore saved candidates, excluding SANPO unlabeled (class 0) pixels.

Initial run scores included void pixels as negatives. Preserve initial outputs;
write corrected metrics in evaluated/ and record exact differences. No inference.
"""
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

start=time.perf_counter()
root=Path('artifacts.local/work/assistive-thesis-screen-20261015/thin_structure')
cv2.setNumThreads(2)
combined={}
for run in ['run1','run2_cityscapes']:
    folder=root/run
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    prior=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    old=list(csv.DictReader((folder/'frame_metrics.csv').open(encoding='utf8')))
    rows=[]
    correction=defaultdict(int)
    current_sample=None
    for r in old:
        i=int(r['sample_index'])
        if current_sample!=i:
            m=manifest['samples'][i]
            path=Path(m['assets']['segmentation_masks']['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==m['assets']['segmentation_masks']['sha256']
            mask=np.asarray(Image.open(path).convert('RGB'))
            truth=mask[:,:,0]==24
            valid=mask[:,:,0]!=0
            current_sample=i
        name=r['method']
        if name.startswith('oracle_mask_'):
            s=int(name.rsplit('_',1)[1])
            low=cv2.resize(truth.astype(np.float32),(s,s),interpolation=cv2.INTER_AREA)
            pred=cv2.resize((low>=.5).astype(np.uint8),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            pred=np.asarray(Image.open(folder/'predictions'/f'{i:02d}_{name}.png'))>0
        # Independently reconstruct the confusion matrix from serialized predictions.
        codes=truth.astype(np.uint8)*2+pred.astype(np.uint8)
        counts=np.bincount(codes[valid],minlength=4)
        tn,fp,fn,tp=[int(x) for x in counts]
        assert tp==int(r['tp']) and fn==int(r['fn'])
        assert fp<=int(r['fp'])
        correction[name]+=int(r['fp'])-fp
        nr={**r,'tp':tp,'fp':fp,'fn':fn,'gt_pixels':int(truth.sum()),'valid_pixels':int(valid.sum()),'excluded_pixels':int((~valid).sum()),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'iou':tp/(tp+fp+fn) if tp+fp+fn else None}
        rows.append(nr)
    dest=folder/'evaluated';dest.mkdir(exist_ok=False)
    with (dest/'frame_metrics.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={**prior,'evaluation_contract':'native mask 2208x1242; target class24; ignore class0 unlabeled; all other annotated classes negative','raw_scores_preserved':True,'unlabeled_fp_removed':dict(correction),'audit_reconstructed_candidates':len(rows),'metrics':{},'by_session':[],'paired_instance_hits':{},'selector_coverage':{},'timing_note':'forward-only wall measurements; crop calls sequential, not end-to-end/device latency'}
    for name in prior['methods']:
        rs=[r for r in rows if r['method']==name]
        tp,fp,fn=[sum(int(r[k]) for r in rs) for k in ['tp','fp','fn']]
        summary['metrics'][name]={**prior['methods'][name],'tp':tp,'fp':fp,'fn':fn,'iou':tp/(tp+fp+fn) if tp+fp+fn else None,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
        for session in sorted(set(r['session'] for r in rs)):
            ss=[r for r in rs if r['session']==session]
            a,b,c=[sum(int(r[k]) for r in ss) for k in ['tp','fp','fn']]
            summary['by_session'].append({'session':session,'method':name,'frames':len(ss),'gt_pixels':a+c,'tp':a,'fp':b,'fn':c,'iou':a/(a+b+c) if a+b+c else None,'recall':a/(a+c) if a+c else None})
    inst=list(csv.DictReader((folder/'instance_metrics.csv').open(encoding='utf8')))
    key=lambda r:(r['sample_index'],r['instance'])
    for baseline in ['low256','full512']:
        base={key(r):r for r in inst if r['method']==baseline}
        for method in ['entropy2','rgbdetail2']+(['full1024'] if run=='run2_cityscapes' else []):
            arm={key(r):r for r in inst if r['method']==method}
            for metric in ['any_hit','recall_ge_20pct']:
                pairs=[(base[k][metric]=='True',arm[k][metric]=='True') for k in base]
                summary['paired_instance_hits'][baseline+'__'+method+'__'+metric]={'total':len(pairs),'rescue':sum(not a and b for a,b in pairs),'loss':sum(a and not b for a,b in pairs),'both_hit':sum(a and b for a,b in pairs),'neither':sum(not a and not b for a,b in pairs)}
    selectors=json.loads((folder/'selectors.json').read_text(encoding='utf8'))
    for name in ['entropy2','rgbdetail2']:
        ss=[x for x in selectors if x['selector']==name]
        numerator=sum(x['gt_pole_pixels_covered'] for x in ss)
        denom=sum(x['gt_pole_pixels'] for x in ss)
        summary['selector_coverage'][name]={'area_fraction':.5,'pole_pixel_coverage':numerator/denom,'pole_pixels':denom,'covered_pixels':numerator}
    summary.pop('methods')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    combined[run]=summary
    print(json.dumps({'run':run,'removed_false_positive_counts':dict(correction),'metrics':summary['metrics'],'paired':summary['paired_instance_hits'],'selector':summary['selector_coverage']}),flush=True)
(root/'evaluation_audit.json').write_text(json.dumps({'status':'PASS','runs':list(combined),'candidate_masks_reconstructed':sum(s['audit_reconstructed_candidates'] for s in combined.values()),'elapsed_s':time.perf_counter()-start,'original_tp_fn_identical':True,'correction':'ignore class0 unlabeled; initial raw metrics preserved'},indent=2),encoding='utf8')

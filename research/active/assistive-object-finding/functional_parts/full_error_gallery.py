"""Automatic native-label validation error galleries and training curves.

Gallery is explanatory selection, never the performance denominator. No test GT
is opened: predictions and scored records must both declare val/validation.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from full_evaluate_baselines import identifier, load_mask, metrics_from_confusion, native_prediction, read_rows, score_arrays

CLASS_NAMES = ['background','contain','cut','display','engine','grasp','hit','pound','support','wrap-grasp']
PALETTE = np.array([[20,20,24],[255,158,61],[219,65,82],[174,112,255],[79,177,255],
                    [74,215,149],[247,218,72],[255,111,193],[55,211,219],[160,202,72]],dtype=np.uint8)


def image_errors(row):
    ps=[p for p in row['points'] if p['hit'] is not None]
    misses=[p for p in ps if not p['hit']]
    iou=metrics_from_confusion(row['confusion'],list(range(1,len(row['confusion']))))['mIoU_foreground']
    return {'native_label_mixup':sum(p['native_label_mixup'] for p in ps),
            'point_on_background':sum(p['native_label_at_point']==0 for p in misses),
            'small_region_point_failure':sum(p['small_region'] for p in misses),
            'multi_class_joint_failure':sum(p['joint_success'] is False for p in row['pairs']),
            'segmentation_error':1-float(iou) if iou is not None else 0,
            'point_failures':len(misses), 'point_queries':len(ps), 'per_image_foreground_mIoU':iou}


def choose_cases(rows, maximum, per_type):
    types=['native_label_mixup','point_on_background','small_region_point_failure',
           'multi_class_joint_failure','segmentation_error']
    errors={r['source_id']:image_errors(r) for r in rows}
    selected={}
    rankings={}
    for kind in types:
        candidates=sorted((r for r in rows if errors[r['source_id']][kind]>0),
                          key=lambda r:(-errors[r['source_id']][kind],
                                        -errors[r['source_id']]['segmentation_error'],r['source_id']))
        rankings[kind]=[r['source_id'] for r in candidates[:per_type]]
        for r in candidates[:per_type]:
            sid=r['source_id']
            if sid not in selected and len(selected)>=maximum:
                continue
            selected.setdefault(sid,{'source_id':sid,'selected_for':[],'errors':errors[sid]})['selected_for'].append(kind)
    # No fallback filling with successful cases: the requested output is errors.
    return list(selected.values()),errors,rankings


def font(size):
    for p in ['C:/Windows/Fonts/consola.ttf','C:/Windows/Fonts/arial.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default(size=size)


def panel(array, points, size=(376,270), mask=False):
    if mask:
        rgb=np.full((*array.shape,3),128,dtype=np.uint8)
        valid=(array>=0)&(array<len(PALETTE));rgb[valid]=PALETTE[array[valid]]
        image=Image.fromarray(rgb)
    else:
        image=Image.fromarray(array).convert('RGB')
    w,h=image.size
    scale=min(size[0]/w,size[1]/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    resized=image.resize((nw,nh),Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    output=Image.new('RGB',size,(240,240,240));ox=(size[0]-nw)//2;oy=(size[1]-nh)//2
    output.paste(resized,(ox,oy));draw=ImageDraw.Draw(output)
    for p in points:
        x,y=p['point_xy'];x=ox+(x+.5)*nw/w;y=oy+(y+.5)*nh/h
        draw.ellipse((x-6,y-6,x+6,y+6),fill='white',outline='black',width=2)
        draw.text((x+7,y-8),str(p['class_id']),fill='white',stroke_width=2,stroke_fill='black',font=font(13))
    return output


def gallery(args):
    rows=read_rows(args.per_image)
    predictions=read_rows(args.predictions)
    # Gate before looking at any paths or loading RGB/GT.
    if not rows or any(r.get('split') not in ['val','validation'] for r in rows+predictions):
        raise ValueError('Gallery only accepts nonempty validation records; test is not authorized here')
    scored={r['source_id']:r for r in rows};pred={identifier(r):r for r in predictions}
    if len(scored)!=len(rows) or len(pred)!=len(predictions) or set(scored)!=set(pred):
        raise ValueError('Scored/prediction IDs must be unique and equal for the full validation set')
    records={identifier(r):r for r in read_rows(args.records)}
    maximum=min(max(args.max_images,0),48)
    cases,errors,rankings=choose_cases(rows,maximum,args.per_type)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary={'split':'val','all_validation_images':len(rows),'gallery_selected_images':len(cases),
             'selection':'deterministic descending error count then per-image segmentation error then source ID; up to per_type each; no hand selection',
             'max_images':maximum,'per_type':args.per_type,'ranking_heads':rankings,
             'all_image_error_counts':{k:sum(v[k]>0 for v in errors.values()) for k in ['native_label_mixup','point_on_background','small_region_point_failure','multi_class_joint_failure','segmentation_error']},
             'all_point_queries':sum(v['point_queries'] for v in errors.values()),
             'all_point_failures':sum(v['point_failures'] for v in errors.values()),
             'claim_limit':'Native semantic-label diagnostics only. Gallery is intentionally worst-ranked and not representative prevalence. Use full evaluator summary for performance.',
             'palette':{str(i):{'name':name,'rgb':PALETTE[i].tolist()} for i,name in enumerate(CLASS_NAMES)},
             'sheets':[]}
    sheet=None
    for ix,case in enumerate(cases):
        local=ix%6
        if local==0:
            remaining=min(6,len(cases)-ix)
            sheet=Image.new('RGB',(1152,98+remaining*370),(255,255,255))
            draw=ImageDraw.Draw(sheet)
            draw.text((12,8),f'{args.model} | automatically selected validation failures | {len(rows)} images scored',fill='black',font=font(16))
            draw.text((12,32),'Native label colors: '+', '.join(f'{i}:{name}' for i,name in enumerate(CLASS_NAMES)),fill='black',font=font(11))
            for j,title in enumerate(['RGB + predicted class points','Native GT + same points','Predicted semantic labels + same points']):
                draw.text((12+j*384,68),title,fill='black',font=font(13))
        sid=case['source_id'];r=scored[sid];p=pred[sid];meta=records.get(sid)
        if meta is None:raise ValueError(f'No image metadata for {sid}')
        gt=load_mask(p.get('gt_path') or meta.get('mask'))
        path=Path(p['prediction_path'])
        if not path.is_absolute():path=Path(args.predictions).resolve().parent/path
        prediction,_=native_prediction(np.load(path,allow_pickle=False),gt.shape,p.get('native_class_ids',list(range(10))))
        with Image.open(meta.get('image') or meta.get('image_path')) as im:
            rgb=np.asarray(im.convert('RGB'))
        if tuple(rgb.shape[:2])!=tuple(gt.shape):raise ValueError('RGB and native GT geometry mismatch')
        y=98+local*370
        for j,arr in enumerate([rgb,gt,prediction]):
            sheet.paste(panel(arr,r['points'],mask=j>0),(8+384*j,y))
        draw=ImageDraw.Draw(sheet)
        draw.text((10,y+274),sid+' | '+', '.join(case['selected_for']),fill='black',font=font(12))
        point_text='; '.join(f"{CLASS_NAMES[t['class_id']]} area={100*t['target_fraction']:.2f}% -> {CLASS_NAMES[t['native_label_at_point']] if 0<=t['native_label_at_point']<10 else 'ignore'} {'hit' if t['hit'] else 'miss'}" for t in r['points'])
        for line_no,line in enumerate(textwrap.wrap(point_text,width=145)[:4]):
            draw.text((10,y+293+15*line_no),line,fill='black',font=font(12))
        case['points']=r['points'];case['native_mask_shape']=list(gt.shape)
        if local==5 or ix==len(cases)-1:
            name=f'failures_{ix//6+1:02d}.jpg';sheet.save(out/name,quality=92)
            summary['sheets'].append(name)
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'error_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_val_images':len(rows),'gallery_images':len(cases),'sheets':summary['sheets']}))


def curves(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.runs)
    paths=sorted(root.glob('*/progress.jsonl'))
    if (root/'progress.jsonl').exists():paths=[root/'progress.jsonl']
    if not paths:raise ValueError('No progress.jsonl found')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    sources=[]
    for p in paths:
        data=[];partial=False
        lines=p.read_text(encoding='utf-8-sig').splitlines()
        for i,line in enumerate(lines):
            if not line.strip():continue
            try:data.append(json.loads(line))
            except json.JSONDecodeError:
                if i==len(lines)-1:partial=True
                else:raise
        # Resume may leave older duplicates: retain last complete epoch event.
        epochs={int(r['epoch']):r for r in data if r.get('stage') in ['epoch','validation']}
        ordered=[epochs[k] for k in sorted(epochs)]
        sources.append({'path':str(p),'parsed_rows':len(data),'partial_last_line_ignored':partial,'complete_epochs':len(ordered)})
        for ax,key in zip(axes,['train_loss','miou_foreground','point_hit']):
            valid=[r for r in ordered if r.get(key) is not None]
            if valid:ax.plot([r['epoch'] for r in valid],[r[key] for r in valid],marker='.',label=p.parent.name)
    for ax,title in zip(axes,['Training loss (epoch mean)','Full validation foreground mIoU','Full validation positive-class point hit']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
        if ax.lines:ax.legend(fontsize=8)
    axes[1].set_ylim(0,1);axes[2].set_ylim(0,1)
    fig.suptitle('Development curves; checkpoint selection uses validation, no test scores')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    (out/'curves_sources.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runs':len(paths),'curve':str(out/'training_curves.png')}))


def selftest(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    gt=np.zeros((32,48),dtype=np.uint8);gt[2:15,2:15]=1;gt[17:30,30:45]=5
    logits=np.full((10,32,48),-5,dtype=np.float32)
    logits[0]=1;logits[5,2:15,2:15]=9;logits[1,17:30,30:45]=9
    rgb=np.full((32,48,3),160,dtype=np.uint8)
    Image.fromarray(rgb).save(root/'synthetic.jpg');Image.fromarray(gt).save(root/'synthetic_mask.png')
    np.save(root/'synthetic_logits.npy',logits)
    pred,scores=native_prediction(logits,gt.shape,list(range(10)))
    row={'source_id':'synthetic_only','split':'val',**score_arrays(gt,pred,scores,list(range(10)))}
    manifest={'source_id':'synthetic_only','split':'val','gt_path':str((root/'synthetic_mask.png').resolve()),
              'prediction_path':str((root/'synthetic_logits.npy').resolve()),'native_class_ids':list(range(10))}
    record={'source_id':'synthetic_only','image':str((root/'synthetic.jpg').resolve())}
    for name,obj in [('per_image',row),('predictions',manifest),('records',record)]:
        (root/f'{name}.jsonl').write_text(json.dumps(obj)+'\n',encoding='utf-8')
    gallery(SimpleNamespace(per_image=root/'per_image.jsonl',predictions=root/'predictions.jsonl',records=root/'records.jsonl',
                            output=root/'gallery',max_images=48,per_type=8,model='SYNTHETIC SANITY ONLY'))
    summary=json.loads((root/'gallery/error_summary.json').read_text())
    assert summary['gallery_selected_images']==1 and summary['all_image_error_counts']['native_label_mixup']==1
    run=root/'runs'/'synthetic';run.mkdir(parents=True,exist_ok=True)
    logs=[{'stage':'validation','epoch':e,'train_loss':1/e,'miou_foreground':e*.1,'point_hit':e*.2} for e in [1,2,3]]
    (run/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs),encoding='utf-8')
    curves(SimpleNamespace(runs=root/'runs',output=root/'curves'))
    for path in [root/'gallery/failures_01.jpg',root/'curves/training_curves.png']:
        with Image.open(path) as image:image.verify()
    print('PASS: synthetic error selection, native-label mixup, gallery encoding, curve encoding; no real data opened')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('gallery')
    for name in ['per-image','predictions','records','output']:g.add_argument('--'+name,required=True)
    g.add_argument('--model',default='model');g.add_argument('--max-images',type=int,default=48)
    g.add_argument('--per-type',type=int,default=8)
    c=sub.add_parser('curves');c.add_argument('--runs',required=True);c.add_argument('--output',required=True)
    t=sub.add_parser('selftest');t.add_argument('--output',required=True)
    args=p.parse_args()
    {'gallery':gallery,'curves':curves,'selftest':selftest}[args.command](args)


if __name__=='__main__':main()

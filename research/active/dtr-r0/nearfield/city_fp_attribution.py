"""Cached-prediction diagnostics only; no model import, inference, or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def ranks(a):
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=len(x), pearson=None, spearman=None, reason='insufficient or constant')
    return dict(n=len(x), pearson=float(np.corrcoef(x, y)[0, 1]),
                spearman=float(np.corrcoef(ranks(x), ranks(y))[0, 1]))


def stats(a):
    a = np.asarray(a)
    return dict(max=float(a.max()), mean=float(a.mean()), area=float((a >= .5).mean())) if a.size else dict(max=None, mean=None, area=None)


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output
    provenance = {}
    def checked(p, expected=None):
        h = sha(p)
        if expected is not None and h != expected:
            raise ValueError(f'Hash mismatch: {p}')
        provenance[str(p)] = h
        return p
    def array(root, entry):
        p = (root / entry['path']).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError('Path escape')
        return np.load(checked(p, entry['sha256']), mmap_mode='r', allow_pickle=False)
    try:
        cm = read(checked(a.cache / 'manifest.json'))
        split = read(checked(a.split, cm['source_split_sha256']))
        labels = read(checked(a.cache / 'evaluator/test.json'))
        ids = cm['partitions']['test']['sample_indices']
        if ids != labels['sample_indices']:
            raise ValueError('Test ordering mismatch')
        spec = read(checked(a.capture / 'evaluator/spec.json', split['input_hashes']['evaluator/spec.json']))
        cases = spec['cases']
        citymeta = [dict(sample_id=str(i), sample_index=i, group_id=cases[i]['group_id'], variant=cases[i]['variant_id']) for i in ids]
        rm = read(checked(a.regression / 'manifest.json'))
        for name, digest in rm['files'].items():
            checked(a.regression / name, digest)
        willowmeta = [dict(sample_id=s['sample_id'], sample_index=s['sample_index'], group_id=s['group_id'],
                           variant=s['sample_id'][len(s['group_id'])+1:-3]) for s in rm['selected_samples']]
        sets = dict(city=(array(a.cache, cm['partitions']['test']['rgb']), array(a.cache, labels['near']), array(a.cache, labels['support']), citymeta),
                    willow_regression=tuple(np.load(a.regression / (n+'.npy'), mmap_mode='r', allow_pickle=False) for n in ('rgb','near','support')) + (willowmeta,))
        thin = {r['file']: r['sha256'] for r in read(checked(a.predictions / 'thin-manifest.json'))}
        rows=[]; paired=[]; summaries={}; predictions={}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for domain, (rgb, near, truth, meta) in sets.items():
            pred={}
            for arm in ('baseline','city_finetuned'):
                name=f'{arm}-{domain}-predictions.npz'
                with np.load(checked(a.predictions/name, thin[name]), allow_pickle=False) as z:
                    pred[arm]={k:z[k].copy() for k in ('near','support')}
                if pred[arm]['near'].shape != near.shape or pred[arm]['support'].shape != truth.shape:
                    raise ValueError('Prediction shape mismatch')
                if any(not np.isfinite(v).all() or v.min()<0 or v.max()>1 for v in pred[arm].values()):
                    raise ValueError('Invalid probabilities')
            predictions[domain]=pred
            logits={arm:np.log(np.clip(p['near'].astype(float),1e-7,1-1e-7)/np.clip(1-p['near'].astype(float),1e-7,1-1e-7)) for arm,p in pred.items()}
            domainrows=[]
            for h, head in enumerate(('BODY','HEAD')):
                variants=sorted({m['variant'] for m in meta})
                fig,axes=plt.subplots(1,len(variants),figsize=(4*len(variants),4),squeeze=False)
                for ax,variant in zip(axes[0],variants):
                    ix=np.array([j for j,m in enumerate(meta) if m['variant']==variant])
                    for label,color in [('TP','green'),('TN','steelblue'),('FP','red'),('FN','orange')]:
                        selected=[j for j in ix if (('TP' if near[j,h]==1 else 'FP') if pred['city_finetuned']['near'][j,h]>=.5 else ('FN' if near[j,h]==1 else 'TN'))==label]
                        ax.scatter(logits['baseline'][selected,h],logits['city_finetuned'][selected,h],s=12,alpha=.6,c=color,label=f'{label} n={len(selected)}')
                    ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
                    ax.set(xlabel='Baseline derived logit',ylabel='Fine-tuned derived logit',title=f'{domain} {head}: {variant}');ax.legend(fontsize=7)
                fig.tight_layout();fig.savefig(out/f'{domain}-{head}-logit-paired.png',dpi=150);plt.close(fig)
                for j, m in enumerate(meta):
                    for arm in pred:
                        p=float(pred[arm]['near'][j,h]); y=int(near[j,h]); mask=pred[arm]['support'][j,h]; known=truth[j,h]>=0
                        r=dict(domain=domain,head=head,arm=arm,**m,target=y,probability=p,derived_logit=float(logits[arm][j,h]),
                               delta_probability=float(pred['city_finetuned']['near'][j,h]-pred['baseline']['near'][j,h]),
                               delta_logit=float(logits['city_finetuned'][j,h]-logits['baseline'][j,h]),
                               confusion=('TP' if y==1 else 'FP') if p>=.5 else ('FN' if y==1 else 'TN'),unknown_fraction=float((~known).mean()))
                        for scope, values in [('all',mask),('known',mask[known]),('unknown',mask[~known])]:
                            r.update({f'support_{scope}_{k}':v for k,v in stats(values).items()})
                        rows.append(r);domainrows.append(r)
                for arm in pred:
                    rr=[r for r in domainrows if r['head']==head and r['arm']==arm]
                    for variant in ['ALL']+sorted({r['variant'] for r in rr}):
                        sub=[r for r in rr if variant=='ALL' or r['variant']==variant]; neg=[r for r in sub if r['target']==0]
                        summaries[f'{domain}/{head}/{arm}/{variant}']=dict(n=len(sub),confusion={c:sum(r['confusion']==c for r in sub) for c in ('TP','FP','FN','TN')},
                            mean_delta_logit=float(np.mean([r['delta_logit'] for r in sub])),negative_support={key:dict(mean=float(np.mean([r[key] for r in neg if r[key] is not None])) if any(r[key] is not None for r in neg) else None,
                            correlation_delta_logit=correlation([r[key] if r[key] is not None else np.nan for r in neg],[r['delta_logit'] for r in neg])) for key in rr[0] if key.startswith('support_')})
                    neg=[r for r in rr if r['target']==0]
                    fig, axes=plt.subplots(1,2,figsize=(9,3.5))
                    for ax,scope in zip(axes,('all','known')):
                        ax.scatter([r[f'support_{scope}_area'] for r in neg],[r['delta_logit'] for r in neg],s=8,alpha=.4)
                        ax.set(xlabel=f'{arm} {scope} support >= .5 fraction',ylabel='Derived near logit change',title=f'{domain} {head}: negatives')
                    fig.tight_layout();fig.savefig(out/f'{domain}-{head}-{arm}-scatter.png',dpi=140);plt.close(fig)
                bygroup={}
                for j,m in enumerate(meta):bygroup.setdefault(m['group_id'],{})[m['variant']]=j
                reference='clear' if domain=='city' else 'neither'
                for group, variants in bygroup.items():
                    if reference not in variants:raise ValueError('Missing group reference')
                    base=variants[reference]
                    for variant,j in variants.items():
                        if variant==reference:continue
                        paired.append(dict(domain=domain,head=head,group_id=group,variant=variant,reference=reference,
                            baseline_contrast=float(logits['baseline'][j,h]-logits['baseline'][base,h]),
                            finetuned_contrast=float(logits['city_finetuned'][j,h]-logits['city_finetuned'][base,h]),
                            contrast_change=float((logits['city_finetuned'][j,h]-logits['city_finetuned'][base,h])-(logits['baseline'][j,h]-logits['baseline'][base,h]))))
        write(out/'per-frame.json',rows);write(out/'paired-contrasts.json',paired)
        with (out/'per-frame.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        candidates=[r for r in rows if r['domain']=='city' and r['head']=='BODY' and r['arm']=='city_finetuned']
        fps=sorted([r for r in candidates if r['confusion']=='FP'],key=lambda r:(-r['probability'],r['sample_id']))[:50]
        tns=sorted([r for r in candidates if r['confusion']=='TN'],key=lambda r:r['sample_id'])
        rng=np.random.default_rng(17);chosen=sorted(rng.choice(len(tns),min(50,len(tns)),replace=False).tolist());tns=[tns[i] for i in chosen]
        selections=dict(seed=17,fp_rule='Top50 BODY fine-tuned near probability descending; sample_id lexical tie-break',tn_rule='Seed17 numpy default_rng choice without replacement from sample_id sorted TN',fp=fps,tn=tns)
        write(out/'selection.json',selections)
        rgb,near,truth,meta=sets['city'];lookup={m['sample_id']:j for j,m in enumerate(meta)};(out/'gallery').mkdir();panels=[]
        font=None
        for category, selection in [('FP',fps),('TN',tns)]:
            for r in selection:
                j=lookup[r['sample_id']];im=Image.fromarray(rgb[j]);panel=Image.new('RGB',(1024,180),'white');d=ImageDraw.Draw(panel)
                maps=[None,predictions['city']['baseline']['support'][j,0],predictions['city']['city_finetuned']['support'][j,0],truth[j,0]]
                titles=['RGB','baseline BODY support','finetuned BODY support','GT: red=1 green=0 gray=?']
                for k,m in enumerate(maps):
                    visual=im.copy()
                    if m is not None:
                        if k==3:
                            color=np.zeros((*m.shape,3),np.uint8);color[m<0]=[130,130,130];color[m==0]=[0,170,0];color[m==1]=[255,0,0]
                        else:color=np.stack([m*255,np.zeros_like(m),(1-m)*255],axis=-1).astype(np.uint8)
                        heat=Image.fromarray(color).resize(im.size,Image.Resampling.NEAREST);visual=Image.blend(im,heat,.48)
                    panel.paste(visual,(k*256,36));d.text((k*256+3,20),titles[k],fill='black')
                d.text((3,3),f"{category} id={r['sample_id']} {r['variant']} baseline={predictions['city']['baseline']['near'][j,0]:.5f} fine={r['probability']:.5f}",fill='black')
                name=f"{category}-{r['sample_id']}.png";panel.save(out/'gallery'/name);panels.append((name,panel))
        sheets=[]
        for start in range(0,len(panels),10):
            sheet=Image.new('RGB',(1024,180*min(10,len(panels)-start)),'white')
            for k,(_,panel) in enumerate(panels[start:start+10]):sheet.paste(panel,(0,k*180))
            name=f'contact-{start//10:02d}.jpg';sheet.save(out/name,quality=92);sheets.append(name)
        (out/'gallery.html').write_text('<html><body><h1>Fixed BODY FP/TN review: no semantic annotation</h1>'+''.join(f'<img loading="lazy" src="gallery/{n}" style="max-width:100%;display:block">' for n,_ in panels)+'</body></html>',encoding='utf-8')
        paired_summary={}
        for domain in sets:
            for head in ('BODY','HEAD'):
                for variant in sorted({p['variant'] for p in paired if p['domain']==domain}):
                    sub=[p for p in paired if p['domain']==domain and p['head']==head and p['variant']==variant]
                    paired_summary[f'{domain}/{head}/{variant}']={k:float(np.mean([p[k] for p in sub])) for k in ('baseline_contrast','finetuned_contrast','contrast_change')}
        matched=[]
        for group in sorted({m['group_id'] for m in meta}):
            ix={m['variant']:j for j,m in enumerate(meta) if m['group_id']==group}
            for negative in ('clear','right_clearance'):
                if 'center' not in ix or negative not in ix:raise ValueError('Missing City paired variant')
                for h,head in enumerate(('BODY','HEAD')):
                    pos=truth[ix['center'],h];neg=truth[ix[negative],h];both=(pos>=0)&(neg>=0)
                    matched.append(dict(group_id=group,head=head,negative_variant=negative,center_near=int(near[ix['center'],h]),negative_near=int(near[ix[negative],h]),
                        both_known_pixels=int(both.sum()),positive_pixels=int((pos==1).sum()),negative_positive_pixels=int((neg==1).sum()),
                        positive_into_negative_unknown=int(((pos==1)&(neg<0)).sum()),unknown_changed_pixels=int(((pos<0)!=(neg<0)).sum()),
                        delta_equals_center_positive_on_both_known=bool(np.array_equal((pos-neg)[both],(pos==1)[both].astype(np.int8)))))
        write(out/'matched-gt-support.json',matched)
        write(out/'summary.json',dict(status='PASS',scope='Cached TEST750 + consumed Willow VAL96 only; TRAIN predictions absent; no inference/training/threshold search; correlation is not causal attribution',
              logits='Inverse sigmoid of stored float probabilities clipped [1e-7,1-1e-7]; saturation and float32 precision prevent recovery of exact model logits',
              matched_gt_support=dict(pairs=len(matched),all_delta_equals_positive_on_both_known=all(r['delta_equals_center_positive_on_both_known'] for r in matched),positive_into_negative_unknown=sum(r['positive_into_negative_unknown'] for r in matched),negative_positive_pixels=sum(r['negative_positive_pixels'] for r in matched)),
              strata=summaries,paired_contrast_means=paired_summary,selection_counts=dict(fp=len(fps),tn=len(tns)),contact_sheets=sheets,seconds=time.perf_counter()-started,source_sha256=sha(__file__),inputs=provenance))
    except BaseException as e:
        write(out/'failure.json',dict(status='FAIL',error=repr(e),seconds=time.perf_counter()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','split','capture','predictions','regression','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

"""Small-array summaries of the fixed HEAD-X0 inference; no model or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_text())


def stats(values):
    x=np.asarray(values,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75))) if len(x) else dict(n=0)


def run(root,out):
    receipt=read(root/'receipt.json')
    assert receipt['status']=='PASS' and receipt['training_steps']==0
    assert receipt['rows_per_model']==1941 and receipt['original_B_backbone_identical']
    inputs=read(root/'inputs.json');selected=inputs['selected']
    byid={r['sample_id']:r for r in selected}
    assert len(selected)==len(byid)==209
    with np.load(root/'ROIs.npz') as masks:
        assert np.array_equal(masks['roi'].sum((1,2)),masks['sham'].sum((1,2)))
        assert not (masks['roi']&masks['sham']).any()
    def cohort(sid):
        row=byid[sid]
        if row['domain']=='plaza':return 'plaza_positive' if row['near_GT'][1]==1 else 'plaza_FP'
        return 'TRAIN_positive' if row['domain'] in ('new','old') else 'DEV_positive'
    result=dict(scope='Paired sensitivity, not causal identification; no geometry labels assigned to edited inputs',arms={})
    for arm in ('original','B','F'):
        rows=read(root/f'{arm}-rows.json')
        z=np.load(root/f'{arm}-counterfactual.npz')
        assert len(rows)==len(z['near'])==1941
        assert np.isfinite(z['logits']).all() and np.isfinite(z['support']).all()
        lookup={(r['sample_id'],r['variant']):i for i,r in enumerate(rows) if 'donor' not in r}
        grouped={}
        for group in ('TRAIN_positive','DEV_positive','plaza_positive','plaza_FP'):
            ids=[sid for sid in byid if cohort(sid)==group]
            data={}
            for variant in ('original','roi_retain_mean','roi_erase_mean','sham_retain_mean','sham_erase_mean','roi_retain_blur','roi_erase_blur','sham_retain_blur','sham_erase_blur'):
                ix=[lookup[sid,variant] for sid in ids]
                data[variant]=dict(probability=stats(z['near'][ix,1]),raw_logit=stats(z['logits'][ix,1]),delta_probability=stats([rows[i]['delta_HEAD_probability'] for i in ix]),delta_logit=stats([rows[i]['delta_HEAD_logit'] for i in ix]))
            for op in ('erase','retain'):
                for fill in ('mean','blur'):
                    data[f'paired_roi_minus_sham_{op}_{fill}']=dict(delta_probability=stats([z['near'][lookup[sid,f'roi_{op}_{fill}'],1]-z['near'][lookup[sid,f'sham_{op}_{fill}'],1] for sid in ids]),delta_logit=stats([z['logits'][lookup[sid,f'roi_{op}_{fill}'],1]-z['logits'][lookup[sid,f'sham_{op}_{fill}'],1] for sid in ids]))
            grouped[group]=data
        paste={}
        for direction in ('TRAIN_to_plaza','plaza_to_TRAIN'):
            ids=[i for i,r in enumerate(rows) if 'donor' in r and (r['donor'].startswith('plaza:'))==(direction=='plaza_to_TRAIN')]
            variants={}
            for variant in ('donor_roi','donor_sham'):
                indices=[i for i in ids if rows[i]['variant']==variant]
                variants[variant]=dict(delta_probability=stats([rows[i]['delta_HEAD_probability'] for i in indices]),delta_logit=stats([rows[i]['delta_HEAD_logit'] for i in indices]))
            roi=[i for i in ids if rows[i]['variant']=='donor_roi'];sham=[i for i in ids if rows[i]['variant']=='donor_sham']
            assert all(rows[i]['pair']==rows[j]['pair'] for i,j in zip(roi,sham))
            variants['paired_roi_minus_sham']=dict(delta_probability=stats(z['near'][roi,1]-z['near'][sham,1]),delta_logit=stats(z['logits'][roi,1]-z['logits'][sham,1]))
            paste[direction]=variants
        result['arms'][arm]=dict(edits=grouped,paste=paste)
    query={r['sample_id']:r for r in read(root/'feature-query-identities.json')}
    result['feature_affinity']={}
    for arm in ('original','F'):
        grouped={}
        for row in read(root/f'{arm}-similarity.json'):
            sid=row['sample_id'];q=query[sid]
            group=cohort(sid) if sid in byid else ('plaza_TN' if q['domain']=='plaza' else 'DEV_negative')
            # Nonselected DEV positive rows still have ground-truth support.
            if q['domain']=='DEV' and q['feature_ROI_kind'].startswith('GT_'):group='DEV_positive'
            key=group+'/'+row['mode']
            vals=grouped.setdefault(key,dict(positive_mean=[],negative_mean=[],mean_gap=[],positive_max=[],negative_max=[],max_gap=[]))
            p=row['reference']['1'];n=row['reference']['0']
            for k,v in [('positive_mean',p['mean']),('negative_mean',n['mean']),('mean_gap',p['mean']-n['mean']),('positive_max',p['max']),('negative_max',n['max']),('max_gap',p['max']-n['max'])]:vals[k].append(v)
        result['feature_affinity'][arm]={k:{n:stats(v) for n,v in vals.items()} for k,vals in grouped.items()}
    result['sham']=dict(overlap_cells=stats([r['sham']['overlap_cells'] for r in selected]),overlap_examples=[r['sample_id'] for r in selected if r['sham']['overlap_cells']>0])
    result['inputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'inputs.json',root/'receipt.json',*[root/f'{a}-counterfactual.npz' for a in ('original','B','F')]]}
    out.write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    groups=['plaza_positive','plaza_FP','plaza_TN']
    for ax,mode in zip(axes,['whole','query_ROI_same_coordinates']):
        for arm,offset,color in [('original',-.17,'#4279ac'),('F',.17,'#db7240')]:
            rows=read(root/f'{arm}-similarity.json');values=[]
            for group in groups:
                values.append([r['reference']['1']['max']-r['reference']['0']['max'] for r in rows if r['mode']==mode and r['sample_id'].startswith('plaza:') and (cohort(r['sample_id']) if r['sample_id'] in byid else 'plaza_TN')==group])
            bp=ax.boxplot(values,positions=np.arange(3)+offset,widths=.27,patch_artist=True,showfliers=False)
            for box in bp['boxes']:box.set_facecolor(color);box.set_alpha(.6)
            ax.plot([],[],color=color,label=arm)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(3),['positive n=15','FP n=171','TN n=564'])
        ax.set_title('Whole-image deep pooling' if mode=='whole' else 'Query-coordinate ROI pooling (conditioned)')
        ax.set_ylabel('Nearest positive cosine - nearest negative cosine');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('HEAD-X0: within-model TRAIN affinity; original/B backbone identical')
    fig.text(.5,.01,'ROI: positive GT support, negatives F peak cell. Contextual 5x8 features; 207 positive / 927 negative references; not causal evidence.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out.with_name('feature-affinity.png'),dpi=170);plt.close(fig)
    print(json.dumps(dict(status='PASS',output=str(out),training_steps=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.probe,a.output)

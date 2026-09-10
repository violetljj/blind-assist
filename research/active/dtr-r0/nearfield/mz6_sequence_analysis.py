"""Post-run scalar audit and zero-activation diagnosis; no fit or changed predictor."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import cv2

from mz5_fixed_ensemble import EVENTS, load, read, sha, write
from mz6_causal_packet import correspondences, HZ


def run(folder):
    output=folder/'analysis';output.mkdir(exist_ok=False)
    obs=load(folder/'observations.npz');ev=load(folder/'evaluator.npz')
    predictions=load(folder/'predictions.npz');result=read(folder/'result.json')
    receipt=read(folder/'score-receipt.json')
    assert sha(folder/'predictions.npz')==receipt['output_sha256']['predictions.npz']
    assert sha(folder/'result.json')==receipt['output_sha256']['result.json']
    truth=ev['truth'];audited=0
    for name,logits in predictions.items():
        p=logits>=0;exact=wf=bh=hb=0
        confusion={e:dict(tp=0,fp=0,fn=0,tn=0) for e in EVENTS}
        for a,b in zip(p.tolist(),truth.tolist()):
            exact+=a==b
            wf+=sum(a[k+1] and b[k] and not b[k+1] for k in (0,2))
            bh+=any(b[:2]) and not any(b[2:]) and any(a[2:])
            hb+=any(b[2:]) and not any(b[:2]) and any(a[:2])
            for k,e in enumerate(EVENTS):confusion[e]['tp' if a[k] and b[k] else 'fp' if a[k] else 'fn' if b[k] else 'tn']+=1
        metrics=result['arms'][name]['metrics']
        assert [exact,wf,bh,hb]==[metrics[k]['numerator'] for k in ('spatial_exact','wrong_far','body_to_head','head_to_body')]
        assert confusion==metrics['event_confusion'];audited+=len(p)
    for condition in ('clean','stress'):
        raw=predictions[condition+'/CURRENT'];expected=[]
        for i in range(len(raw)):
            previous=[j for j in range(max(0,i-2),i+1) if obs['clip'][j]==obs['clip'][i]]
            expected.append(np.mean(raw[previous],axis=0))
        np.testing.assert_array_equal(np.stack(expected),predictions[condition+'/MEAN3'])
    cv2.setNumThreads(1)
    stage={};nearby_raw_opportunities=0
    for condition in ('clean','stress'):
        ranges=obs['ranges' if condition=='clean' else 'stress_ranges']
        valid=obs['valid' if condition=='clean' else 'stress_valid']
        counts=Counter();max_matches=[]
        for i in range(len(ranges)):
            for age in (1,2):
                j=i-age
                if j<0 or obs['clip'][j]!=obs['clip'][i]:continue
                source=valid[j].all(1)&((ranges[j,:,1]-ranges[j,:,0])>=.30)
                dest=valid[i].sum(1)==1
                possible=source[:,None]&dest[None,:]
                counts['raw_source_destination_pairs']+=int(possible.sum())
                current=np.where(valid[i],ranges[i],0).sum(1)
                compatible=(possible&(np.abs(ranges[j,:,1,None]-current[None,:])<=.05+1.5*age/HZ)
                    &(ranges[j,:,0,None]<current[None,:]-.15))
                counts['range_compatible_pairs']+=int(compatible.sum())
                if compatible.any():
                    matches=correspondences(obs['gray'][j],obs['gray'][i])
                    counts['range_compatible_pairs_with_any_match']+=int((compatible&(matches>0)).sum())
                    counts['range_compatible_pairs_with_three_matches']+=int((compatible&(matches>=3)).sum())
                    max_matches.append(int(matches[compatible].max()))
        stage[condition]=dict(counts,maximum_matches_on_compatible_pair=max(max_matches,default=0))
    for i,z in np.argwhere(ev['deleted']):
        if any(i-age>=0 and obs['clip'][i-age]==obs['clip'][i]
            and obs['stress_valid'][i-age,z].all()
            and abs(obs['stress_ranges'][i-age,z,0]-obs['ranges'][i,z,0])<=.05+1.5*age/HZ
            for age in (1,2)):
            nearby_raw_opportunities+=1
    by_clip={}
    for clip in dict.fromkeys(obs['clip'].tolist()):
        mask=obs['clip']==clip
        by_clip[clip]=dict(frames=int(mask.sum()),
            no_valid_packet_frames=int((~obs['valid'][mask].any((1,2))).sum()),
            positive_bits=truth[mask].sum(0).tolist(),
            clean_stress_changed_flags=int(((predictions['clean/CURRENT'][mask]>=0)!=(predictions['stress/CURRENT'][mask]>=0)).sum()),
            clean_stress_max_abs_logit_change=float(np.max(np.abs(predictions['clean/CURRENT'][mask]-predictions['stress/CURRENT'][mask]))))
    report=dict(status='PASS',audit_scalar_prediction_rows=audited,mean3_rows_verified=400,
        training_steps=0,model_forward_passes=0,gate_diagnosis=stage,
        same_zone_recent_raw_near_return_opportunities=nearby_raw_opportunities,
        opportunity_scope='Evaluator diagnostic for39 deletions; no target correspondence or routing authority',
        by_clip=by_clip,source_unknown='No valid return is UNKNOWN, never CLEAR. All event metrics are visible-support classification.',
        clearance_limit='Never-detected far event silence is not successful release. BODY/HEAD union must accompany event bits.',
        episode_count_limit='BODY_NEAR and BODY_ANY delay in the same exit clip are two metric records, one physical episode',
        input_sha256={n:sha(folder/n) for n in ('observations.npz','evaluator.npz','predictions.npz','result.json')},
        source_sha256=sha(__file__))
    write(output/'audit-and-diagnosis.json',report)
    # A compact scientific timeline exposes range confusion and delayed detection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
    pairs=[('approach_head_bar_target',3),('lateral_body_enter_target',1),
           ('thin_pole_background_target',1),('lateral_target_exit_target',0)]
    for ax,(clip,k) in zip(axes.flat,pairs):
        mask=obs['clip']==clip
        matrix=np.stack([truth[mask,k],*(predictions['clean/'+m][mask,k]>=0 for m in ('CURRENT','MEAN3','COMPATIBLE'))])
        ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap='Greens',vmin=0,vmax=1)
        ax.set_yticks(range(4),['Native support','Current','Mean3','Compatible'])
        ax.set_xticks([0,6,12,18,24]);ax.set_xlabel('Nominal sample index (12 Hz; posed capture)')
        ax.set_title(clip.replace('_target','').replace('_',' ')+' / '+EVENTS[k],fontsize=10)
    fig.suptitle('MZ6: visible-support flags; missing-return stress has identical flags',fontsize=13)
    fig.savefig(output/'event-timeline.png',dpi=150);plt.close(fig)
    print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run(args.run.resolve())

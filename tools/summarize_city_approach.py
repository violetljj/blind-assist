"""Analyze cached fixed-model approach observations without fitting or inference."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import evaluate_city_native_route as frozen
from diagnose_city_field import target_decision,confusion,truly_clear

BANDS={'thin_pole':('BODY','HEAD'),'body_protrusion':('BODY',),'head_bar':('HEAD',),'suspended_sign':('HEAD',)}


def validate_label_identity(manifest,receipt,ids,hashes):
    if receipt.get('status')!='PASS' or manifest.get('sample_indices')!=ids:
        raise ValueError('Labels are not PASS or sample order differs')
    expected=dict(manifest=receipt.get('label_manifest_sha256'),near=receipt.get('near_sha256'),
        support=receipt.get('support_sha256'),targets=receipt.get('target_mask_sha256'))
    if not all(expected.values()) or expected!=hashes:raise ValueError('Label payload hash mismatch')


def sequence_summary(rows):
    rows=sorted(rows,key=lambda r:-r['distance_m'])
    eligible=[r for r in rows if r['eligibility']=='ELIGIBLE']
    joints=[r for r in eligible if r['joint']]
    first=max((r['distance_m'] for r in joints),default=None)
    nearer=[r for r in rows if first is not None and r['distance_m']<first]
    missed=[r['distance_m'] for r in nearer if r['eligibility']=='ELIGIBLE' and not r['joint']]
    unknown=[r['distance_m'] for r in nearer if r['eligibility']=='UNKNOWN']
    nearer_eligible=sum(r['eligibility']=='ELIGIBLE' for r in nearer)
    return dict(sampled_distances_m=[r['distance_m'] for r in rows],
        eligible=len(eligible),unknown=sum(r['eligibility']=='UNKNOWN' for r in rows),
        visible_no_eligible_query_support=sum(r['eligibility']=='VISIBLE_NO_ELIGIBLE_QUERY_SUPPORT' for r in rows),
        alert_misses=sum(not r['alert'] for r in eligible),joint_hits=len(joints),
        peak_hits=sum(r['peak_hit'] for r in eligible),
        joint_with_clear_silent=sum(r['joint'] and not r['clear_alert'] for r in eligible),
        farthest_sampled_joint_m=first,
        farthest_sampled_query_alert_m=max((r['distance_m'] for r in rows if r['alert']),default=None),
        nearer_joint_misses_m=missed,nearer_unknown_m=unknown,
        nearer_eligible_count=nearer_eligible,
        all_nearer_eligible_joint=not missed if nearer_eligible else None,
        persistence='OBSERVED_LOSS' if missed else 'UNKNOWN' if unknown else 'HOLDS_AT_SAMPLED_NEARER_POINTS' if nearer_eligible else 'NOT_TESTED',
        continuity='UNKNOWN' if unknown else 'SAMPLED_ONLY_NOT_CONTINUOUS_TIME',rows=rows)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def plots(out,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[];colours={-15:'#2474b5',0:'#d07922',15:'#27844f'}
    for category,heads in BANDS.items():
        for head in heads:
            fig,axes=plt.subplots(1,2,figsize=(12,4.3),sharex=True,sharey=True)
            for ax,method in zip(axes,('g10','g13')):
                for az in (-15,0,15):
                    data=groups[f'{method}/{category}/{head}/{az}']['rows']
                    xs=[r['distance_m'] for r in data];ys=[r['probability'] for r in data]
                    ax.plot(xs,ys,'o-',color=colours[az],label=f'target {az:+d} deg',ms=4)
                    ax.plot(xs,[r['clear_probability'] for r in data],'--',color=colours[az],alpha=.6,label=f'clear {az:+d} deg')
                    for r in data:
                        if r['eligibility']=='UNKNOWN':ax.plot(r['distance_m'],r['probability'],'x',color='black',ms=10)
                        elif r['eligibility']=='ELIGIBLE':ax.plot(r['distance_m'],r['probability'],'o',mfc='none',mec=colours[az],ms=9)
                ax.axhline(data[0]['threshold'],color='#555555',ls=':',label='frozen cutoff')
                ax.set(title=method.upper(),xlabel='Radial anchor distance (m)',ylim=(-.03,1.03),xlim=(6.2,.8));ax.grid(alpha=.2)
            axes[0].set_ylabel('Near-alert probability');axes[1].legend(fontsize=7,loc='best')
            fig.suptitle(f'{category} / {head} | ring: eligible; X: unreliable; dashed: matched clear')
            fig.tight_layout();p=out/f'{category}-{head}.png';fig.savefig(p,dpi=150);plt.close(fig);files.append(str(p))
    return files


def run(capture,evaluation,labels,out):
    capture=capture.resolve(strict=True);evaluation=evaluation.resolve(strict=True);labels=labels.resolve(strict=True);out=out.resolve()
    if out.exists() or out==frozen.ARTIFACTS.resolve() or not out.is_relative_to(frozen.ARTIFACTS.resolve()):raise ValueError('Fresh canonical artifact output required')
    spec=frozen.read(capture/'source/spec.json');ep=frozen.read(evaluation/'protocol.json');er=frozen.read(evaluation/'receipt.json')
    if er.get('status')!='PASS' or frozen.sha(evaluation/'predictions.npz')!=er['predictions_sha256']:raise ValueError('Prediction receipt invalid')
    dataset=frozen.read(capture/'model/dataset.json');ids=[f['sample_index'] for f in dataset['frames']]
    if ep['sample_indices']!=ids:raise ValueError('Prediction sample order mismatch')
    for frame in dataset['frames']:
        if ep['rgb_sha256'][str(frame['sample_index'])]!=frozen.sha(frozen.within(capture/'model',frame['rgb_path'])):raise ValueError('Prediction RGB mismatch')
    lm=frozen.read(labels/'native-route-labels.json');lv=frozen.read(labels/'label-validation.json')
    label_hashes=dict(manifest=frozen.sha(labels/'native-route-labels.json'),
        near=frozen.sha(frozen.within(labels,lm['near'])),support=frozen.sha(frozen.within(labels,lm['support'])),
        targets={t['target_id']:frozen.sha(frozen.within(labels,t['masks'])) for t in lm['targets']})
    validate_label_identity(lm,lv,ids,label_hashes)
    if lm['provenance']['spec_sha256']!=frozen.sha(capture/'source/spec.json') or lm['provenance']['receipt_sha256']!=frozen.sha(capture/'receipt.json'):raise ValueError('Label/capture mismatch')
    prediction=dict(np.load(evaluation/'predictions.npz',allow_pickle=False));near=np.load(labels/lm['near'],allow_pickle=False)
    clear={c['pair_id']:i for i,c in enumerate(spec['cases']) if c['variant']=='clear'}
    target_masks={t['category']:frozen.pooled(np.load(labels/t['masks'],allow_pickle=False)) for t in lm['targets']}
    target_ids={t['category']:t['target_id'] for t in lm['targets']}
    evidence={tid:{r['sample_index']:r for r in rows} for tid,rows in lv['targets'].items()}
    groups=defaultdict(list);rows=[]
    for i,case in enumerate(spec['cases']):
        kind=case['variant']
        if kind=='clear':continue
        ci=clear[case['pair_id']]
        if case['camera']!=spec['cases'][ci]['camera']:raise ValueError('Clear/hazard camera poses differ')
        tid=target_ids[kind];mask=target_masks[kind];ev=evidence[tid][i]
        for head in BANDS[kind]:
            h=('BODY','HEAD').index(head);positive=mask[i,h]==1
            status='UNKNOWN' if ev.get('status')!='EVALUABLE' else 'ELIGIBLE' if positive.any() else 'VISIBLE_NO_ELIGIBLE_QUERY_SUPPORT'
            for method in frozen.METHODS:
                p=float(prediction[method+'_near'][i,h]);cut=ep['methods'][method]['near_thresholds'][h]['value']
                cp=float(prediction[method+'_near'][ci,h]);decision=target_decision(p,prediction[method+'_support'][i,h],mask[i,h],cut)
                row=dict(method=method,category=kind,head=head,azimuth_deg=case['approach_azimuth_deg'],distance_m=case['distance_m'],sample_index=i,
                    target_id=tid,eligibility=status,probability=p,threshold=cut,clear_probability=cp,clear_alert=bool(cp>=cut),paired_probability_delta=p-cp,
                    alert=bool(p>=cut),overlap=bool(decision and decision['overlap']),joint=bool(status=='ELIGIBLE' and decision and decision['joint']),
                    peak_hit=bool(status=='ELIGIBLE' and decision and decision['peak_hit']),
                    native_clear_truth=near[ci].tolist(),scene_truth=near[i].tolist(),raycheck=ev.get('raycheck'),
                    depth_cause='NOT_IDENTIFIABLE_DIRECT_RGB_MODELS')
                rows.append(row);groups[f"{method}/{kind}/{head}/{case['approach_azimuth_deg']}"] .append(row)
    summaries={key:sequence_summary(value) for key,value in groups.items()};overall={}
    for method in frozen.METHODS:
        current=[r for r in rows if r['method']==method];eligible=[r for r in current if r['eligibility']=='ELIGIBLE']
        ci=[i for i,c in enumerate(spec['cases']) if truly_clear(c['variant'],near[i])]
        cuts=np.array([v['value'] for v in ep['methods'][method]['near_thresholds']]);alerts=prediction[method+'_near']>=cuts
        overall[method]=dict(eligibility=dict(Counter(r['eligibility'] for r in current)),alert_misses=sum(not r['alert'] for r in eligible),
            joint_hits=sum(r['joint'] for r in eligible),joint_with_clear_silent=sum(r['joint'] and not r['clear_alert'] for r in eligible),
            known_clear_controls=len(ci),clear_false_alert_frames=int(alerts[ci].any(axis=1).sum()),
            scene={head:confusion(prediction[method+'_near'][:,h],near[:,h],cuts[h]) for h,head in enumerate(('BODY','HEAD'))})
    out.mkdir(parents=True)
    result=dict(schema='city-field-approach-summary-v1',frames=len(ids),overall=overall,sequences=summaries,
        source=dict(capture=str(capture),evaluation=str(evaluation),prediction_sha256=frozen.sha(evaluation/'predictions.npz'),
            spec_sha256=frozen.sha(capture/'source/spec.json'),labels_sha256=frozen.sha(labels/'native-route-labels.json'),
            verified_label_hashes=label_hashes,label_validation_sha256=frozen.sha(labels/'label-validation.json'),scorer_sha256=frozen.sha(Path(__file__))),
        scope='Sampled radial distance, not surface clearance or temporal warning. No eligible target support is not FN; unknown remains excluded; no new inference.',
        paired_comparison='Hazard variants also change feet/clamps/hangers. Probability delta and joint_with_clear_silent describe assembly-associated response, not target-only causal recognition.')
    result['plots']=plots(out,summaries);write(out/'result.json',result);write(out/'rows.json',rows)
    write(out/'receipt.json',dict(status='PASS',result_sha256=frozen.sha(out/'result.json'),model_inference_repeated=False))
    print(json.dumps(dict(status='PASS',overall=overall,output=str(out)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','evaluation','labels','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.capture,a.evaluation,a.labels,a.output)

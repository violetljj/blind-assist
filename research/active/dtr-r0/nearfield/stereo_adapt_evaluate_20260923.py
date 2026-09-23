"""Sealed ordinary/balanced stereo adaptation evaluation; never selects weights."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
import run_surface_contact_20260923 as c
import surface_contact_oracle as oracle
from vpp_geometry_core import ray_hints

ROOT,ART=c.ROOT,c.ART
PREP=ART/'evidence/ba-stereo-adapt-20260923-prepared'
PRED=ART/'evidence/ba-stereo-adapt-20260923-predictions'
OUT=ART/'evidence/ba-stereo-adapt-20260923-evaluation'
OLD=ART/'evidence/ba-vpp-geometry-20260923-frontend-v1'
OLD_TRUTH=ART/'evidence/ba-surface-contact-20260923-evaluation-v1'
ARMS=('baseline','ordinary','balanced')
RIG=c.spatial.RIG
read,sha,write=c.read,c.sha,c.write


def verify(folder):
    receipt=read(folder/'receipt.json');assert receipt['status']=='PASS'
    for key,value in receipt['hashes'].items():
        path=folder/key.replace('\\','/');assert path.resolve().is_relative_to(folder.resolve())
        assert sha(path)==value,path
    return receipt


def pixel_queries(depth,horizon=4.):
    z=np.asarray(depth);yy,xx=np.indices(z.shape)
    f=640/(2*np.tan(np.radians(35)))
    y=(xx-320)*z/f;up=-(yy-180)*z/f
    valid=np.isfinite(z)&(z>=.5)&(z<=horizon)
    valid&=(np.abs(y)<=z*np.tan(np.radians(22.5)))&(np.abs(up)<=z*np.tan(np.radians(20)))
    return {part:valid&(np.abs(y)<=c.WIDTHS[part][1]/2)&(up>=lo)&(up<=hi)
            for part,(lo,hi) in c.BANDS.items()}


def pixel_stats(pred,raw,native):
    """Default camera query, 4m contact horizon; fixed native denominator."""
    assert pred.shape==raw.shape==native.shape==(360,640)
    true=pixel_queries(native);pquery=pixel_queries(pred)
    valid=np.isfinite(pred)&(pred>=.5)&(pred<=4.)
    error=np.abs(pred-native);far=np.isfinite(native)&(native>4.)
    rows={}
    for part,n in true.items():
        values=raw[n];finite=values[np.isfinite(values)&(values>0)]
        v=n&valid
        rows[part]=dict(native_query_pixels=int(n.sum()),predicted_valid=int(v.sum()),
            missing=int((n&~valid).sum()),correct_5cm=int((v&pquery[part]&(error<=.05)).sum()),
            abs_z_5cm=int((v&(error<=.05)).sum()),far_native_estimated_query=int((far&pquery[part]).sum()),
            predicted_query_pixels=int(pquery[part].sum()),
            conditional_abs_z_sum_m=float(error[v].sum(dtype=np.float64)),
            raw_depth=dict(native_query_pixels=int(n.sum()),finite_positive=int(len(finite)),
                missing=int(len(values)-len(finite)),below_half_m=int((finite<.5).sum()),
                above4m=int((finite>4).sum()),quantiles_0_25_50_75_100=np.percentile(finite,[0,25,50,75,100]).tolist() if len(finite) else []))
    yy,xx=np.indices(native.shape);f=640/(2*np.tan(np.radians(35)))
    fov=(np.abs((xx-320)/f)<=np.tan(np.radians(22.5)))&(np.abs((yy-180)/f)<=np.tan(np.radians(20)))
    return rows,dict(far_native_fov_pixels=int((far&fov).sum()),far_native_estimated_near=int((far&fov&valid).sum()))


def pixel_aggregate(rows):
    keys=('native_query_pixels','predicted_valid','missing','correct_5cm','abs_z_5cm',
          'far_native_estimated_query','predicted_query_pixels','conditional_abs_z_sum_m')
    total={k:sum(r[k] for r in rows) for k in keys}
    total['hit_5cm']=total['correct_5cm']/total['native_query_pixels'] if total['native_query_pixels'] else None
    total['conditional_mae_m']=total['conditional_abs_z_sum_m']/total['predicted_valid'] if total['predicted_valid'] else None
    return total


def slices(rows):
    result={'all':rows,'HEAD':[r for r in rows if r['part']=='HEAD'],
        'BODY':[r for r in rows if r['part']=='BODY'],'no_frame_hints':[r for r in rows if r['no_frame_hints']],
        'small_head_HEAD':[r for r in rows if r['family']=='small_head' and r['part']=='HEAD'],
        'nonwall':[r for r in rows if r['family']!='wall']}
    result.update({f:[r for r in rows if r['family']==f] for f in sorted({r['family'] for r in rows})})
    return result


def evaluate(output=OUT,predictions=PRED,prepared=PREP):
    journal=read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])
    assert journal['state']=='running' and journal['reuse_preflight']['status']=='PASS'
    c.fresh(output)
    # Every arm's complete raw outputs are hash-verified before any truth join.
    receipts={arm:verify(predictions/arm) for arm in ARMS};old_receipt=verify(OLD)
    assert receipts['baseline']['frames']==96
    assert receipts['ordinary']['frames']==receipts['balanced']['frames']==672
    public_doc=read(prepared/'eval_observations/examples.json');assert public_doc['rig']==RIG
    public=public_doc['frames']
    assert len(public)==96 and len({r['id'] for r in public})==96
    assert public_doc['split']=='eval' and all(r['split']=='eval' for r in public)
    old_doc=read(c.PUBLIC/'tof-inputs.json');assert old_doc['rig']==RIG
    old_public=old_doc['frames'];assert len(old_public)==576
    entries=[dict(r,panel='new_eval',cohort='new_eval') for r in public]
    entries += [dict(r,cohort='historical') for r in old_public]
    sealed=[]
    for entry in entries:
        panel,id=entry['panel'],entry['id']
        for key in ('ranges','valid'):assert sha(entry[key])==entry[key+'_sha256']
        ranges,valid=np.load(entry['ranges']),np.load(entry['valid'])
        _,seeds=ray_hints(ranges,valid,RIG)
        tpoints=c.fov(c.spatial.tof_points(ranges,valid))
        row=dict(panel=panel,id=id,cohort=entry['cohort'],no_frame_hints=not len(seeds),
                 frame_hints=len(seeds),tof=c.queries(lambda lo,hi,axis:c.point_contact(tpoints,lo,hi,axis)),arms={},paths={})
        for arm in ARMS:
            folder=OLD if arm=='baseline' and panel!='new_eval' else predictions/arm
            receipt=old_receipt if folder==OLD else receipts[arm]
            paths={kind:folder/panel/kind/(id+'.npy') for kind in ('depth','raw_depth')}
            for path in paths.values():assert sha(path)==receipt['hashes'][path.relative_to(folder).as_posix()]
            depth=np.load(paths['depth']);assert depth.shape==(360,640)
            depth=np.where(np.isfinite(depth)&(depth>=.5)&(depth<=4.),depth,np.nan)
            points=c.fov(c.spatial.depth_points(depth))
            row['arms'][arm]=c.queries(lambda lo,hi,axis:c.point_contact(points,lo,hi,axis))
            row['paths'][arm]={k:str(p) for k,p in paths.items()}
        sealed.append(row)
    write(output/'predictions.json',sealed)
    write(output/'prediction-seal.json',dict(frames=672,predictions_sha256=sha(output/'predictions.json'),
        arm_receipts={a:sha(predictions/a/'receipt.json') for a in ARMS},historical_baseline_receipt_sha256=sha(OLD/'receipt.json')))
    # Scene cuboids and native arrays are evaluator-only below this point.
    prior=verify(OLD_TRUTH);historical={(r['panel'],r['id']):r for r in read(OLD_TRUTH/'truth.json')}
    prep_receipt=read(prepared/'receipt.json')
    assert prep_receipt['status']=='PASS' and prep_receipt['eval_frames']==96
    assert sha(prepared/'eval_observations/examples.json')==prep_receipt['hashes']['eval_observations/examples.json']
    truth_path=prepared/'eval_truth/truth.json'
    assert sha(truth_path)==prep_receipt['hashes']['eval_truth/truth.json']
    new_document=read(truth_path);new_truth={r['id']:r for r in new_document['frames']}
    assert set(new_truth)=={r['id'] for r in public}
    cap_receipts={panel:read(path/'receipt.json') for panel,path in c.CAPTURES.items()}
    drows=[];wrows=[];prows=[];frows=[];truth_rows=[]
    for row in sealed:
        panel,id=row['panel'],row['id']
        if panel=='new_eval':
            frame=new_truth[id];native_path=Path(frame['target_depth'])
            assert sha(native_path)==frame['target_depth_sha256']
            extra=tuple(new_document[k] for k in ('background','floor') if new_document.get(k))
            truth=dict(distance={part:[c.finite(oracle.oracle(frame,part,w,extra_bounds=extra)) for w in c.WIDTHS[part]] for part in c.BANDS},
                width={part:c.finite(oracle.oracle_width(frame,part,extra_bounds=extra)) for part in c.BANDS})
            common={k:frame[k] for k in ('family','episode','time_s')}
        else:
            cached=historical[(panel,id)];truth=cached['truth'];common={k:cached[k] for k in ('family','episode','time_s')}
            native_path=c.CAPTURES[panel]/'frame'/id/'native-left-depth.npy'
            assert sha(native_path)==cap_receipts[panel]['hashes'][f'frame/{id}/native-left-depth.npy']
        native=np.load(native_path);assert native.shape==(360,640)
        npoints=c.fov(c.spatial.depth_points(np.where(np.isfinite(native)&(native>=.5)&(native<=4),native,np.nan)))
        visible=c.queries(lambda lo,hi,axis:c.point_contact(npoints,lo,hi,axis))
        common.update(panel=panel,id=id,cohort=row['cohort'],no_frame_hints=row['no_frame_hints'],frame_hints=row['frame_hints'])
        truth_rows.append(dict(common,truth=truth,native_visible=visible))
        for arm in ARMS:
            depth=np.load(row['paths'][arm]['depth']);raw=np.load(row['paths'][arm]['raw_depth'])
            pix,far=pixel_stats(depth,raw,native);frows.append(dict(common,arm=arm,**far))
            for part,value in pix.items():prows.append(dict(common,arm=arm,part=part,**value))
            for suffix in ('','_union'):
                name=arm+suffix;query=row['arms'][arm]
                for part in c.BANDS:
                    for wi,width in enumerate(c.WIDTHS[part]):
                        pred=query['distance'][part][wi]
                        if suffix:pred=c.minimum(pred,row['tof']['distance'][part][wi])
                        drows.append(dict(common,arm=name,part=part,width=width,pred=pred,truth=truth['distance'][part][wi],native_visible=visible['distance'][part][wi]))
                    pred=query['width'][part]
                    if suffix:pred=c.minimum(pred,row['tof']['width'][part])
                    wrows.append(dict(common,arm=name,part=part,pred=pred,truth=truth['width'][part],native_visible=visible['width'][part]))
    summaries={};alerts={};alert_rows=[]
    for cohort in ('new_eval','historical'):
        summaries[cohort]={};alerts[cohort]={}
        for arm in [a+s for a in ARMS for s in ('','_union')]:
            d=[r for r in drows if r['cohort']==cohort and r['arm']==arm]
            w=[r for r in wrows if r['cohort']==cohort and r['arm']==arm]
            detail=dict(distance={k:c.boundary_stats(v) for k,v in slices(d).items()},width={k:c.boundary_stats(v) for k,v in slices(w).items()},
                width_positive=c.boundary_stats([r for r in w if r['truth'] is not None and r['truth']>1e-8]),
                width_zero=c.boundary_stats([r for r in w if r['truth'] is not None and r['truth']<=1e-8]),
                scene_contact_native_absent=c.boundary_stats([r for r in d if r['truth'] is not None and r['native_visible'] is None]))
            summaries[cohort][arm]=detail
            panels={}
            for panel in dict.fromkeys(r['panel'] for r in d):
                q=[r for r in d if r['panel']==panel and r['width']==c.WIDTHS[r['part']][1]]
                assert all(q[i]['id']==q[i+1]['id'] for i in range(0,len(q),2))
                a=np.array([r['pred'] is not None and r['pred']<=3 for r in q]).reshape(-1,2)
                gt=np.array([r['truth'] is not None and r['truth']<=3 for r in q]).reshape(-1,2)
                final=c.spatial.hysteresis(a,[r['episode'] for r in q[::2]])
                panels[panel]=dict(raw=c.counts(a,gt),final=c.counts(final,gt))
                for k,r in enumerate(q):alert_rows.append(dict(cohort=cohort,panel=panel,id=r['id'],part=r['part'],arm=arm,truth=bool(gt.flat[k]),raw=bool(a.flat[k]),final=bool(final.flat[k])))
            alerts[cohort][arm]={stage:{k:sum(p[stage][k] for p in panels.values()) for k in ('TP','FP','FN','TN')} for stage in ('raw','final')}
            alerts[cohort][arm]['panels']=panels
    pixels={cohort:{arm:{k:pixel_aggregate(v) for k,v in slices([r for r in prows if r['cohort']==cohort and r['arm']==arm]).items()}
        for arm in ARMS} for cohort in ('new_eval','historical')}
    fars={cohort:{arm:{k:sum(r[k] for r in frows if r['cohort']==cohort and r['arm']==arm)
        for k in ('far_native_fov_pixels','far_native_estimated_near')} for arm in ARMS} for cohort in ('new_eval','historical')}
    # Reproduce unadapted VPP direct-point contact baseline on original history.
    previous=read(OLD_TRUTH/'summary.json')
    for suffix in ('','_union'):
        for kind in ('distance','width'):
            current=summaries['historical']['baseline'+suffix][kind]['all']
            expected=previous['summaries']['guided_raw'+suffix][kind]['all']
            for key,value in current.items():
                if isinstance(value,float):assert np.isclose(value,expected[key],rtol=0,atol=1e-10),(key,value,expected[key])
                else:assert value==expected[key],(key,value,expected[key])
    for name,values in [('truth.json',truth_rows),('distance-queries.json',drows),('width-queries.json',wrows),('pixel-queries.json',prows),('far-pixels.json',frows),('alerts.json',alert_rows)]:write(output/name,values)
    write(output/'summary.json',dict(status='PASS',frames=dict(new_eval=96,historical=576),summaries=summaries,alerts=alerts,pixels=pixels,far_pixels=fars,
        historical_baseline_parity=True,evidence='NEW_LAYOUT_CONSUMED_DEVELOPMENT_PLUS_REPEATED_HISTORICAL_DIAGNOSTIC',
        definitions=dict(distance='Camera x0.5..4m; three fixed widths; scene LP retains hidden surfaces',
            width='2min|y| horizon3m and fullwidthcap1m; positive and zero truth separated',
            pixels='DefaultBODY0.56/HEAD0.36 camera bands, nativeZ0.5..4m; samequery+Z5cm, missing fails fixed denominator',
            alerts='Defaultwidth contact<=3m then unchangedtwo-on/two-off per episode',
            nohint='Zero whole-frame eligible projected publicToFcentre rays, independent of scene labels',
            raw_depth='Unfiltered positive axial depths on native defaultquery pixels; includes above4m and missing'),
        prediction_seal_sha256=sha(output/'prediction-seal.json')))
    write(output/'receipt.json',dict(status='PASS',evaluator_sha256=sha(__file__),hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--predictions',type=Path,default=PRED);p.add_argument('--prepared',type=Path,default=PREP)
    a=p.parse_args();evaluate(a.output,a.predictions,a.prepared)

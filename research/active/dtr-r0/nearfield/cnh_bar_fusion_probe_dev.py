"""Four frozen same-cost cache fusion probes; no sampling, training or inference."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S

OUT=B.ROOT/'artifacts.local/work/cnh-bar-fusion-vertical-dev-20261009/fusion'
LOCAL=B.ROOT/'artifacts.local/work/cnh-bar-local-readout-dev-20261009'
REP=B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008'
FRAMES=np.arange(3,16)


def save(name,value):B.save(OUT/name,value)


def smooth(raw):
    raw=np.asarray(raw,dtype=float);answer=np.empty_like(raw)
    for frame in range(13):
        start=max(0,frame-4);weights=2.**np.arange(frame-start+1)
        answer[...,frame,:]=np.sum(raw[...,start:frame+1,:]*weights[:,None],axis=-2)/weights.sum()
    return answer


def nearest(values,target,floor=None):
    """Joint-slot whole-score ties, higher threshold for equal absolute residual."""
    values=np.asarray(values,dtype=float).reshape(-1);finite=np.sort(values[np.isfinite(values)])
    candidates=np.r_[np.unique(finite),np.inf]
    if floor is not None:candidates=candidates[candidates>=floor]
    costs=len(finite)-np.searchsorted(finite,candidates,side='left')
    chosen=np.lexsort((-candidates,np.abs(costs-target)))[0]
    return float(candidates[chosen]),int(costs[chosen])


def gate_mask(m3,raised,gate):
    return ((m3>=0)&(m3<raised)) if gate=='mid_nonnegative' else m3<raised


def alarm(m3,local,raised,local_theta,gate):
    return (m3>=raised)|(gate_mask(m3,raised,gate)&(local>=local_theta))


def metrics(flags,category):
    joint=flags.any(-1);contact=category=='contact';clear=(category=='clear').all(1)
    timely=flags[:,:,:11].any(2)&contact[:,None,:]
    late=flags[:,:,11:].any(2)&contact[:,None,:]&~timely
    cc=joint[clear]
    summary=dict(counts=timely.sum((0,1)).tolist(),late_counts=late.sum((0,1)).tolist(),
        clear_slots=int(cc.sum()),clear_denominator=int(cc.size),clear_segments=int(cc[:,:,0].sum()+((~cc[:,:,:-1])&cc[:,:,1:]).sum()),
        clear_clips=int(cc.any(2).sum()),physical_contact_any_height=int(joint[:,:,:11].any(2)[contact.any(1)].sum()),
        pass_clips=int(joint[(category=='pass').any(1)&~contact.any(1)].any(2).sum()))
    return summary,timely


def compare(a,b,category,ids=None):
    result=[]
    for q in (0,1):
        indices=np.flatnonzero(category[:,q]=='contact');aa=a[indices,:,q];bb=b[indices,:,q]
        gains=np.argwhere(~aa&bb);losses=np.argwhere(aa&~bb)
        result.append(dict(height=('HEAD','BODY')[q],denominator=int(aa.size),baseline=int(aa.sum()),candidate=int(bb.sum()),
            gain=len(gains),loss=len(losses),net=int(bb.sum()-aa.sum()),
            gain_keys=[[int(indices[i]),int(k)] for i,k in gains],loss_keys=[[int(indices[i]),int(k)] for i,k in losses],
            gain_scene_keys=[[int(ids[indices[i]] if ids is not None else indices[i]),int(k)] for i,k in gains],
            loss_scene_keys=[[int(ids[indices[i]] if ids is not None else indices[i]),int(k)] for i,k in losses]))
    return result


def prepare():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve frozen fusion PLAN')
    inputs=[S.OUT/f for f in ('PLAN.json','physical.npz','evaluated.npz')]+[
        LOCAL/f for f in ('PLAN.json','cached_scores.npz','mc_scores.npz','result_analysis.json')]+[
        REP/'fp16/PLAN.json']
    sources=[Path(__file__),Path(B.__file__),Path(S.__file__)]
    plan=dict(task='CNH_BAR_FUSION_PROBE_DEV_20261009',lane='EXPLORE consumed Development cache',
        authorization='User advanced roundtable first two recommendations; fusion cache branch delegated by primary agent',
        budget_CPU_wall_seconds=180,goal='Four declared same-clear-slot-cost M3/local fusion probes with all paired losses retained',
        adjustable_scope='Implementation repair, cache statistics and focused checks only; no new noise/model/geometry/threshold/patch sweep',
        primary='492 original scenes xK4 x13 frames f3..15; timely f3..13, latef14..15; HEAD/BODY688 contact events each',
        workpoints=[dict(k=k,gate=gate) for k in (5,10) for gate in ('mid_nonnegative','unbounded_below')],
        raise_rule='Original M3 threshold fixed; raised threshold>=originaltheta selects whole-score-tie joint-clear slot cost nearest46-k, higher threshold on equal absolute residual.',
        fusion_rule='alarm=(smoothM3>=raisedtheta) OR (gate AND smoothLocal>=newlocaltheta); mid gate M3 in[0,raisedtheta), unbounded gate M3<raisedtheta. Both gates predeclared; no event-based choice.',
        local_rule='On clear slots not already alarmed by raised-M3, max eligible local score over two queries. Select whole-score ties nearest totalbaseline46, higher threshold on equal absolute residual. Shared query threshold. Preserve observed residual.',
        baseline_rules='Original M3theta and original standalone localthreshold remain unchanged for comparison and timely-event M3-only/local-only/both/neither decomposition.',
        supplementary='Reuse37 saved conditional endpoints x64 and full13 M3/local outputs. Apply frozen primary thresholds; report observed clearcost, no recalibration or fresh geometry claim.',
        grouping='Evaluator-only family/variant/height/dark tags joined after fixed score/gate formula; gain/loss keys and earliest timely frames retained. Slot-score CSVs are not event denominators.',
        decision_check='Four points all reported without winner selection or promotion guardrail. A failed local replacement does not rule out fusion. Development reuse is not fresh confirmation.',
        stop='Complete within cumulative180s including prepare and failed attempts; preserve failures/partial work; no training/sampling/inference or Git delivery by this branch',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},source_sha256={B.logical_path(p):B.sha(p) for p in sources})
    plan['prepare_seconds']=time.monotonic()-began
    save('PLAN.json',plan);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',plan['prepare_seconds'],flush=True)


def write_csv(name,records):
    with (OUT/name).open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def run():
    began=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text())
    spent=plan['prepare_seconds']+sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('failure_*.json'))
    def check():
        if spent+time.monotonic()-began>180:raise TimeoutError('fusion cumulative CPU wall180s')
    points=[]
    try:
        for name,digest in {**plan['inputs_sha256'],**plan['source_sha256']}.items():
            assert B.sha(B.ROOT/name)==digest,('Changed frozen fusion input/source',name)
        # Focused math fixture: inclusive ties and missing-gate slots.
        th,cost=nearest([1.,1.,2.,3.],3);assert th==2. and cost==2
        thzero,czero=nearest([-np.inf,-np.inf],0);assert np.isinf(thzero) and czero==0
        sm=smooth(np.arange(13,dtype=float).reshape(1,1,13,1));assert sm[0,0,0,0]==0 and abs(sm[0,0,4,0]-98/31)<1e-12
        localold=json.loads((LOCAL/'result_analysis.json').read_text());theta0=float(localold['baseline']['threshold']);localth0=float(localold['candidate']['threshold'])
        with np.load(S.OUT/'evaluated.npz') as d:category=d['category'];m3=d['scores']
        with np.load(S.OUT/'physical.npz') as d:original_raw=d['raw']
        np.testing.assert_allclose(smooth(original_raw),m3,rtol=1e-13,atol=1e-13)
        with np.load(LOCAL/'cached_scores.npz') as d:local=smooth(d['raw'])
        with np.load(LOCAL/'mc_scores.npz') as d:
            ids=d['endpoint_ids'];mc_m3=smooth(d['m3_raw']);mc_local=smooth(d['raw'])
        rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
        darkids=json.loads((REP/'fp16/PLAN.json').read_text())['contact_scene_ids'];dark=np.zeros(492,bool);dark[darkids]=True
        clear=(category=='clear').all(1);mc_category=np.r_[category[ids[:-1]],np.array([['clear','clear']])]
        oldflags=m3>=theta0;localflags=local>=localth0;old,oldt=metrics(oldflags,category);standalone,localt=metrics(localflags,category)
        mc_oldflags=mc_m3>=theta0;mc_old,mc_oldt=metrics(mc_oldflags,mc_category)
        assert old['clear_slots']==46 and old['clear_denominator']==4576 and old['counts']==[517,406]
        assert standalone['counts']==[530,423] and standalone['clear_slots']==46
        groups=np.where(oldt,np.where(localt,'both','M3_only'),np.where(localt,'local_only','neither'))
        eventgroups=[]
        for q in (0,1):
            for group in ('M3_only','local_only','both','neither'):
                mask=(category[:,q]=='contact')[:,None]&(groups[:,:,q]==group)
                eventgroups.append(dict(height=('HEAD','BODY')[q],group=group,events=int(mask.sum()),
                    event_keys=[[int(i),int(k)] for i,k in np.argwhere(mask)]))
        write_csv('baseline_event_groups.csv',[dict(scene=int(i),replica=int(k),height=('HEAD','BODY')[q],group=groups[i,k,q],
            baseline=int(oldt[i,k,q]),local=int(localt[i,k,q]),family=rows[i]['family'],variant=rows[i]['variant'],dark4cm=int(dark[i]))
            for i,k,q in np.argwhere(np.broadcast_to((category=='contact')[:,None,:],oldt.shape))])
        outputs=[];ledgers=[]
        for wp in plan['workpoints']:
            check();k=wp['k'];gate=wp['gate'];name=f'k{k}_{gate}'
            raised,raised_cost=nearest(m3[clear].max(-1),old['clear_slots']-k,floor=theta0)
            base=(m3>=raised);gateflag=gate_mask(m3,raised,gate)
            basejoint=base[clear].any(-1)
            eligible=np.where(gateflag[clear],local[clear],-np.inf).max(-1)
            theta,added_cost=nearest(eligible[~basejoint],old['clear_slots']-raised_cost)
            flags=alarm(m3,local,raised,theta,gate);summary,ft=metrics(flags,category)
            assert summary['clear_slots']==raised_cost+added_cost
            comparison=compare(oldt,ft,category)
            families=[]
            for family in S.FAMILIES:
                mask=np.array([r['family']==family for r in rows]);families.append(dict(family=family,paired=compare(oldt[mask],ft[mask],category[mask],np.flatnonzero(mask))))
            darkpaired=compare(oldt[dark],ft[dark],category[dark],np.flatnonzero(dark))
            lengthgroups=[]
            for variant in ('length0.3_thick0.04','length0.9_thick0.04'):
                mask=dark&np.array([r['variant']==variant for r in rows]);lengthgroups.append(dict(variant=variant,paired=compare(oldt[mask],ft[mask],category[mask],np.flatnonzero(mask))))
            exchange=[]
            for q in (0,1):
                for group in ('M3_only','local_only','both','neither'):
                    mask=(category[:,q]=='contact')[:,None]&(groups[:,:,q]==group);aa=oldt[:,:,q][mask];bb=ft[:,:,q][mask]
                    exchange.append(dict(height=('HEAD','BODY')[q],group=group,events=int(mask.sum()),baseline=int(aa.sum()),candidate=int(bb.sum()),
                        gain=int((~aa&bb).sum()),loss=int((aa&~bb).sum())))
            mcflags=alarm(mc_m3,mc_local,raised,theta,gate);mc_summary,mc_t=metrics(mcflags,mc_category)
            mccomparison=compare(mc_oldt,mc_t,mc_category,ids)
            item=dict(workpoint=name,k=k,gate=gate,gate_lower_bound=0 if gate=='mid_nonnegative' else None,
                original_M3_threshold=theta0,original_local_threshold=localth0,raised_M3_threshold=raised,new_local_threshold=theta,
                raised_clear_target=46-k,raised_clear_slots=raised_cost,raised_clear_residual=raised_cost-(46-k),added_clear_slots=added_cost,
                metrics=summary,total_clear_residual=summary['clear_slots']-old['clear_slots'],paired=comparison,
                families=families,dark4cm=darkpaired,dark_length_groups=lengthgroups,timely_source_exchange=exchange,
                MC_metrics=mc_summary,MC_clear_residual_vs_frozen_M3=mc_summary['clear_slots']-mc_old['clear_slots'],MC_paired=mccomparison,
                MC_thresholds_recalibrated=False)
            points.append(item);outputs.append(flags)
            for i,replica,q in np.argwhere(np.broadcast_to((category=='contact')[:,None,:],oldt.shape)):
                bf=np.flatnonzero(oldflags[i,replica,:11,q]);ff=np.flatnonzero(flags[i,replica,:11,q]);rf=np.flatnonzero(base[i,replica,:11,q])
                lf=np.flatnonzero((gateflag&(local>=theta))[i,replica,:11,q])
                ledgers.append(dict(workpoint=name,scene=int(i),replica=int(replica),height=('HEAD','BODY')[q],family=rows[i]['family'],variant=rows[i]['variant'],placement=rows[i]['placement'],rho=rows[i]['rho'],dark4cm=int(dark[i]),
                    source_group=groups[i,replica,q],baseline=int(bf.size>0),candidate=int(ff.size>0),gain=int(bf.size==0 and ff.size>0),loss=int(bf.size>0 and ff.size==0),
                    baseline_first_timely_frame=int(FRAMES[bf[0]]) if bf.size else '',fusion_first_timely_frame=int(FRAMES[ff[0]]) if ff.size else '',
                    raised_M3_first_timely_frame=int(FRAMES[rf[0]]) if rf.size else '',local_gate_first_timely_frame=int(FRAMES[lf[0]]) if lf.size else ''))
            save(name+'.json',item)
            print('POINT',name,'cost',summary['clear_slots'],'raised/add',raised_cost,added_cost,'paired',[(p['height'],p['gain'],p['loss']) for p in comparison],flush=True)
        write_csv('event_ledger.csv',ledgers)
        for label,mask in (('clear',np.broadcast_to(clear[:,None,None,None],m3.shape)),('contact',np.broadcast_to((category=='contact')[:,None,None,:],m3.shape))):
            records=[dict(scene=int(i),replica=int(replica),frame=int(FRAMES[f]),height=('HEAD','BODY')[q],
                timely_window=int(f<11),M3_score=float(m3[i,replica,f,q]),local_score=float(local[i,replica,f,q]),
                M3_original_alarm=int(oldflags[i,replica,f,q]),local_original_alarm=int(localflags[i,replica,f,q]))
                for i,replica,f,q in np.argwhere(mask)]
            write_csv('score_'+label+'.csv',records)
        np.savez_compressed(OUT/'scores.npz',m3_smooth=m3,local_smooth=local,workpoints=np.array([p['workpoint'] for p in points]),fusion_alarm=np.stack(outputs),
            original_M3_alarm=oldflags,original_local_alarm=localflags,MC_endpoint_ids=ids,MC_M3_smooth=mc_m3,MC_local_smooth=mc_local,
            MC_fusion_alarm=np.stack([alarm(mc_m3,mc_local,p['raised_M3_threshold'],p['new_local_threshold'],p['gate']) for p in points]))
        # Focused conservation checks do not depend on any apparent gain.
        assert len(ledgers)==4*1376
        for point in points:
            for q,pair in enumerate(point['paired']):
                assert pair['candidate']-pair['baseline']==pair['gain']-pair['loss']
                assert sum(g['events'] for g in point['timely_source_exchange'] if g['height']==pair['height'])==688
                assert sum(g['loss'] for g in point['timely_source_exchange'] if g['height']==pair['height'])==pair['loss']
                assert sum(g['paired'][q]['denominator'] for g in point['families'])==688
        checks=dict(status='PASS',math_tie_fixture=True,missing_gate_fixture=True,smoothing_fixture=True,
            original_M3_smooth_max_abs=float(np.max(np.abs(smooth(original_raw)-m3))),workpoints=4,event_ledger_rows=len(ledgers),
            event_group_conservation=True,paired_gain_loss_conservation=True,family_denominator_conservation=True,
            source_boundary='Fusion consumes two fixed score arrays and predeclared thresholds/gates. category used only for clear calibration and reporting; geometry/expectation not consumed by alarm().')
        save('focused_checks.json',checks)
        check();result=dict(status='COMPLETE',seconds=time.monotonic()-began,cumulative_seconds=spent+time.monotonic()-began,
            baseline=old,standalone_local=standalone,baseline_MC=mc_old,event_groups=eventgroups,workpoints=points,
            cost_matching='Joint-clear time slots only; segments/clips/pass remain independently reported',
            decision='ALL_FOUR_FIXED_POINTS_REPORTED; no automatic winner selection or M3/policy promotion',new_samples=0,model_inference_examples=0,training=0)
        save('result.json',result);print('COMPLETE',result['cumulative_seconds'],flush=True)
    except BaseException as error:
        save('failure_'+str(time.time_ns())+'.json',dict(error=repr(error),seconds=time.monotonic()-began,completed_workpoints=len(points)));raise


if __name__=='__main__':globals()[sys.argv[1]]()

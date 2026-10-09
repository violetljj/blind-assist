"""Cache-only query error cost decomposition and focused readout controls."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S
import cnh_bar_fusion_probe_dev as F

OUT=B.ROOT/'artifacts.local/work/cnh-bar-query-cost-dev-20261009'
POSE=B.ROOT/'artifacts.local/work/cnh-bar-transfer-dev-20261009/pose'
ANGLES=(-3,-2,-1,0,1,2,3)
T0=.8557642486787612
TM=.9404184587540165
TL=4.625390338985158
def branch(a):return 'ideal' if a==0 else f'query_{abs(a)}_{int(np.sign(a)):+d}'
def write(name,obj):B.save(OUT/name,obj)
def csvwrite(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load():
    with np.load(S.OUT/'geometry.npz') as d:cat=d['category']
    arrays={}
    for a in ANGLES:
        with np.load(POSE/(branch(a)+'.npz')) as d:arrays[a]=(F.smooth(d['m3_raw']),F.smooth(d['local_raw']))
    return cat,arrays,json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
def prepare():
    t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve original query cost PLAN')
    inputs=[S.OUT/'PLAN.json',S.OUT/'geometry.npz',POSE/'PLAN.json',POSE/'result_analysis.json',*[POSE/(branch(a)+'.npz') for a in ANGLES]]
    write('PLAN.json',dict(task='CNH_BAR_QUERY_COST_DEV_20261009',lane='EXPLORE consumed simulated cache',base_commit='2848e983',
        authorization='User 推进 after recommendation to investigate query direction error clear cost',
        goal='Separate original M3/raised M3/local rescue in query-yaw clear cost; compare useful readout controls after this diagnosis',
        budget_CPU_wall_seconds=240,independent_geometry_budget_seconds=120,independent_audit_budget_seconds=120,
        unit='Cumulative stage wall seconds including source failures and focused checks; no sampling/training/model inference/GPU',
        data='Seven saved query angles -3..3deg, original492scene/K4/full13output/bothqueries. Contacttruth stays true ideal query.',
        diagnosis='Fixed original and fusion thresholds; jointclear raised-only/local-only/both/neither, original removal/addition, ideal-to-error added/lost slots and scenes by frame/family/side/height. Evaluation-only true boxes versus estimated query separately.',
        controls='After inspecting primary decomposition, select one mechanism-specific control with explicit scope and stop in CONTROLS_PLAN. No larger patch/threshold/angle sweep or policy promotion.',
        baseline='Original M3 theta=.8557642486787612; fixed k5 OR raised M3 .9404184587540165 / local4.625390338985158',
        decision_check='Positive relative timely alone cannot justify: actualclear slots/segments/clips and ideal retention/paired samecondition losses matter. Clearjoint4576, eachheight688/dark56; smallest unit1event/1slot. Case-specific cost rematching is diagnostic not observable error adaptation.',
        stop='Complete diagnosis plus one focused control and relevant audit within cap; no unseen-geometry/pose robustness claim. Preserve discarded cases and failed runs.',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},source_sha256=B.sha(Path(__file__)),prepare_seconds=time.monotonic()-t))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',flush=True)
def diagnose():
    t=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());spent=plan['prepare_seconds']+sum(json.loads(x.read_text())['seconds'] for x in OUT.glob('failure_*.json'))
    try:
        for p,digest in plan['inputs_sha256'].items():assert B.sha(B.ROOT/p)==digest,p
        cat,arrays,rows=load();clear=(cat=='clear').all(1);ids=np.flatnonzero(clear)
        im,il=arrays[0];ideal=(im>=TM)|(il>=TL);idealjoint=ideal.any(-1)
        summary=[];slots=[];events=[]
        for a,(m,l) in arrays.items():
            original=m>=T0;raised=m>=TM;strong=l>=TL;flags=raised|strong
            om,ot=F.metrics(original,cat);rm,rt=F.metrics(raised,cat);lm,lt=F.metrics(strong,cat);fm,ft=F.metrics(flags,cat)
            oj=original.any(-1);rj=raised.any(-1);lj=strong.any(-1);fj=flags.any(-1)
            removed=oj&~fj;added=~oj&fj
            groups={'raised_only':rj&~lj,'local_only':~rj&lj,'both':rj&lj,'neither':~rj&~lj}
            counts={k:int(v[clear].sum()) for k,v in groups.items()}
            assert counts['raised_only']+counts['local_only']+counts['both']==fm['clear_slots']
            summary.append(dict(angle=a,branch=branch(a),original=om,raised=rm,strong_local=lm,fusion=fm,
                clear_groups=counts,removed_original_clear=int(removed[clear].sum()),added_local_clear=int(added[clear].sum()),
                added_vs_ideal_clear=int((~idealjoint&fj)[clear].sum()),lost_vs_ideal_clear=int((idealjoint&~fj)[clear].sum()),
                same_angle_paired=F.compare(ot,ft,cat)))
            for i in ids:
                for k in range(4):
                    for j in range(13):
                        slots.append(dict(angle=a,scene=int(i),replica=k,frame=j+3,family=rows[i]['family'],side=rows[i]['side'],placement=rows[i]['placement'],context=rows[i]['context'],
                            M3=int(oj[i,k,j]),raised=int(rj[i,k,j]),local=int(lj[i,k,j]),fusion=int(fj[i,k,j]),
                            original_removed=int(removed[i,k,j]),local_added=int(added[i,k,j]),ideal=int(idealjoint[i,k,j]),
                            ideal_added=int(not idealjoint[i,k,j] and fj[i,k,j]),ideal_lost=int(idealjoint[i,k,j] and not fj[i,k,j]),
                            M_HEAD=float(m[i,k,j,0]),M_BODY=float(m[i,k,j,1]),L_HEAD=float(l[i,k,j,0]),L_BODY=float(l[i,k,j,1])))
            for i,k,q in np.argwhere(np.broadcast_to((cat=='contact')[:,None,:],ot.shape)):
                events.append(dict(angle=a,scene=int(i),replica=int(k),height=('HEAD','BODY')[q],M3=int(ot[i,k,q]),fusion=int(ft[i,k,q]),gain=int(not ot[i,k,q] and ft[i,k,q]),loss=int(ot[i,k,q] and not ft[i,k,q])))
        csvwrite('clear_slots.csv',slots);csvwrite('event_ledger.csv',events)
        write('result_diagnosis.json',dict(status='COMPLETE',seconds=time.monotonic()-t,summary=summary,clear_slot_rows=len(slots),event_rows=len(events),
            conclusion='Fixed threshold decomposition only, all7angles preserved; local-only counts must not be confused with all strong-local alerts or with event gains.'))
        for v in summary:print(v['angle'],'clear orig/raised/local/fusion',*[v[x]['clear_slots'] for x in ('original','raised','strong_local','fusion')],'groups',v['clear_groups'],flush=True)
        assert spent+time.monotonic()-t<240
    except BaseException as e:write('failure_'+str(time.time_ns())+'.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise
def control_prepare():
    if (OUT/'CONTROLS_PLAN.json').exists():raise FileExistsError('Preserve control plan')
    diagnosis=json.loads((OUT/'result_diagnosis.json').read_text())
    assert diagnosis['status']=='COMPLETE'
    write('CONTROLS_PLAN.json',dict(task='Same query cost diagnosis, one M3 angular-agreement control',
        selected_after_diagnosis=True,reason='At+3deg170/175 jointclear slots already raised M3; center-local-only5 unchanged fromideal. Test direction-sensitive M3 support only.',
        mechanism='Smooth each of3 M3 query directions using originallast5, then score_min=min(Ma-1,Ma,Ma+1); alarm=(score_min>=fixedTM) OR (center_local>=fixedTL). Local branch stays center. Allquery thresholds unchanged.',
        centers=[-2,-1,0,1,2],neighbor_offset_degrees=[-1,0,1],
        missing='Center±3 requires saved±4 absent; explicitly NOT_EVALUABLE. No substitute one-sided agreement or oracle-known-error gate.',
        budget='Part of initial240s cumulativeCPU, no additional model inference/sampling/GPU. No further aggregation/offset/threshold selection.',
        primary='Paired center-fixed-fusion vsguard slot/segment/clip reductions and original timely losses; compare with samecenter originalM3 separately. Guard must subset fixedfusion, cannot rescue events versus it.',
        cost='Deployment would need3 query projections+3 frozen5model ensembles (15 single-model forwards), versus1query+5 forwards now; cache replay does not measure added latency/energy.',
        decision='Do not recommend replacement fromclear reduction alone. Report all5 centers inclideal retention. This diagnosis does not establish robustness at worst±3.',
        source_sha256=B.sha(Path(__file__)),diagnosis_sha256=B.sha(OUT/'result_diagnosis.json')))
    (OUT/'control_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('CONTROL PREPARED fixed3query M3 agreement only',flush=True)
def control():
    t=time.monotonic();plan=json.loads((OUT/'CONTROLS_PLAN.json').read_text())
    assert plan['source_sha256']==B.sha(Path(__file__))
    cat,arrays,rows=load();summary=[];out={};ledger=[]
    for a in plan['centers']:
        m,l=arrays[a];low=np.minimum.reduce([arrays[a+delta][0] for delta in (-1,0,1)])
        original=m>=T0;fixed=(m>=TM)|(l>=TL);guard=(low>=TM)|(l>=TL)
        assert not np.any(guard&~fixed)
        om,ot=F.metrics(original,cat);fm,ft=F.metrics(fixed,cat);gm,gt=F.metrics(guard,cat)
        summary.append(dict(center=a,original=om,fixed=fm,guard=gm,
            versus_fixed=F.compare(ft,gt,cat),versus_M3=F.compare(ot,gt,cat),
            clear_slots_removed=int((fixed.any(-1)&~guard.any(-1))[(cat=='clear').all(1)].sum())))
        out[f'center_{a:+d}_flags']=guard
        for i,k,q in np.argwhere(np.broadcast_to((cat=='contact')[:,None,:],ot.shape)):
            ledger.append(dict(center=a,scene=int(i),replica=int(k),height=('HEAD','BODY')[q],M3=int(ot[i,k,q]),fixed=int(ft[i,k,q]),guard=int(gt[i,k,q]),
                gain_vs_fixed=int(not ft[i,k,q] and gt[i,k,q]),loss_vs_fixed=int(ft[i,k,q] and not gt[i,k,q]),gain_vs_M3=int(not ot[i,k,q] and gt[i,k,q]),loss_vs_M3=int(ot[i,k,q] and not gt[i,k,q])))
    csvwrite('control_events.csv',ledger);np.savez_compressed(OUT/'control_flags.npz',**out)
    previous=json.loads((OUT/'PLAN.json').read_text())['prepare_seconds']+json.loads((OUT/'result_diagnosis.json').read_text())['seconds']
    previous+=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('failure_*.json'))
    assert previous+time.monotonic()-t<240
    write('result_control.json',dict(status='COMPLETE',seconds=time.monotonic()-t,cumulative_primary_seconds=previous+time.monotonic()-t,
        summary=summary,event_rows=len(ledger),not_evaluable_centers=[-3,3],training=0,new_inference=0,new_photons=0,
        limitation='One prescribed3querymin control, postdiagnosisDevelopment. Subset entails no gains againstcenterfixedfusion. No matchedcost,±3result or production compute/latency claim.'))
    for x in summary:print('CENTER',x['center'],'clear',x['fixed']['clear_slots'],'->',x['guard']['clear_slots'],'H/B',x['fixed']['counts'],'->',x['guard']['counts'],'loss',[q['loss'] for q in x['versus_fixed']],flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()

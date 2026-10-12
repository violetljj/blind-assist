"""Additive light alerts from current/untracked/tracked native-peak evidence.

No scoring fit. Fixed raw-evidence comparisons, cal-only extra slot budgets,
all original strong/full-single light grades retained; consumed Development.
"""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

ROOT = C.ROOT
TASK = ROOT/'artifacts.local/work/cnh-graded-peak-track-dev-20261010'
OUT = TASK/'alert'
RULES = ('frame_peak','bag5_sum','tracked_sum')
BUDGETS = ((5,32),(15,64),(30,128))  # additional clear/pass candidate union slots


def evidence(archive):
    names = archive['names'].tolist()
    f,v = archive['features'],archive['valid']
    def get(name):
        i = names.index(name)
        return f[...,i].astype(float),v[...,i]
    amplitude = archive['candidate_amplitude'][...,0].astype(float)
    current = archive['candidate_valid'][...,0]&np.isfinite(amplitude)
    bag,bv = get('untracked_top1sum5');bh,bhv = get('untracked_positive_frames5')
    track,tv = get('tracked_sum');hit,hv = get('hits')
    return dict(frame_peak=np.where(current,amplitude,np.nan),
        bag5_sum=np.where(current&bv&bhv&(bh>=2),bag,np.nan),
        tracked_sum=np.where(tv&hv&(hit>=2)&(archive['best_track_id']>=0),track,np.nan))


def calibrate(score,eligible,category,mask,clear_cap,pass_cap):
    s,e,c = score[mask],eligible[mask],category[mask]
    passed = (c=='pass').any(1)&~(c=='contact').any(1)
    clear = (c=='clear').all(1)
    candidate = np.where(e&np.isfinite(s),s,-np.inf)
    theta = max(C.at_most(candidate[clear].max(-1),clear_cap),C.at_most(candidate[passed].max(-1),pass_cap))
    # Evidence is nonnegative. No finite candidates/ties or a nonbinding cap
    # uses its defined lower bound; invalid evidence remains excluded explicitly.
    if not np.isfinite(theta): theta = 0.
    flags = e&np.isfinite(s)&(s>=theta)
    ac,ap = int(flags[clear].any(-1).sum()),int(flags[passed].any(-1).sum())
    assert ac<=clear_cap and ap<=pass_cap
    return dict(theta=float(theta),clear_cap=clear_cap,pass_cap=pass_cap,
        actual_extra_candidate_clear_slots=ac,actual_extra_candidate_pass_slots=ap,
        clear_unused=clear_cap-ac,pass_unused=pass_cap-ap,
        clear_slot_denominator=int(clear.sum())*4*13,pass_slot_denominator=int(passed.sum())*4*13)


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve prior alert exploration')
    C.save(OUT/'PLAN.json',dict(task='CNH_GRADED_PEAK_TRACK_DEV_20261010',base_commit='c782b6bd',
        authorization='User continues public peak tracks and intermittent support',lane='EXPLORE consumed simulated Development',
        goal='Timely weak contact rescue/advance from matched peak accumulation versus current and unmatched window controls, plus pass duration/cost',
        budgets=dict(total_CPU_command_wall_seconds_cap=900,tracks_cap_seconds=300,track_audit_cap_seconds=120,alert_audit_cap_seconds=60,main_cap_seconds=300,GPU_seconds=0),
        backend='Small native extraction/greedy association/cache statistics on CPU; prior CUDA count mismatch preserved, no GPU required',
        adjustable_scope='Fixed native association, fixed rules/budgets, implementation fixes, focused verification/report',
        stop='Finish declared rules and all seeds/budgets then deliver or phase/total caps; no old stop reopening/new draws/model fit/backbone inference/protected480/device',
        decision_check='Fixed fullsingle light has HEAD351/344/346 BODY307/310/312 of384 with pass149/168/158 of256; single event and frame are observable units. Seek rescues/earlier alerts with measured slot/clip/duration cost, not strict contact/pass separation or zero-cost gate.',
        rules=list(RULES),extra_budgets_clear_pass_slots=[list(v) for v in BUDGETS],
        evidence='All FP32 archives promoted FP64 before comparisons. Frame rank0 current amplitude; bag sum past5 distinct current rank0 and>=2 observed frames, requires currentcandidate. Track best actualcurrent-hit>=2 observations in5 frames, gap<=2; no repeated past8 or monotone distance condition.',
        policy='Only addgrade1 at prior query grade0. Original ordinaryOR grade2 and fullsingle grade1/time retained.',
        calibration='Cal9/11 only, cutoff max of complete-tie minima under two fixed additional candidatequery-union slot caps. Eligibility excludes baseline queryalerts and invalid scores. Nonnegative score lower bound0 if no binding finite cutoff. No validation choices.',
        cost='Caps bound added candidatequery unions, including slot where anotherquery alreadyalerts. Total joint slot/clip increase may be smaller; highestgrade light cost and longest runs reported separately.',
        limits='No target attribution/coverage/free/independent confirmation. Tracking starts f3, window warmup disclosed. Static/background/noise association can also persist.',
        inputs_sha256={str(p.relative_to(ROOT)):C.sha(p) for p in (
            TASK/'tracks/cal_tracks.npz',TASK/'tracks/validation_tracks.npz',TASK/'tracks/schema.json',
            C.OUT/'input_manifest.json',C.PARENT/'thresholds.json')},source_sha256=C.sha(Path(__file__))))
    try:
        ds = C.load();fit,partition = C.split_cal(ds['cal']['rows'])
        C.save(OUT/'cal_partition.json',dict(**partition,fit='NOT_RUN; only cal9/11 used for new raw-evidence cutoffs'))
        th = C.read(C.PARENT/'thresholds.json');scores = {}
        for split,d in ds.items():
            with np.load(TASK/'tracks'/f'{split}_tracks.npz',allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['scene_ids'],d['scene_ids'])
                scores[split] = evidence(archive)
        metrics,summary,cuts,ledger = {},[],{},[]
        saved = {split:[] for split in ds}
        for si,seed in enumerate(G.SEEDS):
            if time.monotonic()-began>=300: raise TimeoutError('Alert command300s cap')
            refs = {}
            for split,d in ds.items():
                ordinary = d['candidates'][0,si]
                strong = E.old_fusion(d['m3'],d['local'])|(ordinary>=th[str(seed)]['addition'])
                light = ~strong&(ordinary>=th[str(seed)]['single'])
                baseline = np.where(strong,2,np.where(light,1,0)).astype(np.int8)
                refs[split] = dict(baseline=baseline,strong=strong,prior=baseline>0)
            for rule in RULES:
                eligible = {split:(r['baseline']==0)&np.isfinite(scores[split][rule]) for split,r in refs.items()}
                for clear_cap,pass_cap in BUDGETS:
                    key = f'{seed}/{rule}/c{clear_cap}_p{pass_cap}'
                    cut = calibrate(scores['cal'][rule],eligible['cal'],ds['cal']['category'],~fit,clear_cap,pass_cap)
                    cuts[key] = cut
                    for split,d in ds.items():
                        r = refs[split];added = eligible[split]&(scores[split][rule]>=cut['theta'])
                        grade = np.where(added,1,r['baseline']).astype(np.int8)
                        assert np.all(grade>=r['baseline'])
                        np.testing.assert_array_equal(grade==2,r['strong'])
                        comparison = dict(prior_light=r['prior'],ordinary_OR=r['strong'],M3=d['m3']>=E.M3_THETA,old_fusion=E.old_fusion(d['m3'],d['local']))
                        result = G.describe(grade,d,comparison);metrics[f'{split}/{key}'] = result
                        base = G.describe(r['baseline'],d,comparison)
                        p = result['paired_any']['prior_light'];cat=d['category']
                        clear=(cat=='clear').all(1);passed=(cat=='pass').any(1)&~(cat=='contact').any(1)
                        summary.append(dict(split=split,seed=seed,rule=rule,clear_cap=clear_cap,pass_cap=pass_cap,
                            HEAD=result['any']['counts'][0],BODY=result['any']['counts'][1],
                            HEAD_rescue=p[0]['rescue'],BODY_rescue=p[1]['rescue'],HEAD_loss=p[0]['loss'],BODY_loss=p[1]['loss'],
                            HEAD_earlier=p[0]['earlier'],BODY_earlier=p[1]['earlier'],HEAD_later=p[0]['later'],BODY_later=p[1]['later'],
                            clear_slots=result['any']['clear_slots'],pass_clips=result['any']['pass_clips'],
                            extra_total_clear_slots=result['any']['clear_slots']-base['any']['clear_slots'],
                            extra_total_pass_clips=result['any']['pass_clips']-base['any']['pass_clips'],
                            extra_candidate_clear_slots=int(added[clear].any(-1).sum()),extra_candidate_pass_slots=int(added[passed].any(-1).sum()),
                            light_pass_slots=result['joint_costs']['pass']['light']['slots'],light_pass_longest_frames=result['joint_costs']['pass']['light']['longest_run_frames']))
                        saved[split].append((key,grade))
                        if split=='validation':
                            before,after = G.first(r['prior']),G.first(grade>0)
                            for n,row in enumerate(d['rows']):
                                for k in range(4):
                                    for q,height in enumerate(E.HEIGHTS):
                                        ledger.append(dict(key=key,scene=int(d['scene_ids'][n]),replica=k,height=height,category=cat[n,q],
                                            family=row['shape_family'],before_first=int(before[n,k,q]),after_first=int(after[n,k,q])))
        for split,records in saved.items():
            np.savez_compressed(OUT/f'{split}_grades.npz',keys=np.array([k for k,v in records]),grades=np.array([v for k,v in records]))
        G.write_csv(OUT/'summary.csv',summary);G.write_csv(OUT/'event_ledger.csv',ledger)
        C.save(OUT/'metrics.json',metrics);C.save(OUT/'calibrations.json',cuts)
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary),calibrations=len(cuts),event_rows=len(ledger),fit=0,GPU_seconds=0,source_sha256=C.sha(Path(__file__))))
        print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary))))
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__=='__main__':
    with threadpool_limits(limits=2): run()

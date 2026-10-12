"""CPU-only fixed EMA notice timing; candidate labels never receive truth."""
from __future__ import annotations
import csv
from pathlib import Path
import time
import numpy as np
import cnh_sector_notice_dev as S

B, E, Y = S.B, S.E, S.Y
OUT = B.A.ROOT/'artifacts.local/work/cnh-ema-bearing-opportunity-dev-20261008'


def quantize(angle):
    a = np.asarray(angle)
    a = (a + 180.) % 360. - 180.
    return np.where(~np.isfinite(a), -1, np.where(a < -10., 0, np.where(a > 10., 2, 1))).astype(np.int8)


def first_notice(alarm, contact, deadline):
    valid = alarm & contact[:, None] & (np.arange(13)[None, :] >= 2) & (np.arange(13)[None, :] <= deadline[:, None])
    return np.where(valid.any(-1), valid.argmax(-1), -1)


def evaluate(labels, first, truth, contact):
    rr = np.flatnonzero(contact & (first >= 0))
    count = truth[rr, first[rr]].sum(-1)
    value = labels[rr, first[rr]]
    finite = value >= 0
    supported = np.zeros(len(rr), bool)
    supported[finite] = truth[rr[finite], first[rr[finite]], value[finite]]
    clean = np.zeros(len(contact), bool); clean[rr] = supported
    unique = np.zeros(len(contact), bool); unique[rr] = supported & (count == 1)
    wrong = np.zeros(len(contact), bool); wrong[rr] = finite & ~supported
    abstain = np.zeros(len(contact), bool); abstain[rr] = ~finite
    opportunity_a = finite & ~supported & (count == 1)
    opportunity_b = finite & ~supported & (count > 1)
    opportunity_c = ~finite & (count > 0)
    return dict(events=int(contact.sum()), timely=int(len(rr)), first_clean=int(clean.sum()),
        first_unique=int(unique.sum()), first_wrong=int(wrong.sum()), first_abstain=int(abstain.sum()),
        truth_empty=int((count == 0).sum()), truth_multiple=int((count > 1).sum()),
        conditional_clean=None if not len(rr) else float(clean.sum()/len(rr)),
        opportunity=dict(a_wrong_unique=int(opportunity_a.sum()), b_wrong_multiple=int(opportunity_b.sum()),
            c_unavailable_supported=int(opportunity_c.sum()), c_unavailable_unique=int((~finite & (count == 1)).sum()),
            clean_oracle_headroom=int((opportunity_a | opportunity_b | opportunity_c).sum()),
            unique_oracle_headroom=int((opportunity_a | (~finite & (count == 1))).sum())),
        clean_oracle_ceiling=int((count > 0).sum()), unique_oracle_ceiling=int((count == 1).sum())), dict(clean=clean, unique=unique, wrong=wrong, abstain=abstain)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = 180.; tick = time.monotonic()
    def check():
        if time.monotonic()-tick >= cap: raise TimeoutError('CPU analysis cap180 wall seconds')
    paths = [Path(__file__), E.OUT/'ledger.npz', S.OUT/'ledger.npz', S.OUT/'result.json']
    B.save(OUT/'PLAN.json', dict(task='CNH_EMA_BEARING_OPPORTUNITY_DEV_20261008', lane='EXPLORE consumed Development',
        goal='Fixed EMA notice times/count; quantify zero-new-information L1 and remaining label opportunity before spatial investment',
        authorization='User: 推进, following ready-reviewed roundtable 20261008073631-e7a7',
        budgets_wall_seconds=dict(cpu_analysis=cap, focused_verification=90.),
        adjustable='Implementation corrections only; no threshold, label boundary, detector or cohort tuning in this check',
        cohort='96 cached units/3840 windows/229 contacts/384 strict-clear; all five yaw conditions, single/dual separately',
        first='Original EMA same-total-notice theta from sector result; first outputs2..deadline inclusive, raw input frame=output+3; do not move first on abstention',
        labels='L1 quantize(-ema_rel[:,3:]) at +/-10deg; exact boundaries CENTER, unavailable=-1. CENTER constant diagnostic. L2 deterministic argmax of stored causally smoothed/fused sector scores; no extra theta.',
        notice='All EMA alarms preserved, including clear/other/startup; candidate label adds no alarms and suppresses none',
        evaluator='Cached all-contact-box footprint legal-sector set only after candidate labels computed; first_clean supported singleton output, first_unique requires unique truth. Wrong/abstain separate, all counts /229.',
        opportunities='a wrong unique truth; b wrong multi truth; c unavailable with nonempty truth. clean ceiling a+b+c; unique ceiling a+c(unique). Empty truth excluded from opportunity, misses retained.',
        decision_check='One event minimum change; inspect each challenge/config individually. Saturated cells retain non-decrease constraint, do not veto other cells. Oracle is upper bound not attainable benefit. Decide spatial work using observed headroom and implementation cost; L2 is not a spatial gate.',
        stop='Cap/missing source/alignment mismatch: preserve partial and no full-cohort ranking. No GPU/render/training/new confirmation.',
        deliverables='L1/CENTER/L2 event table, per-cell opportunity and cost, proportionate checks, research decision and scoped master delivery',
        hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths}))
    status='FAILED'
    try:
        with np.load(E.OUT/'ledger.npz') as f: base={k:f[k] for k in ('score','unit','config','contact','control','deadline')}
        with np.load(S.OUT/'ledger.npz') as f:
            spatial={k:f[k] for k in ('score','truth','unit','config','contact','control','deadline')}
            old_timely={name+'/'+sensor:f[name+'/'+sensor+'/ema/timely'] for name in Y.NAMES for sensor in ('single','dual')}
        for k in ('unit','config','contact','control','deadline'): np.testing.assert_array_equal(base[k],spatial[k])
        uid, cfg, contact, deadline = (base[k] for k in ('unit','config','contact','deadline'))
        assert (len(uid),int(contact.sum()),int(base['control'].sum())) == (3840,229,384)
        labels=np.full((5,3840,13),-1,np.int8); unit_hashes={}; costs=[]
        for u in Y.UNITS:
            check(); t=time.monotonic(); rows=np.flatnonzero(uid==u)
            np.testing.assert_array_equal(cfg[rows],np.arange(40))
            path=E.OUT/'units'/f'unit{u}.npz'; unit_hashes[str(u)]=B.A.sha(path)
            with np.load(path) as f:
                for ni,name in enumerate(Y.NAMES):
                    rel=f[name+'/ema_rel']; assert rel.shape == (40,16)
                    labels[ni,rows]=quantize(-rel[:,3:])
            costs.append(time.monotonic()-t)
            if u == Y.UNITS[0]:
                B.save(OUT/'cost_projection.json',dict(first_unit_seconds=costs[0],projected96_seconds=costs[0]*96,
                    setup_seconds=t-tick,includes='Unit hash/read and all five L1 arrays; first unit reused, no inference'))
        parent=B.A.read(S.OUT/'result.json'); metrics={}; saved={}; table=[]
        for ni,name in enumerate(Y.NAMES):
            for si,sensor in enumerate(('single','dual')):
                check(); key=name+'/'+sensor
                theta=parent['metrics'][key]['ema']['working_point']['threshold']
                alarm=base['score'][ni,si,4]>=theta; first=first_notice(alarm,contact,deadline)
                np.testing.assert_array_equal(first>=0,old_timely[key])
                expected=parent['metrics'][key]['ema']
                assert int(alarm.sum())==expected['total_notices']
                assert int((first>=0).sum())==expected['timely_main']
                cell=dict(threshold=theta,total_notices=int(alarm.sum()),strict_clear_notices=int(alarm[base['control']].sum()),
                    contact_window_notices=int(alarm[contact].sum()),other_window_notices=int(alarm[~contact & ~base['control']].sum()))
                saved[key+'/first']=first; saved[key+'/alarm']=alarm
                arms=dict(L1=labels[ni],CENTER=np.ones((3840,13),np.int8),L2=spatial['score'][ni,si].argmax(-1).astype(np.int8))
                for arm,values in arms.items():
                    rec,e=evaluate(values,first,spatial['truth'][ni],contact); cell[arm]=rec
                    assert rec['first_clean']+rec['first_wrong']+rec['first_abstain']==rec['timely']
                    saved[key+'/'+arm+'/labels']=values
                    for metric,val in e.items(): saved[key+'/'+arm+'/'+metric]=val
                    for row in np.flatnonzero(contact):
                        ff=int(first[row]); legal=[] if ff<0 else np.flatnonzero(spatial['truth'][ni,row,ff]).tolist()
                        value=-1 if ff<0 else int(values[row,ff])
                        table.append(dict(name=name,sensor=sensor,arm=arm,unit=int(uid[row]),config=int(cfg[row]),
                            deadline=int(deadline[row]),first_output=ff,first_input_frame=-1 if ff<0 else ff+3,
                            label='NONE' if value<0 else S.LABELS[value],supported='|'.join(S.LABELS[i] for i in legal),
                            timely=ff>=0,clean=bool(e['clean'][row]),unique=bool(e['unique'][row]),wrong=bool(e['wrong'][row]),abstain=bool(e['abstain'][row])))
                metrics[key]=cell
        B.atomic_npz(OUT/'ledger.npz',unit=uid,config=cfg,contact=contact,deadline=deadline,**saved)
        with (OUT/'events.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
        B.save(OUT/'result.json',dict(status='COMPLETE',metrics=metrics,events=229,units=96,
            unit_hashes=unit_hashes,unit_read_seconds=sum(costs),ledger_sha256=B.A.sha(OUT/'ledger.npz'),
            limits='Consumed Development artificial yaw and inherited footprint geometry; no real hardware/user-benefit/object-association claim. L1 uses no spatial observation. L2 is cached diagnostic, not spatial gate. Oracle headroom not attainable gain; matched all13 message count is not audio burden. EMA noisy5Hz reset and ideal origin inherited.'))
        status='COMPLETE'
        for k,v in metrics.items(): print(k, {a: {m:v[a][m] for m in ('timely','first_clean','first_unique','first_wrong','first_abstain','opportunity')} for a in arms})
    finally:
        B.save(OUT/'runtime.json',dict(status=status,seconds=time.monotonic()-tick,budget_seconds=cap,backend='CPU numpy, no inference'))


if __name__=='__main__': run()

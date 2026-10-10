"""Focused independent burst recurrence, grade, timing/cost and ledger audit."""
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E
import audit_cnh_graded_peak_body_only_dev as A

OUT = C.ROOT/'artifacts.local/work/cnh-graded-raw-burst-dev-20261010'
PARENT = C.ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'


def pair(before, after, category, end):
    a, b = A.clock(before, end), A.clock(after, end)
    reports = []
    for q, height in enumerate(E.HEIGHTS):
        aa, bb = a[category[:,q] == 'contact', :, q], b[category[:,q] == 'contact', :, q]
        result = A.pair_row(aa, bb, height)
        result['both'] = result.pop('both_timely')
        result.pop('median_advance_frames')
        result.update(before=int((aa >= 0).sum()), after=int((bb >= 0).sum()))
        reports.append(result)
    return reports


def run():
    began = time.monotonic()
    audit = OUT/'verification'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve audit attempt')
    C.save(audit/'PLAN.json',dict(task='FIXED_RAW_BURST_FOCUSED_AUDIT', CPU_command_wall_seconds=90, GPU_seconds=0,
        producer_imported=False, goal='Independently reconstruct all54 grade cells, recurrence, first clocks, costs and165888 ledger rows',
        source_sha256=C.sha(Path(__file__)), helpers_sha256=C.sha(Path(A.__file__))))
    try:
        plan = C.read(OUT/'PLAN.json')
        for path, digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'input/'+path)
        A.equal(C.sha(Path(__file__).with_name('cnh_graded_raw_burst_dev.py')),plan['source_sha256'],'producer hash')
        A.equal(plan['mechanisms'],['fixed','raw_single','raw_k2of3'],'fixed mechanisms')
        A.equal(plan['policies'],['baseline','both','head50'],'fixed working points')
        data, published = C.load(), C.read(OUT/'metrics.json')
        thresholds = C.read(C.PARENT/'thresholds.json')
        ledger = A.read_csv(OUT/'ledger.csv')
        indexed = defaultdict(list)
        for row in ledger: indexed[(row['split'],int(row['seed']),row['policy'],row['mechanism'])].append(row)
        cohort_counts = defaultdict(Counter)
        for split,d in data.items():
            with np.load(PARENT/f'{split}_grades.npz',allow_pickle=False) as a: parents = dict(zip(a['keys'].tolist(),a['grades']))
            with np.load(OUT/f'{split}_grades.npz',allow_pickle=False) as a:
                saved = dict(zip(a['keys'].tolist(),a['grades']))
                A.equal(a['scene_ids'],d['scene_ids'],'scene ids')
                A.equal(a['category'],d['category'],'categories')
            for seed in G.SEEDS:
                raw = E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz','ordinary',seed,split,'ideal')
                crossing = raw >= thresholds[str(seed)]['single']
                # Independent recurrence by one/two-step shifts; no rolling-window helper.
                prior1,prior2 = np.zeros_like(crossing,dtype=np.int8),np.zeros_like(crossing,dtype=np.int8)
                prior1[...,1:,:] = crossing[...,:-1,:]
                prior2[...,2:,:] = crossing[...,:-2,:]
                persistent = (crossing.astype(np.int8)+prior1+prior2) >= 2
                assert not persistent[...,0,:].any()
                # Synthetic first/reset semantics, including separated impulses and isolated streams.
                sample = np.array([[1,0,1,0,0],[0,1,0,0,0]],dtype=np.int8)
                padded = np.pad(sample,((0,0),(2,0)))
                expected = np.stack([padded[:,t:t+3].sum(1)>=2 for t in range(5)],1)
                A.equal(expected,np.array([[0,0,1,0,0],[0,0,0,0,0]],dtype=bool),'first/reset fixture')
                for policy in ('baseline','both','head50'):
                    before = parents[f'{seed}/{policy}']
                    for mechanism in ('fixed','raw_single','raw_k2of3'):
                        if time.monotonic()-began >= 90: raise TimeoutError('Focused audit90s cap')
                        grade = before.copy()
                        if mechanism != 'fixed': grade[(before == 0) & (crossing if mechanism == 'raw_single' else persistent)] = 1
                        key = f'{seed}/{policy}/{mechanism}'
                        A.equal(saved[key],grade,key+'/grades')
                        A.equal(grade == 2,before == 2,key+'/strong')
                        before_full,before_timely,before_strong = A.clock(before > 0),A.clock(before > 0,11),A.clock(before == 2)
                        full,timely,strong = A.clock(grade > 0),A.clock(grade > 0,11),A.clock(grade == 2)
                        outcomes = []
                        for q,height in enumerate(E.HEIGHTS):
                            f,t = full[d['category'][:,q]=='contact',:,q],timely[d['category'][:,q]=='contact',:,q]
                            outcomes.append(dict(height=height,denominator=int(f.size),timely=int((t>=0).sum()),
                                late=int(((t<0)&(f>=0)).sum()),silent=int((f<0).sum())))
                        expected_metrics = dict(outcomes=outcomes,paired_timely=pair(before>0,grade>0,d['category'],11),
                            paired_full=pair(before>0,grade>0,d['category'],13),
                            physical_contact_timely=A.physical_pair(before>0,grade>0,d['category']),
                            new_cost=A.new_costs(before>0,grade>0,d['category']),
                            added_cost=A.addition_costs((grade>0)&(before==0),d['category']),
                            total_cost=A.addition_costs(grade>0,d['category']),light_cost=A.addition_costs(grade==1,d['category']),
                            strong_cost=A.addition_costs(grade==2,d['category']))
                        A.equal(published[f'{split}/{key}'],expected_metrics,key+'/metrics')
                        scene_index = {int(s):n for n,s in enumerate(d['scene_ids'])}
                        entries = indexed[(split,seed,policy,mechanism)]
                        A.equal(len(entries),384*4*2,key+'/ledger count')
                        for entry in entries:
                            n,k,q = scene_index[int(entry['scene'])],int(entry['replica']),E.HEIGHTS.index(entry['height'])
                            for name,values in [('before_first',before_full),('after_first',full),('before_timely',before_timely),
                                    ('after_timely',timely),('before_first_strong',before_strong),('after_first_strong',strong)]:
                                A.equal(int(entry[name]),int(values[n,k,q]),key+'/'+name)
                            A.equal(entry['grades'],''.join(map(str,grade[n,k,:,q].tolist())),key+'/grade string')
                            old = 'timely' if before_timely[n,k,q]>=0 else 'late' if before_full[n,k,q]>=0 else 'silent'
                            new = 'timely' if timely[n,k,q]>=0 else 'late' if full[n,k,q]>=0 else 'silent'
                            A.equal(entry['before_outcome'],old,key+'/old outcome');A.equal(entry['after_outcome'],new,key+'/new outcome')
                            A.equal(entry['category'],str(d['category'][n,q]),key+'/truth reporting')
                            A.equal(entry['shape_family'],d['rows'][n]['shape_family'],key+'/shape reporting')
                            if entry['category'] == 'contact':
                                for family in ('ALL',entry['shape_family']):
                                    cohort_counts[(split,seed,policy,mechanism,entry['height'],old,family)][new]+=1
        for row in A.read_csv(OUT/'cohorts.csv'):
            count = cohort_counts[(row['split'],int(row['seed']),row['policy'],row['mechanism'],row['height'],row['before_outcome'],row['shape_family'])]
            A.equal(int(row['denominator']),sum(count.values()),'cohort denom')
            for outcome in ('timely','late','silent'): A.equal(int(row['after_'+outcome]),count[outcome],'cohort '+outcome)
        A.equal(len(published),54,'all cells')
        A.equal(len(ledger),165888,'all ledger')
        C.save(audit/'receipt.json',dict(status='PASS',cells=54,ledger_rows=len(ledger),checks=A.CHECKS,
            seconds=time.monotonic()-began,producer_imported=False,source_sha256=C.sha(Path(__file__)),GPU_seconds=0))
        print(f'PASS cells54 ledger{len(ledger)} checks{A.CHECKS} seconds{time.monotonic()-began:.3f}')
    except BaseException as error:
        C.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__ == '__main__': run()

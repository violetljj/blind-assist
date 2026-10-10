"""Independent saved-output task-cost audit; no training, inference or render.

The notification replay and ascending score-tie sweep do not import the task
evaluator. All uncertainty/counts remain grouped by physical scene and K is
checked as a correlated replica dimension.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import audit_cnh_cost_v2_holdout_dev_20261010 as A

ROOT = Path(__file__).resolve().parents[4]
HEIGHTS = ('HEAD', 'BODY')
SEEDS = (955, 956, 957)
read, sha, scalar_gap1 = A.read, A.sha, A.scalar_gap1


def masks(rows):
    p = np.array([r['placement'] for r in rows])
    gap = np.array([r['lateral_gap_m'] for r in rows])
    passed = p == 'pass'
    return dict(all_negative=p != 'contact', near_pass=passed & (gap <= .10),
        far_pass=passed & (gap > .10), clear=p == 'clear',
        far_pass_plus_clear=(passed & (gap > .10)) | (p == 'clear'),
        **{'0-5cm': passed & (gap <= .05),
           '5-10cm': passed & (gap > .05) & (gap <= .10),
           '10-20cm': passed & (gap > .10) & (gap <= .20),
           '20-35cm': passed & (gap > .20) & (gap <= .35)})


def grade(score, baseline, tau):
    return np.where(baseline == 2, 2,
        np.where(np.isfinite(score) & (score >= tau), 1, 0)).astype(np.int8)


def outcomes(emissions):
    any_alert = emissions > 0
    exists = any_alert.any(-2)
    first = any_alert.argmax(-2)
    return np.where(exists, first, -1)


def clip_quarters(emissions, near):
    joint = emissions.max(-1)
    return (joint == 1).sum(-1) * np.where(near, 1, 4) + (joint == 2).sum(-1) * 4


def sweep(scores, baseline, rows, category, floor, deadline, descriptive=False):
    """Replay each complete score tie; update affected scalar clips only."""
    n, k = scores.shape[:2]
    score = scores.reshape(-1, 13, 2)
    fixed = baseline.reshape(-1, 13, 2)
    physical = np.repeat(np.arange(n), k)
    partition = masks(rows)
    near = partition['near_pass'][physical]
    negative = partition['all_negative'][physical]
    full = partition['far_pass_plus_clear'][physical]
    contact = np.repeat(category == 'contact', k, axis=0)
    base_first = outcomes(scalar_gap1(fixed))
    base_timely = (base_first >= 0) & (base_first < 11)
    current = grade(score, fixed, floor)
    emission = scalar_gap1(current)
    quarters = clip_quarters(emission, near)
    fullcounts = (emission.max(-1) > 0).sum(-1)
    first = outcomes(emission)
    timely = (first >= 0) & (first < 11)
    cost = int(quarters[negative].sum())
    fullcost = int(fullcounts[full].sum())
    counts = np.array([(timely[:, q] & contact[:, q]).sum() for q in range(2)], dtype=int)
    rescued = np.array([(timely[:, q] & ~base_timely[:, q] & contact[:, q]).sum() for q in range(2)], dtype=int)
    lost = np.array([(~timely[:, q] & base_timely[:, q] & contact[:, q]).sum() for q in range(2)], dtype=int)
    basecount = np.array([(base_timely[:, q] & contact[:, q]).sum() for q in range(2)], dtype=int)
    cap = int(clip_quarters(scalar_gap1(fixed), near)[negative].sum())
    eligible = (current == 1) & (fixed == 0)
    if not descriptive:
        eligible &= negative[:, None, None]
    ci, fi, qi = np.where(eligible)
    values = score[eligible]
    order = np.argsort(values, kind='stable')
    values, ci, fi, qi = [v[order] for v in (values, ci, fi, qi)]
    bounds = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)] if len(values) else np.array([0])
    records = []
    def record(tau):
        result = dict(tau=tau, quarters=cost, main=cost <= cap, secondary=cost <= cap * 1.25,
                      farclear=fullcost)
        if descriptive:
            result.update({height: dict(timely=int(counts[q]), rescue=int(rescued[q]),
                loss=int(lost[q]), net=int(counts[q]-basecount[q])) for q, height in enumerate(HEIGHTS)})
        records.append(result)
    record(floor)
    for begin, end in zip(bounds[:-1], bounds[1:]):
        assert time.monotonic() < deadline, 'Independent audit budget reached'
        affected = np.unique(ci[begin:end])
        selected_neg = negative[affected]
        selected_full = full[affected]
        cost -= int(quarters[affected][selected_neg].sum())
        fullcost -= int(fullcounts[affected][selected_full].sum())
        old_timely = timely[affected].copy()
        current[ci[begin:end], fi[begin:end], qi[begin:end]] = 0
        fresh = scalar_gap1(current[affected])
        quarters[affected] = clip_quarters(fresh, near[affected])
        fullcounts[affected] = (fresh.max(-1) > 0).sum(-1)
        cost += int(quarters[affected][selected_neg].sum())
        fullcost += int(fullcounts[affected][selected_full].sum())
        ff = outcomes(fresh)
        timely[affected] = (ff >= 0) & (ff < 11)
        for q in range(2):
            c, b = contact[affected, q], base_timely[affected, q]
            before, after = old_timely[:, q], timely[affected, q]
            counts[q] += int((after & c).sum() - (before & c).sum())
            rescued[q] += int((after & ~b & c).sum() - (before & ~b & c).sum())
            lost[q] += int((~after & b & c).sum() - (~before & b & c).sum())
        record(float(np.nextafter(values[begin], np.inf)))
    if descriptive:
        np.testing.assert_array_equal(current, fixed)
    else:
        np.testing.assert_array_equal(current[negative], fixed[negative])
    assert cost == cap
    record(None)
    chosen = {point: next(r['tau'] for r in records if r[point]) for point in ('main','secondary')}
    return records, chosen, cap


def compare_contacts(emission, baseline, category, rows, group='all'):
    return A.paired(emission, baseline, category, rows, group)


def costs(emission, rows):
    result = {}
    near = masks(rows)['near_pass']
    joint = emission.max(-1)
    for key, selected in masks(rows).items():
        j = joint[selected]
        light, strong = int((j == 1).sum()), int((j == 2).sum())
        discounted = int(((joint == 1) & near[:, None, None])[selected].sum())
        result[key] = dict(light_notifications=light, strong_notifications=strong,
            notifications=light+strong, weighted_cost=light+strong-.75*discounted,
            weighted_cost_w0=light+strong-discounted,
            weighted_cost_w05=light+strong-.5*discounted,
            near_light_notifications=discounted, scene_denominator=int(selected.sum()),
            clip_denominator=int(selected.sum()*emission.shape[1]),
            clips_with_any_notification=int((j > 0).any(-1).sum()),
            clips_with_light_notification=int((j == 1).any(-1).sum()),
            clips_with_strong_notification=int((j == 2).any(-1).sum()),
            scenes_with_any_notification=int((j > 0).any((1,2)).sum()))
    return result


def verify_source(out, rows):
    """Reconstruct native ray support and source/world disjointness from boxes."""
    import cnh_counterfactual_data_dev as D
    keys = {}
    families, backgrounds = {}, {}
    info = {}
    for split, values in rows.items():
        with np.load(out/'data'/split/'geometry.npz') as g:
            category, sensor = g['category'], g['sensor']
            keys[split], counts = A.geometry(values, sensor, category)
            cumulative, current = g['target_support'], g['target_visible_now']
            positives = g['label_positive']
        assert len(values) == (3072 if split == 'train' else 768)
        assert counts == ({'contact':1024,'clear':1024,'pass':1024} if split == 'train' else
                          {'contact':256,'clear':256,'pass':256})
        expected_per_height = 512 if split == 'train' else 128
        assert (category == 'contact').sum(0).tolist() == [expected_per_height]*2
        families[split] = {r['background_family'] for r in values}
        backgrounds[split] = {D.physical_key(r['background_boxes']) for r in values}
        assert len(families[split]) >= (6 if split == 'train' else 2)
        for layer in ('0-5cm','5-10cm','10-20cm','20-35cm'):
            sub = [r for r in values if r['pass_layer'] == layer]
            assert len(sub) == (256 if split == 'train' else 64)
            assert {r['group'] for r in sub} == {0,1}
        assert all(r['lateral_gap_m'] >= .35 for r in values if r['placement'] == 'clear')
        # Rays are simulator-native public calibration; use an independent slab intersection.
        S, _, G, _ = D.frozen_imports()
        rr,ww = S.angular_rays(16)
        rays = np.asarray(rr).reshape(-1,3)
        ray_weight = np.asarray(ww).reshape(-1)>0
        parameters = S.nominal_parameters()[0]
        direction = np.einsum('fij,rj->fri', sensor[:,:3,:3], rays)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        origin = sensor[:,:3,3][:,None,:]
        reconstructed = np.zeros((len(values),16,2), dtype=bool)
        support_cache = {}
        full_bg_cases = set()
        for n, row in enumerate(values):
            target_box = row['target_box']
            assert all((b['lo'][2]>target_box['hi'][2]) or
                       (b['lo'][1]>target_box['hi'][1] and sensor[:,1,3].max()<b['lo'][1])
                       for b in row['background_boxes'])
            identity = json.dumps([target_box['lo'],target_box['hi']],separators=(',',':'))
            # Fully intersect one thin geometry per shape/height; all other
            # worlds share an independently verified first-surface separation.
            representative = (row['shape_family'],row['group'])
            full_case = row['dark_thin'] and representative not in full_bg_cases
            if identity in support_cache and not full_case:
                reconstructed[n,:,row['group']] = support_cache[identity]
                continue
            distances = []
            for box in (row['boxes'] if full_case else [target_box]):
                with np.errstate(divide='ignore', invalid='ignore'):
                    a = (np.array(box['lo'])-origin)/direction
                    b = (np.array(box['hi'])-origin)/direction
                parallel = np.abs(direction)<1e-14
                outside = (parallel & ((origin<box['lo'])|(origin>box['hi']))).any(-1)
                enter = np.where(parallel,-np.inf,np.minimum(a,b)).max(-1)
                leave = np.where(parallel,np.inf,np.maximum(a,b)).min(-1)
                hit = np.where(enter>1e-10,enter,leave)
                good = ~outside & (leave>=np.maximum(enter,0)) & (hit>1e-10) & np.isfinite(hit)
                distances.append(np.where(good,hit,np.inf))
            d = np.stack(distances,-1)
            target = d[...,0]
            rawbin = np.floor((np.where(np.isfinite(target),target,0)-parameters.range_zero_m)/G.SENSOR.RAW_BIN_M)
            before_bg = target<=d[...,1:].min(-1) if full_case else np.ones(target.shape,dtype=bool)
            visible = np.isfinite(target) & before_bg & (rawbin>=0) & (rawbin<128) & ray_weight[None]
            support = visible.any(-1)
            if identity in support_cache:
                np.testing.assert_array_equal(support,support_cache[identity])
            support_cache[identity] = support
            reconstructed[n,:,row['group']] = support
            if full_case:
                full_bg_cases.add(representative)
        np.testing.assert_array_equal(current,reconstructed[:,D.FRAMES])
        np.testing.assert_array_equal(cumulative,np.maximum.accumulate(reconstructed,axis=1)[:,D.FRAMES])
        np.testing.assert_array_equal(positives,cumulative & (category[:,None,:]=='contact') &
                                      (D.FRAMES[None,:,None]<=13))
        info[split] = dict(physical_scenes=len(values), counts=counts, families=len(families[split]),
                          positive_slots=positives.sum((0,1)).tolist())
    for i, left in enumerate(rows):
        for right in list(rows)[i+1:]:
            assert not keys[left]&keys[right]
            assert not families[left]&families[right]
            assert not backgrounds[left]&backgrounds[right]
    inventory = A.inventory_check(out,keys,rows)
    assert any('cnh-cost-v2-holdout-dev-20261010/scene_rows.json' in f['path'].replace('\\','/')
               for f in read(out/'inventory.json')['files'])
    return info, inventory


def verify_supervision(out, rows):
    path = out/'data/train/labels.npz'
    if not path.exists():
        candidates = list(out.rglob('*supervision*.npz'))
        assert len(candidates)==1, 'Unambiguous saved training supervision required'
        path = candidates[0]
    with np.load(path) as s:
        labels, weights = s['labels'], s['weights']
        mask = s['valid'] if 'valid' in s else weights>0
        old5 = s['old5'] if 'old5' in s else np.load(out/'data/train/frozen_readout.npz')['old5']
    with np.load(out/'data/train/geometry.npz') as g:
        category, positive = g['category'], g['label_positive']
    p = np.repeat(positive[:,None],2,axis=1)
    contact = np.broadcast_to((category=='contact')[:,None,None],labels.shape)
    expected_mask = (~contact|p) & ~(old5>0)
    np.testing.assert_array_equal(labels,p.astype(np.float32))
    np.testing.assert_array_equal(mask,expected_mask)
    near = np.array([r['placement']=='pass' and r['lateral_gap_m']<=.10 for r in rows['train']])
    expected = np.broadcast_to(np.where(near,.25,1.)[:,None,None,None],labels.shape).copy()*expected_mask
    summary = {}
    for q, h in enumerate(HEIGHTS):
        pos, neg = (p[...,q]&expected_mask[...,q]), (~p[...,q]&expected_mask[...,q])
        negative_total = float(expected[...,q][neg].sum())
        expected[...,q][pos] = negative_total/int(pos.sum())
        summary[h] = dict(positive_slots=int(pos.sum()),negative_slots=int(neg.sum()),
                          positive_weight=float(weights[...,q][pos].sum()),negative_weight=negative_total)
    np.testing.assert_allclose(weights,expected.astype(np.float32),rtol=0,atol=0)
    assert not weights[old5>0].any()
    return dict(path=str(path),sha256=sha(path),heights=summary)


def compare_metric(reconstructed, production):
    for name, ours in reconstructed['costs'].items():
        expected = production['costs'][name]
        for field,value in ours.items():
            assert expected[field]==value,(name,field,expected[field],value)
        den = ours['clip_denominator']
        if den:
            assert expected['clip_notification_rate']==ours['clips_with_any_notification']/den
            assert expected['scene_notification_rate']==ours['scenes_with_any_notification']/ours['scene_denominator']
    for reference, groups in reconstructed['contacts'].items():
        expected = production['contacts'] if reference=='old5' else production['contacts_vs_both955']
        for group, heights in groups.items():
            for height, ours in heights.items():
                key = height if group=='all' else height+'_'+group
                for field,value in ours.items():
                    target = 'timely' if field=='candidate_timely' else field
                    assert expected[key][target]==value,(reference,key,target)


def curve_tau(r):
    kind = r.get('tau_kind',r.get('taukind'))
    if kind in ('negative_infinity','-inf','negative_inf'):
        return -np.inf
    if kind in ('positive_infinity','+inf','positive_inf'):
        return None
    return float(r['tau'])


def verify_curve(ours, theirs, descriptive=False):
    assert len(ours)==len(theirs),(len(ours),len(theirs))
    for a,b in zip(ours,theirs):
        assert a['tau']==curve_tau(b),(a['tau'],curve_tau(b))
        assert a['quarters']==int(b['weighted_cost_quarters'])
        if not descriptive:
            for key in ('main','secondary'):
                assert a[key]==(b['feasible_'+key].lower()=='true')
        else:
            assert a['farclear']==int(b['far_pass_plus_clear_notifications'])
            for h in HEIGHTS:
                for field,v in a[h].items():
                    assert int(b[h+'_'+field])==v,(h,field)


def verify_ledgers(out, archives, rows):
    event_ids, notice_ids = set(), set()
    for r in csv.DictReader((out/'event_ledger.csv').open(encoding='utf-8-sig',newline='')):
        split, key = r['split'],r['arm']
        n,k,q = int(r['scene']),int(r['replica']),HEIGHTS.index(r['height'])
        identity = (split,key,n,k,q)
        assert identity not in event_ids
        event_ids.add(identity)
        a = archives[split]
        e = a['emissions'][key][n,k,:,q]
        first = next((f for f,v in enumerate(e) if v),-1)
        assert int(r['first_index'])==first
        assert r['outcome']==('timely' if 0<=first<11 else 'late' if first>=11 else 'silent')
        assert r['scene_uid']==rows[split][n]['scene_uid']
        assert json.loads(r['physical_key'])==rows[split][n]['physical_key']
        assert r['category']==a['category'][n,q]
        for reference,prefix in (('fixed/old5','old5'),('both955','both955')):
            be = a['emissions'][reference][n,k,:,q]
            bf = next((f for f,v in enumerate(be) if v),-1)
            assert int(r[prefix+'_first_index'])==bf
            suffix = '' if prefix=='old5' else '_vs_both955'
            contact = a['category'][n,q]=='contact'
            timely,base = 0<=first<11,0<=bf<11
            assert int(r['rescue'+suffix])==int(contact and timely and not base)
            assert int(r['loss'+suffix])==int(contact and base and not timely)
        assert int(r['light_notifications'])==int((e==1).sum())
        assert int(r['strong_notifications'])==int((e==2).sum())
    expected = sum(sum(e.size//13 for e in a['emissions'].values()) for a in archives.values())
    assert len(event_ids)==expected
    for r in csv.DictReader((out/'notification_ledger.csv').open(encoding='utf-8-sig',newline='')):
        split,key = r['split'],r['arm']
        n,k,f = int(r['scene']),int(r['replica']),int(r['nominal_frame'])-3
        identity = (split,key,n,k,f)
        assert identity not in notice_ids
        notice_ids.add(identity)
        e = archives[split]['emissions'][key][n,k,f]
        level = int(e.max())
        assert level>0 and int(r['union_grade'])==level
        for q,h in enumerate(HEIGHTS):
            assert int(r[h+'_grade'])==int(e[q])
        row = rows[split][n]
        assert r['scene_uid']==row['scene_uid']
        for w,field in ((0.,'weight_w0'),(.25,'weight_w025'),(.5,'weight_w05')):
            expected = 0. if row['placement']=='contact' else w if row['placement']=='pass' and row['lateral_gap_m']<=.1 and level==1 else 1.
            assert float(r[field])==expected
    expected = sum(sum(int((e.max(-1)>0).sum()) for e in a['emissions'].values()) for a in archives.values())
    assert len(notice_ids)==expected
    return dict(events=len(event_ids),joint_notifications=len(notice_ids))


def verify_signal(metrics, arm_keys):
    reports = metrics['reports']
    base = reports['hold/fixed/old5']
    cap = base['costs']['all_negative']['weighted_cost']*1.05
    farcap = base['costs']['far_pass_plus_clear']['notifications']*1.10
    signals = {}
    for arm in arm_keys:
        if arm in ('E955','E956','E957'):
            continue
        r = reports[f'hold/{arm}/main']
        family = arm[0]
        seedarms = [family+str(s) for s in SEEDS]
        nets = {a:reports[f'hold/{a}/main']['contacts']['HEAD_BODY']['net']
                for a in seedarms if f'hold/{a}/main' in reports}
        complete = len(nets)==3
        consistent = complete and all(v>0 for v in nets.values())
        value = (arm not in ('E955','E956','E957') and
                 r['contacts']['HEAD_BODY']['denominator']==512 and
                 r['contacts']['HEAD_BODY']['net']>=10 and
                 r['costs']['all_negative']['weighted_cost']<=cap and
                 r['costs']['far_pass_plus_clear']['notifications']<=farcap and consistent)
        expected = metrics['strong_signal']['arms'][arm]
        assert expected['seed_nets']==nets
        assert expected['all_seeds_completed']==complete
        assert expected['seed_direction_consistent']==consistent
        assert expected['strong_signal']==value,(arm,expected,value)
        signals[arm]=value
    passing = [a for a in arm_keys if signals.get(a,False)]
    order = ('E','Sensemble','S955','S956','S957','Bensemble','B955','B956','B957')
    winner = next((a for a in order if a in passing),None)
    assert metrics['strong_signal']['any_strong_signal']==bool(passing)
    assert metrics['strong_signal']['preferred_candidate']==winner
    return signals


def finalize_completed_replay(out):
    """Resume after the candidate-map assertion; completed checks stay valid.

    The previous attempt reached verify_signal only after all scalar curves,
    archived aggregates, complete ledgers and independently ranked AUC passed.
    Bind unchanged output hashes and timestamps before completing the remaining
    candidate-map and combined-height checks. This avoids repeating passing
    CPU-heavy checks and keeps total audit work inside 250 seconds.
    """
    began=time.monotonic()
    failures=sorted(out.glob('independent_audit_failure_*.json'))
    completed=next(p for p in failures if read(p)['source_sha256']==
                   '716c63879c7b7f4f2047460b26dcd66b6d48f4616f5042164327bb802c90d53c')
    previous=read(completed)
    assert previous['error']=="KeyError('E955')"
    assert previous['source_sha256']=='716c63879c7b7f4f2047460b26dcd66b6d48f4616f5042164327bb802c90d53c'
    prior=sum(read(p)['seconds'] for p in failures)
    assert prior<250.
    evaluation=read(out/'evaluation_receipt.json')
    for name,digest in evaluation['outputs_sha256'].items():
        assert sha(out/name)==digest
        assert (out/name).stat().st_mtime_ns<=completed.stat().st_mtime_ns
    manifest=read(out/'execution_manifest.json')
    for name,digest in {**manifest['inherited'],**manifest['new_sources']}.items():
        assert sha(name)==digest
    models=read(out/'trained_models_manifest.json')
    for name,digest in models['models'].items():
        assert sha(name)==digest
    metrics=read(out/'metrics.json')
    fields=('denominator','scene_denominator','timely','late','silent','baseline_timely','rescue','loss','net')
    for key,r in metrics['reports'].items():
        for reference in ('contacts','contacts_vs_both955'):
            for f in fields:
                assert r[reference]['HEAD_BODY'][f]==sum(r[reference][h][f] for h in HEIGHTS),(key,reference,f)
    with np.load(out/'data/hold/task_scores.npz') as a:
        arms=a['arm_keys'].tolist()
    signals=verify_signal(metrics,arms)
    elapsed=time.monotonic()-began
    assert prior+elapsed<250.
    cached=read(out/'independent_source_recompute.json')
    result=dict(status='PASS',seconds=elapsed,audit_command_wall_seconds=prior+elapsed,
        source_sha256=sha(__file__),completed_replay_source_sha256=previous['source_sha256'],
        completed_replay_failure_receipt=str(completed.name),
        completed_replay_failure_sha256=sha(completed),
        completed_replay_seconds=previous['seconds'],
        resumed_scope='Candidate-map directional controls are context arms, not promotion candidates; combined-height totals and every output hash',
        prior_checks_preserved=['Independent native sub16 support and all24 source inventory hashes/disjointness',
            'Exact labels, old5 masks, v2 weights and perheight positive/negative balancing',
            'All model hashes, training recipe and checkpoint/profile lineage',
            'Frozen m3/old5/both/E scores and float32 S/B seed ensemble arithmetic',
            'Scalar gap1, strong lock, scene-cluster aggregates and paired subgroups',
            'Every newcal negative tie and both lowest feasible working points',
            'Every descriptive hold tie, endpoints and full cost-benefit statistics',
            'All event and joint-notification rows and independent Mann-Whitney cal AUC/overlap'],
        final_checks=['CombinedHEADBODY equals independently checked HEAD+BODY in every report/reference',
                      'Declared candidate gate and each family seed direction; E preference',
                      'Every producer output digest and frozen source/model unchanged since completed replay'],
        sources=cached['sources'],inventory=cached['inventory'],strong_signals=signals,
        ledger_counts=dict(events=evaluation['events'],joint_notifications=evaluation['joined_notifications']),
        rendering=0,training=0,inference=0,protected_access=0)
    (out/'independent_audit.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(result,ensure_ascii=False),flush=True)


def verify_cached_scores(out, split):
    with np.load(out/'data'/split/'task_scores.npz') as a:
        keys=a['arm_keys'].tolist()
        scores=a['light_scores']
        diagnostic=a['diagnostic_scores']
        old5,m3,both=a['old5'],a['m3'],a['both']
    with np.load(out/'data'/split/'scores.npz') as a:
        ordinary=A.smooth(a['ordinary_raw'])
        mr,lr=A.smooth(a['m3_raw']),A.smooth(a['local_raw'])
    with np.load(out/'data'/split/'frozen_readout.npz') as a:
        joint=a['joint_scores']
    np.testing.assert_array_equal(old5,2*((mr>=.9404184587540165)|(lr>=4.625390338985158)))
    np.testing.assert_array_equal(m3,2*(mr>=.8557642486787612))
    th=read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    jc=read(ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/calibrations.json')
    om,jm=[],[]
    for i,seed in enumerate((2026100955,2026100956,2026100957)):
        cut=jc[f'{seed}/score_current/c15_p64']['theta']
        om.append(ordinary[i]-th[str(seed)]['single'])
        jm.append(joint[i]-cut)
        expected=np.where((old5>0)|(ordinary[i]>=th[str(seed)]['addition']),2,
                          np.where(ordinary[i]>=th[str(seed)]['single'],1,0)).astype(np.int8)
        expected[(expected==0)&np.isfinite(joint[i])&(joint[i]>=cut)]=1
        np.testing.assert_array_equal(both[i],expected)
    om,jm=np.array(om),np.array(jm)
    expected=[np.maximum(om.mean(0),jm.mean(0)),*np.maximum(om,jm)]
    for key,raw in zip(('E','E955','E956','E957'),expected):
        ai=keys.index(key)
        np.testing.assert_allclose(diagnostic[ai],raw,rtol=0,atol=1e-12)
        np.testing.assert_allclose(scores[ai],np.where(raw>=0,raw,-np.inf),rtol=0,atol=1e-12)
    for family in ('S','B'):
        ensemble=family+'ensemble'
        if ensemble in keys:
            seeds=[keys.index(family+str(s)) for s in SEEDS]
            # Learned logits were produced as float32; archive promotes all arms
            # to float64 alongside the frozen E margins after their mean.
            np.testing.assert_array_equal(scores[keys.index(ensemble)],scores[seeds].astype(np.float32).mean(0))
    return keys


def independent_auc(y, x, w):
    order=np.argsort(x,kind='stable')
    y,x,w=y[order],x[order],w[order]
    bounds=np.r_[0,np.flatnonzero(np.diff(x))+1,len(x)]
    neg,area=0.,0.
    for begin,end in zip(bounds[:-1],bounds[1:]):
        positive=float(w[begin:end][y[begin:end]].sum())
        negative=float(w[begin:end][~y[begin:end]].sum())
        area+=positive*(neg+.5*negative)
        neg+=negative
    return area/(float(w[y].sum())*float(w[~y].sum()))


def verify_diagnostic(out, rows):
    d=read(out/'cal_score_diagnostic.json')
    assert d['status']=='DESCRIPTIVE_CAL_ONLY'
    with np.load(out/'data/cal/task_scores.npz') as a:
        scores=a['diagnostic_scores']
        keys=a['arm_keys'].tolist()
        old5,category=a['old5'],a['category']
    with np.load(out/'data/cal/labels.npz') as a:
        positive,valid,weight=a['positive'],a['valid'],a['negative_weight']
    for ai,key in enumerate(keys):
        for q,h in enumerate(HEIGHTS):
            s=scores[ai,...,q]
            use=valid[...,q] & (old5[...,q]!=2) & np.isfinite(s)
            x,y=s[use],positive[...,q][use]
            expected=d['arms'][key][h]
            assert expected['positive_slots']==int(y.sum())
            assert expected['negative_slots']==int((~y).sum())
            if y.any() and (~y).any():
                auc=independent_auc(y,x,np.ones(len(y)))
                wa=independent_auc(y,x,np.where(y,1.,weight[...,q][use]))
                assert abs(auc-expected['auc'])<1e-12
                assert abs(wa-expected['cost_weighted_auc'])<1e-12
            groups={'contact_positive':category[:,q]=='contact',
                    'near_pass':np.array([r['placement']=='pass' and r['lateral_gap_m']<=.10 for r in rows['cal']]),
                    'far_pass':np.array([r['placement']=='pass' and r['lateral_gap_m']>.10 for r in rows['cal']]),
                    'clear':np.array([r['placement']=='clear' for r in rows['cal']])}
            for name,scene in groups.items():
                pick=use & scene[:,None,None]
                if name=='contact_positive':pick &= positive[...,q]
                values=s[pick]
                theirs=expected['class_score_overlap'][name]
                assert theirs['slots']==len(values)
                assert theirs['scene_denominator']==int(scene.sum())
                for p,v in theirs['quantiles'].items():
                    assert float(np.quantile(values,float(p)))==v
    return dict(status='PASS',arms=len(keys),auc='Independent weighted Mann-Whitney tie ranking',
                interpretation='Correlated slot ranking descriptor; scene cluster retained, no iid interval')


def run(out):
    began = time.monotonic()
    prior_seconds=sum(read(p)['seconds'] for p in out.glob('independent_audit_failure_*.json'))
    deadline = began+max(0.,250.-prior_seconds)
    assert not (out/'independent_audit.json').exists(), 'Preserve completed audit'
    rows = read(out/'scene_rows.json')
    plan = read(out/'PLAN.json')
    assert plan['near_pass_light_weight']==.25 and plan['near_pass_max_gap_m']==.10
    manifest=read(out/'execution_manifest.json')
    for name,digest in {**manifest['inherited'],**manifest['new_sources']}.items():
        assert sha(name)==digest,name
    model_manifest=read(out/'trained_models_manifest.json')
    for name,digest in model_manifest['models'].items():
        assert sha(name)==digest,name
    assert model_manifest['plan_sha256']==sha(out/'PLAN.json')
    assert model_manifest['manifest_sha256']==sha(out/'execution_manifest.json')
    source_cache=out/'independent_source_recompute.json'
    if source_cache.exists():
        cached=read(source_cache)
        assert cached['manifest_sha256']==sha(out/'execution_manifest.json')
        source_info,inventory=cached['sources'],cached['inventory']
    else:
        source_info, inventory = verify_source(out,rows)
        source_cache.write_bytes((json.dumps(dict(sources=source_info,inventory=inventory,
            manifest_sha256=sha(out/'execution_manifest.json'),source_sha256=sha(__file__),
            stage='Independent geometry/support/inventory complete'),ensure_ascii=False,indent=2)+'\n').encode())
    print('AUDIT_STAGE_SOURCE',round(time.monotonic()-began,3),flush=True)
    supervision = verify_supervision(out,rows)
    seal = read(out/'sealed_calibration.json')
    for name,digest in seal['bindings'].items():
        assert sha(out/name)==digest
    hold_hist = out/'data/hold/hist.npy'
    assert (out/'sealed_calibration.json').stat().st_mtime_ns<=hold_hist.stat().st_mtime_ns
    assert (out/'sealed_calibration.json').stat().st_mtime_ns<=(out/'data/hold/task_scores.npz').stat().st_mtime_ns
    models = []
    for path in sorted((out/'models').glob('*')):
        if path.suffix in ('.pt','.pickle'):
            models.append(dict(path=path.relative_to(out).as_posix(),sha256=sha(path)))
    assert len([m for m in models if m['path'].startswith('models/S_')])==6
    # Check parameter and lineage metadata without loading any checkpoint network.
    sr,br = read(out/'S_training_receipt.json'),read(out/'B_training_receipt.json')
    assert len([m for m in models if m['path'].startswith('models/B_') and
                not m['path'].endswith('_partial_step.pt')])==sum('path' in j for j in br['jobs'])
    assert sr['recipe']==dict(max_iter=100,learning_rate=.1,max_leaf_nodes=7,max_depth=3,
                            min_samples_leaf=50,l2_regularization=1.,early_stopping=False)
    assert (out/'B_profile_receipt.json').stat().st_mtime_ns<=(out/'B_training_receipt.json').stat().st_mtime_ns
    for receipt in (sr,br):
        for job in receipt['jobs']:
            if 'path' in job:
                assert sha(job['path'])==job['sha256']
    archives, reconstructed, curve_details = {}, {}, {}
    prod = read(out/'metrics.json')
    cal_csv = list(csv.DictReader((out/'calibration_curve.csv').open(encoding='utf-8-sig',newline='')))
    hold_csv = list(csv.DictReader((out/'hold_cost_benefit_curve.csv').open(encoding='utf-8-sig',newline='')))
    for split in ('cal','hold'):
        verify_cached_scores(out,split)
        with np.load(out/f'{split}_grades_notifications.npz') as a:
            keys = a['keys'].tolist()
            grades = dict(zip(keys,a['grades']))
            saved = dict(zip(keys,a['notifications']))
            category = a['category']
            scores,arms = a['rawlight_scores'],a['arm_keys'].tolist()
        with np.load(out/'data'/split/'task_scores.npz') as a:
            np.testing.assert_array_equal(a['category'],category)
            np.testing.assert_array_equal(a['light_scores'],scores)
            np.testing.assert_array_equal(a['old5'],grades['fixed/old5'])
            np.testing.assert_array_equal(a['m3'],grades['fixed/m3'])
            for i,seed in enumerate(SEEDS):
                np.testing.assert_array_equal(a['both'][i],grades[f'both{seed}'])
        emissions = {key:scalar_gap1(g) for key,g in grades.items()}
        for key,e in emissions.items():
            np.testing.assert_array_equal(e,saved[key])
            if key not in ('fixed/m3','fixed/old5','both955','both956','both957'):
                np.testing.assert_array_equal(grades[key]==2,grades['fixed/old5']==2)
            independent = dict(costs=costs(e,rows[split]),contacts={
                ref:{group:compare_contacts(e,emissions[base],category,rows[split],group)
                     for group in ('all','dark_thin','sign_edge')}
                for ref,base in (('old5','fixed/old5'),('both','both955'))})
            compare_metric(independent,prod['reports'][f'{split}/{key}'])
            reconstructed[f'{split}/{key}']=independent
        for ai,arm in enumerate(arms):
            score = scores[ai]
            curves,chosen,cap = sweep(score,grades['fixed/old5'],rows[split],category,-np.inf,
                                      deadline,descriptive=(split=='hold'))
            selected = [r for r in (cal_csv if split=='cal' else hold_csv) if r['arm']==arm]
            verify_curve(curves,selected,descriptive=(split=='hold'))
            if split=='cal':
                for point,multiplier in (('main',1.),('secondary',1.25)):
                    record = seal['calibrations'][arm][point]
                    expected_tau = curve_tau(record)
                    assert chosen[point]==expected_tau,(arm,point,chosen[point],expected_tau)
                    assert record['weighted_cap_quarters']==cap*multiplier
                curve_details[arm] = dict(ties=len(curves)-2,
                    main_tau=seal['calibrations'][arm]['main'],secondary_tau=seal['calibrations'][arm]['secondary'])
            for point in ('main','secondary'):
                tau = curve_tau(seal['calibrations'][arm][point])
                np.testing.assert_array_equal(grade(score,grades['fixed/old5'],np.inf if tau is None else tau),
                                              grades[f'{arm}/{point}'])
        archives[split] = dict(emissions=emissions,category=category)
        print('AUDIT_STAGE',split,round(time.monotonic()-began,3),flush=True)
    ledger = verify_ledgers(out,archives,rows)
    diagnostic = verify_diagnostic(out,rows)
    signals = verify_signal(prod,arms)
    evaluation = read(out/'evaluation_receipt.json')
    for name,digest in evaluation['outputs_sha256'].items():
        assert sha(out/name)==digest
    result = dict(status='PASS',seconds=time.monotonic()-began,source_sha256=sha(__file__),
        checks=['Independent native geometry support and all physical-source inventory hashes/disjointness',
                'Task labels, masks, cost weights and exact per-height positive/negative balance',
                'Saved model hashes, fixed shallow HGB recipe and profile before long B training',
                'Scalar gap1 and exact old5 strong lock for every arm/seed/workpoint',
                'All new-cal negative ties and both lowest feasible gates independently replayed',
                'All hold ties, +/-inf and cost/benefit/far-clear curves independently replayed',
                'Every event/notification ledger and scene-cluster aggregate reconstructed',
                'Declared gate, family seed directions and preferred candidate independently checked'],
        sources=source_info,inventory=inventory,supervision=supervision,models=models,
        calibration=curve_details,ledger_counts=ledger,strong_signals=signals,separability=diagnostic,
        rendering=0,training=0,inference=0,protected_access=0)
    (out/'independent_audit.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','calibration')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--finalize-completed-replay',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010')
    args=parser.parse_args()
    out=args.out
    started=time.monotonic()
    try:
        finalize_completed_replay(out) if args.finalize_completed_replay else run(out)
    except BaseException as error:
        out.mkdir(parents=True,exist_ok=True)
        (out/f'independent_audit_failure_{time.time_ns()}.json').write_bytes((json.dumps(
            dict(status='FAILED',seconds=time.monotonic()-started,error=repr(error),source_sha256=sha(__file__)),
            ensure_ascii=False,indent=2)+'\n').encode())
        raise

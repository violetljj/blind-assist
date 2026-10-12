"""Independent CPU recount of retained finite-support diagnostics.

Reads candidates, source scores/masks and frozen EMA timing. Does not invoke the
space runner or its evaluator, regenerate support/coverage or geometry masks, or
run inference. Output audit.json is exclusive; existing evidence is preserved.
"""
import csv
import hashlib
import json
import pathlib
import time
import numpy as np


def audit():
    started = time.monotonic()
    root = pathlib.Path(__file__).resolve().parents[4]
    out = root/'artifacts.local/work/cnh-bearing-support-cross-dev-20261008'
    sec = root/'artifacts.local/work/cnh-sector-notice-dev-20261008'
    opp = root/'artifacts.local/work/cnh-ema-bearing-opportunity-dev-20261008'
    names = ('zero','const_pos','const_neg','pulse_pos','pulse_neg')
    sensors = ('single','dual')

    def read(p):
        return json.loads(p.read_text(encoding='utf8'))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def check():
        if time.monotonic()-started >= 60:
            raise TimeoutError('Independent audit60s CPU cap')

    result = read(out/'result.json')
    plan = read(out/'PLAN.json')
    assert result['status'] == 'COMPLETE'
    for p, expected in plan['hashes'].items():
        assert sha(root/p) == expected, p
    assert sha(out/'candidate.npz') == result['candidate_sha256']
    assert sha(out/'ledger.npz') == result['ledger_sha256']
    with np.load(out/'candidate.npz') as z:
        cand = {k:z[k] for k in z.files}
    with np.load(out/'ledger.npz') as z:
        ledger = {k:z[k] for k in z.files}
    with np.load(sec/'ledger.npz') as z:
        truth, score, units, configs = (z[k] for k in ('truth','score','unit','config'))
    with np.load(opp/'ledger.npz') as z:
        old = {k:z[k] for k in z.files}
    contact, deadline = old['contact'], old['deadline']
    assert (len(units),sum(contact)) == (3840,229)
    for k in ('unit','config'):
        np.testing.assert_array_equal(cand[k],old[k])
        np.testing.assert_array_equal(ledger[k],old[k])
    np.testing.assert_array_equal(ledger['contact'],contact)
    np.testing.assert_array_equal(ledger['deadline'],deadline)
    assert len(np.unique(units)) == 96
    with (out/'events.csv').open(encoding='utf-8-sig',newline='') as f:
        csvrows = list(csv.DictReader(f))
    lookup = {}
    seen = set()
    for row in csvrows:
        key = (row['name'],row['sensor'],int(row['unit']),int(row['config']))
        assert key not in seen
        seen.add(key)
        lookup[key] = row
    assert len(csvrows) == 2290
    cells = {}
    all_firsts = 0
    event_masks = {}
    empty_truth_counts = {}
    for ni, name in enumerate(names):
        for si, sensor in enumerate(sensors):
            check()
            key = name+'/'+sensor
            reported = result['metrics'][key]
            source_alarm = old[key+'/alarm']
            first_all = np.full(3840,-1,int)
            for row in range(3840):
                found = [f for f in range(2,13) if source_alarm[row,f]]
                if found:
                    first_all[row] = found[0]
            np.testing.assert_array_equal(first_all,cand['first'][ni,si])
            event_first = np.full(3840,-1,int)
            for row in np.flatnonzero(contact):
                if 0 <= first_all[row] <= deadline[row]:
                    event_first[row] = first_all[row]
            np.testing.assert_array_equal(event_first,old[key+'/first'])
            valid = first_all >= 0
            # All candidate windows, not only evaluator contact events.
            for row in np.flatnonzero(valid):
                f = int(first_all[row])
                base = int(np.argmax(score[ni,si,row,f]))
                assert int(cand['old'][ni,si,row]) == base
                assert int(cand['runner'][ni,si,row]) == int(np.argsort(-score[ni,si,row,f],kind='stable')[1])
                mask = int(cand['mask'][ni,si,row])
                state = int(cand['state'][ni,si,row])
                count = int(cand['cluster_count'][ni,si,row])
                assert 0 <= mask <= 7 and mask == int(cand['cluster_sector_union'][ni,si,row])
                if count == 0:
                    assert state == 0 and mask == 0
                elif count == 1:
                    assert state == (1 if mask in (1,2,4) else 2)
                else:
                    assert state == 3
                veto = -1 if not (mask & (1 << base)) and bool(cand['coverage'][ni,si,row,base]) else base
                switch = {1:0,2:1,4:2}[mask] if state == 1 else base
                output_set = mask if mask else 1 << base
                assert int(cand['veto'][ni,si,row]) == veto
                assert int(cand['switch'][ni,si,row]) == switch
                assert int(cand['set'][ni,si,row]) == output_set
                if sensor == 'single':
                    assert not bool(cand['disagree'][ni,si,row])
            all_firsts += int(valid.sum())
            rr = np.flatnonzero(event_first >= 0)
            arms = {}
            flags = {}
            association = runner_legal = boundary_wrong = extra_labels = 0
            cross = {}
            anatomy = {}
            wrong_anatomy = {}
            empty_truth_counts[key] = 0
            for arm in ('old','veto','switch','set','disagree'):
                metrics = {k:0 for k in ('clean','unique','wrong','abstain')}
                outcome = {k:np.zeros(3840,bool) for k in metrics}
                masks = np.zeros(3840,int)
                for row in rr:
                    selected = int(cand['old' if arm == 'disagree' else arm][ni,si,row])
                    if arm == 'set':
                        selected_mask = selected
                    elif arm == 'disagree' and bool(cand['disagree'][ni,si,row]):
                        selected_mask = 0
                    else:
                        selected_mask = 0 if selected < 0 else 1 << selected
                    masks[row] = selected_mask
                    f = int(event_first[row])
                    legal = [i for i in range(3) if truth[ni,row,f,i]]
                    emitted = [i for i in range(3) if selected_mask & (1 << i)]
                    clean = bool(emitted) and all(i in legal for i in emitted)
                    unique = clean and len(legal) == 1
                    values = dict(clean=clean,unique=unique,wrong=bool(emitted) and not clean,abstain=not emitted)
                    for field, value in values.items():
                        outcome[field][row] = value
                        metrics[field] += int(value)
                for field, value in outcome.items():
                    np.testing.assert_array_equal(value,ledger[key+'/'+arm+'/'+field])
                assert metrics == reported['arms'][arm], (key,arm)
                assert metrics['clean']+metrics['wrong']+metrics['abstain'] == len(rr)
                arms[arm] = metrics
                flags[arm] = outcome
                event_masks[key+'/'+arm] = masks
            np.testing.assert_array_equal(flags['old']['clean'],old[key+'/L2/clean'])
            np.testing.assert_array_equal(flags['old']['unique'],old[key+'/L2/unique'])
            paired = {}
            for arm in ('veto','switch','set','disagree'):
                e, b = flags[arm], flags['old']
                pair = dict(clean_rescued=int((contact & e['clean'] & ~b['clean']).sum()),
                    clean_lost=int((contact & ~e['clean'] & b['clean']).sum()),
                    clean_diff=arms[arm]['clean']-arms['old']['clean'],
                    unique_diff=arms[arm]['unique']-arms['old']['unique'],
                    wrong_removed=int((contact & b['wrong'] & ~e['wrong']).sum()),
                    new_wrong=int((contact & ~b['wrong'] & e['wrong']).sum()),
                    new_abstain=int((contact & e['abstain'] & ~b['abstain']).sum()))
                assert pair == reported['paired'][arm], (key,arm,pair,reported['paired'][arm])
                assert pair['clean_rescued']-pair['clean_lost'] == pair['clean_diff']
                paired[arm] = pair
            for row in np.flatnonzero(contact):
                f = int(event_first[row])
                record = lookup[(name,sensor,int(units[row]),int(configs[row]))]
                assert int(record['first_output']) == f
                assert record['timely'] == str(f >= 0)
                assert record['status'] == ('EVALUABLE' if f >= 0 else 'NOT_TIMELY')
                if f < 0:
                    continue
                legal = truth[ni,row,f]
                legal_mask = sum(1 << i for i in range(3) if legal[i])
                empty_truth_counts[key] += int(legal_mask == 0)
                base = int(cand['old'][ni,si,row])
                mask = int(cand['mask'][ni,si,row])
                wrong = bool(flags['old']['wrong'][row])
                assert record['selected_supported_by_target'] == str(bool(legal[base]))
                assert record['target_cross_boundary'] == str(int(legal.sum()) > 1)
                adjacent = not bool(legal[base]) and any(abs(base-i) == 1 for i in range(3) if legal[i])
                assert record['adjacent_to_target_sector'] == str(adjacent)
                assert int(record['mask']) == mask
                has_competitor = record['selected_has_competitor'] == 'True'
                selected_count = int(record['selected_competitor_count'])
                assert has_competitor == (selected_count > 0)
                assoc = wrong and has_competitor and bool(mask & legal_mask)
                association += int(assoc)
                assert record['association_oracle_opportunity'] == str(assoc)
                run_legal = bool(legal[int(cand['runner'][ni,si,row])])
                runner_legal += int(wrong and run_legal)
                assert record['runner_legal'] == str(run_legal)
                boundary = float(cand['boundary_distance'][ni,si,row])
                yaw = float(cand['yaw_span'][ni,si,row])
                test_boundary = bool(np.isfinite(boundary) and yaw >= boundary)
                assert record['yaw_ge_boundary'] == str(test_boundary)
                boundary_wrong += int(wrong and test_boundary)
                chosen_covered = bool(cand['coverage'][ni,si,row,base])
                selected_support = bool(mask & (1 << base))
                crosskey = 'state'+str(int(cand['state'][ni,si,row]))+'/coverage'+str(int(chosen_covered))+'/support'+str(int(selected_support))+'/wrong'+str(int(wrong))
                cross[crosskey] = cross.get(crosskey,0)+1
                extra_labels += int(cand['set'][ni,si,row]).bit_count()-1
                for field in ('selected_supported_by_target','target_cross_boundary','adjacent_to_target_sector','selected_has_competitor','selected_geometry_empty','competitor_closer'):
                    if record[field] == 'True':
                        anatomy[field] = anatomy.get(field,0)+1
                        if wrong:
                            wrong_anatomy[field] = wrong_anatomy.get(field,0)+1
                for arm in flags:
                    for field in flags[arm]:
                        assert record[arm+'_'+field] == str(bool(flags[arm][field][row]))
            for field, value in (('events',229),('timely',len(rr)),('not_timely',229-len(rr)),('not_evaluable',0),('association_opportunity',association),('wrong_runner_legal',runner_legal),('wrong_yaw_ge_boundary',boundary_wrong),('extra_set_labels',extra_labels),('all_window_firsts',int(valid.sum()))):
                assert reported[field] == value, (key,field)
            assert reported['cross_table'] == cross
            assert reported['flags'] == anatomy and reported['wrong_flags'] == wrong_anatomy
            assert sum(cross.values()) == len(rr)
            cells[key] = dict(arms=arms,paired=paired,association_opportunity=association,
                timely=len(rr),all_window_firsts=int(valid.sum()),extra_set_labels=extra_labels,
                first_truth_empty=empty_truth_counts[key])
    routes = {}
    for sensor in sensors:
        four = [cells[name+'/'+sensor] for name in names[1:]]
        recomputed = {}
        for arm in ('veto','disagree'):
            pairs = [c['paired'][arm] for c in four]
            recomputed[arm] = any(p['wrong_removed'] >= 3 for p in pairs) and all(p['clean_lost'] == 0 and p['new_abstain'] <= 10 for p in pairs)
        pairs = [c['paired']['switch'] for c in four]
        recomputed['switch'] = any(p['clean_rescued'] >= 3 for p in pairs) and all(p['clean_lost'] <= 1 and p['clean_diff'] >= 0 and p['unique_diff'] >= 0 for p in pairs)
        pairs = [c['paired']['set'] for c in four]
        recomputed['set'] = any(p['clean_rescued'] >= 3 for p in pairs) and all(p['clean_lost'] <= 1 and p['clean_diff'] >= 0 and p['unique_diff'] >= 0 and p['new_wrong'] <= 10 and p['new_abstain'] == 0 for p in pairs)
        recomputed['association'] = any(c['association_opportunity'] >= 3 for c in four)
        assert recomputed == result['decisions'][sensor], (sensor,recomputed,result['decisions'][sensor])
        routes[sensor] = recomputed
    assert all_firsts == 21019
    audit_result = dict(status='PASS',seconds=time.monotonic()-started,backend='CPU numpy independent summary; no spatial-operator rerun',
        units=96,events_denominator=229,conditions=5,configurations=2,candidate_window_descriptors=all_firsts,
        first_contact_event_csv_rows=2290,arms_including_original=5,candidate_arms=4,
        exact_notifications_source_sha256=sha(opp/'ledger.npz'),
        candidate_sha256=result['candidate_sha256'],ledger_sha256=result['ledger_sha256'],
        metrics=cells,decisions=routes,
        checks=['Frozen sources hashes','All-window earliest main EMA source alarms match candidates','Contact deadline clipping preserves O.first',
            'Candidate old/runner labels reproduce source score order','Candidate veto/switch/set rules reconstructed on all21019 descriptors',
            'SET uses alllabelslegal subset, not intersection','clean/unique/wrong/abstain and fourarm paired counts independently reconstructed',
            'wrong_removed and clean_rescued remain distinct','CSV nonexclusive flags/association oracle and cross-table tally',
            'Fourchallenge routes reproduce frozen PLAN per configuration'],
        limits='Spatial support/coverage/cluster arrays and evaluator masks are retained inputs, not independently regenerated; competitor flag accepted from event anatomy. Association is a truth-conditioned diagnostic opportunity, not actual repair. Source EMA alarm ledger preserved; no extra inference or candidate tuning.')
    with (out/'audit.json').open('x',encoding='utf8') as f:
        json.dump(audit_result,f,ensure_ascii=False,indent=2)
    print('SUPPORT_INDEPENDENT_AUDIT_PASS',json.dumps(dict(seconds=audit_result['seconds'],descriptors=all_firsts,decisions=routes)))
    for key, cell in cells.items():
        print(key,json.dumps(dict(paired=cell['paired'],association=cell['association_opportunity'],truth_empty=cell['first_truth_empty'])))


if __name__ == '__main__':
    audit()

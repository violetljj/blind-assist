"""Independent v1 admission/integrity audit; never opens eval method payloads."""
import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
from rgb_body_query_fixed_grid import independent_xyz_labels
from audit_sync_rgb_tof_feasibility_dev import interval


def load(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def chronological_windows(stamps):
    """Scalar reference search, deliberately independent of acquisition vectorization."""
    times = np.asarray(stamps)
    if not len(times):
        return []
    grid = times[0] + .2*np.arange(int((times[-1]-times[0])/.2)+1)
    chosen = []
    start = 0
    while start+16 <= len(grid) and len(chosen) < 32:
        indices = [int(np.argmin(abs(times-g))) for g in grid[start:start+16]]
        if len(set(indices)) == 16 and all(abs(times[i]-g) <= .10000001 for i,g in zip(indices, grid[start:start+16])):
            chosen.extend((float(g), float(times[i])) for i,g in zip(indices, grid[start:start+16]))
            # chosen is flattened; two windows =32 timestamps.
            if len(chosen) == 32:
                break
            start += 16
        else:
            start += 1
    return chosen


def classify(depth, k, queries):
    records = []
    labels = []
    for q in queries:
        label, _ = independent_xyz_labels(depth, k, q, np.ones(depth.shape, bool))
        pos, free, unk = (int((label == v).sum()) for v in [1, 0, 2])
        state = 'POSITIVE' if pos >= 16 else ('FREE_ON_SAMPLED_RAYS' if free >= 16 and pos == 0 and unk == 0 else 'UNKNOWN')
        records.append(dict(state=state, positive_pixels=pos, free_ray_pixels=free, unknown_pixels=unk))
        labels.append(label)
    return records, np.asarray(labels)


def extra_tables(root):
    """Independent all-source/common-source aggregation, including compact report."""
    buckets={}
    with (root/'train_cal_fusion_queries.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            assert r['role'] in ('train','cal')
            subsets=['all_grid']+(['FARO_available'] if r['source_available']=='True' else [])
            for sub in subsets:
                for role in [r['role'],'combined']:
                    key=(sub,role,r['noise_arm'],int(r['repeat']),r['arm'],r['band'])
                    b=buckets.setdefault(key,dict(queries=0,POS=0,W=0,FREE=0,F=0,UNKNOWN=0,U=0))
                    b['queries']+=1
                    state=r['state']
                    if state=='POSITIVE':b['POS']+=1;b['W']+=r['witness']=='True'
                    elif state=='FREE_ON_SAMPLED_RAYS':b['FREE']+=1;b['F']+=r['support']=='True'
                    else:assert state=='UNKNOWN';b['UNKNOWN']+=1;b['U']+=r['support']=='True'
    checked=0
    for path in root.glob('*_fusion_FARO_available_summary.csv'):
        prefix=path.name.removesuffix('_fusion_FARO_available_summary.csv')
        role,rest=prefix.split('_',1); noise,rep=rest.rsplit('_k',1)
        with path.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                b=buckets['FARO_available',role,noise,int(rep),r['arm'],r['band']]
                assert all(int(r[k])==v for k,v in b.items())
                checked+=1
    compact=0
    with (root/'train_cal_compact.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            b=buckets[r['subset'],'combined',r['noise_arm'],int(r['K']),r['method'],r['band']]
            assert all(int(r[k])==v for k,v in b.items())
            compact+=1
    # Frozen train/cal aggregation only. Never open sealed_eval method files.
    frozen=load(root/'frozen_inference/train_cal_description.json')['rows']
    n=0
    for s in load(root/'frozen_inference/train_cal_summary.json')['rows']:
        rr=[r for r in frozen if r['role']==s['role'] and r['arm']==s['arm']]
        assert len(rr)==s['sequences']
        assert sum(r['frames'] for r in rr)==s['per_height_slot_denominator']
        for k in s:
            if k.startswith('head_body_'):
                assert np.array_equal(np.sum([r[k] for r in rr],axis=0),s[k])
        n+=1
    return dict(status='PASS',FARO_available_summary_rows=checked,compact_rows=compact,
                frozen_summary_rows=n,eval_method_payloads_opened=0)


def describe_check(root, manifest, refs):
    """All table aggregations plus first frame per visit pixel checks, no eval."""
    fusion = root/'train_cal_fusion_queries.csv'
    synthesis_path = root/'synthesis_manifest.json'
    if not fusion.exists() or not synthesis_path.exists():
        return dict(status='NOT_AVAILABLE', eval_method_payloads_opened=0)
    with fusion.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert all(r['role'] in ('train','cal') for r in rows)
    bykey = {(r['frame'],r['noise_arm'],int(r['repeat']),r['query'],r['arm']):r for r in rows}
    assert len(bykey) == len(rows)
    prediction_paths = {}
    for model in ['dav2','unidepth']:
        candidates = [p for p in root.rglob('predictions.json') if p.parent.name == model]
        assert len(candidates) == 1, (model,candidates)
        index = load(candidates[0])['rows']
        assert all(r['role'] != 'eval' for r in index)
        prediction_paths[model] = {r['rgb_path']:r for r in index}
    source = {(r['capture'],r['frame']):r for r in refs if r['role'] != 'eval'}
    frames = {}
    ordinals = {}
    for e in manifest['entries']:
        if e['role'] == 'eval':
            continue
        ordinals[e['capture']] = e['accepted_ordinal']
        for ordinal,r in enumerate(e['frames']):
            frames[f"{e['capture']}_{r['window_id']}_{ordinal:03d}"] = source[e['capture'],r['frame']]
    checked = 0
    visited, faro_visited = set(), set()
    pixel_frames = 0
    for fr in load(synthesis_path)['frames']:
        if fr['role'] == 'eval':
            # Only manifest metadata is read; no eval raw payload is opened.
            continue
        first = fr['visit_id'] not in visited
        first_faro = bool(fr['source_available']) and fr['visit_id'] not in faro_visited
        if not (first or first_faro):
            continue
        visited.add(fr['visit_id'])
        if fr['source_available']:
            faro_visited.add(fr['visit_id'])
        pixel_frames += 1
        r = frames[fr['frame_id']]
        k = np.asarray(r['depth_K']); shape = tuple(r['depth_shape'])
        yy,xx = np.indices(shape); x=(xx-k[0,2])/k[0,0]; y=(yy-k[1,2])/k[1,1]
        edge = np.tan(np.deg2rad(22.5)); fov=(abs(x)<=edge)&(abs(y)<=edge)
        ix=np.clip(np.floor((x+edge)/(2*edge)*8).astype(int),0,7)
        iy=np.clip(np.floor((y+edge)/(2*edge)*8).astype(int),0,7)
        model_depth = {}
        for model in prediction_paths:
            pr = prediction_paths[model][r['rgb_path']]
            assert sha(pr['path']) == pr['sha256']
            with np.load(pr['path'],allow_pickle=False) as f:
                model_depth[model]=f['depth']
        with np.load(r['reference_path'],allow_pickle=False) as f:
            labels=f['labels']
        for arm in fr['arms']:
            assert sha(arm['path']) == arm['sha256']
            with np.load(arm['path'],allow_pickle=False) as f:
                seed=int(np.random.SeedSequence([20261011,ordinals[fr['capture']],r['window_id'],r['window_frame'],arm['repeat']]).generate_state(1)[0])
                assert int(f['seed'])==seed
                h=f['hist']; bg=f['background']; cov=f['coverage']
                assert np.array_equal(h,f['counts']-bg)
                peak=h.argmax(-1); height=np.take_along_axis(h,peak[...,None],-1)[...,0]
                back=np.take_along_axis(bg,peak[...,None],-1)[...,0]
                snr=height/np.sqrt(np.maximum(height,0)+2*back+1)
                valid=fov&(cov[iy,ix]>=.75)&(snr[iy,ix]>=3)
                z=(peak[iy,ix]+.5)*.3002784/np.sqrt(1+x*x+y*y)
                tof=np.full(shape,np.inf)
                if np.isfinite(f['grid_pose']).all():
                    xyz=np.stack([x[valid]*z[valid],y[valid]*z[valid],z[valid],np.ones(int(valid.sum()))])
                    camera=np.linalg.inv(f['rgb_pose'])@f['grid_pose']@xyz
                    uv=k@camera[:3]; u,v=np.rint(uv[:2]/uv[2]).astype(int)
                    ok=(camera[2]>0)&(u>=0)&(u<shape[1])&(v>=0)&(v<shape[0])
                    np.minimum.at(tof.ravel(),v[ok]*shape[1]+u[ok],camera[2,ok])
            for j,q in enumerate(manifest['queries']):
                lo,hi,domain=interval(x,y,q); near=q['high'][2]<=.8
                masks={}
                for name,z,cut,active in [('tof',tof,0.,True),('dav',model_depth['dav2'],.24403834342956543,not near),('uni',model_depth['unidepth'],.09616100788116455,near)]:
                    masks[name]=domain&np.isfinite(z)&(z>0)&(np.minimum(z-lo,hi-z)>=cut)&active
                rgb=masks['uni' if near else 'dav']
                masks.update(rgb_banded=rgb,OR=masks['tof']|rgb,AND=masks['tof']&rgb)
                for method,mask in masks.items():
                    rr=bykey[fr['frame_id'],arm['arm'],arm['repeat'],q['name'],method]
                    n,w=int(mask.sum()),int((mask&(labels[j]==1)).sum())
                    assert (n,w)==(int(rr['support_pixels']),int(rr['positive_pixels']))
                    assert (rr['support']=='True')==(n>=16) and (rr['witness']=='True')==(w>=16)
                    checked+=1
    summary_checks = 0
    buckets = {}
    for rr in rows:
        key = (rr['role'],rr['noise_arm'],int(rr['repeat']),rr['arm'],rr['band'])
        b = buckets.setdefault(key,dict(queries=0,POS=0,W=0,FREE=0,F=0,UNKNOWN=0,U=0))
        b['queries'] += 1
        if rr['state'] == 'POSITIVE':
            b['POS'] += 1; b['W'] += rr['witness']=='True'
        elif rr['state'] == 'FREE_ON_SAMPLED_RAYS':
            b['FREE'] += 1; b['F'] += rr['support']=='True'
        else:
            assert rr['state']=='UNKNOWN'
            b['UNKNOWN'] += 1; b['U'] += rr['support']=='True'
    for role,noise,repeat in sorted({key[:3] for key in buckets}):
        path = root/f'{role}_{noise}_k{repeat}_fusion_summary.csv'
        with path.open(encoding='utf-8-sig') as f:
            for s in csv.DictReader(f):
                calc = buckets[role,noise,repeat,s['arm'],s['band']]
                assert all(int(s[k])==v for k,v in calc.items())
                summary_checks += 1
    return dict(status='PASS', pixel_rows_recomputed=checked, pixel_frames=pixel_frames,
                total_rows_aggregated=len(rows), summary_rows_recomputed=summary_checks,
                sampling='First chronological frame plus first source-available FARO frame of each train/cal visit, all5arms x K2 x27queries x6methods; source-only selection',
                eval_method_payloads_opened=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--description-only', action='store_true')
    ap.add_argument('--extra-tables-only', action='store_true')
    ap.add_argument('--report-only', type=Path)
    a = ap.parse_args()
    root = a.root
    start = time.perf_counter()
    if a.report_only:
        with (root/'train_cal_compact.csv').open(encoding='utf-8-sig') as f:
            idx={(r['subset'],r['noise_arm'],r['K'],r['method'],r['band']):r for r in csv.DictReader(f)}
        body=a.report_only.read_text(encoding='utf8');subset='all_grid';n=0
        for line in body.splitlines():
            if line.startswith('### 共同FARO'):subset='FARO_available'
            cells=[c.strip() for c in line.split('|')[1:-1]]
            if len(cells)!=6 or cells[0] not in {key[1] for key in idx}:continue
            for band,value in zip(['0.3-0.8m','0.8-1.5m','1.5-3m'],cells[3:]):
                r=idx[subset,cells[0],cells[1],cells[2],band]
                assert value==f"{r['W']}/{r['POS']}；{r['F']}/{r['FREE']}；{r['U']}/{r['UNKNOWN']}"
                n+=1
        assert n==180
        assert '每zone16 sub-ray' not in body, 'Subray count typo; actual16x16=256'
        out=dict(status='PASS',numeric_table_cells_checked=n,report_sha256=sha(a.report_only),
                 elapsed_s=time.perf_counter()-start,eval_method_payloads_opened=0,
                 scope='Text boundaries reviewed plus all180method-table cells compared to independently audited compact CSV')
        (root/'report_review_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
        print(json.dumps(out));return
    if a.extra_tables_only:
        out=extra_tables(root);out['elapsed_s']=time.perf_counter()-start
        (root/'independent_extra_tables_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
        print(json.dumps(out));return
    if a.description_only:
        description=describe_check(root,load(root/'dataset_manifest.json'),load(root/'train_cal_reference_roster.json')['rows'])
        description['elapsed_s']=time.perf_counter()-start
        (root/'independent_description_audit.json').write_text(json.dumps(description,indent=2)+'\n',encoding='utf8')
        print(json.dumps(description))
        return
    plan = load(root/'PLAN.json')
    assert sha(root/'PLAN.json') == '4f87e9875d1c990dc1a6ad1dd2e7606ca3c6476431b1f5f69dcea46f2b4e619e'
    with (root/'official/metadata.csv').open(encoding='utf-8-sig') as f:
        metadata = list(csv.DictReader(f))
    inv = load(root/'consumed_inventory.json')
    consumed = set(inv['visit_ids'])
    history = load(root/'historical_arkit_ledger_rows.json')
    historical_ids = {c for row in history['rows'] for c in re.findall(r'(?:^|[\\/])(\d{8})$', row['session_root'])}
    known_ids = {r['video_id'] for r in metadata}
    assert historical_ids & known_ids <= set(inv['capture_ids'])
    history_check = dict(rows=len(history['rows']), capture_ids=sorted(historical_ids),
                         metadata_mapped_capture_ids=sorted(historical_ids & known_ids),
                         unresolved_capture_ids=sorted(historical_ids-known_ids))
    # Propagating every listed capture to visit must be complete.
    assert {r['visit_id'] for r in metadata if r['video_id'] in inv['capture_ids']} <= consumed
    expected = [(i, r['video_id'], r['visit_id']) for i,r in enumerate(metadata)
                if r['is_in_upsampling'] == 'True' and r['visit_id'] not in consumed and r['visit_id'] != 'NA']
    ordered = load(root/'ordered_candidates.json')
    actual = [(r['metadata_index'], r['video_id'], r['visit_id']) for r in ordered]
    # Inventory may list only the first eligible capture per visit.
    first = []
    seen = set()
    for item in expected:
        if item[2] not in seen:
            first.append(item); seen.add(item[2])
    assert actual in (expected, first), 'Candidate order differs from official physical order'
    m = load(root/'dataset_manifest.json')
    candidates = load(root/'candidate_gate_records.json') if (root/'candidate_gate_records.json').exists() else []
    assert [r['metadata_index'] for r in candidates] == sorted(r['metadata_index'] for r in candidates)
    closed_visits = set(consumed)
    selection_checks, gaps = [], []
    for r in candidates:
        if r['visit_id'] in closed_visits:
            # Later-capture transport failures from attempt1 remain in lineage
            # after an earlier capture of that visit was recovered.
            assert r['status'] == 'SKIP_SOURCE_FAILURE'
            assert not (root/'source'/r['capture']/'selection.json').exists()
            continue
        for previous in r.get('previous_attempts', []):
            assert previous['status'] not in ('SKIP_REFERENCE_GATE','ACCEPTED'), 'Scientific gate reopened during acquisition recovery'
        source = root/'source'/r['capture']
        path = source/'selection.json'
        if path.exists():
            s = load(path)
            independently_selected = chronological_windows(s['native_common_timestamps_s'])
            assert len(independently_selected) == 32
            observed = [(x['grid_timestamp_s'], x['rgb_timestamp_s']) for x in s['rows']]
            assert np.allclose(observed, independently_selected, atol=1e-9)
            assert len({x['source_id'] for x in s['rows']}) == 32, 'Overlapping native frames across windows'
            assert s['selected_before_labels'] and s['FARO_not_selection_input']
            closed_visits.add(r['visit_id'])
            selection_checks.append(dict(capture=r['capture'], frames=32, status='PASS'))
            if 'selection_sha256' in r:
                assert sha(path) == r['selection_sha256']
    refs = load(root/'reference_roster.json')['rows']
    reference_by_key = {(r['capture'], r['frame']): r for r in refs}
    splits, counts = {}, {'train': 0, 'cal': 0, 'eval': 0}
    gates = []
    for ordinal,e in enumerate(m['entries']):
        expected_role = plan['source']['split_by_accepted_ordinal'][ordinal]
        assert e['role'] == expected_role and e['accepted_ordinal'] == ordinal
        assert e['visit_id'] not in consumed and e['visit_id'] not in splits
        splits[e['visit_id']] = e['role']
        assert len(e['frames']) == 32
        near = midfar = 0
        for fr in e['frames']:
            assert fr['role'] == e['role'] and fr['visit_id'] == e['visit_id']
            assert abs(fr['rgb_timestamp_s']-fr['grid_timestamp_s']) <= .10000001
            for path_key, hash_key in [('rgb_path','rgb_sha256'), ('native_depth_path','native_depth_sha256')]:
                assert sha(fr[path_key]) == fr[hash_key]
            ref = reference_by_key[e['capture'], fr['frame']]
            assert sha(ref['reference_path']) == ref['reference_sha256']
            with np.load(ref['reference_path'], allow_pickle=False) as f:
                states, labels = classify(f['depth'], np.asarray(fr['depth_K']), m['queries'])
                assert np.array_equal(labels, f['labels'])
            for q,calc,stored in zip(m['queries'], states, ref['queries']):
                for key in calc:
                    assert calc[key] == stored[key]
                near += q['low'][2] == .3 and calc['state'] == 'POSITIVE'
                midfar += q['low'][2] >= .8 and calc['state'] == 'FREE_ON_SAMPLED_RAYS'
            counts[e['role']] += 1
        assert near == e['near_POS'] and midfar == e['midfar_strict_FREE']
        assert near >= 16 and midfar >= 32
        gates.append(dict(capture=e['capture'], near_POS=near, midfar_strict_FREE=midfar))
    accepted_order = [next(r['metadata_index'] for r in candidates if r['capture'] == e['capture']) for e in m['entries']]
    assert accepted_order == sorted(accepted_order), 'Final roles must follow recovered official order'
    for r in candidates:
        if r['status'] == 'SKIP_REFERENCE_GATE':
            assert r['near_POS'] < 16 or r['midfar_strict_FREE'] < 32
            gatepath = root/'source'/r['capture']/'gate_inputs.npz'
            if not gatepath.exists():
                gaps.append(dict(capture=r['capture'], note='Rejected gate raw input not retained; count-only gate assertion'))
                continue
            near = midfar = 0
            with np.load(gatepath, allow_pickle=False) as f:
                for depth, k in zip(f['depth'], f['K']):
                    states, _ = classify(depth, k, m['queries'])
                    near += sum(q['low'][2] == .3 and s['state'] == 'POSITIVE' for q,s in zip(m['queries'],states))
                    midfar += sum(q['low'][2] >= .8 and s['state'] == 'FREE_ON_SAMPLED_RAYS' for q,s in zip(m['queries'],states))
            assert near == r['near_POS'] and midfar == r['midfar_strict_FREE']
            gates.append(dict(capture=r['capture'], near_POS=near, midfar_strict_FREE=midfar, rejected=True))
    method_tables = list(root.glob('train_cal*queries.csv'))
    for path in method_tables:
        with path.open(encoding='utf-8-sig') as f:
            assert all(r['role'] in ('train','cal') for r in csv.DictReader(f)), 'Eval method row forbidden'
    description = describe_check(root,m,refs)
    out = dict(status='PASS_WITH_DECLARED_GAPS' if gaps else 'PASS', elapsed_s=time.perf_counter()-start,
               plan_sha256=sha(root/'PLAN.json'), candidate_order_entries=len(actual), candidate_records=len(candidates),
               chronological_selections=selection_checks, accepted_gates=gates, split_frames=counts,
               historical_ledger_check=history_check,
               train_cal_description=description,
               accepted_visits=len(splits), frames=sum(counts.values()), queries=sum(counts.values())*27,
               target_frame_shortfall=384-sum(counts.values()), target_query_shortfall=10368-sum(counts.values())*27,
               gaps=gaps, eval_method_payloads_opened=0, protected_access=0, training=0,
               scope='Official ordering, chronological first windows, accepted reference-only gates, visit isolation and source hashes; no eval method payload opened')
    (root/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out))


if __name__ == '__main__':
    main()

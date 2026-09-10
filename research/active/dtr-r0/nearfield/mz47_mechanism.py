"""CPU-only saved-output/supervision diagnosis; no learned forward or fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from contact_retina_spec import BODY_BOXES

EVENTS = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
ARMS = ('FROZEN', 'REPLAY', 'ENRICH')
CONDITIONS = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024): result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def selected_boolean_rows(path, key, indices):
    """Stream NPY members in bounded blocks; never load dense visual arrays."""
    indices = np.asarray(indices, dtype=np.int64)
    assert np.array_equal(indices, np.unique(indices))
    parts = []
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        assert not fortran and not dtype.hasobject
        row_bytes = int(np.prod(shape[1:]))*dtype.itemsize
        for begin in range(0, shape[0], 64):
            count = min(64, shape[0]-begin)
            blob = stream.read(count*row_bytes)
            assert len(blob) == count*row_bytes
            take = indices[(indices >= begin) & (indices < begin+count)]-begin
            if len(take):
                block = np.frombuffer(blob, dtype=dtype).reshape((count,)+shape[1:])
                parts.append(block[take] > 0)
    result = np.concatenate(parts)
    assert len(result) == len(indices)
    return result


def packet(ranges, valid, condition):
    ranges, valid = ranges.copy(), valid.copy()
    if condition == 'DROP_CLOSE':
        close = valid.all(2) & ((ranges[:,:,1].astype(float)-ranges[:,:,0].astype(float)) < .600)
        ranges[close] = 0; valid[close] = False
    else: assert condition == 'IDEAL'
    return ranges, valid


def eligibility(ranges, valid, rays):
    """Only the frozen ray/query geometry; no visual/head computation."""
    good = valid & np.isfinite(ranges) & (ranges > 0) & (ranges <= 4)
    clean = np.where(good, ranges, np.float32(0))
    points = clean[...,None,None]*rays[None,:,None,:,:]
    points += np.array([0,0,1.7], np.float32)
    masks = []
    for low, high in BODY_BOXES:
        for half, start in enumerate((high[0], high[0]+1.5)):
            end = points[...,0] < start+1.5 if half == 0 else points[...,0] <= start+1.5
            masks.append((points[...,0] >= start) & end & (points[...,1] >= low[1])
                         & (points[...,1] <= high[1]) & (points[...,2] >= low[2]) & (points[...,2] <= high[2]))
    return np.stack(masks,-1) & good[...,None,None]


def coverage(ranges, valid, query, known, truth, rays, condition):
    fields = {key: [] for key in ('candidate','witness','selected_native','local_positive_cells','eligible_positive_cells','local_negative_cells')}
    for begin in range(0, len(truth), 64):
        end = min(begin+64, len(truth))
        r,v = packet(ranges[begin:end], valid[begin:end], condition)
        eligible = eligibility(r,v,rays)
        k = known[begin:end,:,None,:,None] & v[:,:,:,None,None]
        positive = query[begin:end] & k
        witness = positive & eligible
        values = dict(candidate=eligible.any((1,2,3)), witness=witness.any((1,2,3)),
                      selected_native=positive.any((1,2,3)), local_positive_cells=positive.sum((1,2,3)),
                      eligible_positive_cells=witness.sum((1,2,3)),
                      local_negative_cells=k.sum((1,2,3))-positive.sum((1,2,3)))
        for key, value in values.items(): fields[key].append(value)
    return dict(truth=truth, **{key:np.concatenate(value) for key,value in fields.items()})


def coverage_summary(data, indices):
    y = data['truth'][indices]
    result = dict(frame_presentations=len(indices), unique_frame_ids=int(len(np.unique(indices))),
                  positive_events=y.sum(0).tolist())
    for name in ('candidate','witness','selected_native'):
        result['positive_with_'+name] = (y & data[name][indices]).sum(0).tolist()
        result['positive_without_'+name] = (y & ~data[name][indices]).sum(0).tolist()
    for name in ('local_positive_cells','eligible_positive_cells','local_negative_cells'):
        result[name] = data[name][indices].sum(0).tolist()
    return result


def tally(scores, truth, selected):
    pred, y = scores[selected] >= 0, truth[selected]
    return dict(tp=(pred & y).sum(0).tolist(), fp=(pred & ~y).sum(0).tolist(), fn=(~pred & y).sum(0).tolist())


def run(task):
    start = time.perf_counter(); source = task/'run-v1'; out = task/'diagnostic-v1'
    out.mkdir(parents=True, exist_ok=True)
    work = task.parent
    inputs, expected = {}, {}
    receipt = read(source/'receipt.json'); assert receipt['status'] == 'PASS'
    for name, digest in receipt['outputs'].items(): expected[(source/name).resolve()] = digest
    for path, digest in receipt['inputs'].items(): expected[Path(path).resolve()] = digest
    def bind(path):
        path = Path(path).resolve(); digest = sha(path)
        if path in expected: assert digest == expected[path], str(path)
        inputs[str(path)] = digest
        return path
    bind(source/'receipt.json')
    p = np.load(bind(source/'predictions.npz'), allow_pickle=False)
    rich = np.load(bind(source/'rich-local-supervision.npz'), allow_pickle=False)
    schedule = np.load(bind(source/'schedule.npz'), allow_pickle=False)
    fits = read(bind(source/'fits.json')); preparation = read(bind(source/'preparation.json'))
    parity = read(bind(source/'parity.json')); assert parity['status'] == 'PASS'
    scored = read(bind(task/'score-v1/result.json'))
    groups = read(bind(source/'groups.json'))['records']
    np.testing.assert_array_equal(p['rich/frame_ids'], rich['frame_ids'])
    assert preparation['exact_extra_truth_match']
    state = torch.load(bind(source/'REPLAY.pt'), map_location='cpu', weights_only=True)
    other = torch.load(bind(source/'ENRICH.pt'), map_location='cpu', weights_only=True)
    original_folder = work/'mz20-rank-objective-20260910/run-v1'
    original = torch.load(bind(original_folder/'BODY_RANK.pt'), map_location='cpu', weights_only=True)
    for key in ('rays','grid','relative'):
        assert torch.equal(state[key], other[key]) and torch.equal(state[key], original[key])
    rays = state['rays'].numpy()
    del state, other, original
    cutoffs = dict(FROZEN=np.load(bind(original_folder/'BODY_RANK-cutoff.npy')),
                   **{arm:np.load(bind(source/(arm+'-cutoff.npy'))) for arm in ('REPLAY','ENRICH')})
    masks = dict(fit=np.arange(44)<12, nonfit=np.arange(44)>=12,
                 form_transfer=np.array([r['family']=='birch' for r in groups]),
                 distance_transfer=np.array([r['block']=='far' and r['family'] in ('pipe','ladder','pouch') for r in groups]),
                 context_control=np.array([r['family']=='oblique_rod' for r in groups]))
    y = p['rich/truth']
    stages, changes, swaps, score_changes = {}, [], {}, {}
    for condition in CONDITIONS:
        prefix = 'rich/'+condition+'/'
        stages[condition], swaps[condition], score_changes[condition] = {}, {}, {}
        for group, mask in masks.items():
            record = dict(positive_events=y[mask].sum(0).tolist())
            for arm in ARMS:
                a = prefix+arm+'/'
                record[arm] = {key:tally(p[a+key], y, mask) for key in ('local_before','local_after','MZ37')}
                support, restricted = p[a+'support'], p[a+'restricted_support']
                gaps = dict(no_candidate=(y[mask]&~support[mask]).sum(0).tolist(),
                            candidate_lost_by_bank=(y[mask]&support[mask]&~restricted[mask]).sum(0).tolist())
                if condition != 'MERGE_CLOSE':
                    wb,wa = p[a+'native_witness_before'], p[a+'native_witness_after']
                    gaps.update(no_witness_before=(y[mask]&~wb[mask]).sum(0).tolist(),
                                witness_fully_lost_by_bank=(y[mask]&wb[mask]&~wa[mask]).sum(0).tolist(),
                                witness_survives_but_below_cutoff=(y[mask]&wa[mask]&(p[a+'local_after'][mask]<0)).sum(0).tolist())
                before, after = p[a+'local_before']>=0, p[a+'local_after']>=0
                gaps['tp_scores_removed_by_bank'] = (y[mask]&before[mask]&~after[mask]).sum(0).tolist()
                record[arm]['gaps'] = gaps
                np.testing.assert_array_equal(record[arm]['MZ37']['tp'], scored['conditions']['rich'][condition][group][arm+'/MZ37']['tp'])
            stages[condition][group] = record
        for arm in ('REPLAY','ENRICH'):
            # Existing fixed cutoffs only. These are diagnostic local-score
            # comparisons, not calibrated new candidates or model evaluations.
            for reference in ('REPLAY','ENRICH'):
                a = prefix+arm+'/'
                scores = np.where(p[a+'restricted_support'], p[a+'restricted_raw'].astype(float)-cutoffs[reference], -1e6)
                swaps[condition][arm+'_raw_at_'+reference+'_existing_cutoff'] = tally(scores,y,masks['nonfit'])
        for q,name in enumerate(EVENTS):
            mask = masks['nonfit'] & y[:,q] & p[prefix+'REPLAY/support'][:,q]
            delta_raw = p[prefix+'ENRICH/raw'][mask,q]-p[prefix+'REPLAY/raw'][mask,q]
            delta_banked = p[prefix+'ENRICH/restricted_raw'][mask,q]-p[prefix+'REPLAY/restricted_raw'][mask,q]
            score_changes[condition][name] = dict(events=int(mask.sum()), cutoff_delta=float(cutoffs['ENRICH'][q]-cutoffs['REPLAY'][q]),
                raw_improved=int((delta_raw>0).sum()), raw_worsened=int((delta_raw<0).sum()),
                raw_mean_delta=float(delta_raw.mean()), banked_mean_delta=float(delta_banked.mean()),
                banked_improved=int((delta_banked>0).sum()), banked_worsened=int((delta_banked<0).sum()))
        for i,q in zip(*np.nonzero((p[prefix+'REPLAY/MZ37']>=0)!=(p[prefix+'ENRICH/MZ37']>=0))):
            row = dict(condition=condition, index=int(i), frame_id=str(p['rich/frame_ids'][i]), query=EVENTS[q],
                       truth=bool(y[i,q]), fit=bool(masks['fit'][i]), group=groups[i], arms={})
            for arm in ARMS:
                a=prefix+arm+'/'
                row['arms'][arm] = {key:float(p[a+key][i,q]) for key in ('raw','restricted_raw','local_before','local_after','MZ37')}
                row['arms'][arm].update(cutoff=float(cutoffs[arm][q]), support=bool(p[a+'support'][i,q]))
                if condition != 'MERGE_CLOSE':
                    row['arms'][arm].update(witness_before=bool(p[a+'native_witness_before'][i,q]), witness_after=bool(p[a+'native_witness_after'][i,q]))
            row['baseline_mz5'] = float(p[prefix+'MZ5'][i,q])
            row['diagnostic_enrich_banked_score_at_replay_cutoff'] = float(p[prefix+'ENRICH/restricted_raw'][i,q]-cutoffs['REPLAY'][q])
            changes.append(row)

    calibration = {}
    for q,name in enumerate(EVENTS):
        calibration[name] = {}
        for arm in ARMS:
            a='DEV/DROP_CLOSE/'+arm+'/'
            eligible = p[a+'support'][:,q] & (p['DEV/DROP_CLOSE/MZ5'][:,q]<0) & ~p['DEV/truth'][:,q]
            indices = np.flatnonzero(eligible)
            ranked = indices[np.argsort(p[a+'raw'][indices,q])[::-1]]
            if arm != 'FROZEN':
                assert np.nextafter(np.float64(p[a+'raw'][ranked[0],q]),np.inf) == cutoffs[arm][q]
            calibration[name][arm] = dict(existing_cutoff=float(cutoffs[arm][q]), eligible_negative_frames=len(indices),
                top_negative_examples=[dict(global_id=int(p['DEV/frame_ids'][i]), raw=float(p[a+'raw'][i,q]),
                    banked_raw=float(p[a+'restricted_raw'][i,q]), original_support=bool(p[a+'support'][i,q]),
                    banked_support=bool(p[a+'restricted_support'][i,q])) for i in ranked[:5]])

    legacy = {}
    for cohort in ('DEV','relation10000','distance5000','mz36'):
        legacy[cohort] = {}
        for condition in CONDITIONS:
            actual = scored['conditions'][cohort][condition]['all']
            legacy[cohort][condition] = {arm:{key:actual[arm+'/MZ37'][key] for key in ('tp','fp','fn')} for arm in ARMS}

    # Load only small packet/truth NPZ members; stream selected local labels.
    old_dir=work/'mz8-attribution-20260910/cache-v5'
    new_dir=work/'mz15-shared-support-20260910/cache-v1'
    old_obs=np.load(bind(old_dir/'observations.npz')); new_obs=np.load(bind(new_dir/'observations.npz'))
    old_eval=np.load(bind(old_dir/'evaluator.npz')); new_eval=np.load(bind(new_dir/'evaluator.npz'))
    all_ranges=np.concatenate([old_obs['ranges'],new_obs['ranges']])
    all_valid=np.concatenate([old_obs['valid'],new_obs['valid']])
    all_truth=np.concatenate([old_eval['truth'],new_eval['truth']])
    ids=np.unique(np.concatenate([schedule['old'].ravel(),schedule['replay'].ravel()]))
    old_count=len(old_eval['truth']); old_ids=ids[ids<old_count]; new_ids=ids[ids>=old_count]-old_count
    old_labels=bind(work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz')
    query=np.concatenate([selected_boolean_rows(old_labels,'query_counts',old_ids),
                          selected_boolean_rows(new_dir/'evaluator.npz','query_presence',new_ids)])
    known=np.concatenate([selected_boolean_rows(old_labels,'cell_known_counts',old_ids),
                          selected_boolean_rows(new_dir/'evaluator.npz','known',new_ids)])
    train_cov=coverage(all_ranges[ids],all_valid[ids],query,known,all_truth[ids],rays,'DROP_CLOSE')
    del query,known
    rich_cov={c:coverage(rich['ranges'],rich['valid'],rich['query'],rich['local_known'],rich['truth'],rays,c) for c in ('IDEAL','DROP_CLOSE')}
    for c in rich_cov:
        np.testing.assert_array_equal(rich_cov[c]['candidate'],p['rich/'+c+'/FROZEN/support'])
        np.testing.assert_array_equal(rich_cov[c]['witness'],p['rich/'+c+'/FROZEN/native_witness_before'])
    positions={key:np.searchsorted(ids,schedule[key].ravel()) for key in ('old','replay')}
    np.testing.assert_array_equal(all_truth[schedule['replay']],rich['truth'][schedule['enrich']])
    training=dict(common_old=coverage_summary(train_cov,positions['old']),
                  REPLAY_extra=coverage_summary(train_cov,positions['replay']),
                  ENRICH_extra=coverage_summary(rich_cov['DROP_CLOSE'],schedule['enrich'].ravel()))
    for arm in ('REPLAY','ENRICH'):
        for key,saved in [('positive_without_candidate','positive_presentations_without_candidate'),
                          ('positive_without_witness','positive_presentations_without_eligible_native_witness')]:
            count=np.array(training['common_old'][key])+training[arm+'_extra'][key]
            np.testing.assert_array_equal(count,fits[arm][saved])
    fit_signals=[]
    for i in np.flatnonzero(masks['fit']):
        fit_signals.append(dict(frame_id=str(rich['frame_ids'][i]), group=groups[i], truth=y[i].tolist(),
            IDEAL={key:rich_cov['IDEAL'][key][i].tolist() for key in ('candidate','witness','selected_native','local_positive_cells')},
            DROP_CLOSE={key:rich_cov['DROP_CLOSE'][key][i].tolist() for key in ('candidate','witness','selected_native','local_positive_cells')},
            banked_witness=p['rich/DROP_CLOSE/FROZEN/native_witness_after'][i].tolist()))
    weight=np.bincount(schedule['enrich'].ravel(),minlength=44)
    assert np.array_equal(weight[:12],np.full(12,100)) and not weight[12:].any()
    result=dict(status='PASS', event_order=EVENTS, scope='One saved two-arm single-seed consumed Development run; diagnostic only',
        boundary=dict(new_fits=0,new_model_forward_frames=0,gpu_operations=0,threshold_searches=0,captures=0,dense_visual_arrays_opened=0),
        stage_counts=stages, fixed_cutoff_local_diagnostics=swaps, positive_score_changes=score_changes,
        existing_cutoffs={key:value.tolist() for key,value in cutoffs.items()}, calibration_negative_tails=calibration,
        training_signal_coverage=training, fit_frame_signals=fit_signals, changed_rich_events=changes,
        legacy_query_counts=legacy, checks=dict(saved_score_counts_match=True,frozen_geometry_buffers_equal=True,
            numpy_geometry_support_and_witness_match_saved_rich=True,training_coverage_matches_saved_fit_counters=True,
            existing_new_arm_cutoff_maxima_confirmed=True,extra_truth_patterns_equal=True),
        limits=['MERGE_CLOSE contributor witnesses are not inferred from original slots.',
                'Candidate-level learned fields/argmax coordinates were not saved; retained witness existence cannot identify whether a removed maximum was a witness.',
                'A witness bound is a bound on grounded local recovery, not the predictive ceiling of RGB/global guessing.',
                'Existing-cutoff swaps diagnose score versus calibration effects and are not new selected candidates.'])
    write(out/'result.json',result)
    bind(Path(__file__))
    for path,digest in inputs.items(): assert sha(path)==digest,path
    write(out/'receipt.json',dict(status='PASS',backend='CPU saved scores and fixed ray geometry; TASK_NOT_GPU_SUITABLE',
        seconds=time.perf_counter()-start,inputs=inputs,outputs={'result.json':sha(out/'result.json')},**result['boundary']))
    print(json.dumps(dict(training=training,drop_local_swaps=swaps['DROP_CLOSE'],
        drop_changes=[r for r in changes if r['condition']=='DROP_CLOSE'],seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',type=Path,required=True)
    run(parser.parse_args().task.resolve())

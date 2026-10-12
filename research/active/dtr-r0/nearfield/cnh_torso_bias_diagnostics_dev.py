"""Read-only mechanism diagnostics of the frozen torso-bias Development run.

Does not fit, select, rerun alarms or alter any estimator arm. Output is exclusive.
"""
import argparse
import json
import time

import numpy as np

import cnh_torso_bias_dev as T


def describe(values):
    v = np.asarray(values, float)
    return None if not len(v) else dict(n=len(v), mean=float(v.mean()),
        p50=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))


def rms(values):
    v = np.asarray(values, float)
    return None if not len(v) else float(np.sqrt(np.mean(v*v)))


def streak_lengths(mask):
    mask = np.asarray(mask, bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])+1
    return ends-starts


def main(output_name='mechanism.json'):
    tick = time.monotonic()
    if output_name not in ('mechanism.json','mechanism_onset_repair.json'):
        raise ValueError('Diagnostic output must be an explicit known filename')
    out = T.OUT/output_name
    if out.exists():
        raise FileExistsError('Preserve existing mechanism diagnostic')
    result = json.loads((T.OUT/'result_angles.json').read_text(encoding='utf8'))
    plan = json.loads((T.OUT/'PLAN.json').read_text(encoding='utf8'))
    onsets = json.loads((T.OUT/'turn_onsets.json').read_text(encoding='utf8'))
    config = result['config']
    s = dict(np.load(T.OUT/'series.npz'))
    records = []; groups = {}; blocks = []; lag = {}; onset_detail = []
    gate_by_group = {}; gate_rate = {}
    config_frames = sum(x['pid'] in T.EVAL for x in plan['files'])*600
    for fi, rec in enumerate(plan['files']):
        if rec['pid'] not in T.EVAL:
            continue
        x = np.load(T.B.SRC/rec['name'])
        m = s['file_index'] == fi
        d = {k:v[m] for k,v in s.items()}
        v, up, bias = d['valid'], d['updated'], d['bias_deg']
        age = d['frame_index']; seen = np.cumsum(up)>0
        p = x[:, 0, :2]; torso = T.torso_yaw(x); ids = np.arange(60,600)
        chord = p[ids]-p[ids-60]
        half1 = T.B.yaw(p[ids]-p[ids-30]); half0 = T.B.yaw(p[ids-30]-p[ids-60])
        speed_ok = np.linalg.norm(chord,axis=1)>=config['min_speed']
        direction_ok = np.abs(T.B.wrap(half1-half0))/.5<=config['straight_rate_deg_s']
        body_ok = np.abs(T.B.wrap(torso[ids]-torso[ids-60]))<=config['torso_rate_deg_s']
        expected = speed_ok & direction_ok & body_ok
        assert np.array_equal(expected, up[ids])
        valid_ids = v[ids]
        residual = T.B.wrap(torso[ids-30]-T.B.yaw(chord))
        past_rate = np.abs(T.B.wrap(half1-half0))/.5
        changes = np.diff(np.r_[0.,bias])
        for group, groupid in [('straight',0),('slowturn',1),('onset',2),('otherturn',3)]:
            mask=valid_ids&(d['turn_group'][ids]==groupid)
            g=gate_by_group.setdefault(group,dict(frames=0,speed_pass=0,direction_pass=0,torso_pass=0,all_pass=0))
            for name,passed in [('frames',np.ones(len(ids),bool)),('speed_pass',speed_ok),
                ('direction_pass',direction_ok),('torso_pass',body_ok),('all_pass',expected)]:
                g[name]+=int((passed&mask).sum())
            gate_rate.setdefault(group,[]).extend(past_rate[mask].tolist())
        oracle = T.B.wrap(d['torso'][v]-d['oracle'][v])
        oracle_bias = float(oracle[0]) if len(oracle) else None
        first = float(np.flatnonzero(up)[0]/60) if up.any() else None
        gate_counts = dict(causal_frames=len(ids), causal_valid_frames=int(valid_ids.sum()),
            speed_pass=int(speed_ok.sum()), direction_pass=int(direction_ok.sum()), torso_pass=int(body_ok.sum()),
            all_pass=int(expected.sum()), speed_valid_pass=int((speed_ok&valid_ids).sum()),
            direction_valid_pass=int((direction_ok&valid_ids).sum()), torso_valid_pass=int((body_ok&valid_ids).sum()),
            all_valid_pass=int((expected&valid_ids).sum()),
            direction_fail_after_speed_valid=int((speed_ok&~direction_ok&valid_ids).sum()),
            torso_fail_after_speed_direction_valid=int((speed_ok&direction_ok&~body_ok&valid_ids).sum()))
        lengths = streak_lengths(up); blocks.extend(lengths.tolist())
        record = dict(clip=rec['name'],pid=rec['pid'],valid_frames=int(v.sum()),
            updated_valid_frames=int((up&v).sum()),never_updated_yet_valid_frames=int((v&~seen).sum()),
            no_update_clip=bool(not up.any()),first_update_seconds=first,
            total_update_seconds=float(up.sum()/60),final_bias_deg=float(bias[-1]),
            saturated_update_frames=int((up&np.isclose(np.abs(changes),config['max_bias_rate_deg_s']/60)).sum()),
            oracle_bias_deg=oracle_bias,valid_abs_bias=describe(np.abs(bias[v])),
            eligible_residual_deg=describe(residual[expected]),gate=gate_counts,
            persistent={k:T.persistent(d[k],v) for k in T.ARMS})
        records.append(record)
        masks = dict(all=v,cold1to2=v&(age<120),age2to4=v&(age>=120)&(age<240),
            age4plus=v&(age>=240),never_updated_yet=v&~seen,updated_yet=v&seen,
            straight=v&(d['turn_group']==0),slowturn=v&(d['turn_group']==1),
            onset=v&(d['turn_group']==2),otherturn=v&(d['turn_group']==3))
        for name, mask in masks.items():
            g = groups.setdefault(name,{k:[] for k in (*T.ARMS,'abs_bias','updated','never_updated_yet')})
            for k in T.ARMS:g[k].extend(d[k][mask].tolist())
            g['abs_bias'].extend(np.abs(bias[mask]).tolist())
            g['updated'].extend(up[mask].tolist());g['never_updated_yet'].extend((~seen[mask]).tolist())
        common = v&(age>=120)
        corrected = T.B.wrap(torso-bias)
        for seconds in (.5,1.,1.5,2.):
            back = np.maximum(age-int(seconds*60),0)
            disp = T.B.yaw(p-p[back])
            q=lag.setdefault(str(seconds),{k:[] for k in ('torso','corrected','e1')})
            head_e1=T.B.yaw(x[:,6,:2]-x[np.maximum(age-60,0),6,:2])
            for k, estimate in [('torso',torso),('corrected',corrected),('e1',head_e1)]:
                q[k].extend(T.B.wrap(estimate-disp)[common].tolist())
        for row in onsets:
            if row['clip']!=rec['name']:continue
            onset=row['frame'];before=slice(max(0,onset-120),onset)
            after=slice(onset,min(600,onset+60));postvalid=v[after]
            old=np.abs(d['torso'][after][postvalid]);new=np.abs(d['corrected'][after][postvalid])
            old_window = slice(onset-120,onset-60) if onset>=120 else None
            onset_detail.append(dict(**row,bias_at_onset_deg=float(bias[onset]),
                pre2s_abs_bias_max_deg=float(np.abs(bias[before]).max()),
                full_pre2s=bool(onset>=120),pre2s_frames=len(bias[before]),before_available_frames=min(onset,120),
                pre1to2s_update_frames=int(up[old_window].sum()) if old_window is not None else None,
                pre1to2s_abs_bias_change_deg=float(np.abs(changes[old_window]).sum()) if old_window is not None else None,
                pre1to2s_net_bias_change_deg=float(changes[old_window].sum()) if old_window is not None else None,
                post_valid_frames=len(old),post_delta_mae_deg=float((new-old).mean()) if len(old) else None,
                post_delta_rms_deg=rms(new)-rms(old) if len(old) else None))
    group_results={}
    for name,g in groups.items():
        n=len(g['torso'])
        group_results[name]=dict(frames=n,seconds=n/60,
            updated_frames=int(sum(g['updated'])),never_updated_yet_frames=int(sum(g['never_updated_yet'])),
            abs_bias_deg=describe(g['abs_bias']),
            error={k:dict(rms_deg=rms(g[k]),gt15_frames=int((np.abs(g[k])>15).sum())) for k in T.ARMS})
    usable=[r for r in onset_detail if r['post_valid_frames']]
    anyup=[r for r in usable if r['pre2s_update_fraction']>0]
    full=[r for r in usable if r['full_pre2s']]
    early=[r for r in usable if not r['full_pre2s']]
    def onset_summary(rows):
        return dict(onsets=len(rows),post_valid_frames=sum(r['post_valid_frames'] for r in rows),
            pre2s_updated_onsets=sum(r['pre2s_update_fraction']>0 for r in rows),
            abs_bias_at_onset=describe([abs(r['bias_at_onset_deg']) for r in rows]),
            pre2s_abs_bias_change=describe([r['pre2s_abs_bias_change_deg'] for r in rows]),
            post_delta_mae=describe([r['post_delta_mae_deg'] for r in rows]),
            improved_onsets=sum(r['post_delta_mae_deg'] < -1e-9 for r in rows),
            worsened_onsets=sum(r['post_delta_mae_deg'] > 1e-9 for r in rows))
    persistent={k:{key:sum(r['persistent'][k][key] for r in records)
        for key in ('runs_ge1s','frames_in_runs_ge1s')} for k in T.ARMS}
    gates={k:sum(r['gate'][k] for r in records) for k in records[0]['gate']}
    first=[r['first_update_seconds'] for r in records if r['first_update_seconds'] is not None]
    oracle=[abs(r['oracle_bias_deg']) for r in records if r['oracle_bias_deg'] is not None]
    validrecords=[r for r in records if r['oracle_bias_deg'] is not None]
    elapsed=time.monotonic()-tick
    prior_seconds = 0.
    if output_name=='mechanism_onset_repair.json':
        prior_seconds=json.loads((T.OUT/'mechanism.json').read_text(encoding='utf8'))['seconds']
    assert result['seconds']+prior_seconds+elapsed<=plan['budget']['estimator_cpu_wall_seconds']
    payload=dict(status='DIAGNOSTIC_ONLY_NO_NEW_ARMS',seconds=elapsed,
        original_estimator_seconds=result['seconds'],prior_diagnostic_seconds=prior_seconds,
        combined_estimator_diagnostic_seconds=result['seconds']+prior_seconds+elapsed,
        source_sha256=T.sha(__file__),series_sha256=T.sha(T.OUT/'series.npz'),config=config,
        evaluation_clips=len(records),evaluation_all_frames=config_frames,
        clips_with_no_updates=sum(r['no_update_clip'] for r in records),
        clips_with_no_updates_but_valid=sum(r['no_update_clip'] and r['valid_frames']>0 for r in records),
        clips_with_no_valid=sum(r['valid_frames']==0 for r in records),
        first_update_seconds=describe(first),eligible_streak_seconds=describe(np.asarray(blocks)/60),
        clip_total_update_seconds=describe([r['total_update_seconds'] for r in records]),
        final_abs_bias_deg=describe([abs(r['final_bias_deg']) for r in records]),
        oracle_abs_bias_deg=describe(oracle),
        saturated_update_frames=sum(r['saturated_update_frames'] for r in records),
        final_vs_oracle_bias_abs_difference=describe([abs(T.B.wrap(r['final_bias_deg']-r['oracle_bias_deg'])) for r in validrecords]),
        gates=gates,gate_by_evaluator_group={k:dict(**v,past_half_chord_rate_deg_s=describe(gate_rate[k])) for k,v in gate_by_group.items()},
        groups=group_results,persistent=persistent,
        onsets=dict(all=onset_summary(usable),pre2s_anyupdate=onset_summary(anyup),
            full_pre2s=onset_summary(full),early_clip_truncated=onset_summary(early),
            full_history_pre1to2s=dict(onsets=len(full),frames=len(full)*60,
                updated_onsets=sum(r['pre1to2s_update_frames']>0 for r in full),
                updated_frames=sum(r['pre1to2s_update_frames'] for r in full),
                abs_bias_change=describe([r['pre1to2s_abs_bias_change_deg'] for r in full]),
                net_bias_change_abs=describe([abs(r['pre1to2s_net_bias_change_deg']) for r in full]))),
        observed_metric_label_defect='Original pre2s fields mean available up-to-2s before onset. 56/424 early-clip onsets lack full2s history. Derived early/full2s and pre1to2s groups repair interpretation without altering frozen inputs.',
        corrected_minus_past_chord_common_support={sec:{k:dict(frames=len(v),rms_deg=rms(v),median_abs_deg=float(np.median(np.abs(v))))
            for k,v in q.items()} for sec,q in lag.items()},
        limits='Consumed Development; observational diagnostic. Future path target is same Xsens model. Group IDs nominal. No upstream-causality, sensor-independence or deployment claim.',
        clip_rows=records,onset_rows=onset_detail)
    with out.open('x',encoding='utf8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('clip_rows','onset_rows')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',choices=['mechanism.json','mechanism_onset_repair.json'],default='mechanism.json')
    main(parser.parse_args().output)

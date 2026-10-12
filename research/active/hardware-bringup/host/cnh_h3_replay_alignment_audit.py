"""Read-only frozen H3 replay alignment diagnostic; never calibrates real alerts.

Uses existing 01/04 capture, unchanged empirical features/models/thresholds.
The all-zero input probe is a network-reference diagnostic, not a real frame or
an alternative algorithm. Background self-replay is descriptive (not held out).
"""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import cnh_h3_live_demo as live
from capture import validate_frame
from cnh_track_a_readout import query_weights, BOXES, START, WIDTH, END


def quantiles(x):
    return {str(q): float(np.percentile(x, q)) for q in (0, 1, 5, 50, 95, 99, 100)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--models', type=Path, nargs=3, required=True)
    p.add_argument('--thresholds', type=Path, required=True)
    p.add_argument('--saved-replay', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=True)
    bg = a.capture/'01-background-attempt1/tof/frames.jsonl'
    mean, variance, noise = live.noise_model(bg)
    tq = np.eye(4)
    angle = np.deg2rad(-10)
    tq[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    sup = (query_weights(tq) >= .75).reshape(6, 8, 8, 16)
    th = json.loads(a.thresholds.read_text())['A2_thresholds']
    thresholds = np.array([th['HEAD' if q % 2 == 0 else 'BODY'] for q in range(6)])
    report = dict(schema='cnh.replay-alignment-audit.v1',
        scope='Existing real-input replay diagnostic; no new data, training, thresholds, calibration, event truth or transfer-validity claim',
        empirical_contract='r=h-scene_mean; z1=r/sqrt(empirical_var1); z4=sum(last<=4 r)/sqrt(empirical_var_k); ambient not used',
        frozen_contract='r=h-bias; v=16*ambient+max(bias,0); z=r/sqrt(v), transported accumulated window; feature float16 then float32 then squash',
        noise=noise, thresholds=thresholds.tolist(), boxes=BOXES.tolist(),
        query_support=dict(cells_per_query=sup.sum((1, 2, 3)).tolist(),
            cells_per_bin=sup.sum((1, 2)).tolist(),
            cell_geometry=dict(start_m=START, width_m=WIDTH, end_m=END)),
        sources={str(path):live.sha(path) for path in [Path(__file__),Path(live.__file__),bg,a.thresholds,*a.models,
            live.NEAR/'cnh_learned_features.py',live.NEAR/'cnh_learned_readout.py',
            live.NEAR/'cnh_track_a_readout.py',live.NEAR/'cnh_route_sensor.py',
            Path(live.__file__).parent/'capture.py']},
        segments={})
    all_arrays = dict(background_mean=mean, background_variance=variance, support=sup)
    for name in ('01-background-attempt1', '04-restored-attempt4'):
        path = a.capture/name/'tof/frames.jsonl'
        report['sources'][str(path)] = live.sha(path)
        records = list(live.causal_sample(live.rows(path)))
        t0 = records[0]['sample_tick_ns']
        records = [r for r in records if (r['sample_tick_ns']-t0)/1e9 < 30]
        engine = live.Engine(a.models, mean, variance, tq)
        outputs, z1s, z4s, hs, amb, distances, statuses, scalers = [], [], [], [], [], [], [], []
        raw_extrema = [2**31, -2**31]
        raw_at_integer_limits = 0
        for row in records:
            derived = validate_frame(row['sensor'])
            h = live.hist(row)
            direct = np.ldexp(np.asarray(row['sensor']['hist_raw'], float), -np.asarray(row['sensor']['hist_scaler'], int)).reshape(8, 8, 16)
            raw = np.asarray(row['sensor']['hist_raw'],np.int64)
            raw_extrema = [min(raw_extrema[0],int(raw.min())),max(raw_extrema[1],int(raw.max()))]
            raw_at_integer_limits += int(((raw == -2**31) | (raw == 2**31-1)).sum())
            assert np.array_equal(h, direct), 'decode mismatch'
            if 'derived' in row and 'hist_normalized' in row['derived']:
                assert np.array_equal(h, np.asarray(row['derived']['hist_normalized']).reshape(8, 8, 16))
            output, z4 = engine.step(row)
            output['alarms'] = (np.array(output['A2']) >= thresholds).tolist() if output['history'] >= 4 else [False]*6
            outputs.append(output)
            z1s.append(engine.history[-1]/np.sqrt(variance[0])); z4s.append(z4)
            hs.append(h); amb.append(np.asarray(derived['ambient_normalized']).reshape(8, 8))
            distances.append(np.asarray(row['sensor']['distance_mm']).reshape(8, 8))
            statuses.append(np.asarray(derived['range_valid']).reshape(8, 8))
            scalers.append(np.asarray(row['sensor']['hist_scaler']))
        z1s, z4s, hs, amb = map(np.asarray, (z1s,z4s,hs,amb))
        valid = np.array([o['history'] >= 4 for o in outputs])
        nn, a2 = (np.asarray([o[k] for o in outputs]) for k in ('NN','A2'))
        nz = z4s[valid]
        med = np.median(nz, axis=0)
        top = np.argsort(med.ravel())[-12:][::-1]
        s = dict(raw_frames=sum(1 for _ in live.rows(path)), sampled_frames=len(records),
            eligible_frames=int(valid.sum()), resets=sum(o['reset'] for o in outputs),
            alarm_counts=(a2[valid] >= thresholds).sum(0).tolist(),
            unsmoothed_nn_alarm_counts=(nn[valid] >= thresholds).sum(0).tolist(),
            logits_nn=[quantiles(nn[valid,q]) for q in range(6)],
            logits_a2=[quantiles(a2[valid,q]) for q in range(6)],
            h=quantiles(hs), ambient=quantiles(amb), histogram_scaler=quantiles(scalers),
            histogram_raw_min_max=raw_extrema, histogram_raw_integer_limit_cells=raw_at_integer_limits,
            z1=quantiles(z1s[valid]), z4=quantiles(nz),
            valid_range_fraction=float(np.mean(statuses)),
            valid_range_mm=quantiles(np.asarray(distances)[np.asarray(statuses)]),
            histogram_bin_median_mean=np.median(hs.mean((1,2)),axis=0).tolist(),
            per_query_z4=[dict(support_cells=int(sup[q].sum()),quantiles=quantiles(nz[:,sup[q]]),
                median_max=float(np.median(nz[:,sup[q]].max(-1)))) for q in range(6)],
            top_median_z4_cells=[dict(row=int(i//128),col=int(i//16%8),bin=int(i%16),
                z4_median=float(med.ravel()[i]), z1_median=float(np.median(z1s[valid].reshape(-1,1024)[:,i])),
                h_mean=float(hs.reshape(-1,1024)[:,i].mean()),background_mean=float(mean.ravel()[i]),
                background_std1=float(np.sqrt(variance[0].ravel()[i])),
                background_std_k=np.sqrt(variance.reshape(4,1024)[:,i]).tolist(),
                support_queries=np.flatnonzero(sup.reshape(6,1024)[:,i]).tolist()) for i in top])
        if name.startswith('04'):
            saved = list(live.rows(a.saved_replay/'inference.jsonl'))
            assert len(saved) == len(outputs), 'saved replay length changed'
            assert all((x['seq'], x['history'], x['reset']) == (y['seq'], y['history'], y['reset']) for x,y in zip(saved,outputs))
            s['saved_replay_max_abs_diff'] = {k:float(np.max(np.abs(np.asarray([o[k] for o in outputs])-np.asarray([o[k] for o in saved])))) for k in ('NN','A2')}
            assert all(saved[i]['alarms']==outputs[i]['alarms'] for i in range(len(saved))), 'saved alarms changed'
            s['saved_replay_alarm_exact'] = True
            report['sources'][str(a.saved_replay/'inference.jsonl')] = live.sha(a.saved_replay/'inference.jsonl')
        report['segments'][name] = s
        prefix = name[:2]
        all_arrays.update({f'{prefix}_{k}':v for k,v in dict(z1=z1s,z4=z4s,h=hs,ambient=amb,eligible=valid,nn=nn,a2=a2,
            seq=np.array([o['seq'] for o in outputs]),distance=np.asarray(distances),range_valid=np.asarray(statuses)).items()})
        (a.out/f'{prefix}_inference.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in outputs),encoding='utf-8')
    report['localized_background_shift'] = {prefix:dict(
        ambient_zone_quantiles=quantiles(all_arrays[f'{prefix}_ambient'][:,4,2]),
        range_zone_mm_quantiles=quantiles(all_arrays[f'{prefix}_distance'][:,4,2]),
        histogram_bin_1_quantiles=quantiles(all_arrays[f'{prefix}_h'][:,4,2,1]),
        histogram_bin_2_quantiles=quantiles(all_arrays[f'{prefix}_h'][:,4,2,2])) for prefix in ('01','04')}
    # Frozen-network input zero point: same models/support, no sensor interpretation.
    torch = engine.torch
    with torch.inference_mode():
        zero = torch.zeros((1,2,8,8,16),device=engine.device)
        logits = torch.stack([m(zero,engine.support) for m in engine.models]).mean(0)[0].cpu().numpy()
    report['zero_input_probe'] = dict(logits=logits.tolist(),above_frozen_threshold=(logits>=thresholds).tolist(),
        scope='All-zero z reference only; neither real physical absence nor an alternative algorithm')
    # One localized post-observation counterfactual to identify the strong input
    # residual; no real operating point or background correction is selected.
    counter_x = np.stack((engine.squash(all_arrays['04_z4']),engine.squash(all_arrays['04_z1'])),axis=1).astype(np.float32)
    counter_x[:,:,4,2,1:3] = 0
    with torch.inference_mode():
        counter_nn = torch.stack([m(torch.as_tensor(counter_x,device=engine.device),
            engine.support.expand(len(counter_x),-1,-1,-1,-1)) for m in engine.models]).mean(0).cpu().numpy()
    history = deque(maxlen=5)
    counter_a2 = []
    saved = list(live.rows(a.out/'04_inference.jsonl'))
    for row,logit in zip(saved,counter_nn):
        if row['reset']: history.clear()
        history.append(logit); counter_a2.append(live.smooth_logits(history))
    counter_a2 = np.asarray(counter_a2)
    eligible = all_arrays['04_eligible']
    report['localized_counterfactual'] = dict(zeroed_cells_zero_based=[[4,2,1],[4,2,2]],
        eligible_frames=int(eligible.sum()),alarm_counts=(counter_a2[eligible]>=thresholds).sum(0).tolist(),
        unsmoothed_nn_alarm_counts=(counter_nn[eligible]>=thresholds).sum(0).tolist(),
        nn_median=np.median(counter_nn[eligible],axis=0).tolist(),
        scope='Post-observation feature ablation of two selected residual cells; necessary-input diagnosis only, not a proposed live correction, calibration, truth evaluation or gain')
    all_arrays['04_counterfactual_nn'] = counter_nn
    all_arrays['04_counterfactual_a2'] = counter_a2
    report['unit_scale_check'] = dict(global_positive_gain_cancels_in_empirical_z=True,
        reason='h*c, mean*c, var*c*c gives identical z before numerical floors; factor-of-two manual/example discrepancy alone cannot explain unchanged empirical replay')
    report['backend']=str(engine.device)
    report['wall_seconds']=time.perf_counter()-started
    np.savez_compressed(a.out/'arrays.npz',**all_arrays)
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(out=str(a.out),wall_seconds=report['wall_seconds'],
        segments={k:{m:v[m] for m in ('sampled_frames','eligible_frames','alarm_counts','unsmoothed_nn_alarm_counts','z4')} for k,v in report['segments'].items()},
        zero_input_probe=report['zero_input_probe']),indent=2))


if __name__ == '__main__':
    main()

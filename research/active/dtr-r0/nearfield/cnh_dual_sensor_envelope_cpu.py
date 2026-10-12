"""CPU-only alternating-rate calibration and independent prior-run fact audit.

Reads immutable pilot2/dual payloads, rebuilds alternating smoothing from raw
inferred sampled-observation logits, then fits exactly one clear-cost cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PREVIOUS = ROOT / 'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
PILOT = ROOT / 'artifacts.local/work/cnh-readout-pilot2-20261005'
OUT = ROOT / 'artifacts.local/work/cnh-dual-sensor-envelope-20261005'
FRAMES = np.arange(3, 16)
THRESHOLD = .8557642486787612
BOOT_SEED = 2026100525
BOOT_N = 2000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def rebuild(raw, sensor):
    """Independent scalar-history reconstruction of acquired-only smoothing."""
    acquired = FRAMES % 2 == sensor
    np.testing.assert_array_equal(np.isfinite(raw).all(axis=(0, 1, 3)), acquired)
    np.testing.assert_array_equal(np.isnan(raw).all(axis=(0, 1, 3)), ~acquired)
    result = np.full_like(raw, -np.inf, dtype=float)
    history = []
    last = None
    for t in range(13):
        if acquired[t]:
            history.append(t)
            ids = history[-5:]
            weights = 2. ** np.arange(len(ids))
            current = sum(raw[..., tick, :] * weight for tick, weight in zip(ids, weights)) / weights.sum()
            last = t
        if last is not None and (t - last) * .2 <= .4 + 1e-12:
            result[..., t, :] = current
    return result


def cutoff(values, budget):
    """Enumerate all attainable >= sets, retaining the lowest feasible cutoff."""
    unique = np.unique(values)
    candidates = np.r_[np.nextafter(unique, np.inf), unique.min()]
    feasible = [float(t) for t in candidates if np.count_nonzero(values >= t) <= budget]
    threshold = min(feasible)
    stops = int(np.count_nonzero(values >= threshold))
    excluded = float(values[values < threshold].max()) if stops < len(values) else None
    included = float(values[values >= threshold].min()) if stops else None
    tied = int(np.count_nonzero(values == excluded)) if excluded is not None else 0
    assert stops <= budget
    assert excluded is None or stops + tied > budget
    assert excluded is None or threshold == float(np.nextafter(excluded, np.inf))
    return dict(threshold=threshold, operator='>=', allowed_stops=budget, actual_stops=stops,
                n=int(len(values)), unattained_stops=budget - stops,
                next_excluded_score=excluded, next_excluded_tie_size=tied,
                lowest_included_score=included,
                rule='Largest complete tied-score alarm set <= single calibration cost; lowest representable >= cutoff')


def event(score, threshold, ranges, covered):
    alarm = score >= threshold
    stopped = alarm.any(-1)
    first = alarm.argmax(-1)
    distance = np.take_along_axis(ranges[:, None, None, :], first[..., None], -1)[..., 0]
    return dict(alarm=alarm, stopped=stopped, first_index=first,
                first_range=distance, timely=stopped & (distance >= .9) & covered[..., None])


def interval(values):
    values = values[np.isfinite(values)]
    return np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]


def metric(flags, baseline, den, keep, draws):
    flags = flags & den
    baseline = baseline & den
    axes = tuple(range(1, flags.ndim))
    counts = flags.sum(axes) * keep
    base_counts = baseline.sum(axes) * keep
    denominators = den.sum(axes) * keep
    n = int(denominators.sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        rates = draws @ counts / (draws @ denominators)
        deltas = draws @ (counts - base_counts) / (draws @ denominators)
    return dict(stops=int(counts.sum()), n=n, rate=float(counts.sum()/n) if n else None,
                ci95=interval(rates), valid_bootstrap_draws=int(np.isfinite(rates).sum()),
                paired_minus_single=dict(delta_stops=int(counts.sum()-base_counts.sum()),
                    delta_rate=float((counts.sum()-base_counts.sum())/n) if n else None,
                    ci95=interval(deltas), rescues=int((flags & ~baseline)[keep].sum()),
                    losses=int((baseline & ~flags)[keep].sum())))


def audit(previous, ledger, units, scenes, full):
    evaluation = np.isin(units, previous['split_units']['evaluation'])
    fov_out = np.array([not s['fov_in'] for s in scenes])
    mode = np.array([s['mode'] for s in scenes])
    out = fov_out & evaluation
    assert np.all(mode[fov_out] == 0)
    physical_yaws, channel_yaws, target_sides = [], [], []
    for i, unit in enumerate(units):
        with np.load(PILOT/'observations/evaluation'/f'unit{unit}.npz', allow_pickle=False) as obs:
            head = obs['sensor'][:, :3, :3]
            physical_yaws.append(np.rad2deg(np.arctan2(head[:, 0, 2], head[:, 2, 2])))
        with np.load(PREVIOUS/'templates'/f'unit{unit}.npz', allow_pickle=False) as template:
            matrix = template['physical'][..., :3, :3]
            channel_yaws.append(np.rad2deg(np.arctan2(matrix[..., 0, 2], matrix[..., 2, 2])))
        truth = read(PILOT/'truth/evaluation'/f'unit{unit}.json')
        target_sides.append(int(np.sign(sum([truth['boxes'][0][0]['lo'][0], truth['boxes'][0][0]['hi'][0]]))))
    physical_yaws, channel_yaws = np.asarray(physical_yaws), np.asarray(channel_yaws)
    source = {}
    for arm in ('dual_frozen', 'dual_matched'):
        stopped = ledger[arm+'_stopped'][:, :2]
        timely = ledger[arm+'_timely'][:, :2]
        first = ledger[arm+'_first_index'][:, :2]
        counts = dict(L=0, R=0, tie=0, timely_L=0, timely_R=0, timely_tie=0)
        for i in np.flatnonzero(out):
            q = scenes[i]['group']
            for d in (0, 1):
                for k in range(4):
                    if not stopped[i,d,k]:
                        continue
                    left, right = full[i,:,d,k,first[i,d,k],q]
                    name = 'L' if left > right else 'R' if right > left else 'tie'
                    counts[name] += 1
                    counts['timely_'+name] += int(timely[i,d,k])
        source[arm] = counts
    cells = previous['sequence']
    return dict(status='PASS',
        FOV_OUT=dict(all_scenes=int(fov_out.sum()), evaluation_scenes=int(out.sum()),
            all_mode0=bool(np.all(mode[fov_out]==0)), target_sides=np.asarray(target_sides)[fov_out].tolist(),
            physical_head_yaw_min_deg=float(physical_yaws[fov_out].min()),
            physical_head_yaw_max_deg=float(physical_yaws[fov_out].max()),
            left_sensor_yaw_max_abs_deg=float(np.abs(channel_yaws[fov_out,0]).max()),
            evaluation_shallow_first_reports=source,
            interpretation='Constant +15deg head and -15deg left splay return left optical yaw to0. 48 left first reports include47 timely; this is alignment construction, not arbitrary-head tolerance.'),
        mode1=dict(physical_head_yaw_range_deg=[float(physical_yaws[mode==1].min()),float(physical_yaws[mode==1].max())],
            dual_sensor_yaw_range_deg=[float(channel_yaws[mode==1].min()),float(channel_yaws[mode==1].max())],
            evaluation={arm:cells['evaluation']['mode1'][arm]['branches']['shallow']['timely'] for arm in ('single','dual_frozen','dual_matched')},
            all={arm:cells['all']['mode1'][arm]['branches']['shallow']['timely'] for arm in ('single','dual_frozen','dual_matched')},
            correction='Physical head sweeps +/-20deg; +/-35deg denotes outer optical axis. 92/96->96/96 is evaluation matched-threshold, frozen-threshold95/96.'),
        clear_all_frozen={arm:cells['all']['all'][arm]['joint_clear'] for arm in ('single','dual_frozen')},
        deep_evaluation={arm:cells['evaluation']['all'][arm]['branches']['deep']['timely'] for arm in ('single','dual_frozen','dual_matched')},
        limitations=['Pilot clear consists of outer-target, ground and backwall; no natural roadside clutter cost.',
            'Adding side sensors replaces center with two side axes. This is not center+side superset OR, so fewer clear stops are possible without contradiction.',
            'Low alternating frozen-threshold clear counts motivate calibration but do not alone identify distribution shift mechanism.'])


def run(out):
    started = time.monotonic()
    out = Path(out)
    parent_path = out/'PLAN.json'
    parent_sha = sha(parent_path)
    parent = read(parent_path)
    assert time.time() <= parent['deadline_unix'], 'Task deadline elapsed before CPU start'
    destination = out/'alternating'
    if (destination/'result.json').exists():
        raise FileExistsError('Completed CPU payload is immutable')
    previous = read(PREVIOUS/'result.json')
    previous_plan = read(PREVIOUS/'PLAN.json')
    assert previous['provenance']['plan_sha256'] == sha(PREVIOUS/'PLAN.json')
    units = np.asarray(previous_plan['units'])
    scenes = previous['scenes']
    full, alternate, score_hashes = [], [], {}
    for unit in units:
        path = PREVIOUS/'scores'/f'unit{unit}.npz'
        score_hashes[str(path)] = sha(path)
        assert score_hashes[str(path)] == previous['provenance']['score_sha256'][str(path)]
        with np.load(path, allow_pickle=False) as scores:
            raw = scores['raw_alternating'].copy()
        alternate.append([rebuild(raw[s], s) for s in range(2)])
    alternate = np.asarray(alternate)
    with np.load(PREVIOUS/'evaluation/ledger.npz', allow_pickle=False) as store:
        ledger = {k:store[k].copy() for k in store.files}
    np.testing.assert_allclose(alternate, ledger['alternating_sensor_smoothed'], atol=1e-12, rtol=0)
    both = dict(single=ledger['single_both_query_scores'], alternating_frozen=alternate.max(1), alternating_matched=alternate.max(1))
    full = ledger['full_sensor_smoothed']
    clear = ledger['joint_clear_den']
    covered = ledger['covered']
    ranges = np.asarray([s['front_range_m'] for s in scenes])
    masks = dict(all=np.ones(len(units),bool), calibration=np.isin(units,previous_plan['calibration_units']),
                 evaluation=np.isin(units,previous_plan['evaluation_units']))
    peaks = {arm:values[:,[5,6]].max(axis=(-2,-1)) for arm,values in both.items()}
    cal_den = clear & masks['calibration'][:,None,None]
    budget = int((peaks['single'][cal_den]>=THRESHOLD).sum())
    assert budget == 17
    match = cutoff(peaks['alternating_matched'][cal_den], budget)
    thresholds = dict(single=THRESHOLD, alternating_frozen=THRESHOLD, alternating_matched=match['threshold'])
    events = {arm:event(np.asarray([value[i,...,s['group']] for i,s in enumerate(scenes)]),thresholds[arm],ranges,covered)
              for arm,value in both.items()}
    for field in ('alarm','stopped','timely','first_index'):
        np.testing.assert_array_equal(events['single'][field],ledger['single_'+field])
        np.testing.assert_array_equal(events['alternating_frozen'][field],ledger['alternating_frozen_'+field])
    groups = dict(all=np.ones(len(units),bool),FOV_OUT=np.asarray([not s['fov_in'] for s in scenes]),
                  FOV_IN=np.asarray([s['fov_in'] for s in scenes]))
    groups.update({f'mode{m}':np.asarray([s['mode']==m for s in scenes]) for m in range(3)})
    sequence = {}
    for split_index,(split,mask) in enumerate(masks.items()):
        ids = np.flatnonzero(mask)
        rng = np.random.default_rng(BOOT_SEED+split_index)
        draws = np.asarray([np.bincount(rng.choice(ids,len(ids),replace=True),minlength=len(units)) for _ in range(BOOT_N)])
        sequence[split] = {}
        for group,group_mask in groups.items():
            keep = mask & group_mask
            sequence[split][group] = {}
            for arm,ev in events.items():
                cell = dict(scenes=int(keep.sum()),branches={})
                for name,ds in (('shallow',[0,1]),('deep',[2])):
                    den = np.broadcast_to(covered[:,ds,None],(len(units),len(ds),4))
                    cell['branches'][name] = dict(timely=metric(ev['timely'][:,ds],events['single']['timely'][:,ds],den,keep,draws))
                joint = peaks[arm]>=thresholds[arm]
                met = metric(joint,peaks['single']>=THRESHOLD,clear,keep,draws)
                minutes = met['n']*13*.2/60
                cell['joint_clear'] = dict(**met,proxy_minutes=minutes,
                    first_stops_per_proxy_minute=met['stops']/minutes if minutes else None)
                sequence[split][group][arm] = cell
    # Direct loops independently cross-check all reported integer numerators/denominators.
    comparisons = 0
    for split,mask in masks.items():
        for group,gm in groups.items():
            ids = [i for i in range(len(units)) if mask[i] and gm[i]]
            for arm in both:
                for name,ds in (('shallow',[0,1]),('deep',[2])):
                    n = sum(int(covered[i,d]) for i in ids for d in ds for k in range(4))
                    count = sum(int(events[arm]['timely'][i,d,k] and covered[i,d]) for i in ids for d in ds for k in range(4))
                    met = sequence[split][group][arm]['branches'][name]['timely']
                    assert (met['stops'],met['n']) == (count,n)
                    comparisons += 2
                count = sum(int(clear[i,d,k] and peaks[arm][i,d,k]>=thresholds[arm]) for i in ids for d in range(2) for k in range(4))
                n = sum(int(clear[i,d,k]) for i in ids for d in range(2) for k in range(4))
                assert (sequence[split][group][arm]['joint_clear']['stops'],sequence[split][group][arm]['joint_clear']['n']) == (count,n)
                comparisons += 2
    assert sha(parent_path) == parent_sha
    audit_result = audit(previous,ledger,units,scenes,full)
    result = dict(status='COMPLETE',task='CPU 2.5Hz alternating clear-cost recalibration on consumed synthetic Development',
        thresholds=thresholds,matching=match,sequence=sequence,split_units=previous['split_units'],
        bootstrap=dict(n=BOOT_N,seed=BOOT_SEED,unit='Whole paired scenes, shared resamples and frozen threshold'),
        interpretation='5Hz-trained M3 fed2.5Hz observations; original <=1.4s voxel history; five acquired-score smoothing spans1.6s endpoint-to-endpoint (versus0.8s at5Hz), conventionally2s versus1s five-sample window. Not hardware throughput.',
        provenance=dict(parent_plan_sha256=parent_sha,parent_deadline_unix=parent['deadline_unix'],
            previous_plan_sha256=sha(PREVIOUS/'PLAN.json'),previous_result_sha256=sha(PREVIOUS/'result.json'),
            previous_ledger_sha256=sha(PREVIOUS/'evaluation/ledger.npz'),score_sha256=score_hashes,source_sha256=sha(__file__)),
        checks=dict(raw_rebuilt_to_previous_ledger_atol=1e-12,integer_comparisons=comparisons,
            cutoff_boundary_verified=True,calibration_single_cost=budget),seconds=time.monotonic()-started)
    write_new(destination/'audit.json',audit_result)
    write_new(destination/'result.json',result)
    lines=['# 交替2.5Hz校准重评','',f"阈值 {match['threshold']:.17g}；校准联合clear {match['actual_stops']}/{match['n']}，single预算{budget}。",'',
           '| 评价指标 | single | 交替冻结 | 交替匹配 |','|---|---:|---:|---:|']
    for group in ('all','FOV_OUT','FOV_IN'):
        for branch in ('shallow','deep'):
            values=[sequence['evaluation'][group][a]['branches'][branch]['timely'] for a in thresholds]
            lines.append('| '+group+'/'+branch+' | '+' | '.join(f"{v['stops']}/{v['n']}" for v in values)+' |')
    vals=[sequence['evaluation']['all'][a]['joint_clear'] for a in thresholds]
    lines.append('| 联合clear | '+' | '.join(f"{v['stops']}/{v['n']}" for v in vals)+' |')
    lines.extend(['',result['interpretation'],'','逐值重建旧平滑ledger通过；'+str(comparisons)+'项计数独立循环核对通过。'])
    with (destination/'REPORT.md').open('x',encoding='utf8') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(threshold=match, evaluation={g:sequence['evaluation'][g] for g in ('all','FOV_OUT','FOV_IN')},seconds=result['seconds']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)

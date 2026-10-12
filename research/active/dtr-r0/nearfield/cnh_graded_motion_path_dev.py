"""Matched fixed shallow readouts of signed native candidate history.

Public motion-aligned paths versus repeated current native indices. Only labels,
negative calibration costs and evaluation join authored truth, after extraction.
"""
from pathlib import Path
import pickle
import time
import numpy as np
from threadpoolctl import threadpool_limits
import cnh_graded_weak_mass_dev as M

C, J, H, G, E, I = M.C, M.J, M.H, M.G, M.E, M.I
OUT = C.ROOT/'artifacts.local/work/cnh-graded-motion-path-dev-20261010'
ARMS = ('static_path', 'aligned_path')


def read_banks(data):
    current, banks, parents, controls = {}, {}, {}, {}
    for split, d in data.items():
        with np.load(J.OUT/'features'/f'{split}_features.npz') as a:
            np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
            current[split], current_names = J.descriptors(a, 'current')
        with np.load(OUT/'features'/f'{split}_features.npz') as a:
            np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
            names = a['names'].tolist()
            banks[split] = {}
            for arm, prefix in zip(ARMS, ('static', 'aligned')):
                x, v = a[prefix].astype(float), a[prefix+'_valid']
                assert x.shape == (384, 4, 13, 2, len(names))
                banks[split][arm] = np.concatenate((np.where(v, x, np.nan), (~v).astype(float)), -1)
        with np.load(H.OUT/f'{split}_grades.npz') as a:
            parents[split] = dict(zip(a['keys'].tolist(), a['grades']))
        with np.load(M.OUT/f'{split}_scores.npz') as a:
            controls[split] = dict(zip(a['keys'].tolist(), a['scores']))
    return current, banks, parents, controls, current_names, names


def run():
    started = time.monotonic()
    if (OUT/'FIT_PLAN.json').exists():
        raise FileExistsError('Preserve fitted motion-path attempt')
    paths = [OUT/'TASK.json', OUT/'features/schema.json', Path(__file__), Path(M.__file__),
             Path(J.__file__), Path(I.__file__), C.PARENT/'thresholds.json', M.OUT/'calibrations.json']
    for split in ('cal', 'validation'):
        paths += [OUT/'features'/f'{split}_features.npz', J.OUT/'features'/f'{split}_features.npz',
                  H.OUT/f'{split}_grades.npz', M.OUT/f'{split}_scores.npz']
    C.save(OUT/'FIT_PLAN.json', dict(task_sha256=C.sha(OUT/'TASK.json'), params=I.PARAMS,
        dimensions=dict(static_path=303, aligned_path=303),
        feature_path=(OUT/'features').relative_to(C.ROOT).as_posix(),
        inputs_sha256={p.relative_to(C.ROOT).as_posix(): C.sha(p) for p in paths},
        source_sha256=C.sha(Path(__file__))))
    try:
        data = C.load()
        fit_mask, partition = C.split_cal(data['cal']['rows'])
        C.save(OUT/'cal_partition.json', partition)
        th = C.read(C.PARENT/'thresholds.json')
        current, banks, parents, controls, current_names, names = read_banks(data)
        C.save(OUT/'feature_names.json', dict(
            current=['ordinary_smooth_margin', 'ordinary_raw_slope5', 'ordinary_raw_detrended_fluctuation5']+current_names,
            extra=names+[n+'_missing' for n in names]))
        current_cuts = C.read(M.OUT/'calibrations.json')
        metrics, cohorts, cuts, models, summary = {}, {}, {}, {}, []
        grades_saved, scores_saved = {s: [] for s in data}, {s: [] for s in data}
        for si, seed in enumerate(G.SEEDS):
            base, head50, tensors, control_grade = {}, {}, {}, {}
            ct = current_cuts[f'{seed}/current_control']
            ct = -np.inf if ct['nonbinding'] else ct['theta']
            for split, d in data.items():
                base[split] = H.baseline(d, si, th[str(seed)])
                head50[split] = parents[split][f'{seed}/head50']
                raw = E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz', 'ordinary', seed, split, 'ideal')
                s = C.build_score_features(raw, d['candidates'][0, si], th[str(seed)]['single'])
                control = np.concatenate((s, current[split]), -1)
                tensors[split] = {a: np.concatenate((control, banks[split][a]), -1) for a in ARMS}
                control_grade[split] = np.where((base[split] == 0)&(controls[split][f'{seed}/current_control'] >= ct), 1, base[split]).astype(np.int8)
            arm_grades = {}
            for arm in ARMS:
                score = {s: np.empty(b.shape, float) for s, b in base.items()}
                for q, height in enumerate(E.HEIGHTS):
                    if time.monotonic()-started >= 240:
                        raise TimeoutError('Fit/evaluation240 CPU-command-wall seconds cap')
                    eligible = base['cal'][fit_mask, ..., q] == 0
                    fit_values = tensors['cal'][arm][fit_mask, ..., q, :][eligible]
                    keep = np.isfinite(fit_values).any(0)
                    model, record = J.fit_height(tensors['cal'][arm][..., keep], data['cal']['category'], base['cal'], fit_mask, q)
                    path = OUT/f'models/{seed}_{arm}_{height}.pickle'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(pickle.dumps(model, protocol=5))
                    models[f'{seed}/{arm}/{height}'] = dict(**record, path=path.relative_to(OUT).as_posix(), sha256=C.sha(path),
                        dimensions=int(keep.sum()), input_dimensions=len(keep), retained_feature_indices=np.flatnonzero(keep).tolist(),
                        all_missing_training_feature_indices=np.flatnonzero(~keep).tolist())
                    for split in data:
                        x = tensors[split][arm][..., q, :][..., keep]
                        score[split][..., q] = model.decision_function(x.reshape(-1, int(keep.sum()))).reshape(x.shape[:-1])
                cut = M.calibrate(score['cal'], base['cal'], head50['cal'], data['cal']['category'], ~fit_mask)
                cuts[f'{seed}/{arm}'] = cut
                theta = -np.inf if cut['nonbinding'] else cut['theta']
                arm_grades[arm] = {}
                for split, d in data.items():
                    grade = np.where((base[split] == 0)&np.isfinite(score[split])&(score[split] >= theta), 1, base[split]).astype(np.int8)
                    np.testing.assert_array_equal(grade == 2, base[split] == 2)
                    assert np.all(grade >= base[split])
                    arm_grades[arm][split] = grade
                    refs = dict(prior_light=head50[split] > 0, ordinary_OR=base[split] == 2,
                        M3=d['m3'] >= E.M3_THETA, old_fusion=E.old_fusion(d['m3'], d['local']))
                    report = G.describe(grade, d, refs)
                    report.update(cost=M.costs(grade > 0, d['category']), head50_cost=M.costs(head50[split] > 0, d['category']),
                        paired_vs_head50=G.paired_timing(head50[split] > 0, grade > 0, d['category']),
                        paired_vs_current_control=G.paired_timing(control_grade[split] > 0, grade > 0, d['category']),
                        physical_vs_head50=H.B.physical_pair(head50[split] > 0, grade > 0, d['category']))
                    key = f'{split}/{seed}/{arm}'
                    metrics[key] = report
                    cohorts[key] = dict(outcomes=M.outcomes(grade, d['category'], d['rows']),
                        paired_vs_head50=M.paired_cohorts(head50[split], grade, d['category'], d['rows']),
                        paired_vs_current_control=M.paired_cohorts(control_grade[split], grade, d['category'], d['rows']))
                    summary.append(dict(split=split, seed=seed, arm=arm, HEAD=report['any']['counts'][0], BODY=report['any']['counts'][1],
                        HEAD_rescue=report['paired_vs_head50'][0]['rescue'], HEAD_loss=report['paired_vs_head50'][0]['loss'],
                        BODY_rescue=report['paired_vs_head50'][1]['rescue'], BODY_loss=report['paired_vs_head50'][1]['loss'], **report['cost']))
                    grades_saved[split].append((f'{seed}/{arm}', grade))
                    scores_saved[split].append((f'{seed}/{arm}', score[split]))
            for split, d in data.items():
                old, new = arm_grades['static_path'][split], arm_grades['aligned_path'][split]
                key = f'{split}/{seed}/aligned_path'
                metrics[key]['paired_vs_static_path'] = G.paired_timing(old > 0, new > 0, d['category'])
                cohorts[key]['paired_vs_static_path'] = M.paired_cohorts(old, new, d['category'], d['rows'])
            print('seed', seed, 'COMPLETE', round(time.monotonic()-started, 3), flush=True)
        for split, d in data.items():
            np.savez_compressed(OUT/f'{split}_grades.npz', keys=np.array([k for k, v in grades_saved[split]]), grades=np.array([v for k, v in grades_saved[split]]), scene_ids=d['scene_ids'])
            np.savez_compressed(OUT/f'{split}_scores.npz', keys=np.array([k for k, v in scores_saved[split]]), scores=np.array([v for k, v in scores_saved[split]]), scene_ids=d['scene_ids'])
        G.write_csv(OUT/'summary.csv', summary)
        for name, value in [('metrics', metrics), ('cohorts', cohorts), ('calibrations', cuts), ('models', models)]:
            C.save(OUT/f'{name}.json', value)
        C.save(OUT/'receipt.json', dict(status='COMPLETE', seconds=time.monotonic()-started, fits=12, reused_control_fits=6,
            cells=12, source_sha256=C.sha(Path(__file__)), persistent_resources=0, new_backbone_forward=0, new_data=0))
        print('COMPLETE', round(time.monotonic()-started, 3), flush=True)
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json', dict(error=repr(error), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()

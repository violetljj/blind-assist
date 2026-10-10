"""Evaluator-only first-score/native-path support diagnosis; no policy changes."""
from collections import defaultdict
import argparse
import csv
from pathlib import Path
import time
import traceback

import numpy as np

import cnh_graded_motion_path_dev as M
import cnh_graded_motion_path_features_dev as F

C, H, G, E = M.C, M.H, M.G, M.E
ROOT = C.ROOT
OUT = M.OUT / 'diagnostic'
TARGET = ROOT / 'artifacts.local/work/cnh-graded-target-support-dev-20261010/repair3/validation_target_geometry.npz'


def finite(value):
    return float(value) if np.isfinite(value) else None


def outcome(flags):
    return 'timely' if flags[:11].any() else 'late' if flags.any() else 'silent'


def first_frame(flags):
    index = np.flatnonzero(flags)
    return int(index[0] + 3) if len(index) else None


def describe(values):
    valid = np.array([x for x in values if x is not None], dtype=float)
    valid = valid[np.isfinite(valid)]
    return dict(n=len(valid), median=float(np.median(valid)) if len(valid) else None,
                minimum=float(valid.min()) if len(valid) else None,
                maximum=float(valid.max()) if len(valid) else None)


def run(output=OUT):
    OUT = output
    start = time.monotonic()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Preserve previous evaluator-only diagnostic')
    if not (M.OUT / 'receipt.json').exists():
        raise FileNotFoundError('Model/evaluation receipt required first')
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), M.OUT / 'receipt.json', M.OUT / 'calibrations.json',
             M.OUT / 'validation_scores.npz', M.OUT / 'validation_grades.npz',
             M.OUT / 'features/validation_features.npz', M.OUT / 'features/validation_maps.npz',
             H.OUT / 'validation_grades.npz', TARGET,
             C.SOURCE / 'scene_rows.json', C.SOURCE / 'data/validation/geometry.npz']
    try:
        C.save(OUT / 'PLAN.json', dict(role='Posthoc evaluator-only consumed ideal Development',
            goal='First score margins, timely native-compatible candidate support and public-path overlap',
            scope='Validation contact horizontal/sign_edge, all existing seeds and HEAD/BODY; no fit or threshold change',
            CPU_command_wall_seconds_cap=120, inputs_sha256={str(p): C.sha(p) for p in paths},
            boundary='Target direct mask joined only here; compatible noisy bins do not identify a target echo',
            source_sha256=C.sha(Path(__file__))))
        d = C.load()['validation']
        with np.load(TARGET, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
            direct = a['direct']
        with np.load(M.OUT / 'features/validation_features.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(a['scene_ids'], d['scene_ids'])
            candidates = a['candidate_native_index']
            candidate_valid = a['candidate_valid']
            aligned = a['aligned'].reshape(384, 4, 13, 2, 8, len(F.METRICS))
            static = a['static'].reshape(aligned.shape)
        with np.load(M.OUT / 'features/validation_maps.npz', allow_pickle=False) as a:
            maps = {k: a[k] for k in ('aligned_index', 'aligned_weight', 'aligned_visible', 'aligned_history_frame')}
        with np.load(M.OUT / 'validation_scores.npz', allow_pickle=False) as a:
            scores = dict(zip(a['keys'].tolist(), a['scores']))
        with np.load(M.OUT / 'validation_grades.npz', allow_pickle=False) as a:
            grades = dict(zip(a['keys'].tolist(), a['grades']))
        with np.load(H.OUT / 'validation_grades.npz', allow_pickle=False) as a:
            parents = dict(zip(a['keys'].tolist(), a['grades']))
        cuts = C.read(M.OUT / 'calibrations.json')
        ledger = []
        for seed in G.SEEDS:
            key = f'{seed}/aligned_path'
            cut = cuts[key]
            theta = -np.inf if cut['nonbinding'] else cut['theta']
            grade, parent, score = grades[key], parents[f'{seed}/head50'], scores[key]
            for n, author in enumerate(d['rows']):
                if author['shape_family'] not in ('horizontal', 'sign_edge'):
                    continue
                for q, height in enumerate(E.HEIGHTS):
                    if d['category'][n, q] != 'contact':
                        continue
                    for replica in range(4):
                        if time.monotonic() - start >= 118:
                            raise TimeoutError('Diagnostic cap120 seconds reached')
                        indices = np.maximum(candidates[n, replica, :, q], 0)
                        compatible = direct[n, F.C.FRAMES[:, None], indices] & candidate_valid[n, replica, :, q]
                        support = compatible.any(-1)
                        first = np.flatnonzero(support[:11])
                        margin = score[n, replica, :, q] - theta
                        row = dict(seed=int(seed), scene=int(d['scene_ids'][n]), replica=replica,
                            height=height, family=author['shape_family'], rho=author['rho'],
                            parent_outcome=outcome(parent[n, replica, :, q] > 0),
                            aligned_outcome=outcome(grade[n, replica, :, q] > 0),
                            parent_first_alert_frame=first_frame(parent[n, replica, :, q] > 0),
                            aligned_first_alert_frame=first_frame(grade[n, replica, :, q] > 0),
                            first_score_margin=finite(margin[0]), timely_max_score_margin=finite(margin[:11].max()),
                            first_target_compatible_top8_frame=int(first[0] + 3) if len(first) else None,
                            target_compatible_timely_frames=int(support[:11].sum()),
                            margin_at_first_target_support=finite(margin[first[0]]) if len(first) else None)
                        snr = aligned[n, replica, :11, q, :, 2]
                        masked = np.where(compatible[:11], snr, -np.inf)
                        if np.isfinite(masked).any():
                            fj, rank = np.unravel_index(np.argmax(masked), masked.shape)
                            seed_bin = int(indices[fj, rank])
                            av, sv = aligned[n, replica, fj, q, rank], static[n, replica, fj, q, rank]
                            overlap, available, hitframes = [], 0, 0
                            for t, h in enumerate(maps['aligned_history_frame'][fj]):
                                if h < 0 or not maps['aligned_visible'][fj, t, seed_bin]:
                                    continue
                                available += 1
                                weights = maps['aligned_weight'][fj, t, seed_bin]
                                ii = maps['aligned_index'][fj, t, seed_bin]
                                value = float((weights * direct[n, int(h), ii]).sum())
                                overlap.append(value)
                                hitframes += value > 0
                            row.update(best_compatible_anchor_frame=int(fj + 3), best_compatible_rank=int(rank + 1),
                                aligned_best_compatible_independent_z=finite(av[2]),
                                static_at_same_anchor_independent_z=finite(sv[2]),
                                aligned_minus_static_independent_z=finite(av[2] - sv[2]),
                                aligned_positive_mass=finite(av[3]), aligned_negative_mass=finite(av[4]),
                                static_positive_mass=finite(sv[3]), static_negative_mass=finite(sv[4]),
                                aligned_valid_path_frames=finite(av[9]), static_valid_path_frames=finite(sv[9]),
                                aligned_path_mean_direct_kernel_overlap=float(np.mean(overlap)) if overlap else None,
                                aligned_path_direct_overlap_frames=hitframes,
                                aligned_path_available_frames=available)
                        else:
                            for name in ('best_compatible_anchor_frame', 'best_compatible_rank',
                                'aligned_best_compatible_independent_z', 'static_at_same_anchor_independent_z',
                                'aligned_minus_static_independent_z', 'aligned_positive_mass', 'aligned_negative_mass',
                                'static_positive_mass', 'static_negative_mass', 'aligned_valid_path_frames',
                                'static_valid_path_frames', 'aligned_path_mean_direct_kernel_overlap',
                                'aligned_path_direct_overlap_frames', 'aligned_path_available_frames'):
                                row[name] = None
                        ledger.append(row)
        with (OUT / 'events.csv').open('w', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
            writer.writeheader(); writer.writerows(ledger)
        grouping = defaultdict(list)
        for row in ledger:
            group = '/'.join(str(row[k]) for k in ('seed', 'height', 'family', 'rho', 'aligned_outcome'))
            grouping[group].append(row)
        summary = {}
        numeric = [k for k in ledger[0] if k not in ('seed', 'scene', 'replica', 'height', 'family', 'rho', 'parent_outcome', 'aligned_outcome')]
        for key, rows in grouping.items():
            summary[key] = dict(events=len(rows), parent_outcomes={o: sum(r['parent_outcome'] == o for r in rows) for o in ('timely', 'late', 'silent')},
                fields={name: describe([r[name] for r in rows]) for name in numeric})
        C.save(OUT / 'groups.json', summary)
        C.save(OUT / 'receipt.json', dict(status='COMPLETE', events=len(ledger), groups=len(summary),
            seconds=time.monotonic() - start, source_sha256=C.sha(Path(__file__)),
            limitation='Posthoc target-compatible noisy current bins and truth-kernel overlap; best anchor selected by evaluator-only compatibility/SNR, never model input or target attribution',
            outputs_sha256={p.name: C.sha(p) for p in OUT.iterdir() if p.is_file()}))
    except BaseException as error:
        C.save(OUT / f'failure_{time.time_ns()}.json', dict(error=repr(error),
            traceback=traceback.format_exc(), seconds=time.monotonic() - start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args().output)

"""Independent expectation/noise-formula audit; bounded CPU sub32 sensitivity.

No model forward, sampling, training or change to frozen configurations. Four
representative quadrature pairs are fixed by background ID order before reading
metric values: alternate .013m/original .017m thickness and left/right side,
then lowest geometry ID. Each pair uses nominal frames3/13 at original rho.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-head-thin-signal-oracle-20261010'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(value, expected, label):
    np.testing.assert_allclose(float(value), expected, rtol=2e-12, atol=1e-9,
                               err_msg=str(label))


def snr_formula(target, present, ambient):
    signal = target.reshape(len(target), -1)
    variance = present + 16*ambient[..., None]
    standardized = signal/np.sqrt(variance.reshape(signal.shape))
    return standardized.max(-1), np.linalg.norm(standardized, axis=-1)


def audit(out, quadrature=True):
    began = time.monotonic()
    attempt_dir = out/'audit_attempts'
    attempt_dir.mkdir(exist_ok=True)
    previous = sum(read(p)['seconds'] for p in attempt_dir.glob('*.json'))
    receipt = dict(status='FAILED', previous_seconds=previous,
                   source_sha256=sha(Path(__file__)), model_forward=0, sampling=0,
                   training=0, threshold_changes=0, protected_access=0)
    def check():
        if previous+time.monotonic()-began >= 120:
            raise TimeoutError('Cumulative independent audit120 command-wall seconds reached')
    try:
        p, run, analysis = read(out/'PLAN.json'), read(out/'run_receipt.json'), read(out/'analysis_receipt.json')
        assert run['status'] == analysis['status'] == 'COMPLETE'
        for path, expected in p.get('source_sha256', {}).items():
            check()
            source = Path(path)
            if not source.is_absolute():
                source = ROOT/source
            assert sha(source) == expected, ('source SHA', source)
        for path, expected in run['input_sha256'].items():
            assert sha(path) == expected, ('input SHA', path)
        assert sha(out/'physical.npz') == run['payload_sha256']
        assert sha(out/'geometries.json') == run['geometries_sha256']
        for name, expected in analysis['outputs_sha256'].items():
            assert sha(out/name) == expected, ('analysis CSV SHA', name)
        geometries = read(out/'geometries.json')
        with np.load(out/'physical.npz', allow_pickle=False) as a:
            endpoints, backgrounds, ambient = a['endpoints'], a['background'], a['ambient']
            occupancy, poses = a['occupancy'], a['sensor']
            endpoint_rho = float(a['endpoint_rho'])
            bgids = a['background_ids'].tolist()
        assert endpoint_rho == .65
        assert endpoints.shape == (len(geometries), 2, 16, 8, 8, 16)
        assert occupancy.shape == (len(geometries), 16, 8, 8)
        assert np.isfinite(endpoints).all() and endpoints.min() >= 0
        assert np.isfinite(backgrounds).all() and backgrounds.min() >= 0
        assert np.isfinite(ambient).all() and ambient.min() >= 0
        assert np.isfinite(occupancy).all() and occupancy.min() >= 0
        for metadata in run['backend']:
            params = metadata['sensor_params']
            assert params['output_gain'] == 1 and params['noise_scale'] == 1
            assert metadata['sub'] == 16
        frames = csv_read(out/'single_frame.csv')
        temporal = csv_read(out/'accumulation.csv')
        grouped = csv_read(out/'grouped.csv')
        source = read(Path(p['source_evaluation'])/'scene_rows.json')['hold']
        selected_source = [r for r in source if r['shape_family'] == 'horizontal' and r['group'] == 0
                           and r['placement'] == 'contact' and r['size_variant'] == 0]
        assert len(selected_source) == 16
        assert len(geometries) == len(selected_source)*len(p['thickness_m'])
        source_ids = {r['scene_id'] for r in selected_source}
        assert {g['source_scene_id'] for g in geometries} == source_ids
        bg_lookup = {int(bg): i for i, bg in enumerate(bgids)}
        frame_lookup = {(int(r['geometry_id']), float(r['rho']), int(r['frame'])): r for r in frames}
        temporal_lookup = {(int(r['geometry_id']), float(r['rho']), int(r['end_frame']), int(r['requested_frames'])): r for r in temporal}
        assert len(frame_lookup) == len(frames) == len(geometries)*len(p['rho_values'])*16
        assert len(temporal_lookup) == len(temporal) == len(geometries)*len(p['rho_values'])*len(p['anchor_frames'])*len(p['windows'])
        fields_verified = 0
        for g in geometries:
            check()
            i = g['geometry_id']
            absent = backgrounds[bg_lookup[g['background_id']]]
            opaque = endpoints[i, 0]
            unit = (endpoints[i, 1]-opaque)/endpoint_rho
            occluded = absent-opaque
            assert unit.min() >= -1e-8 and occluded.min() >= -1e-8
            assert abs(g['target_box']['hi'][1]-g['target_box']['lo'][1]-g['thickness_m']) < 1e-12
            for rho in p['rho_values']:
                target = rho*unit
                present = opaque+target
                delta = present-absent
                np.testing.assert_allclose(delta, target-occluded, rtol=1e-12, atol=1e-9)
                if rho == 0:
                    np.testing.assert_array_equal(present, opaque)
                # Skellam signed = Pois(mu+8A)-Pois(8A): variance is sum of means.
                variance = (present+8*ambient[..., None]) + 8*ambient[..., None]
                pooled_variance = ((present+16*ambient[..., None]) +
                                   (absent+16*ambient[..., None]))/2
                flat_target, flat_var = target.reshape(16, -1), variance.reshape(16, -1)
                flat_delta = delta.reshape(16, -1)
                best, matched = snr_formula(target, present, ambient)
                peak, bestbin = flat_target.argmax(1), (flat_target/np.sqrt(flat_var)).argmax(1)
                pair = np.linalg.norm(flat_delta/np.sqrt(pooled_variance.reshape(16, -1)), axis=1)
                for f in range(16):
                    row = frame_lookup[i, float(rho), f]
                    peak_index = peak[f]
                    expected = dict(target_best_bin_snr=best[f], target_matched_snr=matched[f],
                        pair_separation_proxy=pair[f], target_counts=target[f].sum(),
                        background_loss_counts=occluded[f].sum(), signed_delta_counts=delta[f].sum(),
                        occupancy_max=occupancy[i, f].max(), best_snr_flatbin=bestbin[f],
                        target_peak_flatbin=peak_index, target_peak_counts=flat_target[f, peak_index],
                        background_at_target_peak=absent[f].reshape(-1)[peak_index],
                        opaque_background_at_target_peak=opaque[f].reshape(-1)[peak_index],
                        background_max_counts=absent[f].max(), target_peak_zone=peak_index//16,
                        target_peak_bin=peak_index % 16, front_distance_m=g['target_box']['lo'][2]-poses[f, 2, 3])
                    for name, value in expected.items():
                        close(row[name], value, (i, rho, f, name)); fields_verified += 1
                for end in p['anchor_frames']:
                    for window in p['windows']:
                        row = temporal_lookup[i, float(rho), end, window]
                        start = max(0, end-window+1)
                        t, v = flat_target[start:end+1], flat_var[start:end+1]
                        d = flat_delta[start:end+1]
                        pv = pooled_variance.reshape(16, -1)[start:end+1]
                        choices = (t/np.sqrt(v)).argmax(1)
                        indices = np.arange(len(t))
                        expected = dict(actual_frames=len(t),
                            moving_fixed_bin_snr=np.max(t.sum(0)/np.sqrt(v.sum(0))),
                            privileged_moving_peak_snr=t[indices, choices].sum()/np.sqrt(v[indices, choices].sum()),
                            privileged_moving_matched_snr=np.linalg.norm(t/np.sqrt(v)),
                            privileged_moving_pair_proxy=np.linalg.norm(d/np.sqrt(pv)),
                            static_repeat_peak_snr=np.sqrt(window)*best[end],
                            static_repeat_matched_snr=np.sqrt(window)*matched[end],
                            static_repeat_pair_proxy=np.sqrt(window)*pair[end])
                        for name, value in expected.items():
                            close(row[name], value, (i, rho, end, window, name)); fields_verified += 1
        for row in grouped:
            selected = [r for r in frames if float(r['thickness_m']) == float(row['thickness_m'])
                        and float(r['rho']) == float(row['rho']) and int(r['frame']) == int(row['frame'])]
            assert int(row['worlds']) == len(selected) == 16
            for name in ('target_best_bin_snr', 'target_matched_snr', 'pair_separation_proxy',
                         'target_counts', 'background_loss_counts', 'occupancy_max'):
                values = np.array([float(r[name]) for r in selected])
                for stat, value in [('min', values.min()), ('median', np.median(values)), ('max', values.max())]:
                    close(row[name+'_'+stat], value, ('group', name, stat)); fields_verified += 1
        quadrature_rows = []
        if quadrature:
            import cnh_counterfactual_data_dev as D
            _, cpu, _, _ = D.frozen_imports()
            frames_probe = np.array([3, 13])
            for j, bg in enumerate(sorted(bgids)):
                check()
                thickness = .013 if j % 2 == 0 else .017
                side = -1 if j % 2 == 0 else 1
                choices = [g for g in geometries if g['background_id'] == bg and
                           abs(g['thickness_m']-thickness) < 1e-12 and g['side'] == side]
                chosen = min(choices, key=lambda g: g['geometry_id'])
                i, rho = chosen['geometry_id'], chosen['source_rho']
                base32 = cpu.expected(dict(poses=poses[frames_probe], boxes=chosen['background_boxes']), sub=32)
                zero32 = cpu.expected(dict(poses=poses[frames_probe], boxes=[dict(chosen['target_box'], rho=0), *chosen['background_boxes']]), sub=32)
                present32 = cpu.expected(dict(poses=poses[frames_probe], boxes=[dict(chosen['target_box'], rho=rho), *chosen['background_boxes']]), sub=32)
                np.testing.assert_array_equal(base32['ambient'], ambient[frames_probe])
                np.testing.assert_array_equal(zero32['ambient'], ambient[frames_probe])
                np.testing.assert_array_equal(present32['ambient'], ambient[frames_probe])
                target32 = present32['expectation']-zero32['expectation']
                loss32 = base32['expectation']-zero32['expectation']
                assert target32.min() >= -1e-8 and loss32.min() >= -1e-8
                target16 = rho*(endpoints[i, 1]-endpoints[i, 0])[frames_probe]/endpoint_rho
                absent16 = backgrounds[bg_lookup[bg]][frames_probe]
                present16 = endpoints[i, 0, frames_probe]+target16
                peak16, matched16 = snr_formula(target16, present16, ambient[frames_probe])
                peak32, matched32 = snr_formula(target32, present32['expectation'], ambient[frames_probe])
                for fidx, f in enumerate(frames_probe):
                    a, b = float(target16[fidx].sum()), float(target32[fidx].sum())
                    quadrature_rows.append(dict(geometry_id=i, background_id=bg, side=side,
                        thickness_m=thickness, rho=rho, frame=int(f), sub16_target_counts=a,
                        sub32_target_counts=b, target_counts_relative_change=(b-a)/a if a else None,
                        sub16_peak_snr=float(peak16[fidx]), sub32_peak_snr=float(peak32[fidx]),
                        sub16_matched_snr=float(matched16[fidx]), sub32_matched_snr=float(matched32[fidx]),
                        target_bin_max_abs_difference=float(np.abs(target32[fidx]-target16[fidx]).max())))
                print('ORACLE_AUDIT_SUB32', bg, i, flush=True)
        check()
        receipt.update(status='PASS', fields_verified=fields_verified, single_frame_rows=len(frames),
            temporal_rows=len(temporal), grouped_rows=len(grouped), quadrature_sensitivity=quadrature_rows,
            definitions_verified=['opaque rho0 distinguished from absent target', 'positive target emission vs background loss and signed delta',
                'signed Skellam observation variance mu+16ambient', 'pooled-single-observation Gaussian pair proxy',
                'moving native-bin sum vs authored per-frame peak/template alignment', 'independent identical-pose static sqrt(requested frames)'],
            limits=['Pair proxy uses pooled observation variance; independent two-observation difference variance is twice as large',
                'Static requested window may exceed available moving history; use actual_frames before comparing exposure budgets',
                'Privileged matched SNR has no peak-search false-alarm cost; is not comparable to M3 probabilities/logits or deployable recall',
                'Sub16 and four sub32 probes are finite angular quadrature sensitivity, not convergence or hardware validation',
                'Consumed controlled fixture derivatives; no fresh scientific confirmation or real sensor detectability'],
            model_forward=0, sampling=0, training=0, protected_access=0)
        print('ORACLE_AUDIT_PASS', fields_verified, 'seconds', round(time.monotonic()-began, 3), flush=True)
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['seconds'] = time.monotonic()-began
        receipt['cumulative_seconds'] = previous+receipt['seconds']
        save_new(attempt_dir/f'audit_{time.time_ns()}.json', receipt)
        if receipt['status'] == 'PASS':
            save_new(out/'audit_receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--without-quadrature', action='store_true')
    args = parser.parse_args()
    audit(args.out.resolve(), quadrature=not args.without_quadrature)

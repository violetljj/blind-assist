"""CPU-only scoring of frozen simulated outputs on the exact MZ76 HELD rows."""
import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

RECEIPT_SHA = 'a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
METHODS = ('MZ37', 'OLD_NEG/UNION', 'MZ70/DIVERSE/UNION')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def counts(values, truth, known):
    assert values.shape == truth.shape == known.shape
    assert not np.isnan(values).any(), 'NaN scores require explicit handling'
    pred = values >= 0
    masks = dict(tp=pred & truth & known, fp=pred & ~truth & known,
                 fn=~pred & truth & known, tn=~pred & ~truth & known,
                 unknown=~known, unknown_alert=pred & ~known)
    vector = {name: mask.sum(0).tolist() for name, mask in masks.items()}
    scalar = {name: [0] * 4 for name in masks}
    for i in range(len(values)):
        for q in range(4):
            alert = bool(values[i, q] >= 0)
            if not bool(known[i, q]):
                scalar['unknown'][q] += 1
                scalar['unknown_alert'][q] += int(alert)
            else:
                label = 'tp' if alert and truth[i, q] else 'fp' if alert else 'fn' if truth[i, q] else 'tn'
                scalar[label][q] += 1
    assert vector == scalar
    return dict(per_query=vector, totals={name: sum(v) for name, v in vector.items()})


def run(root, replay=None):
    tick = time.perf_counter()
    work = root / 'artifacts.local/work'
    out = work / 'mz76-simulated-baselines-20260911/analysis-v1'
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or expected == digest, str(path)
        inputs[str(path)] = digest
        return path

    try:
        bind(__file__)
        rp = work / 'mz70-diverse-learning-20260911/run-v1'
        receipt = read(bind(rp / 'receipt.json', RECEIPT_SHA))
        assert receipt['status'] == 'PASS'
        selection = read(bind(work / 'mz76-observed-reliability-20260911/selection.json'))
        results = {}
        missing = {}
        audit_bits = 0
        replay_checks = []
        with ExitStack() as stack:
            archive = stack.enter_context(np.load(bind(rp / 'predictions.npz', receipt['outputs']['predictions.npz']), allow_pickle=False))
            fresh = None
            if replay is not None:
                verification = read(bind(replay / 'result.json'))
                assert verification['status'] == 'PASS_NUMERIC_REPLAY'
                assert verification['frames'] == 2048 and verification['profiles'] == list(PROFILES)
                for path, digest in verification['inputs'].items():
                    # Bind the replay producer snapshot and the shared frozen archive/selection.
                    if Path(path).name in ('mz76_replay_verify.py', 'selection.json', 'predictions.npz'):
                        bind(path, digest)
                fresh = stack.enter_context(np.load(bind(replay / 'predictions.npz', verification['outputs']['predictions.npz']), allow_pickle=False))
            for source in ('mz61', 'mz67'):
                selected = selection['sources'][source]
                ids = np.asarray(selected['held_ids'], dtype=np.int64)
                assert len(ids) == len(np.unique(ids)) == 1024
                assert not set(ids) & set(selected['fit_ids'])
                frames = archive[source + '/frame_ids'][ids]
                np.testing.assert_array_equal(frames, selected['held_frame_ids'])
                if fresh is not None:
                    np.testing.assert_array_equal(fresh[source + '/frame_ids'], frames)
                    np.testing.assert_array_equal(fresh[source + '/original_indices'], ids)
                assert len(set(frames.tolist())) == 1024
                truth, known = (archive[source + '/' + key][ids] for key in ('truth', 'known'))
                assert truth.dtype == known.dtype == np.bool_
                results[source] = dict(frame_ids=frames.tolist(), original_indices=ids.tolist(),
                                       known_query_bits=known.sum(0).tolist(), unknown_query_bits=(~known).sum(0).tolist(), profiles={})
                for profile in PROFILES:
                    prefix = source + '/' + profile + '/'
                    available = [k[len(prefix):] for k in archive.files if k.startswith(prefix)]
                    missing[prefix] = dict(unavailable=[] if fresh is not None else ['RGB-only learned branch', 'ToF-only learned branch', 'MZ5 fixed 0.5 ensemble'], available_saved_keys=available)
                    valid = archive[prefix + 'valid'][ids]
                    ranges = archive[prefix + 'ranges'][ids]
                    assert valid.dtype == np.bool_ and valid.shape == ranges.shape == (1024, 64, 2)
                    assert np.isfinite(ranges[valid]).all()
                    zones = valid.any(axis=2).sum(axis=1)
                    scalar_zones = np.asarray([sum(any(bool(x) for x in zone) for zone in frame) for frame in valid])
                    np.testing.assert_array_equal(zones, scalar_zones)
                    coverage = dict(zero_valid_zone_frames=int((zones == 0).sum()), partial_valid_zone_frames=int(((zones > 0) & (zones < 64)).sum()),
                                    all_64_valid_zone_frames=int((zones == 64).sum()), valid_zone_histogram=np.bincount(zones, minlength=65).tolist(),
                                    total_valid_zones=int(zones.sum()), total_zones=int(zones.size * 64))
                    table = {}
                    for method in METHODS:
                        values = archive[prefix + method][ids]
                        table[method] = counts(values, truth, known)
                        audit_bits += values.size
                    if fresh is not None:
                        actual, expected = fresh[prefix + 'MZ37'], archive[prefix + 'MZ37'][ids]
                        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-6)
                        np.testing.assert_array_equal(actual >= 0, expected >= 0)
                        replay_checks.append(dict(source=source, profile=profile, mz37_sign_changes=0,
                                                  max_abs_error=float(np.abs(actual - expected).max())))
                        np.testing.assert_array_equal(fresh[prefix + 'MZ5'], (fresh[prefix + 'rgb'] + fresh[prefix + 'tof']) * .5)
                        for method in ('rgb', 'tof', 'MZ5'):
                            table[method] = counts(fresh[prefix + method], truth, known)
                            audit_bits += fresh[prefix + method].size
                    results[source]['profiles'][profile] = dict(coverage=coverage, methods=table,
                        packet_sha256=hashlib.sha256(ranges.tobytes() + valid.tobytes()).hexdigest())
        result = dict(status='PASS' if replay is not None else 'PASS_WITH_UNAVAILABLE_BASELINES', query_order=QUERIES, sources=results, missing_saved_baselines=missing, replay_compatibility=replay_checks,
                      decision_rule='Existing saved margins >= 0; no fitting, inference or cutoff selection.',
                      evidence='Consumed simulated Development on exact MZ76 HELD rows; profiles are sensitivity proxies, not hardware or temporal measurements.',
                      coverage_rule='Observed valid-zone coverage is independent of evaluator known query bits. UNKNOWN is excluded from TP/FP/FN/TN and reported separately.')
        write(out / 'result.json', result)
        lines = ['# Frozen simulated baseline comparison', '',
                 'Exact MZ76 HELD: 1,024 frames per source. Entries are query-event counts. Existing saved margins use >= 0.', '',
                 ('RGB-only, ToF-only and MZ5 fixed 0.5 ensemble come from the hash-bound verified replay; existing MZ37 and unions remain the sealed MZ70 outputs. Replay MZ37 signs match on every scored bit; MZ5 equals (rgb + tof) / 2 exactly.' if replay is not None else 'RGB-only, ToF-only and MZ5 fixed 0.5 ensemble are unavailable in this archive. MZ37 is retained under its original name, not treated as one of those methods.'), '',
                 '| Source | Profile | Method | TP | FP | FN | TN | HEAD_NEAR TP/FP |',
                 '|---|---|---|---:|---:|---:|---:|---:|']
        for source, data in results.items():
            for profile, row in data['profiles'].items():
                for method, metric in row['methods'].items():
                    t, q = metric['totals'], metric['per_query']
                    lines.append(f"| {source} | {profile} | {method} | {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} | {q['tp'][2]}/{q['fp'][2]} |")
        lines += ['', 'Coverage (zero / partial / all 64 valid zones), separately from evaluator UNKNOWN:']
        for source, data in results.items():
            lines.append(f"- {source}: evaluator UNKNOWN per query {data['unknown_query_bits']}.")
            for profile, row in data['profiles'].items():
                c = row['coverage']
                lines.append(f"- {source} {profile}: {c['zero_valid_zone_frames']} / {c['partial_valid_zone_frames']} / {c['all_64_valid_zone_frames']} frames.")
        lines += ['', 'This scoring script performs no inference, fit or threshold search. Optional simple branch inputs come from the separately recorded frozen replay. Consumed simulated Development only; no new source, temporal sequence or real-device claim. End-to-end normalized latency is unavailable; no real-time claim. Full per-query counts, frame identities, input hashes and valid-zone histograms are in result.json and receipt.json.']
        (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        write(out / 'receipt.json', dict(status='PASS', inputs=inputs, outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()},
                                        metric_bits_independently_checked=audit_bits, distinct_frames=2048, profile_frames=6144,
                                        backend='CPU', placement_reason='TASK_NOT_GPU_SUITABLE', model_inferences=0, fits=0, cutoff_searches=0,
                                        seconds=time.perf_counter() - tick))
        print(out)
        print('PASS', audit_bits, 'independently scored query bits')
    except BaseException as exc:
        write(out / 'failure.json', dict(status='FAIL', error=repr(exc), inputs=inputs))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.replay.resolve() if args.replay else None)

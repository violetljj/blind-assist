"""Independent contract audit; never fits models, geometry or thresholds.

Expected arithmetic is scalar count accounting, separate from the producer's
vectorized adapter. Scientific evaluation is saved-output recomputation. CUDA
checks are explicitly opt-in and bounded by --seconds.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/local-signed-bias-contract-dev-20261011'
PRIOR = ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
PILOT = ROOT/'artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save(path, obj):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def scalar_real(h, ambient, signed_bias, noise):
    """Independent scalar physical count budget, no producer calls."""
    h = np.asarray(h, float)
    b = np.asarray(signed_bias, float).reshape(h.shape[-3:])
    aa = np.broadcast_to(np.asarray(ambient, float)[..., None], h.shape)
    if not all(np.isfinite(x).all() for x in (h, aa, b)):
        raise ValueError('UNKNOWN nonfinite observation')
    coeff = noise['adapter_coefficients_a_b_c']
    floor = float(noise['numerical_variance_floor'])
    output = np.empty_like(h)
    for ix in np.ndindex(h.shape):
        raw = float(h[ix]); background = max(float(b[ix[-3:]]), 0.)
        if raw <= 0.:
            residual = raw
        elif raw <= background:
            residual = 0.
        else:
            residual = raw-background
        variance = coeff[0]*background+coeff[1]*max(float(aa[ix]), 0.)+coeff[2]
        output[ix] = residual/np.sqrt(max(variance, floor))
    return output


def original_sim(h, ambient, bias):
    """Independent inherited float32-then-float16 operation order."""
    h = np.asarray(h, np.float32)
    b = np.asarray(bias, np.float32)
    a = np.asarray(ambient, np.float32)
    return ((h-b)/np.sqrt(np.maximum(16*a[..., None]+np.maximum(b, np.float32(0)), 1e-9))).astype(np.float16)


def expected_sim(h, ambient, bias):
    z = original_sim(h, ambient, bias)
    raw = np.asarray(h)
    flat = raw.reshape(-1, 1024)
    background = np.broadcast_to(np.asarray(bias), raw.shape).reshape(-1, 1024)
    zero = ((flat == 0).all(1) | (flat == background).all(1)).reshape(raw.shape[:-3])
    z[zero] = 0
    return z


def expected_native(z, deleted=()):
    out = np.array(z, copy=True)
    if not np.isfinite(out).all():
        raise ValueError('UNKNOWN nonfinite observation')
    out[..., list(deleted)] = 0
    frames = out.reshape(-1, 8, 8, 16)
    positive = np.array([any(float(v) > 0 for v in frame.flat) for frame in frames])
    for frame in np.flatnonzero(~positive):
        frames[frame] = 0
    positive = positive.reshape(out.shape[:-3])
    return out, positive


def expected_history(positive, seq, ms):
    positive = np.asarray(positive, bool)
    seq, ms = np.asarray(seq), np.asarray(ms)
    assert seq.shape == ms.shape == positive.shape
    assert np.isfinite(seq).all() and np.isfinite(ms).all()
    assert (seq == np.floor(seq)).all() and (ms == np.floor(ms)).all()
    reset = np.empty(len(seq), bool)
    presence = np.empty(len(seq), bool)
    lengths = np.empty(len(seq), np.int64)
    start = 0
    for i in range(len(seq)):
        reset[i] = i == 0 or seq[i] != seq[i-1]+1 or ms[i] <= ms[i-1] or ms[i]-ms[i-1] > 500
        if reset[i]:
            start = i
        lo = max(start, i-7)
        lengths[i] = i-lo+1
        presence[i] = positive[lo:i+1].any()
    return presence, reset, lengths


def notifications(grades):
    grades = np.asarray(grades)
    assert grades.ndim == 4 and grades.shape[-2:] == (13, 2)
    answer = np.zeros_like(grades, np.int8)
    for scene in range(len(grades)):
        for replica in range(grades.shape[1]):
            for query in range(2):
                peak, empty = 0, 0
                for frame in range(13):
                    current = int(grades[scene, replica, frame, query])
                    assert current in (0, 1, 2)
                    if current == 0:
                        empty += 1
                        if empty > 1:
                            peak = 0
                    else:
                        empty = 0
                        if current > peak:
                            answer[scene, replica, frame, query] = current
                        peak = max(peak, current)
    return answer


def metrics(grades, reference, category, rows):
    emit, base = notifications(grades), notifications(reference)
    category = np.asarray(category)
    negative = ~(category == 'contact').any(1)
    near = (category == 'pass').any(1) & np.array([float(x['lateral_gap_m']) <= .10+1e-10 for x in rows])
    far = negative & ~near
    joint = emit.max(-1)
    cost = np.where((joint == 1) & near[:, None, None], .25, 1.)
    timely = (emit[:, :, :11] > 0).any(2)
    baseline = (base[:, :, :11] > 0).any(2)
    result = dict(weighted_cost=float(np.where(negative[:, None, None] & (joint > 0), cost, 0).sum()),
                  far_clear_notified_clips=int((joint[far] > 0).any(-1).sum()),
                  far_clear_clip_denominator=int(far.sum()*grades.shape[1]), contacts={})
    for q, height in enumerate(('HEAD', 'BODY')):
        ids = category[:, q] == 'contact'
        a, b = timely[ids, :, q], baseline[ids, :, q]
        result['contacts'][height] = dict(denominator=a.size, timely=int(a.sum()),
            rescue=int((a & ~b).sum()), loss=int((~a & b).sum()), net=int(a.sum()-b.sum()))
    return result, emit


def load_adapter():
    path = Path(__file__).with_name('local_signed_bias_adapter.py')
    spec = importlib.util.spec_from_file_location('adapter_under_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapter_units():
    producer = load_adapter()
    noise = read(PILOT/'instrument_parameters.json')['noise']
    # Use deliberately nonuniform, signed templates across every zone and bin.
    b = (np.arange(1024).reshape(8, 8, 16) % 11-5).astype(float)
    bphys = np.maximum(b, 0)
    h = np.stack((np.zeros_like(b), bphys, b, -np.ones_like(b),
                  bphys+.75, np.where(b > 0, b*.5, .25)))
    ambient = np.linspace(2.5, 8., len(h))[:, None, None]*np.ones((1, 8, 8))
    expected = scalar_real(h, ambient, b, noise)
    observed = producer.real_adapter(h, ambient, b, noise)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)
    assert (observed[:2] == 0).all()
    assert (observed[3] < 0).all()
    assert (observed[4] > 0).all()
    for deleted in ((), (0,)):
        want, positive = expected_native(observed, deleted)
        got, flags = producer.native_evidence(observed, deleted)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(flags, positive)
        assert (got[~flags] == 0).all()
    rng = np.random.default_rng(202610111993)
    sim_bias = (np.arange(1024).reshape(8, 8, 16) % 9).astype(np.float32)
    sim = rng.normal(0, 2, (18, 8, 8, 16)).astype(np.float32)+sim_bias
    sim[0] = 0; sim[1] = sim_bias
    a = np.full((len(sim), 8, 8), 4., np.float32)
    np.testing.assert_array_equal(producer.sim_adapter(sim, a, sim_bias), expected_sim(sim, a, sim_bias))
    # Reset and positive-history tests cover expiry, sequence gaps, backwards
    # time and a sensor gap >500 ms, without consulting scene/model outcomes.
    positive = np.zeros(32, bool); positive[[0, 17, 28]] = True
    seq = np.arange(32); seq[15:] += 2
    ms = np.arange(32)*100; ms[23:] += 601; ms[28] = ms[27]-1
    history, reset, lengths = expected_history(positive, seq, ms)
    np.testing.assert_array_equal(producer.history_support(positive, lengths), history)
    assert history[:8].all() and not history[8:15].any()
    assert reset[[0, 15, 23, 28]].all()
    assert not history[15:17].any() and history[17:23].all()
    # Unobservable deleted-only and all-negative frames are exact zero native.
    probes = np.full((3, 8, 8, 16), -1., np.float32)
    probes[0] = 0; probes[0, ..., 0] = 1e6
    probes[2, 0, 0, 3] = 1e-10
    got, flags = producer.native_evidence(probes, (0,))
    want, expected = expected_native(probes, (0,))
    np.testing.assert_array_equal(got, want); np.testing.assert_array_equal(flags, expected)
    assert flags.tolist() == [False, False, True]
    # Invalid samples may not quietly become a no-evidence clear observation.
    rejected = 0
    for name in ('real', 'native'):
        bad = h.copy(); bad[0, 0, 0] = np.nan
        try:
            producer.real_adapter(bad, ambient, b, noise) if name == 'real' else producer.native_evidence(bad, (0,))
        except (ValueError, FloatingPointError):
            rejected += 1
    assert rejected == 2
    return dict(real_cells_checked=h.size, signed_sim_cells_checked=sim.size,
                native_arrays_checked=3, invalid_rejected=rejected,
                zero_background_negative_positive_extremes=True,
                history_expiry_resets=True)


def verify_replay(result, positive, seq, ms, cuts):
    present, resets, lengths = expected_history(positive, seq, ms)
    ready = lengths == 8
    for name, expected in (('frame_positive', positive), ('evidence_present', present),
                           ('reset', resets), ('history', lengths), ('ready', ready),
                           ('eligible', lengths >= 4),
                           ('score_evidence_valid', (present & (lengths >= 4))[:, None])):
        np.testing.assert_array_equal(result[name], expected, err_msg=name)
    m3 = result['m3_score'] >= cuts['m3_tau']
    old5 = (result['m3_score'] >= cuts['raised_tau']) | (result['local_score'] >= cuts['local_tau'])
    s = np.where(old5, 2, np.where(result['s_score'] >= cuts['s_tau'], 1, 0)).astype(np.int8)
    for name, raw in (('ungated_m3', m3), ('ungated_old5', old5), ('ungated_S_grade', s)):
        np.testing.assert_array_equal(result[name], np.where(ready[:, None], raw, 0), err_msg=name)
    allow = ready[:, None] & present[:, None]
    np.testing.assert_array_equal(result['m3'], m3 & allow)
    np.testing.assert_array_equal(result['old5'], old5 & allow)
    np.testing.assert_array_equal(result['S_grade'], np.where(allow, s, 0))
    np.testing.assert_array_equal(result['Sensemble'], (s > 0) & allow)
    return int(ready.sum())


def identities(out):
    seal = read(PRIOR/'sealed_candidates.json')
    counted = {}
    for section in ('scientific_code_sha256', 'old_frozen_dependencies_sha256', 'models_sha256'):
        for name, digest in seal[section].items():
            assert sha(ROOT/name) == digest, ('frozen dependency changed', name)
        counted[section] = len(seal[section])
    assert sha(PRIOR/'sealed_calibration.json') == seal['calibration_sha256']
    manifest = read(PRIOR/'main_real_manifest.json')
    assert sha(PRIOR/'main_real_manifest.json') == seal['real_manifest_hashes']['main']
    cuts = manifest['cuts']
    assert cuts == read(PRIOR/'sealed_calibration.json')['cuts']['no_bin0']
    assert cuts['local_tau'] == 4.625390338985158
    assert manifest['drop_bins'] == [0]
    if (out/'v4_input_seal.json').exists():
        inputs = read(out/'v4_input_seal.json')
        files = dict(plan_sha256=ROOT/'research/active/hardware-bringup/LOCAL_SIGNED_BIAS_CONTRACT_PLAN_20261011.md',
            adapter_sha256=Path(__file__).with_name('local_signed_bias_adapter.py'),
            replay_sha256=Path(__file__).with_name('local_signed_bias_replay.py'),
            prior_seal_sha256=PRIOR/'sealed_candidates.json', manifest_sha256=PRIOR/'main_real_manifest.json',
            template_source_sha256=PILOT/'instrument_arrays.npz',
            noise_source_sha256=PILOT/'instrument_parameters.json')
        for key, path in files.items():
            assert inputs[key] == sha(path), key
        assert inputs['cuts'] == cuts and inputs['LOCAL_cut_unchanged']
    # The new receipts bind source and old artifacts without silently updating
    # the frozen dependencies. Every listed input identity is independently read.
    new_bindings = 0
    for name in ('sim_input_closure.json', 'sim_evaluation.json', 'real_receipt.json'):
        path = out/name
        if path.exists():
            value = read(path)
            for rel, digest in value.get('bindings_sha256', {}).items():
                assert sha(ROOT/rel) == digest, ('new receipt binding', rel)
                new_bindings += 1
    plan = ROOT/'research/active/hardware-bringup/LOCAL_SIGNED_BIAS_CONTRACT_PLAN_20261011.md'
    return dict(old_sections=counted, new_bindings=new_bindings,
                frozen_cuts_exact=True, frozen_bin_mask_exact=True, plan_sha256=sha(plan)), cuts


def simulation_audit(out, check):
    closure = read(out/'sim_input_closure.json')
    hist = np.load(PRIOR/'data/hold/hist.npy', mmap_mode='r')
    ambient = np.load(PRIOR/'data/hold/ambient.npy')
    bias = np.load(ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy')
    saved = load(out/'sim_presence.npz')
    cells = 0; positives = []
    for begin in range(0, len(hist), 32):
        check()
        h = np.asarray(hist[begin:begin+32])
        leading = h.shape[:-3]
        aa = np.broadcast_to(ambient, (*leading[:-1], *ambient.shape))
        inherited = original_sim(h, aa, bias)
        adapted = expected_sim(h, aa, bias)
        np.testing.assert_array_equal(inherited, adapted)
        inherited[..., 0] = 0
        new, positive = expected_native(adapted, (0,))
        np.testing.assert_array_equal(inherited, new)
        cells += new.size
        positives.append(positive)
    positives = np.concatenate(positives)
    history = np.stack([positives[:, :, max(0, f-7):f+1].any(-1) for f in range(3, 16)], -1)
    np.testing.assert_array_equal(saved['native_positive'], positives)
    np.testing.assert_array_equal(saved['history_positive'], history)
    assert history.all()
    assert closure['all_cached_branches_reusable'] and closure['inputs_identical']
    assert closure['masked_input_changed_cells'] == closure['sim_adapter_changed_cells'] == 0
    assert closure['max_abs_input_difference'] == 0.
    assert closure['native_frames'] == positives.size and closure['native_positive_frames'] == positives.sum()
    current = load(out/'sim_grades_notifications.npz')
    previous = load(PRIOR/'hold_grades_notifications.npz')
    keys = list(previous['keys'])
    expected = previous['grades'][keys.index('no_bin0_S')]
    baseline = previous['grades'][keys.index('original_old5')]
    np.testing.assert_array_equal(current['grade'], expected)
    rows = read(PRIOR/'scene_rows.json')['hold']
    category = load(PRIOR/'data/hold/geometry.npz')['category']
    np.testing.assert_array_equal(current['category'], category)
    stats, emit = metrics(current['grade'], baseline, category, rows)
    np.testing.assert_array_equal(current['notifications'], emit)
    prior_stats = read(PRIOR/'independent_audit_real.json')['hold']['statistics']['no_bin0_S']
    assert stats['weighted_cost'] == prior_stats['weighted_cost']
    assert stats['far_clear_notified_clips'] == prior_stats['far_clear_notified_clips']
    for q in ('HEAD', 'BODY'):
        for name in ('denominator', 'timely', 'rescue', 'loss', 'net'):
            assert stats['contacts'][q][name] == prior_stats['contacts'][q][name]
    evaluation = read(out/'sim_evaluation.json')
    produced = evaluation['statistics']
    assert produced['weighted_cost'] == stats['weighted_cost']
    assert produced['far_clear']['notified_clips'] == stats['far_clear_notified_clips']
    for q in ('HEAD', 'BODY'):
        for name in ('denominator', 'timely', 'rescue', 'loss', 'net'):
            assert produced[q][name] == stats['contacts'][q][name], (q, name)
    gain = sum(stats['contacts'][q]['net'] for q in ('HEAD', 'BODY'))
    timely = sum(stats['contacts'][q]['timely'] for q in ('HEAD', 'BODY'))
    den = stats['far_clear_clip_denominator']; fc = stats['far_clear_notified_clips']
    acceptance = dict(gain=gain, gain_floor=136, gain_pass=gain >= 136,
        original_default_far_clear=168, far_clear_denominator=den, far_clear_clips=fc,
        original_default_plus_1pp_pass=fc <= 168+.01*den,
        timely_loss_vs_193=886-timely, timely_loss_cap=10, timely_small_change_pass=886-timely <= 10,
        far_clear_increment_vs_193=fc-152, far_clear_increment_cap=15, far_clear_small_change_pass=fc-152 <= 15)
    assert acceptance == evaluation['acceptance']
    for name in ('sim_forward_receipt.json',):
        value = read(out/name)
        assert value['status'] == 'PASS' and value['fixed_scenes'] == list(range(8))
        assert value['source_input_closure_sha256'] == sha(out/'sim_input_closure.json')
    return dict(native_cells_compared=cells, native_frames=positives.size,
                cached_input_identity=True, cached_grade_identity=True,
                notification_values_checked=emit.size, acceptance=acceptance,
                statistics=stats, fixed_first8_forward_pass=True)


def real_audit(out, maincuts, check):
    import real_pilot_calibration as P
    receipt = read(out/'real_receipt.json')
    assert receipt['status'] == 'COMPLETE'
    assert receipt['input_seal_sha256'] == sha(out/'v4_input_seal.json')
    inputs = read(out/'v4_input_seal.json')
    arrays = load(P.OUT/'instrument_arrays.npz')
    signed = arrays['bias'].reshape(8, 8, 16)
    available = arrays['bias_available'].reshape(8, 8, 16)
    template = load(out/'v4_template.npz')
    np.testing.assert_array_equal(template['physical'].reshape(8, 8, 16), np.maximum(signed, 0))
    np.testing.assert_array_equal(template['statistical'].reshape(8, 8, 16), np.minimum(signed, 0))
    np.testing.assert_array_equal(template['available'].reshape(8, 8, 16), available)
    np.testing.assert_array_equal(template['coverage'], arrays['bias_coverage'])
    assert (signed[~available] == 0).all()
    assert inputs['template_covered'] == int(available.sum()) == 128
    noise = P.read('instrument_parameters.json')['noise']
    pilot = P.data()
    oldmanifest = read(ROOT/'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json')
    oldcuts = dict(m3_tau=.8557642486787612, raised_tau=.9404184587540165,
                   local_tau=4.625390338985158, s_tau=oldmanifest['threshold']['tau'])
    native_index = {(x['arm'], x['condition']): x for x in receipt['natives']}
    count_index = {}; frames = 0; native_cells = 0
    mandatory = {'ZERO_RAW', 'PHYSICAL_BACKGROUND', 'SIGNED_BACKGROUND', 'PURE_AMBIENT',
                 'PURE_CROSSTALK', 'NEGATIVE_ONLY', 'NEGATIVE_BIAS_ZERO', 'BIN0_BACKGROUND_ONLY'}
    deterministic = set()
    for entry in receipt['outputs']:
        check()
        arm, sid, role = entry['arm'], entry['condition'], entry['role']
        assert arm in ('old', 'main')
        path = ROOT/entry['path']
        assert sha(path) == entry['sha256'], str(path)
        result = load(path)
        if role == 'consumed_pilot':
            assert sid in pilot
            raw = pilot[sid]; bias = signed
        else:
            raw = load(out/('fixture_'+sid+'.npz'))
            bias = raw['bias'].reshape(8, 8, 16)
        h = raw['H'].reshape(-1, 8, 8, 16)
        ambient = raw['A'].reshape(-1, 8, 8)
        z = scalar_real(h, ambient, bias, noise).astype(np.float32)
        native, positive = expected_native(z, () if arm == 'old' else (0,))
        detail = native_index[(arm, sid)]
        assert detail['native_sha256'] == hashlib.sha256(native.tobytes()).hexdigest(), (arm, sid, 'native identity')
        assert detail['positive_frames'] == int(positive.sum())
        assert detail['native_nonzero'] == np.count_nonzero(native)
        assert detail['total_frames'] == len(native)
        np.testing.assert_array_equal(result['seq'], raw['seq'])
        np.testing.assert_array_equal(result['ms'], raw['ms'])
        n = verify_replay(result, positive, raw['seq'], raw['ms'], maincuts if arm == 'main' else oldcuts)
        assert detail['evidence_present_frames'] == int(result['evidence_present'].sum())
        frames += len(native); native_cells += native.size
        if sid == 'POSITIVE_THEN_ZERO':
            assert result['frame_positive'].tolist() == [True]+[False]*15
            assert result['evidence_present'].tolist() == [True]*8+[False]*8
        if role == 'deterministic_no_surface' and arm == 'main':
            deterministic.add(sid)
            assert n == 9
        ready = result['ready']
        for q, height in enumerate(('HEAD', 'BODY')):
            for model, key in (('M3', 'm3'), ('old5', 'old5'), ('S', 'S_grade')):
                grade = result[key][ready, q]
                strong = int((grade == 2).sum()) if model == 'S' else int(grade.sum())
                light = int((grade == 1).sum()) if model == 'S' else 0
                count_index[(arm, sid, height, model)] = dict(n=n, strong=strong, light=light)
    assert mandatory == deterministic
    checked = 0
    for row in receipt['tables']:
        item = count_index[(row['arm'], row['condition'], row['height'], row['model'])]
        assert row['ready_frames'] == item['n']
        for kind in ('strong', 'light'):
            assert row[kind] == item[kind]
            assert row[kind+'_rate'] == (item[kind]/item['n'] if item['n'] else None)
        checked += 1
    assert checked == len(count_index)
    with (out/'real_alarm_table.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == checked
    for row, record in zip(csv_rows, receipt['tables']):
        for key in ('arm', 'path', 'condition', 'role', 'height', 'model'):
            assert row[key] == record[key]
        for key in ('ready_frames', 'strong', 'light'):
            assert int(row[key]) == record[key]
    control_pass = all(count_index[('main', sid, height, model)]['strong'] ==
                       count_index[('main', sid, height, model)]['light'] == 0
                       for sid in mandatory for height in ('HEAD', 'BODY') for model in ('M3', 'old5', 'S'))
    presence = {}
    for sid in ('2', '3', '5', '6', '8'):
        heights = ('HEAD',) if sid == '8' else ('HEAD', 'BODY')
        presence[sid] = any(count_index[('main', sid, height, 'S')]['strong']+
                           count_index[('main', sid, height, 'S')]['light'] > 0 for height in heights)
    return dict(table_rows_checked=checked, output_hashes_checked=len(receipt['outputs']),
                native_cells_independently_rebuilt=native_cells, frames=frames,
                covered_template_cells=int(available.sum()), mandatory_controls=sorted(mandatory),
                deterministic_controls_pass=control_pass, reminder_present=presence,
                walls_rods_pass=all(presence.values()), exact_history_gate_and_resets=True)


def simulation_controls(out, maincuts):
    receipt = read(out/'sim_controls_receipt.json')
    assert receipt['status'] == 'PASS' and receipt['unchanged_cuts']
    assert receipt['training'] == receipt['protected_access'] == 0
    assert receipt['source_sha256'] == sha(Path(__file__).with_name('local_signed_bias_sim_controls.py'))
    assert receipt['adapter_sha256'] == sha(Path(__file__).with_name('local_signed_bias_adapter.py'))
    assert receipt['bias_sha256'] == sha(ROOT/receipt['bias_source'])
    bias = np.load(ROOT/receipt['bias_source'])
    assert bias.shape == (8, 8, 16) and (bias >= 0).all()
    original = read(ROOT/'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json')
    oldcuts = dict(m3_tau=.8557642486787612, raised_tau=.9404184587540165,
                   local_tau=4.625390338985158, s_tau=original['threshold']['tau'])
    names = {'SIM_EMPTY_SCENE', 'SIM_PURE_AMBIENT', 'SIM_PURE_CROSSTALK'}
    coverage = set(); counts = {}
    native_records = {(x['arm'], x['condition']): x for x in receipt['natives']}
    for record in receipt['outputs']:
        arm, sid = record['arm'], record['condition']
        assert arm in ('old', 'main') and sid in names
        coverage.add((arm, sid))
        assert sha(ROOT/record['path']) == record['sha256']
        raw = np.broadcast_to(bias, (16, 8, 8, 16)).copy() if sid == 'SIM_PURE_CROSSTALK' else np.zeros((16, 8, 8, 16), np.float32)
        ambient = np.full((16, 8, 8), 10.4 if sid == 'SIM_PURE_AMBIENT' else 4., np.float32)
        z = expected_sim(raw, ambient, bias)
        native, positive = expected_native(z, () if arm == 'old' else (0,))
        assert not native.any() and not positive.any()
        assert native_records[(arm, sid)]['positive_frames'] == native_records[(arm, sid)]['native_nonzero'] == 0
        result = load(ROOT/record['path'])
        seq = np.arange(16); ms = seq*200
        np.testing.assert_array_equal(result['seq'], seq)
        np.testing.assert_array_equal(result['ms'], ms)
        assert verify_replay(result, positive, seq, ms, maincuts if arm == 'main' else oldcuts) == 9
        assert not result['m3'].any() and not result['old5'].any() and not result['S_grade'].any()
        for q, height in enumerate(('HEAD', 'BODY')):
            for model in ('M3', 'old5', 'S'):
                counts[(arm, sid, height, model)] = True
    assert coverage == {(arm, sid) for arm in ('old', 'main') for sid in names}
    assert len(receipt['tables']) == len(counts) == 36
    for row in receipt['tables']:
        assert (row['arm'], row['condition'], row['height'], row['model']) in counts
        assert row['ready_frames'] == 9 and row['strong'] == row['light'] == 0
        assert row['strong_rate'] == row['light_rate'] == 0.
    return dict(status='PASS', full_chain_controls=6, table_rows=36, ready_per_control=9,
                all_native_zero=True, all_three_branches_zero=True,
                adapter_source_bias_hashes=True, deterministic_only=True)


def audit(out):
    began = time.monotonic()
    def check():
        if time.monotonic()-began >= 140:
            raise TimeoutError('Independent CPU audit 140-second internal limit')
    identity, cuts = identities(out)
    units = adapter_units()
    check()
    simulation = simulation_audit(out, check)
    real = real_audit(out, cuts, check)
    sim_controls = simulation_controls(out, cuts)
    accepted = simulation['acceptance']
    recommend = bool(real['deterministic_controls_pass'] and real['walls_rods_pass'] and sim_controls['status'] == 'PASS' and
        accepted['gain_pass'] and accepted['original_default_plus_1pp_pass'] and
        accepted['timely_small_change_pass'] and accepted['far_clear_small_change_pass'])
    report = out/'report_receipt.json'
    if report.exists():
        assert read(report)['recommend'] == recommend
    return dict(status='PASS', protected_access=0, training=0, cuts_changed=0,
                identities=identity, units=units, simulation=simulation, simulation_controls=sim_controls, real=real,
                independent_recommendation=recommend)


def forward(out, seconds):
    """Fixed ten short windows (five cases x old/main); no selected frames."""
    from local_signed_bias_adapter import ContractReplay
    import real_pilot_calibration as P
    began = time.monotonic()
    def check():
        if time.monotonic()-began >= seconds:
            raise TimeoutError('Independent forward allocation reached')
    identities(out)
    noise = P.read('instrument_parameters.json')['noise']
    template = load(P.OUT/'instrument_arrays.npz')['bias'].reshape(8, 8, 16)
    pilot = P.data()
    cases = {}
    for name in ('ZERO_RAW', 'NEGATIVE_BIAS_ZERO', 'BIN0_BACKGROUND_ONLY'):
        fixture = load(out/('fixture_'+name+'.npz'))
        cases[name] = (fixture, fixture['bias'].reshape(8, 8, 16))
    for sid in ('4', '8'):
        d = pilot[sid]
        cases[sid] = ({name: np.asarray(d[name])[:16] for name in ('H', 'A', 'seq', 'ms')}, template)
    compared = []
    for arm, manifest in (('old', None), ('main', PRIOR/'main_real_manifest.json')):
        check(); engine = ContractReplay(manifest)
        try:
            for sid, (d, bias) in cases.items():
                check()
                h = d['H'].reshape(-1, 8, 8, 16)
                ambient = d['A'].reshape(-1, 8, 8)
                z = scalar_real(h, ambient, bias, noise).astype(np.float32)
                actual = engine.infer(z, d['seq'], d['ms'], check)
                saved = load(out/'real'/arm/(sid+'.npz'))
                details = {}
                names = ('m3_raw', 'm3_score', 'local_raw', 'local_score', 'ordinary_raw',
                         's_seed_score', 's_score', 'm3', 'old5', 'S_grade', 'Sensemble',
                         'frame_positive', 'evidence_present', 'history', 'reset', 'ready',
                         'score_evidence_valid', 'ungated_m3', 'ungated_old5', 'ungated_S_grade')
                for key in names:
                    value = actual[key]
                    previous = saved[key][:, :len(h)] if key in ('ordinary_raw', 's_seed_score') else saved[key][:len(h)]
                    if value.dtype.kind in 'biu':
                        np.testing.assert_array_equal(value, previous, err_msg=(arm, sid, key))
                        details[key] = dict(exact=True)
                    else:
                        np.testing.assert_allclose(value, previous, rtol=1e-5, atol=1e-4,
                                                   equal_nan=True, err_msg=str((arm, sid, key)))
                        finite = np.isfinite(value) & np.isfinite(previous)
                        delta = np.abs(value[finite]-previous[finite])
                        details[key] = dict(max_abs=float(delta.max()) if len(delta) else 0.)
                native, positive = expected_native(z, () if arm == 'old' else (0,))
                np.testing.assert_array_equal(actual['frame_positive'], positive)
                presence = expected_history(positive, d['seq'], d['ms'])[0]
                np.testing.assert_array_equal(actual['evidence_present'], presence)
                assert (native[~positive] == 0).all()
                compared.append(dict(arm=arm, condition=sid, frames=len(h), comparisons=details))
        finally:
            engine.close()
    check()
    return dict(status='PASS', windows=len(compared), protected_access=0, training=0,
                no_cut_selection=True, fixed_cases=list(cases), comparisons=compared)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--units', action='store_true')
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--seconds', type=float, default=60.)
    args = parser.parse_args()
    began = time.monotonic()
    if args.units:
        result = dict(status='PASS', units=adapter_units())
    elif args.forward:
        result = forward(args.out, args.seconds)
    else:
        result = audit(args.out)
    result.update(seconds=time.monotonic()-began, source_sha256=sha(__file__), independent=True)
    name = 'independent_forward.json' if args.forward else 'independent_units.json' if args.units else 'independent_audit.json'
    save(args.out/name, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

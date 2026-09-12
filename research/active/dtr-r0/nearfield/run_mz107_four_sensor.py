"""Seal observable predictions, then score native geometry separately."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil
import cv2
import numpy as np
import mz107_rgb_association as model

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
from research_backend import BackendCandidate, DeviceObservation, select_backend


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def readrows(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def truth(frame):
    origin = np.asarray(frame['body_origin_m'])
    return any((lambda lo, hi: hi[0] >= .2 and lo[0] <= 3.6 and hi[1] >= -.3 and
               lo[1] <= .3 and hi[2] >= .4 and lo[2] <= 2.05)(
               np.asarray(o['center_m'])-o['extent_m']-origin,
               np.asarray(o['center_m'])+o['extent_m']-origin) for o in frame['native_bounds'])


def metrics(t, p):
    return dict(TP=int((t&p).sum()), FP=int((~t&p).sum()), FN=int((t&~p).sum()),
                TN=int((~t&~p).sum()), UNKNOWN=int((~p).sum()))


def run(capture, out):
    capture = capture.resolve(); out = out.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    receipt = json.loads((capture/'receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['frames'] == 96
    for name, digest in receipt['hashes'].items(): assert sha(capture/name) == digest, name
    manifest = json.loads((capture/'manifest.json').read_text())
    assert manifest['rgb_camera_count'] == 1 and not manifest['depth_images_produced']
    rows = readrows(capture/'raw.jsonl')
    assert len(rows) == 96 and len({r['id'] for r in rows}) == 96
    assert len({r['episode_id'] for r in rows}) == 12
    assert all(len(r['tof64_range_m']) == 64 and len(r['radar_range_m']) == 4 for r in rows)
    assert not any(any(s in k for s in ('truth','actor','native','depth')) for r in rows for k in r)
    out.mkdir(parents=True)
    for path in (Path(model.__file__), Path(__file__), Path(__file__).with_name('MZ107_FOUR_SENSOR_PROTOCOL_20260913.md')):
        shutil.copyfile(path, out/path.name)
    images = {}
    for row in rows:
        path = (capture/row['rgb_path']).resolve(); assert path.is_relative_to(capture)
        im = cv2.imread(str(path)); assert im is not None
        assert im.shape[:2] == (row['rgb_intrinsics']['height'], row['rgb_intrinsics']['width'])
        images[row['id']] = im
    backend = select_backend('batch-tensor', cpu=BackendCandidate('opencv-cpu', 'cpu',
        lambda: model.proposals(images[rows[0]['id']]),
        lambda _: DeviceObservation('cpu', 'host CPU', 'OpenCV '+cv2.__version__)),
        cpu_reason='GPU_BACKEND_UNAVAILABLE', record_path=out/'backend.json',
        capabilities={'opencv':cv2.__version__, 'cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),
                      'reason':'Installed OpenCV has no CUDA connected-component/morphology proposal backend'})
    started = time.perf_counter()
    predictions = model.predict(rows, lambda r: images[r['id']])
    elapsed = time.perf_counter()-started
    disabled = model.predict(rows, lambda r: None)
    assert all(d['candidate'] == p['baseline'] == d['baseline'] for p,d in zip(predictions,disabled))
    assert all(not p['tof_support'] or p['candidate'] for p in predictions)
    payload = [dict(id=r['id'], **p) for r,p in zip(rows,predictions)]
    write(out/'predictions.json', payload)
    write(out/'rgb-disabled.json', [dict(id=r['id'], **p) for r,p in zip(rows,disabled)])
    write(out/'prediction-seal.json', dict(status='SEALED_BEFORE_EVALUATOR_READ',
        capture_receipt_sha256=sha(capture/'receipt.json'), observation_sha256=sha(capture/'raw.jsonl'),
        prediction_sha256=sha(out/'predictions.json'), rgb_disabled_sha256=sha(out/'rgb-disabled.json'),
        model_sha256=sha(Path(model.__file__)), runner_sha256=sha(Path(__file__)),
        protocol_sha256=sha(Path(__file__).with_name('MZ107_FOUR_SENSOR_PROTOCOL_20260913.md'))))
    # Evaluator and provenance become available only after immutable predictions.
    evaluation = readrows(capture/'evaluator.jsonl'); provenance = readrows(capture/'provenance.jsonl')
    assert [r['id'] for r in rows] == [r['id'] for r in evaluation] == [r['id'] for r in provenance]
    t = np.asarray([truth(e) for e in evaluation], bool)
    b = np.asarray([p['baseline'] for p in predictions], bool)
    c = np.asarray([p['candidate'] for p in predictions], bool)
    families = {}
    for family in sorted({e['family'] for e in evaluation}):
        mask = np.asarray([e['family'] == family for e in evaluation])
        families[family] = dict(baseline=metrics(t[mask], b[mask]), candidate=metrics(t[mask],c[mask]),
                                old_TP_lost=int((t&b&~c&mask).sum()))
    assoc = []; changed = []
    for row, p, e, prov in zip(rows, predictions, evaluation, provenance):
        if p['baseline'] != p['candidate']: changed.append(dict(id=row['id'], truth=truth(e), **p))
        for a in p['associations']:
            if a['state'] != 'ASSOCIATED': continue
            source = prov['radar_slots'][a['slot']]
            # Evaluator-only identity reconstruction: exact native ray endpoints
            # must belong uniquely to the same bounds as this Radar's origin.
            objects = {row['episode_id']+'/'+o['name']:o for o in e['native_bounds']}
            target = objects.get(source.get('actor_id'))
            correct = False
            if target is not None:
                cam = e['camera']; origin = np.array([cam['x'],cam['y'],cam['z']])
                points = [origin+e['tof_native_ranges_m'][k]*model.ray(row['tof64_theta_deg'][k],
                    row['tof64_phi_deg'][k],cam['pitch'],cam['yaw']) for k in a['tof_zones']]
                owners = []
                for pt in points:
                    owners.append([key for key, obj in objects.items()
                        if np.all(pt >= np.asarray(obj['center_m'])-obj['extent_m']-.002)
                        and np.all(pt <= np.asarray(obj['center_m'])+obj['extent_m']+.002)])
                correct = all(owner == [source['actor_id']] for owner in owners)
            assoc.append(dict(id=row['id'], **a, radar_origin=source['kind'],
                              same_native_surface_from_exact_ray_endpoint=bool(correct)))
    write(out/'association-audit.json', assoc); write(out/'changed-frames.json', changed)
    episode_errors = {}
    for name, pred in (('baseline',b),('candidate',c)):
        eps = [np.asarray([r['episode_id']==ep for r in rows]) for ep in sorted({r['episode_id'] for r in rows})]
        episode_errors[name] = dict(false_alarm_episodes=sum(bool((~t&m&pred).any()) for m in eps),
            missed_positive_episodes=sum(bool((t&m).any() and not (t&m&pred).any()) for m in eps))
    passed = int((~t&c).sum()) < int((~t&b).sum()) and not bool((t&b&~c).any())
    identity_correct = sum(a['same_native_surface_from_exact_ray_endpoint'] for a in assoc)
    passed = passed and bool(assoc) and identity_correct == len(assoc)
    summary = dict(status='CANARY_COMPONENT_GAIN' if passed else 'NO_QUALIFIED_GAIN',
        frames=len(rows), scenes=12, baseline=metrics(t,b), candidate=metrics(t,c),
        old_TP_lost=int((t&b&~c).sum()), new_TP_gained=int((t&~b&c).sum()),
        removed_FP=int((~t&b&~c).sum()), added_FP=int((~t&~b&c).sum()),
        families=families, episode_errors=episode_errors, associations=len(assoc),
        identity_supported=identity_correct, unsupported_associations=len(assoc)-identity_correct,
        proposal_frames=sum(bool(p['proposals']) for p in predictions),
        rgb_disabled_exact_baseline=True, independent_tof_preserved=True,
        tof_packet_gaps=sum(not r['tof_packet_received'] for r in rows),
        nonzero_imu_frames=sum(abs(r['delta_yaw'])>0 for r in rows),
        latency_s_per_frame=elapsed/len(rows), latency_scope='preloaded RGB, proposals plus spatial association; excludes image IO/rendering',
        backend=backend['selected_backend'],
        limits=['controlled contrast and simple shapes', 'hypothetical radar, no RF simulation',
                'geometry correspondence audit is not exact sensor identity truth',
                'current corridor only; no future collision claim', 'no CLEAR output'])
    write(out/'summary.json', summary); print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--capture', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path); a = p.parse_args(); run(a.capture,a.output)

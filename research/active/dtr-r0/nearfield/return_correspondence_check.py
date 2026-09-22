"""Perfect angular association with unchanged public range uncertainty."""
from pathlib import Path
import copy
import platform
import sys
import time

import numpy as np

from inherit_spatial_model import canonical_tof, query_bands, QUERIES, FOCAL
from local_support_diagnostic import require_governed
from return_correspondence_data import ray_layout
from query_occupancy_data import read, write, sha
from ba_camera_corridor_metrics import evaluate_rows
from local_transfer_metrics import comparison

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from tools.research_backend import BackendCandidate, DeviceObservation, select_backend

ARMS = ('A_current', 'local', 'exact_interval', 'sampled_interval', 'native_witness')


def interval_counts(native_indices, zone_ids, tof, selected=None):
    """Association indices are privileged; distances and intervals are public."""
    native_indices, zone_ids = np.asarray(native_indices), np.asarray(zone_ids)
    if selected is not None:
        native_indices, zone_ids = native_indices[selected], zone_ids[selected]
    _, valid, _, low, high = canonical_tof(tof)
    assert valid[zone_ids].all(), 'An unobserved zone cannot supply support'
    yy, xx = np.divmod(native_indices, 640)
    ax, ay = (xx+.5-320)/(2*FOCAL), (yy+.5-180)/(2*FOCAL)
    return [int((~query_bands(ax, ay, low[zone_ids], high[zone_ids], q)[2]).sum())
            for q in QUERIES]


def opportunity(rows, metrics, arm):
    rescued = [r for r in rows if r['truth'] is True and
               r['predictions']['local']['alert'] and not r['predictions']['A_current']['alert']]
    false = [r for r in rows if r['truth'] is False and
             r['predictions']['local']['alert'] and not r['predictions']['A_current']['alert']]
    retained = sum(r['predictions'][arm]['alert'] for r in rescued)
    removed = sum(not r['predictions'][arm]['alert'] for r in false)
    comp = comparison(rows, metrics, arm, 'A_current')
    no_a_loss = not comp['lost_true_frames'] and not any(
        e['lost'] or (e['delay_s'] is not None and e['delay_s'] > 1e-9)
        for e in comp['event_differences'])
    return dict(status='PASS' if retained >= 32 and removed == 2 and no_a_loss else 'FAIL',
                original_rescues=len(rescued), retained=retained, original_added_FP=len(false),
                removed=removed, A_preserved=no_a_loss)


def run(repo, result):
    start = time.perf_counter()
    receipt, _ = require_governed(repo, result)
    out = result.parent
    base = repo/'artifacts.local/evidence'
    old = base/'ba-local-support-20260922-run'
    transfer = base/'ba-local-transfer-20260922-prepared'
    rows_path = base/'ba-local-transfer-20260922-evaluated/frame-results.json'
    seal = read(old/'output-seal.json')
    assert sha(old/'input-seal.json') == seal['input_seal_sha256']
    # Primary and lineage were sealed by the preceding independently audited run.
    files = seal.get('files', seal.get('outputs'))
    assert isinstance(files, dict), 'Unexpected source output seal'
    for name, digest in files.items():
        assert sha(old/name) == digest, name
    inputs = {str(old/'output-seal.json'):sha(old/'output-seal.json'), str(rows_path):sha(rows_path)}
    inputs[str(old/'input-seal.json')] = sha(old/'input-seal.json')
    materialization = read(transfer/'materialization.json')
    hashes = materialization['hashes']
    previous_inputs = read(old/'input-seal.json')['input_file_hashes']
    for path in (rows_path, transfer/'materialization.json'):
        assert sha(path) == previous_inputs[path.relative_to(repo).as_posix()]
        inputs[str(path)] = sha(path)
    for name in ('tof.npy','identities.json'):
        path = transfer/'observations'/name
        assert sha(path) == hashes['observations/'+name]
        assert sha(path) == previous_inputs[path.relative_to(repo).as_posix()]
        inputs[str(path)] = sha(path)
    saved = read(rows_path)
    prior = read(old/'frame-diagnostics.json')
    identities = read(transfer/'observations/identities.json')
    tof = np.load(transfer/'observations/tof.npy', mmap_mode='r', allow_pickle=False)
    assert len(saved) == len(prior) == len(identities) == len(tof) == 576
    layout = ray_layout()
    select_backend('batch-tensor', cpu=BackendCandidate('numpy-cpu','cpu',lambda:np.arange(4),
        lambda _:DeviceObservation('cpu',platform.processor() or 'host CPU','numpy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=out/'backend.json',
        capabilities=dict(reason='Ragged saved contributor arrays and scalar evidence accounting; no fitting',
                          python_executable=sys.executable))
    rows = []
    for i, (source, previous, meta) in enumerate(zip(saved, prior, identities)):
        assert source['id'] == previous['id'] == meta['id']
        line_path = old/f'lineage/frame-{i:04d}.npz'
        with np.load(line_path, allow_pickle=False) as line:
            zones = np.repeat(np.arange(64), np.diff(line['zone_offsets']))
            sampled = np.isin(line['sampled_indices'], layout['sensor_indices'])
            exact = interval_counts(line['native_indices'], zones, tof[i])
            small = interval_counts(line['native_indices'], zones, tof[i], sampled)
            _, _, _, low, high = canonical_tof(tof[i])
            assert np.array_equal(low,line['interval_low']) and np.array_equal(high,line['interval_high'])
        native = [q['observed_contributor_points'] for q in previous['queries']]
        winner = (1,4)[int(np.argmax(np.asarray(source['query_probabilities']['local'])[[1,4]]))]
        assert winner == previous['winner_query']
        row = copy.deepcopy(source)
        for arm, counts in [('exact_interval',exact),('sampled_interval',small),('native_witness',native)]:
            flag = bool(source['predictions']['A_current']['alert'] or
                        (source['predictions']['local_standalone']['alert'] and counts[winner] > 0))
            unknown = source['predictions']['A_current']['unknown']
            row['predictions'][arm] = dict(alert=flag,unknown=unknown,ambiguous=bool(flag and unknown))
        row['correspondence'] = dict(winner_query=winner,exact_counts=exact,sampled_counts=small,native_counts=native)
        rows.append(row)
    metrics = evaluate_rows(rows, arms=ARMS)
    # Freeze the native-XYZ parity control; a mismatch is mechanical, not a result.
    assert [metrics['arms']['native_witness']['frames']['all_known'][k] for k in ('TP','FP','FN')] == [224,4,32]
    opportunities = {a:opportunity(rows,metrics,a) for a in ARMS[2:]}
    primary = opportunities['exact_interval']
    pairs = {}
    for row in rows:
        pairs.setdefault(row['appearance_pair_id'], {})[row['appearance']] = row
    assert len(pairs) == 288 and all(set(p) == {'base','changed'} for p in pairs.values())
    pair_changes = {a:dict(pairs=288,
        alert_flips=sum(p['base']['predictions'][a]['alert'] != p['changed']['predictions'][a]['alert'] for p in pairs.values()),
        positive_lost=sum(p['base']['truth'] is True and p['base']['predictions'][a]['alert'] and not p['changed']['predictions'][a]['alert'] for p in pairs.values()),
        positive_gained=sum(p['base']['truth'] is True and not p['base']['predictions'][a]['alert'] and p['changed']['predictions'][a]['alert'] for p in pairs.values())) for a in ARMS}
    write(out/'frame-results.json', rows)
    write(out/'metrics.json', dict(metrics=metrics,opportunity=opportunities,appearance_pairs=pair_changes,
        comparisons={a+'_vs_'+b:comparison(rows,metrics,a,b) for a in ARMS[2:] for b in ('local','A_current')},
        strata={key:{value:evaluate_rows([r for r in rows if r[key]==value],arms=ARMS)
                     for value in sorted({r[key] for r in rows})}
                for key in ('appearance','layer','layout_relation','base_group_id')}))
    write(out/'input-seal.json', dict(files=inputs,source_output_seal=str(old/'output-seal.json'),
                                    all_source_sealed_files_verified=len(files)))
    write(result, dict(status='PASS',decision='OPPORTUNITY_PRESENT_LEARNABILITY_UNTESTED' if primary['status']=='PASS'
        else 'STOP_BEFORE_STUDENT_INTERVAL_READOUT_OPPORTUNITY_FAILED',
        frames=len(rows),opportunity=opportunities,elapsed_s=time.perf_counter()-start,
        no_fit=True,privileged=True,receipt=receipt,
        metrics={a:{**metrics['arms'][a]['frames']['all_known'],
                    'false_segments':metrics['arms'][a]['false_alert_segment_count'],
                    'events_detected':metrics['arms'][a]['detected_events']} for a in ARMS}))
    write(out/'output-seal.json', dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(primary, flush=True)

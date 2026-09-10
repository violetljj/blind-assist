"""CPU-only MZ48 coverage audit: all compact labels, exactly80 declared native samples."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from data_lightweight import CompactSource
from mz50_source import load_source

QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
BOXES = ((.18, .28, .65, 1.4), (.13, .18, 1.4, 1.85))
CROP = (208, 68, 432, 292)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def geometry():
    yy, xx = np.indices((360, 640), dtype=np.float64)
    focal = 320 / np.tan(np.deg2rad(50.))
    right, up = (xx-319.5)/focal, (179.5-yy)/focal
    az, el = np.rad2deg(np.arctan(right)), np.rad2deg(np.arctan(up))
    angular = (abs(az) <= 22.5) & (abs(el) <= 22.5)
    rect = (xx >= CROP[0]) & (xx < CROP[2]) & (yy >= CROP[1]) & (yy < CROP[3])
    assert not (angular & ~rect).any()
    return xx, yy, right, up, np.sqrt(1+right*right+up*up), angular, rect, focal

def event_masks(depth, right, up, factor):
    d = depth.astype(np.float64)
    valid = np.isfinite(d) & (d > 0) & (d < 100) & (d*factor <= 4)
    lateral, height = d*right, 1.7+d*up
    masks = []
    for front, width, low, high in BOXES:
        for half in range(2):
            end = d < front+1.5 if half == 0 else d <= front+3.
            masks.append(valid & (d >= front+half*1.5) & end & (abs(lateral) <= width)
                         & (height >= low) & (height <= high))
    return np.stack(masks)

def counts_for(mask, truth):
    return {q:dict(positive_events=int(truth[:, i].sum()), missing_witness=int(mask[:, i].sum()))
            for i, q in enumerate(QUERIES)}

def group_stats(records, truth, missing):
    result = {'all': counts_for(missing, truth)}
    for field in ('family', 'range', 'site_id', 'relation', 'support_context', 'role'):
        result[field] = {}
        for value in sorted({str(r[field]) for r in records}):
            select = np.array([str(r[field]) == value for r in records])
            result[field][value] = dict(frames=int(select.sum()), queries=counts_for(missing[select], truth[select]))
    result['family_range'] = {}
    for family, distance in sorted({(r['family'], r['range']) for r in records}):
        select = np.array([r['family'] == family and r['range'] == distance for r in records])
        result['family_range'][family+'/'+distance] = counts_for(missing[select], truth[select])
    return result

def run(root, output):
    started = time.perf_counter()
    output = output.resolve()
    assert output.is_relative_to((root/'artifacts.local').resolve())
    output.mkdir(parents=True, exist_ok=False)
    task = root/'artifacts.local/work/mz48-rich-kilotier-20260911'
    index = read(task/'source-index.json')
    admission = read(task/'source-total-receipt.json')
    assert admission['status'] == 'PASS' and admission['frames'] == 2560
    inputs = {str(task/name):sha(task/name) for name in ('source-index.json', 'source-total-receipt.json')}
    for name in ('mz50_source.py', 'data_lightweight.py', 'multizone64_observation.py',
                 'mz9_contributors.py', 'mz48_auxiliary.py', 'mz16_detail_cache.py', 'contact_retina_spec.py'):
        path = Path(__file__).with_name(name); inputs[str(path)] = sha(path)
    write(output/'start.json', dict(status='STARTED', inputs=inputs, code_sha256=sha(__file__),
          native_budget=80, metadata_frames=2560, backend='NumPy CPU only',
          training_steps=0, model_inference_frames=0, captures=0))
    data = load_source(task, index)
    records, truth, known = data['records'], data['truth'], data['known']
    witness = data['cell_event_presence'].any((1, 2))
    positive = truth & known
    missing = positive & ~witness
    assert missing.sum() == 352
    xx, yy, right, up, factor, angular, rect, focal = geometry()
    xmin, xmax = int(xx[angular].min()), int(xx[angular].max())
    ymin, ymax = int(yy[angular].min()), int(yy[angular].max())
    geometry_result = dict(camera=dict(width=640, height=360, hfov_deg=100, eye_height_m=1.7),
        full_vertical_fov_deg=float(np.rad2deg(2*np.arctan(180/focal))),
        angular45=dict(definition='abs(atan(Y/X))<=22.5 and abs(atan(Z/X))<=22.5 degrees',
                       included_center_bounds_xy=[xmin, ymin, xmax, ymax], pixels=int(angular.sum())),
        rectangle224=dict(pil_exclusive_bounds=list(CROP), pixels=int(rect.sum()),
                          outer_edge_fov_deg=float(np.rad2deg(2*np.arctan(112/focal))),
                          extra_pixels_outside_angular45=int((rect & ~angular).sum())),
        threshold='Global/native event >=3 valid pixels; auxiliary cell witness means >=1 valid query pixel, no echo condition')
    audit, event_rows, native_inputs, examples = [], [], [], []
    selected_witness = np.zeros_like(witness)
    refs = data['predictor']['rgb_refs']
    with ExitStack() as stack:
        archives = {name:stack.enter_context(CompactSource(name)) for name in sorted({r.archive for r in refs})}
        oldlabels = {}
        for name, archive in archives.items():
            inputs[name] = sha(name)
            with np.load(io.BytesIO(archive.read_bytes('evaluator/labels.npz')), allow_pickle=False) as labels:
                oldlabels[name] = labels['query_presence'].any((1, 2, 3))
        for i, row in enumerate(records):
            reference = refs[i]
            selected_witness[i] = oldlabels[reference.archive][row['source_index']]
            if not row['native_audit_sample']:
                continue
            archive = archives[reference.archive]
            member = f'evaluator/native/{row["source_index"]:04d}.npy'
            raw = archive.read_bytes(member)
            assert hashlib.sha256(raw).hexdigest() == row['native_sha256']
            depth = np.load(io.BytesIO(raw), allow_pickle=False)
            assert depth.shape == (360, 640) and depth.dtype == np.float32
            masks = event_masks(depth, right, up, factor)
            full = masks.sum((1, 2)); c45 = (masks & angular).sum((1, 2)); c224 = (masks & rect).sum((1, 2))
            np.testing.assert_array_equal(full, row['event_counts'])
            np.testing.assert_array_equal(full >= 3, truth[i])
            np.testing.assert_array_equal(c45 > 0, witness[i])
            item = dict(index=i, frame_id=row['frame_id'], family=row['family'], range=row['range'],
                        site_id=row['site_id'], relation=row['relation'], support_context=row['support_context'],
                        full_counts=full.tolist(), angular45_counts=c45.tolist(), rectangle224_counts=c224.tolist())
            audit.append(item)
            native_inputs.append(dict(frame_id=row['frame_id'], archive=reference.archive, member=member,
                                      sha256=row['native_sha256']))
            for q in np.flatnonzero(full >= 3):
                y, x = np.nonzero(masks[q])
                directions = [name for name, flag in [('left', (x < xmin).any()), ('right', (x > xmax).any()),
                              ('above', (y < ymin).any()), ('below', (y > ymax).any())] if flag] if c45[q] == 0 else []
                event_rows.append(dict(**{k:item[k] for k in ('index','frame_id','family','range','site_id','support_context')},
                    query=QUERIES[q], full_pixels=int(full[q]), angular45_pixels=int(c45[q]), rectangle224_pixels=int(c224[q]),
                    pixel_bbox_inclusive_xy=[int(x.min()),int(y.min()),int(x.max()),int(y.max())],
                    azimuth_minmax_deg=np.rad2deg(np.arctan(right[y,x][[np.argmin(right[y,x]),np.argmax(right[y,x])]])).tolist(),
                    elevation_minmax_deg=np.rad2deg(np.arctan(up[y,x][[np.argmin(up[y,x]),np.argmax(up[y,x])]])).tolist(),
                    no45_witness_directions=directions))
            # Illustrations never choose the statistical sample: first missing-witness
            # audit frame per family, in frozen source order, maximum4 families.
            if missing[i].any() and len(examples) < 4 and row['family'] not in {e['family'] for e in examples}:
                image_name=f'example-{len(examples)+1}-{row["family"]}.png'
                raw_rgb = archive.read_bytes(reference.member)
                assert hashlib.sha256(raw_rgb).hexdigest() == reference.sha256
                (output/image_name).write_bytes(raw_rgb)
                examples.append(dict(**item, image=image_name, rgb_sha256=reference.sha256,
                                     selection='First predeclared audit frame with missing45 witness in this family, frozen order'))
    assert len(audit) == len(native_inputs) == 80
    full = np.array([r['full_counts'] for r in audit]); c45 = np.array([r['angular45_counts'] for r in audit]); c224=np.array([r['rectangle224_counts'] for r in audit])
    positive80 = full >= 3
    def audit_count(mask):
        return dict(total=int(mask.sum()), by_query=dict(zip(QUERIES, mask.sum(0).tolist())))
    sampling = dict(frames=80, selection='Exactly native_audit_sample=True from frozen source; no extra native files',
        full_positive=audit_count(positive80), no_angular45_point=audit_count(positive80 & (c45 == 0)),
        below_angular45_event_threshold=audit_count(positive80 & (c45 < 3)),
        no_rectangle224_point=audit_count(positive80 & (c224 == 0)),
        below_rectangle224_event_threshold=audit_count(positive80 & (c224 < 3)),
        restored_by_rectangle224_at_least1=audit_count(positive80 & (c45 == 0) & (c224 > 0)),
        restored_by_rectangle224_at_least3=audit_count(positive80 & (c45 == 0) & (c224 >= 3)),
        gap_directions=dict(Counter('+'.join(r['no45_witness_directions']) for r in event_rows if r['angular45_pixels'] == 0)),
        audit_group_counts=group_stats([records[r['index']] for r in audit], positive80, positive80 & (c45 == 0)))
    missing_rows = [dict(index=i, frame_id=r['frame_id'], query=QUERIES[q], family=r['family'],
                    range=r['range'], site_id=r['site_id'], relation=r['relation'], support_context=r['support_context'],
                    full_event_pixels=r['event_counts'][q], valid_echo_slots=int(data['predictor']['valid'][i].sum()))
                    for i,r in enumerate(records) for q in range(4) if missing[i,q]]
    result = dict(status='PASS', geometry=geometry_result, full_source=dict(frames=2560, known_bits=int(known.sum()),
        positive_events=int(positive.sum()), missing45_events=int(missing.sum()),
        positive_with45_but_no_selected_echo=int((positive & witness & ~selected_witness).sum()),
        groups=group_stats(records, positive, missing)), audit80=sampling,
        examples=examples, missing_events=missing_rows, native_audit_rows=audit,
        native_positive_event_rows=event_rows, native_inputs=native_inputs,
        limits=['Full RGB native surface support is not proof of RGB-only discrimination or metric depth recovery.',
                '80 predeclared native audit frames are the only directly recomputed depth coverage sample; no extrapolated224 count for other2480.',
                'A zero45 witness is outside the simulated sensor angular domain; other packet echoes may still correlate but do not directly observe that query surface.',
                'No measurement of surfaces outside full camera view or hidden behind visible surfaces; UNKNOWN is not CLEAR.'])
    write(output/'results.json', result)
    for entry in index['shards']:
        for key in ('auxiliary','auxiliary_receipt','dataset_receipt','dataset_result','capture_spec'):
            if key in entry:
                path=task/entry[key]['path']; actual=sha(path); assert actual==entry[key]['sha256']; inputs[str(path)]=actual
    write(output/'receipt.json', dict(status='PASS', inputs=inputs, code_sha256=sha(__file__),
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='receipt.json'},
        native_frames_read=80, rgb_illustrations_extracted=len(examples), metadata_frames=2560,
        backend='NumPy CPU', training_steps=0, model_inference_frames=0, new_captures=0,
        seconds=time.perf_counter()-started, report_pending=True))
    print(json.dumps(dict(status='PASS', geometry=geometry_result, full_source={k:v for k,v in result['full_source'].items() if k!='groups'},audit80={k:v for k,v in sampling.items() if k!='audit_group_counts'},examples=[e['image'] for e in examples])))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.root.resolve(),a.output)

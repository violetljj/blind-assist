"""One resident frozen engine across continuous windows; eval raw outputs sealed.

No reference labels are opened. Train/cal slot alarm counts are descriptions;
eval skips every threshold/grade/notification/count calculation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[4]


def pack_synthesis(source, output, seal_path):
    """Package public CNH frames; incomplete/invalid-pose windows stay NOT_RUN."""
    from collections import defaultdict
    import cnh_counterfactual_common_dev as C
    from sync_rgb_tof_frozen_readout import validate
    seal = C.read(seal_path)
    if not seal['accepted_visit_ids']:
        raise ValueError('No admitted visits')
    grouped = defaultdict(list)
    for frame in C.read(source)['frames']:
        for arm in frame['arms']:
            key = (frame['role'],str(frame['visit_id']),str(frame['capture']),
                   str(frame['window_id']),arm['arm'],arm['repeat'])
            grouped[key].append((frame,arm))
    output.mkdir(parents=True,exist_ok=False)
    rows, skips = [], []
    arm_order={'faro_rho030_ambient1':0,'faro_rho015_ambient1':1,
               'faro_rho060_ambient1':2,'faro_rho030_ambient3':3,'native_circular_upper':4}
    schedule=sorted(grouped.items(),key=lambda item:(arm_order.get(item[0][4],99),
        {'train':0,'cal':1,'eval':2}[item[0][0]],item[0][1:4],item[0][5]))
    for wi,(key,items) in enumerate(schedule):
        items.sort(key=lambda x:x[0]['grid_timestamp_s'])
        desc = dict(role=key[0],visit_id=key[1],capture=key[2],window_id=key[3],arm=key[4],repeat=key[5])
        if len(items)!=16:
            skips.append(dict(desc,status='NOT_RUN',reason='not16frames'))
            continue
        arrays = []
        for frame,arm in items:
            path = Path(arm['path'])
            if C.sha(path)!=arm['sha256']:
                raise ValueError('CNH synthesis payload changed')
            with np.load(path,allow_pickle=False) as z:
                arrays.append({name:z[name] for name in ('hist','ambient','grid_pose','valid')})
        if not all(bool(z['valid']) for z in arrays):
            skips.append(dict(desc,status='NOT_RUN',reason='window has frame without observed source or valid pose'))
            continue
        data = dict(hist=np.array([z['hist'] for z in arrays])[None],
            ambient=np.array([z['ambient'] for z in arrays])[None],
            sensor=np.array([z['grid_pose'] for z in arrays])[None],
            public_query=np.broadcast_to(np.eye(4),(1,16,4,4)).copy(),
            timestamps=np.array([[f['grid_timestamp_s'] for f,_ in items]]))
        try:
            validate(data)
        except ValueError as exc:
            skips.append(dict(desc,status='NOT_RUN',reason=str(exc)))
            continue
        path=output/f'window{wi:03d}.npz'
        np.savez_compressed(path,**data)
        rows.append(dict(desc,path=str(path),sha256=C.sha(path)))
    manifest=output/'windows.json'
    C.save(manifest,dict(rows=rows,skipped=skips,synthesis_manifest_sha256=C.sha(source),
        execution_schedule='Main FARO rho.3 ambient1K2 first; fixed remaining arm order .15,.6,ambient3,native circular; within arm train,cal,eval and chronological window. No outcome-dependent ordering.'))
    return manifest


def run(a):
    import cnh_counterfactual_common_dev as C
    import cnh_counterfactual_eval_dev as E
    import cnh_graded_corridor_eval_dev as Corr
    import cnh_frozen_e2e_tof_20261010 as F
    import cnh_task_cost_train_20261010 as T
    from sync_rgb_tof_frozen_readout import MANIFEST, validate

    def check():
        if time.monotonic() - START >= a.budget_s - 3:
            raise TimeoutError('Frozen batch GPU command-wall allocation reached')

    seal = C.read(a.plan_seal)
    if not seal['accepted_visit_ids']:
        raise ValueError('No admitted visits: inference forbidden')
    windows = C.read(a.windows)['rows']
    if not windows:
        a.output.mkdir(parents=True,exist_ok=False)
        C.save(a.output/'terminal.json',dict(status='NOT_RUN_NO_VALID_WINDOWS',
            windows_sha256=C.sha(a.windows),gpu_command_wall_s=0,
            cpu_preflight_wall_s=time.monotonic()-START,eval_metrics='NOT_COMPUTED'))
        return
    accepted = {str(x) for x in seal['accepted_visit_ids']}
    if any(str(w['visit_id']) not in accepted or w['role'] not in ('train','cal','eval') for w in windows):
        raise ValueError('Window outside sealed accepted roster')
    manifest = C.read(MANIFEST)
    for entry in (manifest['models'] + manifest['features']['implementation']
                  + manifest['strong_source']['frozen_dependencies']):
        if C.sha(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen dependency changed: ' + entry['path'])
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='STARTING', labels_read=False, training=0,
        eval_metrics='NOT_COMPUTED', query_unit='two native HEAD/BODY corridors, not27-query grid',
        hand_held_trajectory_domain_shift=True, manifest_sha256=C.sha(MANIFEST),
        windows_sha256=C.sha(a.windows), seal_sha256=C.sha(a.plan_seal), budget_s=a.budget_s)
    hashes, split_rows, descriptors = {}, {}, []
    try:
        # Separate split names make each trajectory use its own geometry while
        # retaining a single engine across all windows in the frozen implementation.
        for i, row in enumerate(windows):
            check()
            source = Path(row['path'])
            if C.sha(source) != row['sha256']:
                raise ValueError('Window payload changed')
            with np.load(source, allow_pickle=False) as z:
                data = {key:z[key] for key in ('hist','ambient','sensor','public_query','timestamps')}
            if validate(data) != 1:
                raise ValueError('One sequence per window payload required')
            scope = 'sealed_eval' if row['role']=='eval' else row['role']
            split = f'{scope}/window{i:03d}'
            folder = a.output/'data'/split
            folder.mkdir(parents=True)
            np.save(folder/'hist.npy', data['hist'][:,None])
            np.save(folder/'ambient.npy', data['ambient'][0] if data['ambient'].ndim==4 else data['ambient'])
            np.savez_compressed(folder/'geometry.npz',sensor=data['sensor'][0],public_query=data['public_query'][0])
            for name in ('hist.npy','ambient.npy'):
                hashes[f'data/{split}/{name}'] = C.sha(folder/name)
            split_rows[split] = [{}]
            descriptors.append((row,split,data['timestamps'][0,3:].copy()))
        C.save(a.output/'PLAN.json', dict(parent_seal=str(a.plan_seal),sha256=C.sha(a.plan_seal),labels=False))
        C.save(a.output/'render_receipt.json', dict(source='Sealed externally renderedCNH',outputs_sha256=hashes))
        F.S = SimpleNamespace(SPLITS=tuple(split_rows), K=1)
        forward_partial = False
        def forward_check():
            if time.monotonic()-START >= a.budget_s-35:
                raise TimeoutError('Reserve35s for completed-windowS forward and receipts')
        try:
            F.scientific(a.output, split_rows, forward_check)
        except TimeoutError:
            forward_partial = True
        cuts = C.read(F.THRESHOLDS)
        descriptions, predictions = [], []
        for row,split,timestamps in descriptors:
            check()
            folder = a.output/'data'/split
            if not (folder/'scores_receipt.json').exists():
                continue
            with np.load(folder/'scores.npz') as raw:
                ordinary = raw['ordinary_raw']
                current = raw['current'].astype(float)
                valid = raw['current_valid'] & np.isfinite(current)
                # S inference requires ordinary smooth features, including eval.
                # This is a frozen forward operation, not evaluation.
                spatial = np.concatenate((np.where(valid,current,np.nan),(~valid).astype(float)),-1)
                features = np.array([np.concatenate((Corr.build_score_features(ordinary[si],
                    E.smooth(ordinary[si]),cuts[str(seed)]['single']),spatial),-1)
                    for si,seed in enumerate(T.SEEDS)],np.float32)
                scores = T.predict_s(ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010',features)
                mean = scores.mean(0)
                np.savez_compressed(folder/'s_raw_predictions.npz',seed_scores=scores,s_mean=mean,timestamps=timestamps)
                predictions.append(dict(window_id=row['window_id'],role=row['role'],path=str(folder/'s_raw_predictions.npz'),sha256=C.sha(folder/'s_raw_predictions.npz')))
                # No eval thresholding or alarm counts, even transiently.
                if row['role']=='eval':
                    continue
                m3 = E.smooth(raw['m3_raw'])
                old5 = E.old_fusion(m3,E.smooth(raw['local_raw']))
                grade = np.where(old5,2,np.where(mean>=manifest['threshold']['tau'],1,0)).astype(np.int8)
                emitted = F.emit_fast(grade)
                descriptions.append(dict(window_id=row['window_id'],visit_id=row['visit_id'],role=row['role'],arm=row.get('arm'),repeat=row.get('repeat'),
                    frames=13,head_body_m3_alarm_slots=(m3>=E.M3_THETA).sum(axis=(0,1,2)).tolist(),
                    head_body_old5_strong_slots=old5.sum(axis=(0,1,2)).tolist(),
                    head_body_s_light_slots=(grade==1).sum(axis=(0,1,2)).tolist(),
                    head_body_s_any_alarm_slots=(grade>0).sum(axis=(0,1,2)).tolist(),
                    head_body_s_notification_count=(emitted>0).sum(axis=(0,1,2)).tolist()))
        C.save(a.output/'train_cal_description.json',dict(rows=descriptions,eval_metrics='NOT_COMPUTED',tau=manifest['threshold']['tau']))
        C.save(a.output/'prediction_index.json',dict(rows=predictions,eval_metrics='NOT_COMPUTED'))
        receipt.update(status='PARTIAL_BUDGET' if forward_partial else 'COMPLETE',
            windows_forwarded=len(predictions),train_cal_described=len(descriptions),
            unfinished_windows=len(windows)-len(predictions))
    except BaseException as err:
        receipt.update(status='BUDGET_STOP' if isinstance(err,TimeoutError) else 'FAILED',error=repr(err),traceback=traceback.format_exc())
        raise
    finally:
        receipt['gpu_command_wall_s'] = time.monotonic()-START
        C.save(a.output/'terminal.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--windows',type=Path)
    source.add_argument('--synthesis',type=Path)
    p.add_argument('--plan-seal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=200)
    a=p.parse_args()
    if a.synthesis:
        a.windows=pack_synthesis(a.synthesis,a.output.parent/'frozen_inputs',a.plan_seal)
    run(a)

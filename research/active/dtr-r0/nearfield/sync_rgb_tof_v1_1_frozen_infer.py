"""v1.1 fused-FARO observations through the unchanged v1 frozen readout.

No RGB inference, labels, calibration or model changes. Four FARO arms on
train/cal only; eval is input-integrity-only and receives no model forward.
"""
import argparse
import hashlib
import json
from pathlib import Path

import sync_rgb_tof_v1_frozen_infer as base

ARMS = {'faro_rho030_ambient1','faro_rho015_ambient1',
        'faro_rho060_ambient1','faro_rho030_ambient3'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthesis',type=Path,required=True)
    parser.add_argument('--plan-seal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-s',type=float,default=540)
    args=parser.parse_args()
    manifest=json.loads(args.synthesis.read_text(encoding='utf-8-sig'))
    if not manifest.get('frames'):
        raise ValueError('No completed synthesis frames')
    present={arm['arm'] for frame in manifest['frames'] for arm in frame['arms']}
    if not ARMS <= present:
        raise ValueError('v1.1 requires the four frozen FARO arms')
    # Native circular controls remain in the source dataset, but this task's
    # frozen-model run is explicitly restricted to the four FARO observations.
    selection=dict(source_manifest=str(args.synthesis),
        source_sha256=hashlib.sha256(args.synthesis.read_bytes()).hexdigest(),
        selection='FourFARO arms train/cal only; scope clarified before inference, never outcome-selected',
        eval_status='NOT_RUN_INPUT_INTEGRITY_ONLY',
        eval_input_frames_omitted=sum(frame['role']=='eval' for frame in manifest['frames']),
        frames=[dict(frame,arms=[arm for arm in frame['arms'] if arm['arm'] in ARMS])
                for frame in manifest['frames'] if frame['role'] in ('train','cal')])
    selected=args.output.parent/'frozen_synthesis_selection.json'
    with selected.open('x',encoding='utf8') as stream:
        json.dump(selection,stream,indent=2)
    args.windows=base.pack_synthesis(selected,args.output.parent/'frozen_inputs',args.plan_seal)
    base.run(args)
    description=args.output/'train_cal_description.json'
    rows=json.loads(description.read_text())['rows'] if description.exists() else []
    if any(row['role'] not in ('train','cal') for row in rows):
        raise ValueError('Description must never contain eval')
    summaries=[]
    fields=('head_body_m3_alarm_slots','head_body_old5_strong_slots',
            'head_body_s_light_slots','head_body_s_any_alarm_slots')
    for role in ('train','cal'):
        for arm in sorted(ARMS):
            group=[row for row in rows if row['role']==role and row['arm']==arm]
            denominator=13*len(group)
            for q,height in enumerate(('HEAD','BODY')):
                item=dict(role=role,arm=arm,height=height,noise_sequences=len(group),
                    slot_denominator=denominator,status='DESCRIPTIVE_ONLY' if group else 'NOT_RUN')
                for field in fields:
                    count=sum(row[field][q] for row in group)
                    item[field.removeprefix('head_body_')]=count
                    item[field.removeprefix('head_body_')+'_fraction']=count/denominator if denominator else None
                summaries.append(item)
    with (args.output/'train_cal_summary.json').open('x',encoding='utf8') as stream:
        json.dump(dict(rows=summaries,eval='NOT_RUN_INPUT_INTEGRITY_ONLY',
            scope='FusedFARO syntheticCNH, original twoHEAD/BODY corridors; alarm frequency, not accuracy',
            uncertainty='K2 noise repeats are not independent scenes; handheld trajectory domain shift'),stream,indent=2)


if __name__=='__main__':
    main()

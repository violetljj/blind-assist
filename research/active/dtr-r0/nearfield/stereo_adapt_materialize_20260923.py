"""Separate new TRAIN supervision from held-out public inputs and evaluator truth.

The TRAIN native-Z files are authorized training_ground_truth, never a sensor
input. No held-out target is placed inside a training or eval-public bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[4];ART=REPO/'artifacts.local'
BASE=ART/'evidence/ba-stereo-adapt-20260923'
CAPTURE=ART/'evidence/ba-stereo-adapt-20260923-capture-v1'
PREPARED=ART/'evidence/ba-stereo-adapt-20260923-prepared'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def materialize(result):
    journal=read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])
    assert journal['state']=='running' and journal['reuse_preflight']['status']=='PASS'
    assert not PREPARED.exists() or not any(PREPARED.iterdir())
    assert PREPARED.resolve().is_relative_to(ART.resolve())
    receipt=read(CAPTURE/'receipt.json');spec=read(CAPTURE/'spec.json')
    assert receipt['status']=='PASS' and receipt['frames']==288
    assert sha(CAPTURE/'manifest.json')==receipt['hashes']['manifest.json']
    assert sha(CAPTURE/'spec.json')==receipt['spec_sha256']==sha(BASE/'plan/spec.json')
    assert sha(CAPTURE/'mz101_stereo_capture.py')==receipt['source_sha256']
    manifest=read(CAPTURE/'manifest.json');split=read(BASE/'plan/split-manifest.json')
    assert len(manifest['frames'])==len(spec['frames'])==288
    train_ids=set(split['train_ids']);eval_ids=set(split['eval_ids'])
    assert len(train_ids)==192 and len(eval_ids)==96 and not train_ids&eval_ids
    rgb={'train':[],'eval':[]};tof={'train':[],'eval':[]};train=[];eval_public=[];truth=[];native_stats={'train':[],'eval':[]}
    hashes={};actual_geometry_errors=[]
    for planned,actual in zip(spec['frames'],manifest['frames']):
        id=planned['id'];assert id==actual['id']
        partition='train' if id in train_ids else 'eval';assert id in train_ids|eval_ids
        assert partition==planned['split']
        assert planned['camera']==actual['camera']
        for p,a in zip(planned['objects'],actual['native_bounds'],strict=True):
            assert p['name']==a['name']
            delta=max([abs(p['center_m'][j]-a['center_m'][j]) for j in range(3)]+
                [abs(p['size_m'][j]/2-a['extent_m'][j]) for j in range(3)])
            assert delta<=.002;actual_geometry_errors.append(delta)
        public=PREPARED/('train_examples' if partition=='train' else 'eval_observations')
        rgb_row=dict(panel='adapt',id=id);tof_row=dict(panel='adapt',id=id)
        native_record=None
        for name,file in [('left','left.png'),('right','right.png'),('ranges','tof-range.npy'),('valid','tof-valid.npy'),('native_z','native-left-depth.npy')]:
            src=CAPTURE/'frame'/id/file;digest=receipt['hashes'][f'frame/{id}/{file}'];assert sha(src)==digest
            if name=='native_z':
                dest=(public/'training_ground_truth'/id/file if partition=='train' else PREPARED/'eval_truth/native'/id/file)
            else:dest=public/'observations'/id/file
            dest.parent.mkdir(parents=True,exist_ok=True);os.link(src,dest);assert sha(dest)==digest
            hashes[dest.relative_to(PREPARED).as_posix()]=digest
            if name=='native_z':native_record=dict(path=str(dest),sha256=digest,kind='NATIVE_VISIBLE_AXIAL_Z_METRES_UNTRUNCATED')
            else:
                row=rgb_row if name in ('left','right') else tof_row
                row[name]=str(dest);row[name+'_sha256']=digest
        depth=np.load(native_record['path'],allow_pickle=False)
        assert depth.shape==(360,640) and depth.dtype==np.float32
        valid=np.isfinite(depth)&(depth>0)
        native_stats[partition].append(dict(id=id,finite_positive_pixels=int(valid.sum()),far_gt4m_pixels=int((valid&(depth>4)).sum())))
        rgb[partition].append(rgb_row);tof[partition].append(tof_row)
        example=dict(panel='adapt',id=id,layout_id=planned['layout_id'],split=partition,
                family=planned['family'],appearance=planned['appearance'],
                left=rgb_row['left'],left_sha256=rgb_row['left_sha256'],right=rgb_row['right'],right_sha256=rgb_row['right_sha256'],
                ranges=tof_row['ranges'],ranges_sha256=tof_row['ranges_sha256'],valid=tof_row['valid'],valid_sha256=tof_row['valid_sha256'])
        if partition=='train':
            train.append(dict(example,target_depth=native_record['path'],target_depth_sha256=native_record['sha256']))
        else:
            eval_public.append(example)
            truth.append(dict(id=id,layout_id=planned['layout_id'],split='eval',family=planned['family'],appearance=planned['appearance'],
                episode=planned['episode'],time_s=planned['time_s'],target_depth=native_record['path'],target_depth_sha256=native_record['sha256'],
                camera=actual['camera'],native_bounds=actual['native_bounds']))
    assert len(train)==192 and len(truth)==96
    rig=spec['rig']
    for partition,folder in [('train','train_examples'),('eval','eval_observations')]:
        write(PREPARED/folder/'rgb-inputs.json',dict(rig=rig,frames=rgb[partition],authority='RGB_AND_CALIBRATION_ONLY'))
        write(PREPARED/folder/'tof-inputs.json',dict(rig=rig,frames=tof[partition],authority='TOF_AND_CALIBRATION_ONLY'))
    write(PREPARED/'train_examples/examples.json',dict(schema='stereo-adapt-supervised-train-v1',rig=rig,frames=train,
        authority='SUPERVISED_TRAIN_EXAMPLES_ONLY',target_role='training_ground_truth_NOT_SENSOR_INPUT',
        split='train',native_target='Original rendered visible axialZ, retained beyond4m; never learned depth or analytic scene occupancy',
        target_validity='Trainer derives disparity=fB/Z for finite Z>0,0<d<416,u-d>=0; no4m target clipping'))
    write(PREPARED/'eval_truth/truth.json',dict(schema='stereo-adapt-evaluator-only-v1',rig=rig,frames=truth,
        authority='HELD_OUT_DEVELOPMENT_EVALUATOR_ONLY',background=receipt.get('background'),floor=receipt.get('floor')))
    write(PREPARED/'eval_observations/examples.json',dict(schema='stereo-adapt-public-eval-v1',rig=rig,frames=eval_public,
        authority='PUBLIC_RGB_TOF_CALIBRATION_ONLY_NO_TARGETS',split='eval',
        metadata='family/appearance/layout_id are logging/group metadata, never network features or target availability'))
    for path in PREPARED.rglob('*.json'):hashes[path.relative_to(PREPARED).as_posix()]=sha(path)
    write(PREPARED/'receipt.json',dict(status='PASS',train_frames=192,eval_frames=96,train_layouts=48,eval_layouts=24,
        source_receipt_sha256=sha(CAPTURE/'receipt.json'),source_manifest_sha256=sha(CAPTURE/'manifest.json'),
        split_manifest_sha256=sha(BASE/'plan/split-manifest.json'),producer_sha256=sha(__file__),
        max_actual_bound_difference_m=max(actual_geometry_errors),native_stats=native_stats,
        no_eval_targets_in_train=True,no_eval_targets_in_public=True,hashes=hashes))
    write(result,dict(status='PASS',prepared=str(PREPARED),receipt_sha256=sha(PREPARED/'receipt.json'),
        train_frames=192,eval_frames=96,supervised_train_authority='training_ground_truth_NOT_SENSOR_INPUT',
        eval_authority='PUBLIC_RGB_TOF_ONLY_SEPARATE_FROM_EVALUATOR_TRUTH'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
    materialize(p.parse_args().result)

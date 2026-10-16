"""Observation-only IIT->UMD export, then separately invoked mapped-label scoring.

The frozen trainer/evaluator are imported, never edited. Export reads RGB and
label-path metadata, not label pixels. Evaluate is the only GT-reading command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

import full_train_baselines as trainer
import full_evaluate_baselines as evaluator

SHARED = [1, 2, 5, 7, 8, 9]
# UMD native convention. Assert against metadata's declaration if provided.
UMD_TO_IIT = {0: 0, 1: 5, 2: 2, 3: 255, 4: 1, 5: 7, 6: 8, 7: 9}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def unique(items):
    result = {}
    for item in items:
        sid = evaluator.identifier(item)
        if not sid or sid in result:
            raise ValueError('Missing or duplicate source ID')
        result[sid] = item
    return result


class Images(Dataset):
    def __init__(self, observations, size):
        # Discard any incidental evaluator fields before image loader receives them.
        self.items = [(evaluator.identifier(r), r.get('image') or r.get('image_path')) for r in observations]
        self.size = size
        if any(not image for _, image in self.items):
            raise ValueError('Observation must contain image or image_path')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sid, path = self.items[index]
        with Image.open(path) as im:
            im = im.convert('RGB')
            shape = (im.height, im.width)
            im = im.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 255
        x = (x-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None]
        return x, sid, shape


def collate(batch):
    x, sid, shapes = zip(*batch)
    return torch.stack(x), list(sid), list(shapes)


@torch.inference_mode()
def export(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    terminal_path = checkpoint_path.parent / 'terminal.json'
    terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
    if terminal.get('status') != 'COMPLETE':
        raise ValueError('Source training must have terminal COMPLETE before transfer')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = saved['config']
    if config.get('source_sha256') != trainer.sha(trainer.__file__):
        raise ValueError('Frozen trainer source does not match training checkpoint')
    observations = rows(args.observations)
    obs = unique(observations)
    metadata = unique(rows(args.metadata))
    if not obs or set(obs) - set(metadata):
        raise ValueError('Nonempty observations require matching label-path metadata')
    # Validate metadata paths as strings only; never open, stat, hash or decode GT here.
    gt_paths = {}
    for sid in obs:
        meta = metadata[sid]
        path = meta.get('mask') or meta.get('mask_path') or meta.get('gt_path')
        if not isinstance(path, str) or not path:
            raise ValueError(f'Missing label path metadata: {sid}')
        gt_paths[sid] = path
    device = torch.device(args.device)
    model = trainer.build(config, pretrained=False)
    model.load_state_dict(saved['model'], strict=True)
    model.to(device).eval()
    out = Path(args.output)
    (out/'logits').mkdir(parents=True, exist_ok=True)
    manifest = out/'predictions.jsonl'
    if manifest.exists():
        raise FileExistsError('Refuse overwriting transfer manifest; use a fresh output directory')
    receipt = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': trainer.sha(checkpoint_path),
               'training_terminal_sha256': trainer.sha(terminal_path),
               'trainer_source_sha256': trainer.sha(trainer.__file__), 'export_source_sha256': trainer.sha(__file__),
               'observations_sha256': trainer.sha(args.observations), 'metadata_sha256': trainer.sha(args.metadata),
               'input_size': config['size'], 'native_class_ids': list(range(10)),
               'gt_pixels_read': False, 'fine_tuning_on_transfer_data': False,
               'scope': 'Frozen IIT full-image semantic model; no oracle object prompt'}
    trainer.write(out/'config.json', receipt)
    loader = DataLoader(Images(observations, config['size']), batch_size=args.batch, collate_fn=collate, num_workers=0)
    count = 0
    for images, ids, shapes in loader:
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
            logits = model(images.to(device))
        if logits.ndim != 4 or logits.shape[1] != 10 or not torch.isfinite(logits).all():
            raise ValueError('Invalid model output')
        for logit, sid, shape in zip(logits, ids, shapes):
            target = out/'logits'/(hashlib.sha256(sid.encode()).hexdigest()+'.npy')
            np.save(target, logit.float().cpu().numpy())
            meta = metadata[sid]
            trainer.append(manifest, {'source_id': sid, 'split': 'external_transfer', 'image_source': 'UMD',
                           'prediction_path': str(target.resolve()), 'gt_path': gt_paths[sid],
                           'native_class_ids': list(range(10)), 'original_rgb_shape': list(shape),
                           'point_selection': 'softmax_class_probability_spatial_argmax',
                           'resize': 'bilinear_align_corners_false_to_gt', 'supervision_scope': 'full_image',
                           'source_group': meta.get('source_group'), 'umd_to_iit': UMD_TO_IIT,
                           'gt_label_space': meta.get('gt_label_space', 'umd_native_0_7'),
                           'gt_pixels_read_during_export': False})
            count += 1
        trainer.write(out/'progress.json', {'completed':count,'total':len(obs),'stage':'inference'})
    trainer.write(out/'terminal.json', {'status':'COMPLETE','images':count,**receipt,
                                       'manifest_sha256':trainer.sha(manifest)})


def map_umd(gt):
    # The released clutter MAT files contain -1. The author's get_seg_info.m
    # excludes negative labels; -1 is void, not background or another affordance.
    if not np.isin(gt, [-1, *UMD_TO_IIT]).all():
        raise ValueError('Unknown UMD label; do not silently guess mapping')
    mapped = np.full(gt.shape, 255, dtype=np.int64)
    for old, new in UMD_TO_IIT.items():
        mapped[gt==old] = new
    return mapped


def read_transfer_gt(row, *, with_ignore_reasons=False):
    path = Path(row['gt_path'])
    if path.suffix.lower() == '.mat':
        from scipy.io import loadmat
        gt = np.asarray(loadmat(path, variable_names=['gt_label'])['gt_label'])
        if gt.ndim != 2 or not np.isfinite(gt).all() or not np.equal(gt, np.floor(gt)).all():
            raise ValueError('UMD gt_label must be a finite 2-D integer array')
        gt = gt.astype(np.int64)
    else:
        gt = evaluator.load_mask(path)
    space = row.get('gt_label_space', 'umd_native_0_7')
    if space == 'umd_native_0_7':
        mapped = map_umd(gt)
        reasons = {'native_void_minus1':gt==-1, 'unsupported_scoop3':gt==3}
    elif space == 'iit_shared_ids_ignore255':
        if not np.isin(gt, [0, *SHARED, 255]).all():
            raise ValueError('Unexpected already-mapped IIT shared label')
        mapped = gt
        reasons = {'preexisting_iit_ignore255':gt==255}
    else:
        raise ValueError(f'Unrecognized declared label space: {space}')
    return (mapped, reasons) if with_ignore_reasons else mapped


def evaluate(args):
    predictions = rows(args.predictions)
    unique(predictions)
    if not predictions:
        raise ValueError('Empty manifest')
    results=[]; all_ignored=[]; ignored=0
    ignored_pixels_by_reason={}; ignored_points_by_reason={}
    for row in predictions:
        if row.get('split') != 'external_transfer' or row.get('image_source') != 'UMD':
            raise ValueError('Only explicitly exported UMD transfer inputs accepted')
        if row.get('native_class_ids') != list(range(10)):
            raise ValueError('Must retain all ten IIT output channels')
        if {int(k):v for k,v in row['umd_to_iit'].items()} != UMD_TO_IIT:
            raise ValueError('Mapping changed')
        gt, ignore_reasons = read_transfer_gt(row, with_ignore_reasons=True)
        reason_counts={name:int(mask.sum()) for name,mask in ignore_reasons.items()}
        assert sum(reason_counts.values())==int((gt==255).sum())
        for name,count in reason_counts.items():
            ignored_pixels_by_reason[name]=ignored_pixels_by_reason.get(name,0)+count
        ignored += int((gt==255).sum())
        if not (gt!=255).any():
            all_ignored.append(row['source_id']); continue
        pred, scores = evaluator.native_prediction(np.load(row['prediction_path'],allow_pickle=False),gt.shape,list(range(10)),'softmax')
        scored = evaluator.score_arrays(gt,pred,scores,list(range(10)),255,.01)
        reason_points={name:sum(p['hit'] is None and bool(mask[p['point_xy'][1],p['point_xy'][0]])
                                for p in scored['points']) for name,mask in ignore_reasons.items()}
        assert sum(reason_points.values())==sum(p['hit'] is None for p in scored['points'])
        for name,count in reason_points.items():
            ignored_points_by_reason[name]=ignored_points_by_reason.get(name,0)+count
        group=row.get('source_group')
        results.append({'source_id':row['source_id'],'split':'external_transfer','supervision_scope':'full_image',
                        'image_source':'UMD','annotated_object_count':None,
                        'bootstrap_group':str(group) if group else 'image:'+row['source_id'],
                        'source_group_known':bool(group),'ignored_pixels_by_reason':reason_counts,
                        'ignored_point_queries_by_reason':reason_points,**scored})
    if not results:
        raise ValueError('No evaluable pixels after mapping')
    full=evaluator.aggregate(results,list(range(1,10)))
    shared=evaluator.aggregate(results,SHARED)
    points=[p for r in results for p in r['points']]
    summary={'status':'COMPLETE','input_images':len(predictions),'evaluated_images':len(results),
             'all_pixels_ignored_source_ids':all_ignored,'ignored_pixels_total':ignored,
             'ignored_scoop_pixels':ignored_pixels_by_reason.get('unsupported_scoop3',0),
             'ignored_native_void_minus1_pixels':ignored_pixels_by_reason.get('native_void_minus1',0),
             'ignored_pixels_by_reason':ignored_pixels_by_reason,
             'mapping_umd_to_iit':UMD_TO_IIT,'shared_foreground_ids':SHARED,
             'full10_output_mIoU_diagnostic':full,'shared6_foreground_mIoU_primary':shared,
             'point_queries_including_unevaluable':len(points),
             'point_queries_landing_on_any_ignore':sum(p['hit'] is None for p in points),
             'points_landing_on_ignored_scoop':ignored_points_by_reason.get('unsupported_scoop3',0),
             'points_landing_on_native_void_minus1':ignored_points_by_reason.get('native_void_minus1',0),
             'ignored_point_queries_by_reason':ignored_points_by_reason,
             'point_hits_over_all_positive_queries_conservative':sum(p['hit'] is True for p in points)/len(points) if points else None,
             'prediction_policy':'All ten IIT logits retained; no shared-class masking or renormalization',
             'full10_note':'All-output foreground mean over IDs1..9 with union>0 on mapped valid pixels; not complete UMD task truth',
             'shared6_note':'Same full10 predictions; average IoU only over six mapped foreground IDs. Background excluded from primary mIoU.',
             'ignore_note':'UMD native void -1 and unsupported scoop3 map to ignore255, never background; separately counted; point selection never masked by GT',
             'native_void_provenance':'Released author get_seg_info.m excludes pix_labels<0; only observed -1 additionally accepted; other unknown codes rejected',
             'scope':'Cross-source shared semantic labels, not physical-instance part or blind-user task success',
             'bootstrap_shared':evaluator.bootstrap(results,SHARED,args.bootstrap,args.seed),
             'manifest_sha256':trainer.sha(args.predictions),'evaluator_source_sha256':trainer.sha(evaluator.__file__),
             'wrapper_source_sha256':trainer.sha(__file__)}
    out=Path(args.output)
    trainer.write(out/'summary.json',summary)
    with (out/'per_image.jsonl').open('w',encoding='utf-8') as stream:
        for row in results:
            stream.write(json.dumps(row)+'\n')
    trainer.write(out/'terminal.json',{'status':'COMPLETE','images':len(results),'summary_sha256':trainer.sha(out/'summary.json')})


def selftest():
    raw=np.array([[0,1,2,3],[4,5,6,7]])
    gt=map_umd(raw)
    assert gt.tolist()==[[0,5,2,255],[1,7,8,9]]
    logits=np.full((10,2,4),-4,dtype=np.float32)
    for y in range(2):
        for x in range(4):
            logits[0 if gt[y,x]==255 else gt[y,x],y,x]=4
    pred,scores=evaluator.native_prediction(logits,gt.shape,list(range(10)))
    scored=evaluator.score_arrays(gt,pred,scores,list(range(10)))
    agg=evaluator.aggregate([scored],SHARED)
    assert agg['mIoU_foreground']==1 and agg['point_hits']==6 and scored['ignored_pixels']==1
    # An incorrect unsupported-IIT class is still a false negative for shared truth.
    pred[1,0]=3
    wrong=evaluator.aggregate([evaluator.score_arrays(gt,pred,scores,list(range(10)))],SHARED)
    assert wrong['mIoU_foreground'] < 1
    void_raw=np.array([[0,1,2,3,-1],[4,5,6,7,0]],dtype=np.int16)
    void_gt=map_umd(void_raw)
    assert void_gt[0,3]==void_gt[0,4]==255 and void_gt[0,0]==0
    for unknown in (-2,8,255):
        try:map_umd(np.array([[unknown]],dtype=np.int16))
        except ValueError:pass
        else:raise AssertionError(f'Unknown native code silently accepted: {unknown}')
    void_logits=np.full((10,2,5),-4,dtype=np.float32)
    for y in range(2):
        for x in range(5):void_logits[0 if void_gt[y,x]==255 else void_gt[y,x],y,x]=4
    void_logits[5,0,4]=12
    void_pred,void_scores=evaluator.native_prediction(void_logits,void_gt.shape,list(range(10)))
    void_scored=evaluator.score_arrays(void_gt,void_pred,void_scores,list(range(10)))
    point=next(p for p in void_scored['points'] if p['class_id']==5)
    assert point['point_xy']==[4,0] and point['hit'] is None and void_scored['ignored_pixels']==2
    assert sum(map(sum,void_scored['confusion']))==8
    print(json.dumps({'status':'PASS','mapping':True,'scoop_ignored':True,'native_void_ignored':True,
                      'unknown_native_codes_rejected':True,'point_selection_not_GT_masked':True,
                      'shared_metrics':True,'unsupported_prediction_penalized':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('export')
    for name in ['checkpoint','observations','metadata','output']:
        q.add_argument('--'+name,required=True)
    q.add_argument('--device',default='cuda');q.add_argument('--batch',type=int,default=16)
    q=sub.add_parser('evaluate');q.add_argument('--predictions',required=True);q.add_argument('--output',required=True)
    q.add_argument('--bootstrap',type=int,default=1000);q.add_argument('--seed',type=int,default=20261015)
    sub.add_parser('selftest')
    args=p.parse_args();lock=None
    try:
        if hasattr(args,'output'):
            out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
            candidate=out/'owner.lock'
            with candidate.open('x') as stream: stream.write(str(os.getpid()))
            lock=candidate
        if args.command=='export':export(args)
        elif args.command=='evaluate':evaluate(args)
        else:selftest()
    except BaseException:
        if lock:
            trainer.write(Path(args.output)/'terminal.json',{'status':'FAILED','error':traceback.format_exc()})
        raise
    finally:
        if lock:lock.unlink()


if __name__=='__main__':main()

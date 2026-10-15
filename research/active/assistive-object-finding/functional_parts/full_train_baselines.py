#!/usr/bin/env python3
"""Full-image IIT native-label training. No test access; local weights only.

Examples (run from repository, with the existing research Python):
  python full_train_baselines.py selftest
  python full_train_baselines.py train --model segformer --records .../records.jsonl --output .../segformer
  python full_train_baselines.py profile --model dino --records .../records.jsonl --output .../profile_dino --profile-updates 10
  python full_train_baselines.py export --checkpoint .../best.pt --records .../records.jsonl --output .../val_export

Training checkpoints resume at epoch boundaries, including RNG/optimizer/scaler.
An interrupted epoch is replayed from last.pt, never silently skipped. Epochs may
extend 40 -> 80 with --resume; --schedule-epochs fixes the original LR horizon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter

ROOT = Path(__file__).resolve().parents[4]
SEG_WEIGHTS = ROOT / 'artifacts.local/models/nvidia--segformer-b0-finetuned-ade-512-512/snapshots/489d5cd81a0b59fab9b7ea758d3548ebe99677da'
DINO_WEIGHTS = ROOT / 'artifacts.local/work/functional-parts-pilot-20261015/models/AGD20K-Seen.pth'
CLASSES = ['background', 'contain', 'cut', 'display', 'engine', 'grasp', 'hit', 'pound', 'support', 'wrap-grasp']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def append(path, value):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def checkpoint(path, value):
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def read_records(path):
    # Reading a test row's metadata is not opening its image or mask. Filter before Dataset construction.
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    ids = [str(r['source_id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate source_id in records')
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    if not train or not val:
        raise ValueError('Nonempty official train and val required')
    for r in train + val:
        if not r.get('mask') or not Path(r['image']).is_file() or not Path(r['mask']).is_file():
            raise ValueError(f'Missing materialized input: {r["source_id"]}')
    return train, val


def mask_array(path):
    if Path(path).suffix.lower() in ('.txt', '.csv'):
        a = np.loadtxt(path, dtype=np.int64)
    elif Path(path).suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False).astype(np.int64)
    else:
        with Image.open(path) as im:
            a = np.asarray(im).astype(np.int64)
    if a.ndim != 2 or not np.isin(a, np.arange(10)).all():
        raise ValueError(f'Expected native 2-D mask IDs0..9: {path}')
    return a


class IIT(Dataset):
    def __init__(self, records, size, augment=False):
        if any(r['split'] not in ('train', 'val') for r in records):
            raise ValueError('Test access is prohibited in this harness')
        self.records, self.size, self.augment = records, size, augment
        self.jitter = ColorJitter(.15, .15, .1, .02)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row['image']) as im:
            im = im.convert('RGB')
        m = mask_array(row['mask'])
        if (im.height, im.width) != m.shape:
            raise ValueError(f'RGB/mask size mismatch {row["source_id"]}')
        if self.augment and torch.rand(()).item() < .5:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            m = m[:, ::-1].copy()
        if self.augment:
            im = self.jitter(im)
        im = im.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 255
        x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor([.229, .224, .225])[:, None, None]
        if self.augment:
            m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((self.size, self.size), Image.Resampling.NEAREST)).copy()
        return x, torch.from_numpy(m.copy()).long(), row


def collate(batch):
    xs, ys, rows = zip(*batch)
    return torch.stack(xs), list(ys), list(rows)


class DinoDense(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        import timm
        self.encoder = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0, dynamic_img_size=True)
        if weights:
            state = torch.load(weights, map_location='cpu', weights_only=True)
            if 'state_dict' in state:
                state = state['state_dict']
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            state = {k.removeprefix('vit_model.'): v for k, v in state.items() if k.startswith('vit_model.')}
            if not state:
                raise ValueError('No SelectiveCL DINO state found')
            self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False)
        for block in self.encoder.blocks[-4:]:
            block.requires_grad_(True)
        self.encoder.norm.requires_grad_(True)
        self.decoder = nn.Sequential(nn.Conv2d(384, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
                                     nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
                                     nn.Conv2d(128, 10, 1))

    def forward(self, x):
        z = self.encoder.forward_features(x)[:, 1:]
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        return self.decoder(z.transpose(1, 2).reshape(x.shape[0], 384, h, w))


class SegDense(nn.Module):
    def __init__(self, weights):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(str(weights), local_files_only=True)
        # Retain ADE-pretrained encoder+decoder, explicitly reinitialize only ten-class classifier.
        self.net.decode_head.classifier = nn.Conv2d(self.net.config.decoder_hidden_size, 10, 1)
        nn.init.normal_(self.net.decode_head.classifier.weight, std=.02)
        nn.init.zeros_(self.net.decode_head.classifier.bias)
        self.net.config.num_labels = 10

    def forward(self, x):
        return self.net(pixel_values=x).logits


def build(config, pretrained=True):
    if config['model'] == 'dino':
        return DinoDense(config['dino_weights'] if pretrained else None)
    return SegDense(config['seg_weights'])


def dense_loss(logits, y):
    z = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False).float()
    ce = F.cross_entropy(z, y)
    p = z.softmax(1)
    target = F.one_hot(y, 10).permute(0, 3, 1, 2).float()
    # Macro Dice on foreground classes present in this microbatch; not an inferred negative for missing labels.
    mass = target.sum((0, 2, 3))[1:]
    dice = (2 * (p * target).sum((0, 2, 3))[1:] + 1) / (p.sum((0, 2, 3))[1:] + mass + 1)
    dl = (1 - dice[mass > 0]).mean() if (mass > 0).any() else z.sum() * 0
    return ce + dl


def metrics(cm, point_cm):
    cm = np.asarray(cm, dtype=np.float64)
    den = cm.sum(0) + cm.sum(1) - cm.diagonal()
    ious = np.divide(cm.diagonal(), den, out=np.zeros(10), where=den > 0)
    valid = den > 0
    fg = valid.copy(); fg[0] = False
    point_n = int(np.sum(point_cm))
    return {'miou_foreground': float(ious[fg].mean()) if fg.any() else 0.,
            'miou_all': float(ious[valid].mean()) if valid.any() else 0.,
            'per_class_iou': [float(ious[i]) if valid[i] else None for i in range(10)],
            'pixel_confusion': cm.astype(np.int64).tolist(),
            'point_hit': float(np.trace(point_cm) / point_n) if point_n else 0.,
            'point_denominator_positive_image_classes': point_n,
            'point_hits': int(np.trace(point_cm)), 'role_confusion': np.asarray(point_cm).tolist()}


@torch.inference_mode()
def validate(model, loader, device, amp, export=None):
    model.eval()
    cm = np.zeros((10, 10), dtype=np.int64)
    pcm = np.zeros_like(cm)
    count = 0
    if export:
        (export / 'logits').mkdir(parents=True, exist_ok=True)
        manifest = export / 'predictions.jsonl'
        if manifest.exists():
            raise FileExistsError('Refuse duplicate export manifest')
    for xs, masks, rows in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(xs.to(device, non_blocking=True))
        for logit, gt, row in zip(logits, masks, rows):
            z = F.interpolate(logit[None].float(), size=gt.shape, mode='bilinear', align_corners=False)[0]
            pred = z.argmax(0).cpu().numpy()
            g = gt.numpy()
            cm += np.bincount((g * 10 + pred).ravel(), minlength=100).reshape(10, 10)
            # Query-conditioned point: largest softmax class probability, only evaluator selects positive query classes.
            probability = z.softmax(0)
            for cls in np.unique(g):
                if cls:
                    idx = int(probability[int(cls)].argmax())
                    pcm[int(cls), int(g.flat[idx])] += 1
            if export:
                name = hashlib.sha256(str(row['source_id']).encode()).hexdigest()[:20] + '.npy'
                target = export / 'logits' / name
                # Save the small model-output grid losslessly; evaluator resizes in float32.
                np.save(target, logit.float().cpu().numpy())
                append(manifest, {'source_id': row['source_id'], 'split': 'val', 'gt_path': row['mask'],
                                  'prediction_path': str(target.resolve()), 'native_class_ids': list(range(10)),
                                  'native_gt_shape': list(gt.shape), 'logits_dtype': 'float32',
                                  'resize': 'bilinear_align_corners_false_to_gt',
                                  'point_selection': 'softmax_class_probability_spatial_argmax',
                                  'supervision_scope': 'full_image', 'objects': row.get('objects', [])})
            count += 1
    return dict(metrics(cm, pcm), images=count)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def train(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = vars(args).copy()
    config['records_sha256'] = sha(args.records)
    train_rows, val_rows = read_records(args.records)
    config.update(train_images=len(train_rows), val_images=len(val_rows), native_classes=CLASSES,
                  test_access=False, supervision_scope='full_image', checkpoint_selection='validation_foreground_miou',
                  dino_identity='DINO ViT-S/16 in SelectiveCL AGD20K-Seen checkpoint, NOT DINOv2',
                  segformer_identity='SegFormer-B0 ADE20K-finetuned encoder and decoder; new native10-class classifier',
                  normalization='ImageNet RGB mean/std', batchnorm='frozen running statistics',
                  loss='CE + present-foreground macro Dice, equal weights; Dice reduced per physical microbatch',
                  optimizer='AdamW uniform lr, weight_decay=.01, norm_clip=1; poly power .9 floor .01',
                  source_sha256=sha(__file__))
    if args.effective_batch % args.batch:
        raise ValueError('effective_batch must divide by physical batch')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    model = build(config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # Matched optimizer LR for this baseline comparison; do not silently tune one backbone.
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    config.update(trainable_parameters=sum(p.numel() for p in params), total_parameters=sum(p.numel() for p in model.parameters()),
                  pretrained_sha256=sha(Path(args.seg_weights)/'model.safetensors' if args.model=='segformer' else args.dino_weights))
    start, updates, best = 0, 0, -1.
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        for k in ['model', 'size', 'effective_batch', 'batch', 'seed', 'lr', 'schedule_epochs', 'workers', 'val_every', 'source_sha256', 'records_sha256', 'pretrained_sha256']:
            if saved['config'][k] != config[k]:
                raise ValueError(f'Resume input/config changed: {k}')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        start, updates, best = saved['completed_epochs'], saved['updates'], saved['best']
        restore_rng(saved['rng'])
    elif (out/'last.pt').exists():
        raise FileExistsError('Existing run needs explicit --resume')
    write(out/'config.json', config)
    train_data, val_data = IIT(train_rows, args.size, True), IIT(val_rows, args.size)
    loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            collate_fn=collate, pin_memory=device.type=='cuda')
    accumulated = args.effective_batch // args.batch
    total_updates = math.ceil(len(loader) / accumulated) * args.schedule_epochs
    begin = time.perf_counter(); step_times = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    # Initial checkpoint permits exact replay if the first epoch is interrupted.
    def state(epoch):
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(), 'completed_epochs': epoch, 'updates': updates, 'best': best, 'config': config}
    if not args.resume:
        checkpoint(out/'last.pt', state(0))
    for epoch in range(start, args.epochs):
        model.train()
        for layer in model.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, images = 0., 0
        tick = time.perf_counter()
        for batch_index, (xs, ys, _) in enumerate(loader):
            group_start = (batch_index // accumulated) * accumulated
            group_samples = min(args.effective_batch, len(train_data) - group_start * args.batch)
            with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
                logits = model(xs.to(device, non_blocking=True))
                loss = dense_loss(logits, torch.stack(ys).to(device, non_blocking=True))
            scaler.scale(loss * len(xs) / group_samples).backward()
            loss_sum += float(loss.detach()) * len(xs); images += len(xs)
            if (batch_index + 1) % accumulated == 0 or batch_index + 1 == len(loader):
                lr = args.lr * max(.01, (1 - min(updates, total_updates) / total_updates) ** .9)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                updates += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick; step_times.append(elapsed); tick = time.perf_counter()
                if updates % args.log_every == 0 or args.command=='profile':
                    append(out/'progress.jsonl', {'stage':'train', 'epoch':epoch+1, 'updates':updates,
                                                'epoch_images':images, 'train_images':len(train_data),
                                                'loss':loss_sum/images, 'lr':lr, 'update_seconds':elapsed,
                                                'wall_seconds':time.perf_counter()-begin})
                if args.command=='profile' and len(step_times)>=args.profile_updates:
                    result={'status':'PROFILE_COMPLETE', 'optimizer_updates':len(step_times),
                            'seconds_per_update_including_accumulation':step_times,
                            'median_seconds_excluding_first':float(np.median(step_times[1:] or step_times)),
                            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                            'effective_batch':args.effective_batch, 'physical_batch':args.batch,
                            'validation_run':False, 'scientific_result':False}
                    write(out/'terminal.json',result); print(json.dumps(result)); return
        do_val = (epoch+1) % args.val_every == 0 or epoch+1 == args.epochs
        result = validate(model, val_loader, device, device.type=='cuda') if do_val else None
        improved = result is not None and result['miou_foreground'] > best
        if improved:
            best = result['miou_foreground']
        append(out/'progress.jsonl', dict(stage='validation' if do_val else 'epoch', epoch=epoch+1, updates=updates,
                                         train_loss=loss_sum/images, wall_seconds=time.perf_counter()-begin, **(result or {})))
        snapshot = state(epoch+1)
        checkpoint(out/'last.pt',snapshot)
        if improved:
            checkpoint(out/'best.pt',snapshot)
        if result:
            write(out/'latest_validation.json',dict(epoch=epoch+1, best_miou=best, **result))
        print(json.dumps({'epoch':epoch+1,'updates':updates,'val_miou':result['miou_foreground'] if result else None,'best':best}),flush=True)
    write(out/'terminal.json', {'status':'COMPLETE','epochs':args.epochs,'updates':updates,'best_val_miou':best,
                               'wall_seconds':time.perf_counter()-begin,'test_access':False,
                               'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0})


def export(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if sha(args.records)!=saved['config']['records_sha256']:
        raise ValueError('Manifest differs from training')
    _, rows=read_records(args.records)
    model=build(saved['config'], pretrained=False)
    model.load_state_dict(saved['model']); model.to(args.device)
    loader=DataLoader(IIT(rows,saved['config']['size']),batch_size=1,collate_fn=collate)
    result=validate(model,loader,torch.device(args.device),args.device.startswith('cuda'),out)
    write(out/'terminal.json',dict(status='COMPLETE',test_access=False,checkpoint_sha256=sha(args.checkpoint),**result))


def selftest():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    z=torch.randn(2,10,8,8,requires_grad=True)
    y=torch.randint(0,10,(2,16,16))
    loss=dense_loss(z,y); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    cm=np.eye(10,dtype=np.int64); pcm=cm.copy(); pcm[0]=0
    m=metrics(cm,pcm)
    assert m['miou_foreground']==1 and m['point_hit']==1 and m['point_hits']==9
    # An absent class is omitted rather than spuriously scored as perfect.
    zero=metrics(np.zeros((10,10),dtype=np.int64),np.zeros((10,10),dtype=np.int64))
    assert zero['per_class_iou']==[None]*10 and zero['point_denominator_positive_image_classes']==0
    model=DinoDense()
    out=model(torch.randn(1,3,64,64)); dense_loss(out,torch.randint(0,10,(1,64,64))).backward()
    assert out.shape==(1,10,4,4)
    assert model.encoder.blocks[0].attn.qkv.weight.grad is None
    assert model.encoder.blocks[-1].attn.qkv.weight.grad is not None
    print(json.dumps({'status':'PASS','checks':['finite_CE_Dice_backward','metrics_identity','absent_class','DINO_dynamic_grid','DINO_freeze_boundary']}))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('selftest')
    for command in ('train','profile'):
        q=sub.add_parser(command)
        q.add_argument('--records',required=True); q.add_argument('--output',required=True)
        q.add_argument('--model',choices=['segformer','dino'],default='segformer')
        q.add_argument('--seg-weights',default=str(SEG_WEIGHTS)); q.add_argument('--dino-weights',default=str(DINO_WEIGHTS))
        q.add_argument('--size',type=int,default=448); q.add_argument('--batch',type=int,default=2)
        q.add_argument('--effective-batch',type=int,default=16); q.add_argument('--epochs',type=int,default=40)
        q.add_argument('--schedule-epochs',type=int,default=80); q.add_argument('--lr',type=float,default=6e-5)
        q.add_argument('--seed',type=int,default=20261015); q.add_argument('--workers',type=int,default=0)
        q.add_argument('--cpu-threads',type=int,default=4); q.add_argument('--device',default='cuda')
        q.add_argument('--log-every',type=int,default=10); q.add_argument('--resume')
        q.add_argument('--val-every',type=int,default=2)
        q.add_argument('--profile-updates',type=int,default=10)
    q=sub.add_parser('export'); q.add_argument('--checkpoint',required=True); q.add_argument('--records',required=True)
    q.add_argument('--output',required=True); q.add_argument('--device',default='cuda')
    args=p.parse_args()
    lock = None
    owns_lock = False
    try:
        if hasattr(args, 'output'):
            Path(args.output).mkdir(parents=True, exist_ok=True)
            lock = Path(args.output) / 'owner.lock'
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps({'pid': os.getpid(), 'command': args.command}))
            owns_lock = True
        if args.command=='selftest': selftest()
        elif args.command=='export': export(args)
        else: train(args)
    except BaseException:
        if hasattr(args,'output') and owns_lock:
            Path(args.output).mkdir(parents=True,exist_ok=True)
            write(Path(args.output)/'terminal.json',{'status':'FAILED','traceback':traceback.format_exc(),'test_access':False})
        raise
    finally:
        if owns_lock and lock and lock.exists():
            try:
                owner = json.loads(lock.read_text(encoding='utf-8'))
                if owner.get('pid') == os.getpid():
                    lock.unlink()
            except (OSError, ValueError):
                pass


if __name__=='__main__':
    main()

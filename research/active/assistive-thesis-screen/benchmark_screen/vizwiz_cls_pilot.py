"""VizWiz-Classification pilot: how far are modern classifiers from the CVPR 2023 paper's results, and how much do
image-quality flaws still cost? EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md.

Images: the VizWiz-Classification images that sit in the VizWiz validation split, read from the Hugging Face parquet
mirror Multimodal-Fatima/VizWiz_validation (matched by file name). Labels/flaws: official classification_gt.json.
Paper protocol: top-1 over the 1000 ImageNet classes, correct if it is any of the image's labels. Also reported:
argmax restricted to the 200 dataset categories. Clean = no flaw flag except NON; flawed = any flaw.
Usage: vizwiz_cls_pilot.py <vizwiz_cls dir> <out json> [models]
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

TIMM = {'resnet50_tv': 'resnet50.tv_in1k', 'effb0': 'efficientnet_b0.ra_in1k', 'mnv3l': 'mobilenetv3_large_100.ra_in1k',
        'convnext_b_22k': 'convnext_base.fb_in22k_ft_in1k', 'eva02_b_448': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'}
ZS = {'clip_b16': 'openai/clip-vit-base-patch16', 'siglip2_b16': 'google/siglip2-base-patch16-224'}


def load_images(root):
    import pyarrow.parquet as pq
    from PIL import Image
    gt = json.loads((root / 'classification_gt.json').read_text())
    want = {k for k in gt if k.split('_')[1] == 'val'}
    imgs = {}
    for f in sorted((root / 'hf_val').glob('*.parquet')):
        t = pq.read_table(f, columns=['filename', 'image'])
        for name, im in zip(t.column('filename').to_pylist(), t.column('image').to_pylist()):
            n = Path(str(name)).name
            if n in want and n not in imgs:
                imgs[n] = Image.open(io.BytesIO(im['bytes'])).convert('RGB')
    return gt, imgs


def timm_logits(arch, ims):
    import timm
    import torch
    m = timm.create_model(arch, pretrained=True).eval().cuda()
    cfg = timm.data.resolve_data_config({}, model=m); tf = timm.data.create_transform(**cfg)
    out = []
    for k in range(0, len(ims), 32):
        x = torch.stack([tf(i) for i in ims[k:k + 32]]).cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            out.append(m(x).float().cpu().numpy())
    return np.concatenate(out), sum(p.numel() for p in m.parameters())


def zs_logits(hf, ims):
    import timm
    import torch
    from transformers import AutoModel, AutoProcessor
    names = [timm.data.ImageNetInfo().index_to_description(i).split(',')[0] for i in range(1000)]
    m = AutoModel.from_pretrained(hf).eval().cuda(); p = AutoProcessor.from_pretrained(hf)
    prompts = [f'a photo of a {n}.' for n in names]
    with torch.inference_mode():
        t = p(text=prompts, return_tensors='pt', padding='max_length' if 'siglip' in hf else True, truncation=True)
        tf = m.get_text_features(**{k: v.cuda() for k, v in t.items()})
        tf = torch.nn.functional.normalize(tf if torch.is_tensor(tf) else tf.pooler_output, dim=-1)
        out = []
        for k in range(0, len(ims), 32):
            x = p(images=ims[k:k + 32], return_tensors='pt')['pixel_values'].cuda()
            f = m.get_image_features(pixel_values=x)
            f = torch.nn.functional.normalize(f if torch.is_tensor(f) else f.pooler_output, dim=-1)
            out.append((f @ tf.T).float().cpu().numpy())
    return np.concatenate(out), sum(q.numel() for q in m.parameters())


def main(root, out, models='resnet50_tv,effb0,mnv3l,convnext_b_22k,eva02_b_448,clip_b16,siglip2_b16'):
    root = Path(root)
    gt, imgs = load_images(root)
    names = sorted(imgs); ims = [imgs[n] for n in names]
    labels = [set(gt[n]['labels']) for n in names]
    flawed = np.array([any(v for k, v in gt[n]['flaws'].items() if k != 'NON') for n in names])
    cats = sorted({c for l in labels for c in l})
    print('images', len(names), 'of', sum(k.split('_')[1] == 'val' for k in gt), 'val-split items; flawed', int(flawed.sum()), flush=True)
    res = dict(n=len(names), n_flawed=int(flawed.sum()), n_categories=len(cats))
    for name in models.split(','):
        lg, npar = timm_logits(TIMM[name], ims) if name in TIMM else zs_logits(ZS[name], ims)
        p1000 = lg.argmax(1); p200 = np.array(cats)[lg[:, cats].argmax(1)]
        ok1000 = np.array([p in l for p, l in zip(p1000, labels)]); ok200 = np.array([p in l for p, l in zip(p200, labels)])
        r = dict(params_M=round(npar / 1e6, 1))
        for tag, ok in (('top1_1000', ok1000), ('top1_200', ok200)):
            r[tag] = dict(all=round(100 * ok.mean(), 1), clean=round(100 * ok[~flawed].mean(), 1), flawed=round(100 * ok[flawed].mean(), 1))
        res[name] = r
        print(name, json.dumps(r), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])

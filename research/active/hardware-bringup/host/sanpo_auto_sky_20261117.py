"""Evaluator-side automatic sky masks for SANPO frames without human segmentation
(DIRECTION_A_CONFIRM_SANPO_HEAD_20261117.md). SegFormer-B0 ADE20k (local snapshot, class 2 = sky), default 512x512
preprocessing, logits upsampled to 1280x720, argmax == sky. Masks are written as PNG (255 = sky) per frame.

Usage: sanpo_auto_sky_20261117.py <data dir> <camera> <out dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
MODEL = next((ROOT / 'artifacts.local/models/nvidia--segformer-b0-finetuned-ade-512-512/snapshots').iterdir())
W, H, SKY = 1280, 720, 2


def load():
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(MODEL)
    net = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    return proc, net


@torch.no_grad()
def sky_masks(proc, net, imgs_bgr):
    x = proc(images=[cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs_bgr], return_tensors='pt')['pixel_values'].to(net.device)
    lo = torch.nn.functional.interpolate(net(pixel_values=x).logits, size=(H, W), mode='bilinear', align_corners=False)
    return (lo.argmax(1) == SKY).cpu().numpy()


def main(data, cam, out):
    data, out = Path(data), Path(out); proc, net = load(); n = 0
    for sess in sorted(os.listdir(data)):
        base = data / sess / cam / 'left' / 'video_frames'
        if not base.is_dir():
            continue
        frames = sorted(base.glob('*.png'))
        for k in range(0, len(frames), 8):
            batch = frames[k:k + 8]
            imgs = [cv2.resize(cv2.imread(str(p)), (W, H), interpolation=cv2.INTER_AREA) for p in batch]
            for p, m in zip(batch, sky_masks(proc, net, imgs)):
                dst = out / sess / p.name; dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), m.astype(np.uint8) * 255); n += 1
    print('masks', n)


if __name__ == '__main__':
    main(*sys.argv[1:4])

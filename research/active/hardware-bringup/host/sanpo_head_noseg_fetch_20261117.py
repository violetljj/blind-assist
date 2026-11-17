"""SANPO-Real head sessions that were never in any used subset, for the head-worn confirmation
(DIRECTION_A_CONFIRM_SANPO_HEAD_20261117.md). Selection fixed before listing: sessions with no human segmentation on
either camera and not in the local coverage subset, ordered by SHA-256 of the session id; frames 15, 45, 75, ... (every
2 s), at most 16 per session, needing both the left video frame and the CREStereo depth map; stop at >= 1500 frames.
Sky is removed later with host/sanpo_auto_sky_20261117.py.

Usage: sanpo_head_noseg_fetch_20261117.py <inventory.json> <local coverage dir> <out dir> [--download]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sanpo_head_seg_fetch_20261117 import CAP, ROOT, START, STEP, fetch, ls

TARGET = 1500


def one(s):
    base = f'{ROOT}{s}/camera_head/left/'
    items = {i['name']: i for i in ls(base + 'video_frames/') + ls(base + 'depth_maps/') + ls(f'{ROOT}{s}/description.json')}
    frames = sorted(n.split('/')[-1].split('.')[0] for n in items if n.endswith('.png') and '/video_frames/' in n)
    keep = [f for f in frames if int(f) >= START and (int(f) - START) % STEP == 0
            and f'{base}depth_maps/{f}.float16.gz' in items][:CAP]
    if not keep or f'{ROOT}{s}/description.json' not in items:
        return []
    return [items[f'{ROOT}{s}/description.json']] + [items[f'{base}{d}/{f}.{e}'] for f in keep
                                                    for d, e in (('video_frames', 'png'), ('depth_maps', 'float16.gz'))]


def plan(inv, local):
    order = sorted((k for k, v in inv.items() if v['camera_head'] == 0 and v['camera_chest'] == 0 and k not in local),
                   key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    man, n, i = {}, 0, 0
    with ThreadPoolExecutor(6) as ex:
        while n < TARGET and i < len(order):
            chunk = order[i:i + 12]; i += 12
            for s, files in zip(chunk, ex.map(one, chunk)):
                if files and n < TARGET:
                    man[s] = files; n += (len(files) - 1) // 2
    return man


def main(inv_path, local_dir, out_dir, download=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    inv = json.loads(Path(inv_path).read_text()); local = set(os.listdir(local_dir))
    man_path = out / 'manifest.json'
    man = json.loads(man_path.read_text()) if man_path.exists() else plan(inv, local)
    man_path.write_text(json.dumps(man, indent=0))
    files = [i for v in man.values() for i in v]; n_frames = sum((len(v) - 1) // 2 for v in man.values())
    size = sum(int(i['size']) for i in files)
    print(f'sessions {len(man)} frames {n_frames} files {len(files)} bytes {size} ({size / 2**30:.2f} GiB)', flush=True)
    if download:
        t0 = time.time()
        with ThreadPoolExecutor(8) as ex:
            got = sum(ex.map(lambda i: fetch(i, out), files))
        (out / 'RECEIPT.json').write_text(json.dumps(dict(
            source='gs://gresearch/sanpo_dataset/v0/sanpo-real (public bucket, CC BY 4.0)', retrieved=time.strftime('%Y-%m-%d'),
            selection=f'head sessions without any human segmentation, SHA-256 order, frames START={START} STEP={STEP} CAP={CAP}, >= {TARGET}',
            sessions=len(man), frames=n_frames, files=len(files), bytes_downloaded_this_run=got,
            verification='md5 per file against bucket metadata', script=Path(__file__).name), indent=1))
        print('downloaded', got, 'bytes in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main(*sys.argv[1:4], download='--download' in sys.argv)

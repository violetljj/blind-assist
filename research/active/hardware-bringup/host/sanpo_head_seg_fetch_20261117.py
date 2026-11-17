"""SANPO-Real head-camera subset for the direction-A recheck (DIRECTION_A1_RECHECK_SANPO_20261117.md).

Selection (fixed before any download): every session whose camera_head/left has human segmentation masks (needed to
drop sky, where 40% of CREStereo pseudo-depth falls under 4 m on the local 50 frames), minus the sessions already in the
local coverage subset; frames 15, 45, 75, ... (every 2 s at 15 fps), at most 16 per session.
Files per frame: left video_frames PNG, CREStereo depth_maps, segmentation_masks; plus description.json per session.

Usage: sanpo_head_seg_fetch_20261117.py <inventory.json> <local coverage dir> <out dir> [--download] [--camera camera_chest]
--camera (default camera_head) was added 2026-11-17 for the direction-A confirmation on chest sessions
(DIRECTION_A_CONFIRM_SANPO_CHEST_20261117.md); the head selection is unchanged.
Without --download it only writes manifest.json with exact byte counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = 'https://storage.googleapis.com/storage/v1/b/gresearch/o'
MEDIA = 'https://storage.googleapis.com/gresearch/'
ROOT = 'sanpo_dataset/v0/sanpo-real/'
STEP, START, CAP = 30, 15, 16


def _get(url, raw=False):
    for k in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read() if raw else json.load(r)
        except Exception:
            if k == 5:
                raise
            time.sleep(2 * (k + 1))


def ls(prefix):
    tok, out = None, []
    while True:
        q = dict(prefix=prefix, maxResults=1000, fields='items(name,size,md5Hash),nextPageToken')
        if tok:
            q['pageToken'] = tok
        x = _get(API + '?' + urllib.parse.urlencode(q)); out += x.get('items', [])
        tok = x.get('nextPageToken')
        if not tok:
            return out


def plan(inv, local, cam='camera_head'):
    sess = sorted(k for k, v in inv.items() if v[cam] > 0 and k not in local)

    def one(s):
        base = f'{ROOT}{s}/{cam}/left/'
        items = {i['name']: i for i in ls(base) + ls(f'{ROOT}{s}/description.json')}
        frames = sorted(n.split('/')[-1].split('.')[0] for n in items if '/segmentation_masks/' in n and n.endswith('.png'))
        keep = [f for f in frames if int(f) >= START and (int(f) - START) % STEP == 0][:CAP]
        files = [items[f'{ROOT}{s}/description.json']]
        for f in keep:
            names = [f'{base}video_frames/{f}.png', f'{base}depth_maps/{f}.float16.gz', f'{base}segmentation_masks/{f}.png']
            if all(n in items for n in names):
                files += [items[n] for n in names]
        return s, files

    with ThreadPoolExecutor(6) as ex:
        return dict(ex.map(one, sess))


def fetch(item, out):
    dst = out / item['name'][len(ROOT):]
    if dst.exists() and dst.stat().st_size == int(item['size']):
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = _get(MEDIA + urllib.parse.quote(item['name']), raw=True)
    import base64
    if base64.b64encode(hashlib.md5(data).digest()).decode() != item['md5Hash']:
        raise RuntimeError('md5 mismatch ' + item['name'])
    tmp = dst.with_suffix(dst.suffix + '.part'); tmp.write_bytes(data); os.replace(tmp, dst)
    return len(data)


def main(inv_path, local_dir, out_dir, download=False, cam='camera_head'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    inv = json.loads(Path(inv_path).read_text()); local = set(os.listdir(local_dir))
    man_path = out / 'manifest.json'
    man = json.loads(man_path.read_text()) if man_path.exists() else plan(inv, local, cam)
    man_path.write_text(json.dumps(man, indent=0))
    files = [i for v in man.values() for i in v]
    n_frames = sum((len(v) - 1) // 3 for v in man.values())
    print(f'sessions {len(man)} frames {n_frames} files {len(files)} bytes {sum(int(i["size"]) for i in files)} '
          f'({sum(int(i["size"]) for i in files) / 2**30:.2f} GiB)', flush=True)
    if download:
        t0 = time.time()
        with ThreadPoolExecutor(8) as ex:
            got = sum(ex.map(lambda i: fetch(i, out), files))
        (out / 'RECEIPT.json').write_text(json.dumps(dict(
            source='gs://gresearch/sanpo_dataset/v0/sanpo-real (public bucket, CC BY 4.0)', retrieved=time.strftime('%Y-%m-%d'),
            selection=f'{cam} sessions with segmentation, minus local coverage; frames START={START} STEP={STEP} CAP={CAP}',
            sessions=len(man), frames=n_frames, files=len(files), bytes_downloaded_this_run=got,
            verification='md5 per file against bucket metadata', script=Path(__file__).name), indent=1))
        print('downloaded', got, 'bytes in', round(time.time() - t0), 's')


if __name__ == '__main__':
    a = sys.argv
    main(*a[1:4], download='--download' in a, cam=a[a.index('--camera') + 1] if '--camera' in a else 'camera_head')

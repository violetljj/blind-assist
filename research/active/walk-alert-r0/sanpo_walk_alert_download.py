"""Download depth/segmentation/poses (no RGB) for up to 300 (capped at 8 GB) extra SANPO-Synthetic *train* sessions (walk-alert-r0, 2026-11-16).

Selection is fixed before download: official train_session_ids.txt, minus the 60 sessions of the 2026-09-19 BA-NFO
selection (train/val/test, so nothing previously chosen is reused), sorted by sha256('walk-alert-r0:'+id), first 300.
The official test split is never listed or read. Skipped: left/video_frames, left/temp.
Each file is checked against GCS size and md5Hash; RECEIPT.json records url prefix, selection, per-session totals.
Usage: sanpo_walk_alert_download.py [--dry-run]
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts.local/downloads/sanpo-synthetic-walk-alert-20261116'
OLD = ROOT / 'artifacts.local/work/ba-nfo-20260919/sanpo-selection.json'
API = 'https://storage.googleapis.com/storage/v1/b/gresearch/o'
BASE = 'https://storage.googleapis.com/gresearch/'
PRE = 'sanpo_dataset/v0/sanpo-synthetic/'
N, SALT, BYTE_CAP = 300, 'walk-alert-r0:', 8.0e9
SKIP = ('/left/video_frames/', '/left/temp/')


def get(url, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def listing(prefix):
    out, tok = [], None
    while True:
        q = dict(prefix=prefix, maxResults=1000, fields='items(name,size,md5Hash),nextPageToken')
        if tok:
            q['pageToken'] = tok
        d = json.loads(get(API + '?' + urllib.parse.urlencode(q)))
        out += d.get('items', []); tok = d.get('nextPageToken')
        if not tok:
            return out


def fetch(item):
    rel = item['name'][len(PRE):]; dst = OUT / rel
    want = base64.b64decode(item['md5Hash']).hex()
    if dst.exists() and dst.stat().st_size == int(item['size']) and hashlib.md5(dst.read_bytes()).hexdigest() == want:
        return int(item['size']), 0
    data = get(BASE + urllib.parse.quote(item['name']))
    if len(data) != int(item['size']) or hashlib.md5(data).hexdigest() != want:
        raise ValueError(f'checksum mismatch {rel}')
    dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
    return len(data), 1


def main(dry):
    train = get(BASE + PRE + 'splits/train_session_ids.txt').decode().split()
    old = {s['scene'] for s in json.loads(OLD.read_text())['sessions']}
    pool = sorted((s for s in train if s not in old), key=lambda s: hashlib.sha256((SALT + s).encode()).hexdigest())
    chosen = pool[:N]
    with ThreadPoolExecutor(8) as ex:
        lists = [[i for i in L if not i['name'].endswith('_$folder$') and not any(k in i['name'] for k in SKIP)
                  and int(i['size']) > 0] for L in ex.map(lambda s: listing(PRE + s + '/'), chosen)]
    # User approved 8 GB: keep the longest prefix of the fixed order whose cumulative size fits the cap.
    keep, acc = 0, 0
    for L in lists:
        b = sum(int(i['size']) for i in L)
        if acc + b > BYTE_CAP:
            break
        acc += b; keep += 1
    chosen, lists = chosen[:keep], lists[:keep]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'selection.json').write_text(json.dumps(dict(rule=__doc__.split('\n')[2:5], byte_cap=BYTE_CAP, train_total=len(train),
                                                        excluded_prior=len(old & set(train)), sessions=chosen), indent=1))
    items = [i for L in lists for i in L]
    total = sum(int(i['size']) for i in items)
    print(f'{len(chosen)} sessions, {len(items)} files, {total/1e9:.2f} GB', flush=True)
    if dry:
        return
    t0, done, got = time.time(), 0, 0
    with ThreadPoolExecutor(16) as ex:
        for k, (n, new) in enumerate(ex.map(fetch, items), 1):
            done += n; got += new
            if k % 2000 == 0:
                print(f'{k}/{len(items)} files {done/1e9:.2f} GB {time.time()-t0:.0f}s', flush=True)
    per = {}
    for s, L in zip(chosen, lists):
        per[s] = dict(files=len(L), bytes=sum(int(i['size']) for i in L),
                      depth_frames=sum('/depth_maps/' in i['name'] for i in L))
    (OUT / 'RECEIPT.json').write_text(json.dumps(dict(
        source=BASE + PRE, retrieved=time.strftime('%Y-%m-%d'), license='CC BY 4.0 (SANPO); local research use',
        sessions=len(chosen), files=len(items), bytes=total, newly_downloaded=got, verified='size+md5 per file vs GCS',
        skipped_subdirs=list(SKIP), per_session=per), indent=1))
    print(f'done {total/1e9:.2f} GB in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main('--dry-run' in sys.argv)

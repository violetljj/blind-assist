"""Fetch only the ORBIT frames a CLU-VE pilot needs from the official figshare ZIP with parallel HTTP range requests
(figshare/S3 gives ~10 KB/s per connection here, so the 4.15 GB validation ZIP is not downloaded whole).
EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md.

  fetch_orbit_frames.py index  <out dir> <figshare file id>      -> <out>/zip_index.json (name -> offset, size, method)
  fetch_orbit_frames.py select <out dir> <annotations split dir> [context_every=30] [target_n=200] [seed=0]
  fetch_orbit_frames.py fetch  <out dir> <figshare file id> [workers=32]
Selection follows the ORBIT CLU-VE rules for target frames (drop object_not_present frames, skip clutter videos with
< 50 valid frames, else up to 200 random frames); context = every `context_every`-th frame of each clean video.
"""
from __future__ import annotations

import json
import random
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = 'https://ndownloader.figshare.com/files/{}'


def get_range(fid, a, b, tries=8):
    for t in range(tries):
        try:
            req = urllib.request.Request(URL.format(fid), headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except Exception:
            time.sleep(1 + t)
    raise IOError(f'range {a}-{b} failed')


def total_size(fid):
    req = urllib.request.Request(URL.format(fid), headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers['Content-Range'].split('/')[1])


def index(out, fid, workers=32):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    n = total_size(fid)
    tail = get_range(fid, n - 65536, n - 1)
    e = tail.rfind(b'PK\x05\x06')
    cd_size, cd_off = struct.unpack('<II', tail[e + 12:e + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:                       # ZIP64
        l = tail.rfind(b'PK\x06\x07')
        z64 = struct.unpack('<Q', tail[l + 8:l + 16])[0]
        rec = get_range(fid, z64, z64 + 55)
        cd_size, cd_off = struct.unpack('<QQ', rec[40:56])
    chunk = 1 << 20
    parts = [(a, min(a + chunk, cd_off + cd_size) - 1) for a in range(cd_off, cd_off + cd_size, chunk)]
    print('zip', n, 'central dir', cd_size, 'bytes in', len(parts), 'ranges', flush=True)
    with ThreadPoolExecutor(workers) as ex:
        cd = b''.join(ex.map(lambda p: get_range(fid, *p), parts))
    idx, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[p + 10:p + 12])
        crc, csize, usize, nlen, xlen, clen = struct.unpack('<IIIHHH', cd[p + 16:p + 34])
        off, = struct.unpack('<I', cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode('utf8')
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, off):                              # ZIP64 extra field
            q = 0
            while q < len(extra):
                hid, hlen = struct.unpack('<HH', extra[q:q + 4])
                if hid == 1:
                    vals, k = [], q + 4
                    for f in (usize, csize, off):
                        if f == 0xFFFFFFFF:
                            vals.append(struct.unpack('<Q', extra[k:k + 8])[0]); k += 8
                        else:
                            vals.append(f)
                    usize, csize, off = vals
                q += 4 + hlen
        if not name.endswith('/'):
            idx[name] = [off, csize, method, crc]
        p += 46 + nlen + xlen + clen
    (out / 'zip_index.json').write_text(json.dumps(dict(fid=fid, size=n, files=idx)))
    print('indexed', len(idx), 'files', flush=True)


def select(out, ann_dir, context_every=30, target_n=200, seed=0):
    out, ann_dir = Path(out), Path(ann_dir)
    files = json.loads((out / 'zip_index.json').read_text())['files']
    by_base = {Path(k).name: k for k in files}
    rng = random.Random(int(seed)); pick, videos = [], []
    for f in sorted(ann_dir.glob('*.json')):
        user, obj, vtype, vid = f.stem.split('--', 3)
        ann = json.loads(f.read_text()); frames = sorted(ann)
        if vtype == 'clean':
            chosen = frames[::int(context_every)]
        else:
            valid = [x for x in frames if not ann[x]['object_not_present_issue']]
            if len(valid) < 50:
                videos.append(dict(user=user, obj=obj, type=vtype, video=f.stem, frames=[], skipped=True)); continue
            chosen = sorted(rng.sample(valid, min(int(target_n), len(valid))))
        missing = [x for x in chosen if x not in by_base]
        if missing:
            raise KeyError(f'{len(missing)} frames of {f.stem} not in zip, e.g. {missing[0]}')
        pick += [by_base[x] for x in chosen]
        videos.append(dict(user=user, obj=obj, type=vtype, video=f.stem, frames=chosen,
                           boxes=[ann[x].get('object_bounding_box') for x in chosen] if vtype == 'clutter' else None))
    (out / 'selection.json').write_text(json.dumps(dict(context_every=int(context_every), target_n=int(target_n),
                                                        seed=int(seed), videos=videos)))
    nbytes = sum(files[k][1] for k in pick)
    print('selected', len(pick), 'frames from', len(videos), 'videos,', round(nbytes / 1e6, 1), 'MB', flush=True)
    (out / 'to_fetch.json').write_text(json.dumps(pick))


def fetch(out, fid, workers=32):
    out = Path(out); files = json.loads((out / 'zip_index.json').read_text())['files']
    pick = json.loads((out / 'to_fetch.json').read_text())
    dst = out / 'frames'; dst.mkdir(exist_ok=True)
    todo = [k for k in pick if not (dst / Path(k).name).exists()]
    print('to fetch', len(todo), flush=True)
    t0, done = time.time(), [0]

    def one(k):
        off, csize, method, crc = files[k]
        blob = get_range(fid, off, off + 30 + 512 + csize - 1)               # local header + name/extra slack + data
        nlen, xlen = struct.unpack('<HH', blob[26:30])
        raw = blob[30 + nlen + xlen:30 + nlen + xlen + csize]
        if len(raw) < csize:
            raw += get_range(fid, off + 30 + nlen + xlen + len(raw), off + 30 + nlen + xlen + csize - 1)
        data = raw if method == 0 else zlib.decompress(raw, -15)
        if zlib.crc32(data) & 0xFFFFFFFF != crc:
            raise ValueError(f'CRC mismatch {k}')
        (dst / Path(k).name).write_bytes(data)
        done[0] += 1
        if done[0] % 500 == 0:
            print(done[0], 'frames', round(time.time() - t0), 's', flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(one, todo))
    print('fetched', len(todo), 'in', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    dict(index=index, select=select, fetch=fetch)[sys.argv[1]](*sys.argv[2:])

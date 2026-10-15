"""Fetch the ORBIT test-user frames needed for an official-protocol CLU-VE evaluation from the figshare test_224 ZIP.
EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md. Uses figshare_ranges.Reader (persistent tunnelled connections,
reused presigned URLs): ~25 small ranges/s or ~2.4 MB/s in 2 MB ranges here, versus ~5 ranges/s before.

  index   <out> <fid>                         central directory in 2 MB parallel ranges -> zip_index.json
  plan    <out> <ann dir> [context_every=10]  clutter videos: whole contiguous span (all frames, so the official
                                              50 x 200-frame task sampling can be replayed); clean videos: every
                                              `context_every`-th frame (every 30th = the starter-notebook subset)
  fetch   <out> <fid> [workers=64]            spans in 2 MB chunks + single context frames; CRC-checked extraction
"""
from __future__ import annotations

import json
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figshare_ranges import reader, signed_url  # noqa: E402

CH = 2_000_000


def total_size(fid):
    import urllib.request
    req = urllib.request.Request(signed_url(fid), headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers['Content-Range'].split('/')[1])


def index(out, fid, workers=64):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); fid = int(fid)
    n = total_size(fid); tail = reader(fid).read(n - 65536, n - 1)
    e = tail.rfind(b'PK\x05\x06'); cd_size, cd_off = struct.unpack('<II', tail[e + 12:e + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        l = tail.rfind(b'PK\x06\x07'); z64 = struct.unpack('<Q', tail[l + 8:l + 16])[0]
        rec = reader(fid).read(z64, z64 + 55); cd_size, cd_off = struct.unpack('<QQ', rec[40:56])
    parts = [(a, min(a + CH, cd_off + cd_size) - 1) for a in range(cd_off, cd_off + cd_size, CH)]
    t0 = time.time()
    with ThreadPoolExecutor(int(workers)) as ex:
        cd = b''.join(ex.map(lambda p: reader(fid).read(*p), parts))
    print('central dir', cd_size, 'B in', round(time.time() - t0), 's', flush=True)
    idx, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[p + 10:p + 12])
        crc, csize, usize, nlen, xlen, clen = struct.unpack('<IIIHHH', cd[p + 16:p + 34])
        off, = struct.unpack('<I', cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode('utf8'); extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, off):
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


def plan(out, ann_dir, context_every=10):
    out, ann_dir = Path(out), Path(ann_dir)
    files = json.loads((out / 'zip_index.json').read_text())['files']
    by_video = {}
    for k in files:
        by_video.setdefault(Path(k).parent.name, []).append(k)
    spans, singles, videos = [], [], []
    for f in sorted(ann_dir.glob('*.json')):
        user, obj, vtype, vid = f.stem.split('--', 3)
        ann = json.loads(f.read_text()); members = sorted(by_video[f.stem])
        if vtype == 'clutter':
            offs = sorted(files[m][0] for m in members); last = max(members, key=lambda m: files[m][0])
            a, b = offs[0], files[last][0] + 30 + 1024 + files[last][1]
            body = sum(files[m][1] for m in members)
            spans.append(dict(video=f.stem, start=a, end=b, members=members, overhead=round((b - a) / max(body, 1), 3)))
            frames = sorted(ann)
        else:
            frames = sorted(ann)[::int(context_every)]
            base = {Path(m).name: m for m in members}
            singles += [base[x] for x in frames]
        videos.append(dict(user=user, obj=obj, type=vtype, video=f.stem, n_frames=len(ann),
                           valid=[x for x in sorted(ann) if not ann[x].get('object_not_present_issue', False)] if vtype == 'clutter' else None,
                           boxes={x: ann[x].get('object_bounding_box') for x in sorted(ann)} if vtype == 'clutter' else None,
                           context=frames if vtype == 'clean' else None))
    span_bytes = sum(s['end'] - s['start'] for s in spans); single_bytes = sum(files[k][1] for k in singles)
    (out / 'plan.json').write_text(json.dumps(dict(context_every=int(context_every), spans=spans, singles=singles, videos=videos)))
    print(f'{len(spans)} clutter spans {span_bytes / 1e9:.2f} GB (max overhead {max(s["overhead"] for s in spans)}), '
          f'{len(singles)} context frames {single_bytes / 1e6:.0f} MB', flush=True)


def extract(blob, base, member, files):
    off, csize, method, crc = files[member]
    p = off - base; nlen, xlen = struct.unpack('<HH', blob[p + 26:p + 30])
    raw = blob[p + 30 + nlen + xlen:p + 30 + nlen + xlen + csize]
    data = raw if method == 0 else zlib.decompress(raw, -15)
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        raise ValueError(f'CRC mismatch {member}')
    return data


def fetch(out, fid, workers=64):
    out = Path(out); fid = int(fid); workers = int(workers)
    files = json.loads((out / 'zip_index.json').read_text())['files']; P = json.loads((out / 'plan.json').read_text())
    dst = out / 'frames'; dst.mkdir(exist_ok=True)
    t0 = time.time(); lock = threading.Lock(); stat = dict(bytes=0, done=0)

    def span_job(s):
        if all((dst / Path(m).name).exists() for m in s['members']):
            return 0
        a, b = s['start'], s['end']
        blob = b''.join(reader(fid).read(x, min(x + CH, b) - 1) for x in range(a, b, CH))
        for m in s['members']:
            (dst / Path(m).name).write_bytes(extract(blob, a, m, files))
        with lock:
            stat['bytes'] += len(blob); stat['done'] += 1
            if stat['done'] % 20 == 0:
                print(f"spans {stat['done']}/{len(P['spans'])}, {stat['bytes'] / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
        return len(blob)

    def single_job(m):
        if (dst / Path(m).name).exists():
            return 0
        off, csize, _, _ = files[m]
        blob = reader(fid).read(off, off + 30 + 1024 + csize - 1)
        (dst / Path(m).name).write_bytes(extract(blob, off, m, files))
        with lock:
            stat['done'] += 1
            if stat['done'] % 2000 == 0:
                print(f"singles {stat['done']}, {time.time() - t0:.0f} s", flush=True)
        return len(blob)

    # big spans first on half the workers, small context frames on the other half, concurrently
    with ThreadPoolExecutor(workers // 2) as ex_s, ThreadPoolExecutor(workers - workers // 2) as ex_c:
        fs = ex_s.map(span_job, sorted(P['spans'], key=lambda s: s['start'] - s['end']))
        fc = ex_c.map(single_job, P['singles'])
        nb = sum(fs) + sum(fc)
    print('fetched', round(nb / 1e9, 2), 'GB in', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    dict(index=index, plan=plan, fetch=fetch)[sys.argv[1]](*sys.argv[2:])

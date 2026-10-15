"""Fetch a seeded subset of VizWiz-Classification images straight out of the official VizWiz image ZIPs with HTTP range
requests (no full-archive download). EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md.
Usage: fetch_vizwiz_subset.py <out dir with classification_gt.json> <n> [seed]
"""
from __future__ import annotations

import io
import json
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = 'https://vizwiz.cs.colorado.edu/VizWiz_final/images/{}.zip'


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small read-ahead cache for the ZIP directory."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method='HEAD')
        self.size = int(urllib.request.urlopen(req, timeout=60).headers['Content-Length'])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b''
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(req, timeout=120).read()
                break
            except Exception:
                if attempt == 4:
                    raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def main(out, n, seed=0):
    out = Path(out); n = int(n)
    gt = json.loads((out / 'classification_gt.json').read_text())
    names = sorted(gt); random.Random(int(seed)).shuffle(names); pick = sorted(names[:n])
    (out / f'subset_{n}_seed{seed}.json').write_text(json.dumps(pick))
    img = out / 'images'; img.mkdir(exist_ok=True)
    total = 0
    for split in ('train', 'val', 'test'):
        want = [p for p in pick if p.split('_')[1] == split and not (img / p).exists()]
        if not want:
            continue
        zf = zipfile.ZipFile(io.BufferedReader(HttpFile(BASE.format(split)), buffer_size=1 << 20))
        index = {Path(i.filename).name: i for i in zf.infolist()}

        url = BASE.format(split)

        def get(name):
            # read the local header + payload directly (one range request each); CRC checked after inflating
            import struct
            import zlib
            info = index[name]
            f = HttpFile.__new__(HttpFile); f.url, f.size, f.pos = url, zf.fp.raw.size, info.header_offset
            head = f.read(30)
            nlen, xlen = struct.unpack('<HH', head[26:30])
            f.seek(info.header_offset + 30 + nlen + xlen)
            raw = f.read(info.compress_size)
            data = raw if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
            if zlib.crc32(data) != info.CRC:
                raise ValueError(f'CRC mismatch {name}')
            (img / name).write_bytes(data)
            return len(data)
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(get, want))
        total += sum(got)
        print(split, len(want), 'images', round(sum(got) / 1e6, 1), 'MB', flush=True)
    print('done; new bytes', total)


if __name__ == '__main__':
    main(*sys.argv[1:])

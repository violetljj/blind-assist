"""Fast range reads from a figshare file: each worker keeps a persistent HTTPS connection to the S3 host and reuses the
presigned URL until shortly before its 10 s expiry (one ndownloader round trip per refresh), instead of one redirect +
new TLS handshake per range. EXPLORE helper, 2026-10-15.
  figshare_ranges.py bench <file id> [workers=64] [seconds=20] [bytes=20000]
"""
from __future__ import annotations

import http.client
import os
import sys
import threading
import time
import urllib.parse

NDL = 'ndownloader.figshare.com'
_PROXY = urllib.parse.urlsplit(os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or '')


def connect(host, timeout=120):
    """HTTPS connection to host, tunnelled through HTTPS_PROXY when one is set."""
    if _PROXY.hostname:
        c = http.client.HTTPSConnection(_PROXY.hostname, _PROXY.port or 80, timeout=timeout)
        c.set_tunnel(host, 443)
        return c
    return http.client.HTTPSConnection(host, timeout=timeout)


def signed_url(fid):
    c = connect(NDL, 60)
    c.request('GET', f'/files/{fid}', headers={'Range': 'bytes=0-0'})
    r = c.getresponse(); loc = r.getheader('Location'); r.read(); c.close()
    if not loc:
        raise IOError(f'no redirect ({r.status})')
    return loc


class Reader:
    """One per thread: persistent connection + cached presigned URL."""

    def __init__(self, fid, ttl=7.0):
        self.fid, self.ttl, self.url, self.t, self.conn, self.host = fid, ttl, None, 0., None, None

    def _refresh(self):
        self.url = signed_url(self.fid); self.t = time.time()
        u = urllib.parse.urlsplit(self.url)
        if u.netloc != self.host or self.conn is None:
            if self.conn:
                self.conn.close()
            self.host = u.netloc; self.conn = connect(u.netloc)
        self.path = u.path + '?' + u.query

    def read(self, a, b, tries=8):
        for k in range(tries):
            try:
                if self.url is None or time.time() - self.t > self.ttl:
                    self._refresh()
                self.conn.request('GET', self.path, headers={'Range': f'bytes={a}-{b}'})
                r = self.conn.getresponse(); data = r.read()
                if r.status == 206 and len(data) == b - a + 1:
                    return data
                if r.status in (400, 403):
                    self.url = None                       # expired signature: refresh and retry
                    continue
            except Exception:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn, self.url = None, None
                time.sleep(0.5 * (k + 1))
        raise IOError(f'range {a}-{b} failed')


_local = threading.local()


def reader(fid):
    if getattr(_local, 'r', None) is None or _local.r.fid != fid:
        _local.r = Reader(fid)
    return _local.r


def bench(fid, workers=64, seconds=20, nbytes=20000):
    workers, seconds, nbytes = int(workers), float(seconds), int(nbytes)
    stop = time.time() + seconds; got = [0, 0]; lock = threading.Lock()

    def run(i):
        off = i * 10_000_000
        while time.time() < stop:
            d = reader(fid).read(off, off + nbytes - 1); off += nbytes
            with lock:
                got[0] += 1; got[1] += len(d)
    th = [threading.Thread(target=run, args=(i,)) for i in range(workers)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; dt = time.time() - t0
    print(f'{workers} workers, {nbytes} B ranges: {got[0] / dt:.1f} req/s, {got[1] / dt / 1e3:.0f} KB/s over {dt:.0f} s', flush=True)


if __name__ == '__main__':
    dict(bench=bench)[sys.argv[1]](*sys.argv[2:])

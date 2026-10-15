"""Download ASE train chunks from the signed CDN list (artifacts.local/downloads/aria-ase-20261013), verify SHA1, extract to
scenes/, delete the zip. ASE license: research use; data stays in ignored artifacts.
Usage: ase_fetch_chunks_dev_20261015.py <first chunk> <last chunk>
"""
import hashlib
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[4] / 'artifacts.local/downloads/aria-ase-20261013'


def fetch(entry):
    z = ROOT / entry['filename']
    k = int(entry['filename'].split('_')[-1].split('.')[0])
    have = [str(s) for s in range(10 * k, 10 * k + 10) if (ROOT / 'scenes' / str(s) / 'trajectory.csv').exists()]
    if len(have) == 10 and not z.exists():                            # already extracted (train chunk k holds scenes 10k..10k+9)
        return entry['filename'], have
    for attempt in range(30):                                         # resume with Range after dropped connections
        have = z.stat().st_size if z.exists() else 0
        try:
            with requests.get(entry['cdn'], stream=True, timeout=(15, 120),
                              headers={'Range': f'bytes={have}-'} if have else {}) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0
                with open(z, 'ab' if have else 'wb') as f:
                    for b in r.iter_content(8 << 20):
                        f.write(b)
            break
        except (requests.RequestException, OSError) as exc:
            print(entry['filename'], 'retry', attempt, repr(exc)[:120], flush=True)
    h = hashlib.sha1()
    with open(z, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    if h.hexdigest() != entry['sha']:
        raise ValueError(f'sha1 mismatch {z.name}')
    with zipfile.ZipFile(z) as zf:
        names = {n.split('/')[0] for n in zf.namelist()}
        zf.extractall(ROOT / 'scenes')
    z.unlink()
    return entry['filename'], sorted(names, key=lambda s: int(s) if s.isdigit() else 0)


if __name__ == '__main__':
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    want = {f'train_chunk_{k:07d}.zip' for k in range(lo, hi + 1)}
    entries = [e for e in json.loads((ROOT / 'ase_download_urls.json').read_text()) if e['filename'] in want]
    with ThreadPoolExecutor(4) as ex:
        for name, scenes in ex.map(fetch, entries):
            print(name, 'scenes', scenes, flush=True)

"""Read one pinned public full-histogram/Livox pair without model inference.

This checks file capability, not calibrated depth accuracy or ST CNH equivalence.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/data-potential-dev-20261013/expansion/sp_transient_sample'
REPO = 'shuinb/SP-TransientBench'
REV = '04bbf64cb63ff3eda498c16d96be4a0825628790'
FILES = ['DepthEstimate/Histogram/1.txt', 'DepthEstimate/gt/1.csv',
         'config/config.yaml', 'config/IRF_global.csv', 'config/offset.txt',
         'config/final_extrinsic_sp_to_livox.txt', 'codes/pointcloud_extract/raw2pc.py', 'README.md']


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')


def metadata(parent):
    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/' + urllib.parse.quote(parent, safe='/')
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def download(item):
    path = OUT / 'files' / item['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/' + urllib.parse.quote(item['path'], safe='/')
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=50) as response, partial.open('wb') as stream:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > item['size']:
                    raise ValueError('Response exceeded advertised file size')
                stream.write(chunk)
        if partial.stat().st_size != item['size']:
            raise ValueError('Incomplete file retained for diagnosis')
        partial.rename(path)
    assert path.stat().st_size == item['size']
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    official = item.get('lfs', {}).get('oid')
    if official:
        assert sha == official
    return dict(path=item['path'], url=url, size=item['size'], sha256=sha,
                official_lfs_sha_checked=bool(official))


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for parent in sorted({str(Path(p).parent).replace('\\', '/') for p in FILES}):
        catalog.update({x['path']: x for x in metadata('' if parent == '.' else parent)})
    selected = [catalog[p] for p in FILES]
    assert sum(x['size'] for x in selected) < 110_000_000
    save('selection_before_read.json', dict(revision=REV, files=selected,
          role='source-format Development inspection; first enumerated depth sample',
          constraints='no ST calibration transfer, no paired RGB, no model or accuracy claim'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(download, selected))
    save('download_receipt.json', dict(revision=REV, files=receipts,
                                      bytes=sum(x['size'] for x in receipts)))
    hist = np.loadtxt(OUT / 'files/DepthEstimate/Histogram/1.txt', dtype=np.int32)
    assert hist.shape == (192 * 256, 672) and (hist >= 0).all()
    offset = np.loadtxt(OUT / 'files/config/offset.txt')
    assert offset.size == 192 * 256 and np.isfinite(offset).all()
    gt_path = OUT / 'files/DepthEstimate/gt/1.csv'
    with gt_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        coordinate_names = [next((f for f in fields if f.strip().lower() == axis), None) for axis in ('x', 'y', 'z')]
        count = 0; finite = 0; zero_xyz = 0; nonzero_xyz = []
        for row in reader:
            count += 1
            if all(coordinate_names):
                p = [float(row[f]) for f in coordinate_names]
                if np.isfinite(p).all():
                    finite += 1
                    if any(v != 0 for v in p):
                        nonzero_xyz.append(p)
                    else:
                        zero_xyz += 1
    nonempty = hist.sum(axis=1, dtype=np.int64) > 0
    peaks = hist.max(axis=1)
    peakbin = hist.argmax(axis=1)
    summary = dict(revision=REV, download_bytes=sum(x['size'] for x in receipts),
         histogram_shape=list(hist.shape), nonnegative_integer_counts=True,
         nonempty_pixels=int(nonempty.sum()), pixels=len(hist),
         nonzero_bin_entries=int(np.count_nonzero(hist)), total_bin_entries=int(hist.size),
         maximum_count=int(hist.max()),
         peak_bin_p10_p50_p90=np.quantile(peakbin[nonempty], [.1, .5, .9]).tolist(),
         offset_entries=int(offset.size), gt_csv_columns=fields, gt_rows=count,
         gt_finite_xyz_rows=finite, coordinate_fields=coordinate_names,
         gt_zero_xyz_rows=zero_xyz, gt_finite_nonzero_xyz_rows=len(nonzero_xyz),
         official_lfs_hashes=sum(x['official_lfs_sha_checked'] for x in receipts),
         license='CC-BY-NC-SA-4.0', wall_s=time.perf_counter() - start,
         range_accuracy='NOT_EVALUATED', calibration_admission='NOT_VERIFIED',
         limitation='One external ADS6311 sample; no paired RGB or ST CNH equivalence. Raw parser has bin offset and per-pixel correction; config and parser distortion defaults differ, so no naive bin-to-distance or Livox accuracy claim.')
    if nonzero_xyz:
        summary['livox_nonzero_xyz_norm_p10_p50_p90_m'] = np.quantile(np.linalg.norm(nonzero_xyz, axis=1), [.1, .5, .9]).tolist()
    assert finite == zero_xyz + len(nonzero_xyz)
    canvas = Image.new('RGB', (1024, 430), 'white'); draw = ImageDraw.Draw(canvas)
    for x, values, label in [(0, np.log1p(peaks).reshape(192, 256), 'log(1 + peak photon count)'),
                              (512, peakbin.reshape(192, 256).astype(float), 'peak bin index (not calibrated distance)')]:
        v = values - values.min()
        view = (255 * v / max(v.max(), 1)).astype('uint8')
        canvas.paste(Image.fromarray(view).resize((512, 384)), (x, 40))
        draw.text((x + 8, 10), label, fill='black')
    canvas.save(OUT / 'histogram_overview.png')
    save('summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

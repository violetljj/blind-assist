"""Read only the preselected LLRGBD train depth ZIP entry via bounded ranges.

Requires the previously saved ZIP directory metadata; never extracts val/test.
This inspects encoding and support, without asserting metric ground truth.
"""
from __future__ import annotations

import binascii
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts.local/work/data-potential-dev-20261013'
OUT = BASE / 'round4/llrgbd_depth'


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf8')


def main():
    start = time.perf_counter()
    metadata = json.loads((OUT / 'archive_metadata.json').read_text(encoding='utf8'))
    entries = metadata['entries']
    selected = [e for e in entries if e['name'] == 'train_depth_filter/10.png']
    assert len(selected) == 1
    e = selected[0]
    assert e['uncompressed_bytes'] < 1_000_000
    save('selection_before_depth_read.json', {
        'rule': 'Same explicit train stem 10 as the third-round RGB pair; no outcome selection.',
        'file': e, 'archive_url': metadata['archive_url'], 'etag': metadata['etag'],
        'unit_policy': 'No metric scale or invalid-value meaning inferred from PNG dtype.',
        'gpu_s': 0, 'inference': 0})
    requests = []

    def get(lo, hi):
        req = urllib.request.Request(metadata['archive_url'], headers={
            'Range': f'bytes={lo}-{hi}', 'User-Agent': 'BlindAssist-data-audit'})
        with urllib.request.urlopen(req, timeout=25) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == f"bytes {lo}-{hi}/{metadata['archive_bytes']}"
            body = response.read(hi-lo+2)
            assert len(body) == hi-lo+1
            requests.append(dict(start=lo, end=hi, bytes=len(body), status=206))
            return body

    local = OUT / 'train_depth_filter_10.png'
    if not local.exists():
        header = get(e['header_offset'], e['header_offset']+29)
        fields = struct.unpack('<4s5H3L2H', header)
        assert fields[0] == b'PK\x03\x04' and fields[3] == e['compression']
        name_length, extra_length = fields[-2:]
        raw_name = get(e['header_offset']+30, e['header_offset']+29+name_length)
        assert raw_name.decode('utf8') == e['name']
        lo = e['header_offset']+30+name_length+extra_length
        payload = get(lo, lo+e['compressed_bytes']-1)
        if e['compression'] == 0:
            data = payload
        elif e['compression'] == 8:
            data = zlib.decompress(payload, -15)
        else:
            raise ValueError('Unsupported compression')
        assert len(data) == e['uncompressed_bytes']
        assert binascii.crc32(data) & 0xffffffff == e['crc32']
        local.write_bytes(data)
    data = local.read_bytes()
    assert len(data) == e['uncompressed_bytes'] and binascii.crc32(data) & 0xffffffff == e['crc32']
    with Image.open(local) as im:
        mode = im.mode; a = np.asarray(im).copy()
    assert a.shape == (480, 640), a.shape
    values = a.astype(np.float64)
    roots = {e['name'].split('/')[0] for e in entries}
    train_depth = {Path(e['name']).stem for e in entries if e['name'].startswith('train_depth_filter/') and e['name'].endswith('.png')}
    rgb_metadata = json.loads((BASE/'deeper/thin_sources/fixed_selection.json').read_text(encoding='utf8'))
    assert rgb_metadata['selected_stem'] == Path(e['name']).stem == '10'
    assert len(train_depth) == rgb_metadata['common_stems'] == 415
    result = dict(status='DEPTH_ENCODING_INSPECTED_METRIC_CONTRACT_UNRESOLVED', file=e['name'],
        shape_hw=list(a.shape), png_mode=mode, dtype=str(a.dtype), pixels=int(a.size),
        min=float(values.min()), max=float(values.max()), distinct_values=int(np.unique(a).size),
        zeros=int((a==0).sum()), value_65535=int((a==65535).sum()),
        percentiles_native_units=np.quantile(values,[.01,.1,.5,.9,.99]).tolist(),
        archive_directory_names=sorted(roots), train_depth_count=len(train_depth),
        raw_depth_available_in_archive=any('raw' in n.lower() for n in roots),
        metric_unit='NOT_VERIFIED', invalid_encoding='NOT_VERIFIED',
        sha256=hashlib.sha256(data).hexdigest(), crc32_checked=True,
        new_requests=requests, downloaded_entry_and_header_bytes=sum(x['bytes'] for x in requests),
        previous_directory_metadata_bytes=metadata['metadata_transfer_bytes'],
        limitations='Filtered sensor-derived depth for the existing static illumination pair; raw/refined correspondence, metric encoding, K and validity mask not established. Do not use as independent range truth.')
    canvas = Image.new('RGB', (1280, 1010), 'white'); draw = ImageDraw.Draw(canvas)
    samples = BASE/'deeper/thin_sources/sample'
    view_paths = [samples/'train_rgb_l/10.png', samples/'train_rgb_h/10.png', samples/'train_lbl/10.png']
    for n, path in enumerate(view_paths):
        x=(n%2)*640; y=(n//2)*505
        draw.text((x+5,y+5), ['Low light train RGB','Normal light train RGB','Supplied semantic label (not thin-line GT)'][n], fill='black')
        with Image.open(path) as im:
            canvas.paste(im.convert('RGB'),(x,y+25))
    lo, hi = np.quantile(values,[.01,.99])
    view=np.clip((values-lo)/max(hi-lo,1)*255,0,255).astype('uint8')
    draw.text((645,510),'Filtered depth: display scaled in native units, not meters',fill='black')
    canvas.paste(Image.fromarray(view).convert('RGB'),(640,530))
    canvas.save(OUT/'paired_depth_overview.jpg',quality=90)
    result['wall_s']=time.perf_counter()-start
    save('summary.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()

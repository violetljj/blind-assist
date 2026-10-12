"""Same ARKit wide-camera frame: native VGA versus derived area downsample.

Public timestamp/intrinsics alignment is checked before scoring. References are
the unchanged low-resolution LiDAR values; both arms exclude the same public
image-boundary rays. This is a 640-to-256 intervention, not an equivalence claim
between Apple's separately processed VGA and lowres RGB streams.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from rgb_body_query_3rscan import color_coordinates
from rgb_body_query_input_diagnostic import sha, write

BASE = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/47333462/'
SIZE = 670242432


def resize_k(k, old_shape, new_shape):
    sy, sx = np.asarray(new_shape, float)/np.asarray(old_shape, float)
    result = np.array(k, float, copy=True)
    result[0] *= sx; result[1] *= sy
    result[0, 2] += (sx-1)/2; result[1, 2] += (sy-1)/2
    return result


class RangeZip(io.RawIOBase):
    def __init__(self, url, size, records, deadline):
        self.url, self.size, self.records, self.deadline, self.pos = url, size, records, deadline, 0
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos+n if whence == 1 else self.size+n
        return self.pos
    def read(self, n=-1):
        if time.monotonic() >= self.deadline: raise TimeoutError('Network deadline')
        n = self.size-self.pos if n < 0 else n
        if n == 0: return b''
        a, b = self.pos, min(self.size-1, self.pos+n-1)
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={a}-{b}'})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=min(30, max(1, self.deadline-start))) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {a}-{b}/{self.size}':
                raise ValueError('Server did not supply the exact requested range')
            data = response.read(n+1)
        if len(data) != b-a+1: raise ValueError('Range length mismatch')
        self.records.append(dict(start=a, end=b, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), wall_s=time.monotonic()-start))
        self.pos += len(data)
        return data


def query_metadata(labels, queries):
    rows = []
    for label, q in zip(labels, queries):
        counts = {k: int((label == v).sum()) for k, v in [('positive_pixels', 1), ('free_ray_pixels', 0), ('unknown_pixels', 2)]}
        state = 'POSITIVE' if counts['positive_pixels'] >= 16 else ('FREE_ON_SAMPLED_RAYS' if counts['free_ray_pixels'] >= 16 and not counts['positive_pixels'] and not counts['unknown_pixels'] else 'UNKNOWN')
        rows.append(dict(name=q['name'], state=state, domain_pixels=int((label != 255).sum()), **counts))
    return rows


def prepare(old_sensor, source, output, cpu_budget_s=240, network_budget_s=300):
    start = time.monotonic(); output.mkdir(parents=True, exist_ok=True)
    if (output/'selection.json').exists(): raise FileExistsError('Preserve the previous selection')
    old = json.loads((old_sensor/'dataset_manifest.json').read_text())
    rgb_index = json.loads((source/'vga_rgb_index.json').read_text())
    k_index = json.loads((source/'vga_k_index.json').read_text())
    rows = [r for r in old['rows'] if r['source_id'] in rgb_index and r['source_id'] in k_index]
    if len(rows) != 8: raise ValueError('Expected the eight pre-indexed exact timestamp matches')
    selection = dict(source='ARKitScenes raw Training/47333462 vga_wide', method='All eight exact timestamp intersections with the previously fixed 16 frames; old order; no pixel/depth/outcome selection',
                     frames=[dict(frame=r['frame'], source_id=r['source_id'], timestamp_s=r['timestamp_s']) for r in rows],
                     old_manifest_sha256=sha(old_sensor/'dataset_manifest.json'), rgb_index_sha256=sha(source/'vga_rgb_index.json'),
                     intrinsics_index_sha256=sha(source/'vga_k_index.json'), saved_before_rgb_reference_reads=True)
    write(output/'selection.json', selection)
    records=[]; receipt=dict(status='STARTING', rows=[], network_range_requests=records, source_sha256=sha(Path(__file__)), gpu_s=0)
    arms={a:[] for a in ('native_vga','derived_256')}
    for a in arms: (output/a).mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(source/'vga_wide_intrinsics.zip') as kz, zipfile.ZipFile(RangeZip(BASE+'vga_wide.zip', SIZE, records, start+network_budget_s)) as rz:
            for r in rows:
                if time.monotonic()-start >= cpu_budget_s: raise TimeoutError('Preparation wall budget')
                source_id=r['source_id']; kb=kz.read(k_index[source_id]); w,h,fx,fy,cx,cy=np.fromstring(kb.decode(),sep=' ')
                k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
                oldk=np.asarray(r['color_K']); scaled=k.copy(); scaled[:2]*=.4
                alignment_error=float(np.max(np.abs(scaled-oldk)))
                if (w,h)!=(640,480) or alignment_error > .002: raise ValueError('Public wide-camera K proportionality failed')
                payload=rz.read(rgb_index[source_id]); img=Image.open(io.BytesIO(payload)).convert('RGB')
                if img.size!=(640,480): raise ValueError('Native VGA image size differs from pincam')
                low=img.resize((256,192),Image.Resampling.BOX); klow=resize_k(k,(480,640),(192,256))
                oldimage=np.asarray(Image.open(r['rgb_path']).convert('RGB'),float)
                mad=float(np.mean(np.abs(np.asarray(low,float)-oldimage)))
                corr=float(np.corrcoef(np.asarray(low,float).reshape(-1),oldimage.reshape(-1))[0,1])
                dk=np.asarray(r['depth_K']); shape=r['depth_shape']
                maps={a:color_coordinates(dk,ak,shape) for a,ak in [('native_vga',k),('derived_256',klow)]}
                common=np.ones(shape,bool)
                for a, (mx,my) in maps.items():
                    ch,cw=(480,640) if a=='native_vga' else (192,256)
                    common &= (mx>=0)&(mx<=cw-1)&(my>=0)&(my<=ch-1)
                if sha(r['reference_path']) != r['reference_sha256']: raise ValueError('Original reference changed')
                with np.load(r['reference_path']) as z: ref={key:z[key].copy() for key in z.files}
                original_labels=ref['labels'].copy(); ref['labels'][:,~common]=255
                qrows=query_metadata(ref['labels'],old['queries'])
                for a, ak, im in [('native_vga',k,img),('derived_256',klow,low)]:
                    arm=output/a; rgb_path=arm/(source_id+'.png')
                    if a=='native_vga': rgb_path.write_bytes(payload)
                    else: im.save(rgb_path)
                    ref_path=arm/(source_id+'.npz'); mx,my=maps[a]
                    np.savez_compressed(ref_path,**{**ref,'color_K':ak,'map_x':mx,'map_y':my,'paired_common_image_rays':common})
                    nr={**r,'rgb_path':str(rgb_path),'rgb_sha256':sha(rgb_path),'color_K':ak.tolist(),'color_shape':[im.height,im.width],
                        'reference_path':str(ref_path),'reference_sha256':sha(ref_path),'queries':qrows,
                        'old_reference_sha256':r['reference_sha256'],'source_native_rgb_sha256':hashlib.sha256(payload).hexdigest()}
                    arms[a].append(nr)
                receipt['rows'].append(dict(source_id=source_id, public_k_scaled_max_error_px=alignment_error, exact_timestamp=True,
                    original_lowres_vs_derived_rgb_mad=mad, original_lowres_vs_derived_rgb_correlation=corr,
                    common_public_rays=int(common.sum()), excluded_boundary_rays=int((~common).sum()),
                    original_labels_on_common_rays_unchanged=bool(np.array_equal(ref['labels'][:,common],original_labels[:,common])),
                    source_member=rgb_index[source_id],source_member_crc32=rz.getinfo(rgb_index[source_id]).CRC,
                    source_rgb_sha256=hashlib.sha256(payload).hexdigest(),pincam_sha256=hashlib.sha256(kb).hexdigest()))
        for a, rs in arms.items():
            manifest={**old,'rows':rs,'frames':len(rs),'state_counts_by_split':{'validation':dict(Counter(q['state'] for r in rs for q in r['queries']))},
                      'query_ray_units':{key:sum(q[key] for r in rs for q in r['queries']) for key in ('positive_pixels','free_ray_pixels','unknown_pixels','domain_pixels')},
                      'selection_sha256':sha(output/'selection.json'),'paired_contract':'Identical selected timestamps, lowres LiDAR depth/labels/query bounds; identical public RGB common-ray mask; only native VGA versus its BOX area downsample differs',
                      'alignment_basis':'Official RAW README identifies both lowres_wide and vga_wide as wide camera; exact timestamps; native public K proportionality checked <=.002 lowres pixels; calibrated ray map K_color @ inv(K_depth); no new inferred extrinsic',
                      'limitations':'640x480 native versus generated256x192 only; original lowres RGB is separately processed; same-capture correlated evidence; retained lowres LiDAR is not FARO or body/event reference'}
            arm=output/a; write(arm/'dataset_manifest.json',manifest)
            fields=('environment','scan','split','frame','rgb_path','rgb_sha256','color_K','color_shape','depth_K','depth_shape')
            write(arm/'observations.json',dict(rows=[{k:r[k] for k in fields} for r in rs],queries=old['queries'],dataset_manifest_sha256=sha(arm/'dataset_manifest.json')))
        # Coordinate fixture verifies half-pixel resize for arbitrary calibrated rays.
        mx,my=maps['native_vga']; lx,ly=maps['derived_256']
        error=float(max(np.max(np.abs(lx-((mx+.5)*.4-.5))),np.max(np.abs(ly-((my+.5)*.4-.5)))))
        if error>5e-5: raise ValueError('Half-pixel ray-map fixture failed')
        receipt.update(status='COMPLETE',frames=len(rows),arms={a:dict(observations_sha256=sha(output/a/'observations.json'),manifest_sha256=sha(output/a/'dataset_manifest.json')) for a in arms},
                       half_pixel_mapping_max_error_px=error,zip_crc_checked=True,license_source='Prior cached official Apple LICENSE non-commercial research; no data redistribution',
                       source_url=BASE+'vga_wide.zip',archive_length_bytes=SIZE,source_intrinsics_sha256=sha(source/'vga_wide_intrinsics.zip'))
    except Exception as exc:
        receipt.update(status='FAILED',error=repr(exc)); raise
    finally:
        receipt.update(wall_s=time.monotonic()-start,downloaded_range_bytes=sum(r['bytes'] for r in records),network_request_wall_s=sum(r['wall_s'] for r in records))
        write(output/'preparation_receipt.json',receipt); print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-sensor',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.old_sensor.resolve(),a.source.resolve(),a.output.resolve())

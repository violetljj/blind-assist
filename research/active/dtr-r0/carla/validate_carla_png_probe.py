"""Independently decode every probe PNG and compare to capture-time raw pixels."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def validate(root):
    result=json.loads((root/'result.json').read_text())
    if result['status']!='PASS' or not result['ports_released']:raise ValueError('incomplete_probe')
    totals={};poses={};verified=0
    for name,group in result['groups'].items():
        rows=group['records'];pairs={};total=0;poses[name]={}
        expected={(scene,sample,mode) for scene in range(2) for sample in range(50) for mode in ('rgb','depth')}
        if len(rows)!=200 or {(r['scene'],r['sample'],r['mode']) for r in rows}!=expected:
            raise ValueError('incomplete_or_duplicate_images')
        for row in rows:
            path=root/name/row['path']
            if path.parent.resolve()!=(root/name).resolve():raise ValueError('payload_path_escape')
            if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('payload_hash_mismatch')
            with Image.open(path) as image:
                if image.size!=(1280,720):raise ValueError('dimensions')
                rgba=image.convert('RGBA')
                if hashlib.sha256(rgba.tobytes()).hexdigest()!=row['pixel_sha256']:
                    raise ValueError('decoded_pixels_differ_from_sensor_bytes')
            pairs.setdefault((row['scene'],row['sample']),[]).append(row)
            poses[name][(row['scene'],row['sample'],row['mode'])]=row['pose']
            verified+=1;total+=row['bytes']
        for a,b in pairs.values():
            if any(a[k]!=b[k] for k in ('frame','timestamp','pose')):raise ValueError('unaligned_pair')
        totals[name]=total
    if poses['native_sync']!=poses['fast_sync']:raise ValueError('comparison_camera_paths_differ')
    receipt={'status':'PASS','independent_decoder':'Pillow','verified_images':verified,
             'raw_pixel_mismatches':0,'all_pair_frame_timestamp_pose_matches':True,
             'native_fast_camera_paths_identical':True,'encoded_bytes':totals,
             'size_ratio':totals['fast_sync']/totals['native_sync']}
    with (root/'pixel-validation.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    validate(parser.parse_args().root)

"""Lock native LaMAR LiDAR clear-paving windows before replaying either alert method.

For the registered 32 frames, run scan on the inputs recorded in controls/selection.json;
then run lock with its RGB-reviewed candidate indices. Temporal sampling is about 3.3 Hz.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST.parent / "evalkit"))
import core

WORK = HOST.parents[3] / "artifacts.local/work"
SOURCE = WORK / "paired-low-sites-20261118/lamar"
OUT = WORK / "shallow-edge-20261118/controls"


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf8")


def intr(params):
    fx, fy, cx, cy = params
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


def frame_geometry(r):
    source = Path(r.get("root", SOURCE))
    z = cv2.imread(str(source / r["depth"]), cv2.IMREAD_UNCHANGED).astype("f4") * .001
    conf = cv2.imread(str(source / r["confidence"]), cv2.IMREAD_UNCHANGED)
    valid = (conf >= 1) & (z > .30) & (z < 10.)
    K = intr(r["K_depth"])
    cloud = core.rays(K, *z.shape) * z.ravel()[:, None]
    p = cloud[valid.ravel()]
    rng = np.random.default_rng(r['timestamp_us'])
    down = np.asarray(r["down_camera"])
    floor = core.ground_plane(p[::4], down, rng, tries=80, tol=.025, min_inliers=250, max_deg=14)
    if floor is None:
        return None
    n, h = floor
    # Refine the plane using independent, confidence-filtered sensor measurements.
    for _ in range(2):
        pp = p[np.abs(p @ n + h) < .025]
        if len(pp) < 1000:
            return None
        mid = np.mean(pp, axis=0)
        eig = np.linalg.eigh(np.cov((pp - mid).T))[1][:, 0]
        n = eig if eig @ down < 0 else -eig
        h = -n @ mid
    if not .5 <= h <= 2.2 or -n @ down < np.cos(np.deg2rad(14)):
        return None
    fwd = np.array([0., 0., 1.]) - n * n[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(-n, fwd)
    above, along, side = h + p @ n, p @ fwd, p @ right
    scope = (abs(side) <= .35) & (along >= .30) & (along <= 2.5)
    coverage = [int(np.sum(scope & (along <= 1.5) & (abs(above) < .05) &
                           (side >= a) & (side < b)))
                for a, b in [(-.35, -.35/3), (-.35/3, .35/3), (.35/3, .35001)]]
    occ = int(np.sum(scope & (above >= .05) & (above < 2.)))
    return dict(ground=[n.tolist(), float(h)], occupied_pixels_5cm_to_2m=occ,
                near_floor_lcr=coverage, floor_inliers=len(pp), n_valid=len(p),
                empty=occ == 0 and min(coverage) >= 10)


def sheet(rows_by_window, path):
    out = Image.new('RGB', (6*155, len(rows_by_window)*178), '#202326')
    draw = ImageDraw.Draw(out)
    for line, rows in enumerate(rows_by_window):
        for col in range(6):
            i = col*(len(rows)-1)//5
            r = rows[i]
            pic = Image.open(Path(r.get('root', SOURCE)) / r['rgb'])
            pic.thumbnail((148, 147))
            x, y = col*155, line*178
            out.paste(pic,(x,y))
            draw.text((x,y+147),f'{r["scene"]} {line:02d} / {i}',fill='white')
    out.save(path,quality=83)


def scan(paths):
    windows, retained = [], {}
    for path in paths:
        rows=read(path)
        results=[frame_geometry(r) for r in rows]
        name=path.parent.name
        retained[name]=dict(path=str(path), results=results)
        for i,r in enumerate(rows):
            if not results[i] or not results[i]['empty']:
                continue
            j=i
            while j+1<len(rows) and rows[j+1]['timestamp_us']-r['timestamp_us']<=2_150_000 and rows[j+1]['timestamp_us']-rows[j]['timestamp_us']<=505_000:
                j+=1
            if rows[j]['timestamp_us']-r['timestamp_us']<1_800_000 or not all(q and q['empty'] for q in results[i:j+1]):
                continue
            poses=np.array([np.asarray(q['T_local'])[:3,3] for q in rows[i:j+1]])
            displacement=float(np.linalg.norm((poses[-1]-poses[0])[[0,2]]))
            if displacement<.40:
                continue
            if any(w['case']==name and i<=w['end']+2 for w in windows):
                continue
            windows.append(dict(case=name,scene=r['scene'],seq=r['seq'],start=i,end=j,displacement_m=displacement))
    write(OUT/'candidate_geometry.json',retained)
    write(OUT/'candidates.json',windows)
    if windows:
        for offset in range(0,len(windows),12):
            page=windows[offset:offset+12]
            sheet([read(Path(retained[w['case']]['path']))[w['start']:w['end']+1] for w in page], OUT/f'paving_{offset:03d}.jpg')
    print('accepted potential windows',len(windows),flush=True)
    for i,w in enumerate(windows):print(i,w,flush=True)


def lock(which):
    all_windows=read(OUT/'candidates.json')
    scanned=read(OUT/'candidate_geometry.json')
    manifest_rows,windows,evidence=[],[],[]
    for i in which:
        w=all_windows[i]
        note=scanned[w['case']]
        rows=read(Path(note['path']))
        window_id=f"{w['scene']}-{w['case']}-{w['start']}"
        window=dict(id=window_id,site_id='LaMAR-'+w['scene'], frames=[],max_gap_s=.505)
        window_geo=[]
        for j in range(w['start'],w['end']+1):
            r=rows[j]; ref=note['results'][j];assert ref['empty']
            native=Path(r.get('root',SOURCE))
            width,height=Image.open(native/r['depth']).size
            ts=r['timestamp_us']
            row=dict(seq=r['seq'],i=ts,t=ts/1e6,rgb=str(native/r['rgb']),depth=str(native/r['depth']),
                     confidence=str(native/r['confidence']),min_confidence=1,ref_scale=.001,size=[width,height],
                     rgb_native_size=list(Image.open(native/r['rgb']).size),rgb_K=intr(r['K_rgb']).tolist(),
                     K=intr(r['K_depth']).tolist(),native_scale=5,mount='handheld',ref_grade='sparse_geometry',
                     down=r['down_camera'],ground=ref['ground'],T_local=r['T_local'])
            assert all(Path(row[k]).is_file() for k in ('rgb','depth','confidence'))
            manifest_rows.append(row)
            window['frames'].append(dict(seq=r['seq'],i=ts,eligible=True))
            window_geo.append(dict(seq=r['seq'],i=ts,**ref))
        windows.append(window)
        evidence.append(dict(id=window_id,site=window['site_id'],source=note['path'],**w,
                             duration_s=(window['frames'][-1]['i']-window['frames'][0]['i'])/1e6,
                             reference=window_geo))
    for row in manifest_rows:
        row['eye_height']=float(np.median([x['ground'][1] for x in manifest_rows if x['seq']==row['seq']]))
    assert len(set((r['seq'],r['i']) for r in manifest_rows))==len(manifest_rows)
    manifest_rows.sort(key=lambda r:(r['seq'],r['t']))
    loc=OUT/'lamar'
    write(loc/'manifest.json',dict(dataset='lamar-edge-controls',data_root='.',frames=manifest_rows))
    write(loc/'labels.json',dict(manifest_sha256=hashlib.sha256((loc/'manifest.json').read_bytes()).hexdigest(),cases=[],windows=windows))
    write(loc/'evidence.json',dict(windows=evidence,selection='Reference-only native ARKit LiDAR; no original points 5cm–2m above ground within ±0.35m, 0.3–2.5m. Three lateral sectors have observed floor to 1.5m. Sites and RGB reviewed before inference.',source='https://github.com/microsoft/lamar-benchmark/blob/main/RAW-DATA.md',aperture='Native iPhone rectangular sensor image, no virtual-camera border added.'))
    sheet([read(Path(scanned[all_windows[i]['case']]['path']))[all_windows[i]['start']:all_windows[i]['end']+1] for i in which],loc/'review.jpg')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['scan','lock']);p.add_argument('inputs',nargs='*')
    args=p.parse_args();cv2.setNumThreads(1)
    if args.stage=='scan':
        paths=[Path(q) for q in args.inputs] if args.inputs else [OUT/q for q in read(OUT/'selection.json')['scan_inputs']]
        scan(paths)
    else:
        lock(list(map(int,args.inputs)))

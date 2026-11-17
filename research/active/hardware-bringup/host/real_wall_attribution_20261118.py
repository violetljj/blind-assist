"""Paired V6 alarm-event audit on the five consumed head-worn development batches."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT
import frozen_fusion_eval_v2 as V2
import frozen_fusion_eval_v6 as V6

WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'sidewall-attribution-20261118/real'
ORIGINAL = WORK / 'fusion-v6-dev-20261013'
PITCHFIX = WORK / 'hindsight-truth-dev-20261014/pitchfix'
CACHE = WORK / 'fusion-refinement-dev-20261013'
RAW = {
    'm12': 'phone-imu-capture-20261012/raw', 'a12': 'rgb-real-feasibility-20261012/raw',
    'm13': 'frozen-eval-capture-20261013/raw', 'a13': 'frozen-eval-capture-20261013b/raw',
    'e13': 'frozen-eval-capture-20261013c/raw_eval',
}
# Visual adjudications of the rendered voter masks, not inferred from the clip labels.
VISUAL_REVIEW = {
    '020702': 'Open hall context. Only ToF strong votes contributed. Physical reflecting surface unknown.',
    '073635': 'Both BODY events, both pitches: accepted pixels on the near brown region at the bottom, spatially separated from walls, steps and the narrow pole. Object clearance unmeasured.',
    '032738': 'Original BODY voters 60 and 61: the accepted pixels are on the polished floor in front of the camera.',
    '054233': 'Original HEAD voters 7 through 12: the accepted pixels lie at the lower edge of the hanging exit sign and adjacent light.',
}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def report(root, batch):
    return read(root / batch / 'frozen_fusion_eval_v6.json')


def streams(folder):
    tof = [r['received_elapsed_ns'] for r in V6.V1.F.lines(folder / 'tof.index.jsonl') if r.get('valid_frame')]
    rgb = list(V6.V1.F.lines(folder / 'rgb.jsonl'))
    stamps = np.array([r['received_elapsed_ns'] for r in rgb], dtype=np.int64)
    return tof, rgb, stamps


def runs(frames, times):
    events = []
    for pos, row in enumerate(frames):
        if not any(row['fused']):
            continue
        idx = row['tof_index']
        if (pos == 0 or not any(frames[pos-1]['fused'])
                or frames[pos-1]['tof_index'] != idx-1 or times[idx] - times[idx-1] > 500_000_000):
            events.append(dict(indices=[], voters={'rgb': {}, 'tof_strong': {}}, bands=set()))
        event = events[-1]
        event['indices'].append(idx)
        for q, name in enumerate(('HEAD', 'BODY')):
            if row['fused'][q]:
                event['bands'].add(name)
                for src, bands in event['voters'].items():
                    indices = bands.setdefault(name, set())
                    indices.update(x['tof_index'] for x in frames[max(0, pos-V2.N+1):pos+1] if x[src][q])
    for e in events:
        e['bands'] = sorted(e['bands'])
        e['voters'] = {src: {b: sorted(ix) for b, ix in bands.items()} for src, bands in e['voters'].items()}
        e['span_seconds_first_to_last'] = (times[e['indices'][-1]] - times[e['indices'][0]]) * 1e-9
    return events


def photo(folder, rgb, stamps, time, cam):
    j = int(np.argmin(np.abs(stamps-time)))
    return cam.undistort_upright(cv2.imread(str(folder / rgb[j]['file']))), float((stamps[j]-time)*1e-6)


def tile(im, title, width=265):
    pic = cv2.resize(im, (width, round(im.shape[0]*width/im.shape[1])), interpolation=cv2.INTER_AREA)
    pic = cv2.copyMakeBorder(pic, 29, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32,32,32))
    cv2.putText(pic, title, (6,20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)
    return pic


def context_sheet(cam, folder, indices, key, timing):
    tof, rgb, stamps = timing
    tiles = []
    for i in indices:
        im, ms = photo(folder, rgb, stamps, tof[i], cam)
        tiles.append(tile(im, f'{key}   ToF {i}   RGB {ms:+.0f} ms'))
    cv2.imwrite(str(OUT / f'scene-{key}.jpg'), cv2.hconcat(tiles))


def rgb_mask(disp, valid, dirs, pose, rotation, fit, band):
    a,b = fit
    inv_depth = (disp.reshape(-1)-b)/a
    z = np.where(inv_depth>1e-3, 1/np.maximum(inv_depth,1e-3), np.inf)
    ok = np.isfinite(z) & valid.ravel()
    points = (pose @ (rotation.T @ (dirs[ok]*z[ok,None]).T)).T
    lo, hi = V6.V1.F.BOXES[band]
    inside = ((np.abs(points[:,0])<=.30) & (points[:,1]>=lo) & (points[:,1]<=hi)
              & (points[:,2]>=.30) & (points[:,2]<=V6.V1.Z_MAX))
    mask = np.zeros(len(dirs),np.uint8)
    mask[np.flatnonzero(ok)[inside]] = 1
    depths = np.full(len(dirs),np.nan)
    depths[ok] = points[:,2]
    n, cc = cv2.connectedComponents(mask.reshape(disp.shape), connectivity=8)
    accepted = np.zeros_like(cc, bool)
    for c in range(1,n):
        blob = cc==c
        if blob.sum() >= V6.MIN_BLOB_PX and np.median(depths[blob.ravel()]) <= V6.BLOB_MAX_MEDIAN_Z:
            accepted |= blob
    return accepted


def camera_dirs(cam):
    k = cam.K_upright
    w,h = cam.upright_size
    v,u = np.mgrid[:h,:w]
    return np.stack([(u-k[0,2])/k[0,0], (v-k[1,2])/k[1,1], np.ones_like(u,float)],-1).reshape(-1,3)


def advance_floor_sampler(pose, sensor_dirs, height, rng):
    # floor_anchor consumes these exact draws whether RANSAC succeeds or fails.
    y = pose[1] @ sensor_dirs
    invz = y/height
    n = int(np.count_nonzero((y>.1)&(invz>1/6)&(invz<1/.8)))
    if n>=500:
        count = min(4000,n)
        rng.choice(n,count,replace=False)
        for _ in range(64):
            rng.choice(count,2,replace=False)


def mask_tile(image, mask, heading, label):
    overlay = image.copy()
    overlay[mask] = (.22*overlay[mask] + .78*np.array([50,30,255])).astype(np.uint8)
    return cv2.hconcat([tile(image,heading), tile(overlay,label)])


def corrected_side_masks(cam, name, voters, pitch0):
    """Recover the same floor RANSAC state in batch order from archived anchor choices."""
    cache = read(CACHE/'cache_manifest.json')
    evidence = report(PITCHFIX/'v6' if pitch0 else ORIGINAL, 'a12')
    extrinsic = read(PITCHFIX/'camera_tof_extrinsic_pitch0.json' if pitch0 else V6.V1.EXTRINSIC)
    rotation = np.array(extrinsic['rotation'])
    pitch_deg = extrinsic['camera_from_tof_euler_deg']['pitch']
    height = evidence['head_height_m']
    assert height == cache['height']
    dirs = camera_dirs(cam)
    sensor_dirs = rotation.T @ dirs.T
    rng = np.random.default_rng(20261013)
    images, metrics = [], []
    folder = WORK/RAW['a12']/name
    tof, rgb, stamps = streams(folder)
    for sid, session in sorted(evidence['sessions'].items()):
        meta = cache['sessions'][sid[17:23]]['frames']
        assert [f['tof_index'] for f in session['frames']] == [m['i'] for m in meta]
        for f, m in zip(session['frames'], meta):
            i = m['i']
            if not (m['matched'] and m['pose_ok']) or f['anchor'] in ('tof','robust_tof'):
                continue
            pose = V6.V1.F.rot(m['pitch'],m['roll'])
            if sid == name and i in voters:
                assert f['anchor']=='floor' and f['rgb'][1]
                with np.load(CACHE/m['cache']) as sample:
                    disp = sample['disp']
                    fit = V6.V1.S.floor_anchor(disp, dirs, pose, rotation, height, rng)
                    mask = rgb_mask(disp,sample['valid'],dirs,pose,rotation,fit,'BODY')
                n = int(mask.sum())
                assert (n>=V6.V1.MIN_PX) == f['rgb'][1]
                image, ms = photo(folder,rgb,stamps,tof[i],cam)
                assert abs(ms)<=100 and stamps[m['rgb_index']] == m['rgb_t_ns']
                heading = f'H6 #{i} pitch {pitch_deg:g} deg  {ms:+.0f} ms'
                images.append(mask_tile(image,mask,heading,f'V6 BODY, {n} retained px'))
                yy,xx = np.where(mask)
                metrics.append(dict(tof_index=i,rgb_index=m['rgb_index'],rgb_minus_tof_ms=ms,
                                    retained_pixels=n,rectified_bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
                                    affine=list(fit)))
                continue
            advance_floor_sampler(pose,sensor_dirs,height,rng)
        if sid==name:
            break
    assert [s['tof_index'] for s in metrics]==sorted(voters)
    for i in range(0,len(images),2):
        part = images[i:i+2]
        knob='pitch0' if pitch0 else 'original'
        cv2.imwrite(str(OUT/f'{knob}-H6-body-{metrics[i]["tof_index"]}.jpg'),cv2.vconcat(part))
    return metrics


def original_corridor_masks(cam, batch, name, band, voters, model):
    """Re-infer only eight causal RGB votes, compare retained pixels to the archived blob dump."""
    import torch
    from atom_imu import load_mounting
    v1 = V6.V1
    dirs = camera_dirs(cam)
    rotation = np.array(read(v1.EXTRINSIC)['rotation'])
    sensor_dirs = rotation.T @ dirs.T
    evidence = report(ORIGINAL,batch)
    height = evidence['head_height_m']
    rng = np.random.default_rng(20261013)
    folder = WORK/RAW[batch]/name
    tof, rgb, stamps = streams(folder)
    x = v1.R.load(folder,np.array(load_mounting(v1.R.MOUNTING)['tof_from_imu_rotation']))
    dist = np.where(np.isin(x['S'],(5,6,9,10)),x['D'],np.nan).reshape(-1,8,8)[:,::-1,::-1].reshape(-1,64)/1000
    images, metrics = [], []
    q = ('HEAD','BODY').index(band)
    for sid, session in sorted(evidence['sessions'].items()):
        hist = read(WORK/'own-blob-dump-20261013'/f'{sid}.json')['frames']
        assert [f['tof_index'] for f in session['frames']] == [m['tof_index'] for m in hist]
        last = None
        for f, m in zip(session['frames'],hist):
            i = f['tof_index']
            if not m['pose_ok'] or abs(m['rgb_t_ns']-m['t_ns'])>100_000_000:
                continue
            pose = v1.F.rot(m['pitch'],m['roll'])
            if sid!=name or i not in voters:
                if f['anchor'] not in ('tof','robust_tof'):
                    advance_floor_sampler(pose,sensor_dirs,height,rng)
                continue
            assert m['anchor']==f['anchor'] and f['rgb'][q]
            image,ms = photo(folder,rgb,stamps,tof[i],cam)
            # This checkout retains the official-weight FP32 DAV2-S ONNX export.
            norm = cv2.resize(image[:,:,::-1]/255., (518,686), interpolation=cv2.INTER_CUBIC)
            norm = (norm-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
            tensor = np.ascontiguousarray(norm.transpose(2,0,1)[None].astype(np.float32))
            pred = model.run(['disparity'],{'image':tensor})[0]
            disp = torch.nn.functional.interpolate(torch.from_numpy(pred)[:,None],size=image.shape[:2],
                                                    mode='bilinear',align_corners=True)[0,0].numpy()
            anchor = f['anchor']
            if anchor=='tof':
                dz,zfac = v1.S.zone_disparity(disp,cam.K_upright,rotation,v1.X.zone_rays())
                valid = np.isfinite(dist[i]) & (dist[i]>.3) & (dist[i]<4)
                denom = dist[i]*zfac
                inv = np.divide(1.,denom,out=np.full_like(denom,np.nan),where=valid & np.isfinite(denom) & (denom>0))
                fit = v1.S.fit_affine(dz,inv)
            else:
                trial = v1.S.floor_anchor(disp,dirs,pose,rotation,height,rng)
                assert (trial is not None)==(anchor=='floor')
                fit = trial if trial is not None else last
            assert fit is not None
            last = fit
            mask = rgb_mask(disp,image.sum(2)>0,dirs,pose,rotation,fit,band)
            pixels = int(mask.sum())
            archived = sum(px for px,md,*_ in m['blobs'][q][1] if md<=V6.BLOB_MAX_MEDIAN_Z)
            assert pixels>=v1.MIN_PX and archived>=v1.MIN_PX
            images.append(mask_tile(image,mask,f'{sid[17:23]} ToF {i}, RGB {ms:+.0f} ms',f'V6 {band}, {pixels} retained px'))
            metrics.append(dict(tof_index=i,rgb_minus_tof_ms=ms,retained_pixels=pixels,archived_pixels=archived))
        if sid==name:
            break
    assert [s['tof_index'] for s in metrics]==sorted(voters)
    for k in range(0,len(images),2):
        cv2.imwrite(str(OUT/f'original-{name[17:23]}-{metrics[k]["tof_index"]}.jpg'),cv2.vconcat(images[k:k+2]))
    return metrics


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cam = AtomCamera()
    totals = Counter()
    sessions, controls, corridor = [], [], {}
    model = None
    for batch in RAW:
        old = report(ORIGINAL,batch)['sessions']
        fixed = report(PITCHFIX/'v6',batch)['sessions']
        assert set(old)==set(fixed)
        for name, left in old.items():
            right = fixed[name]
            assert left['cls']==right['cls'] and left['scene']==right['scene']
            cls = left['cls']
            if cls in ('head','ahead'):
                controls.append(dict(batch=batch,session=name,cls=cls,original=left['pass'],pitch0=right['pass'],
                                     end_original=left['end_window_head_alert' if cls=='head' else 'end_window_any_alert'],
                                     end_pitch0=right['end_window_head_alert' if cls=='head' else 'end_window_any_alert']))
            if cls not in ('quiet','side'):
                continue
            frames = left['frames']
            folder = WORK/RAW[batch]/name
            timing = streams(folder)
            for l,r in zip(frames,right['frames'],strict=True):
                assert all(l[x]==r[x] for x in ('tof_index','pose_ok','tof_strong'))
            by_pitch = {}
            for knob, rows in [('original',frames),('pitch0',right['frames'])]:
                confirm = np.column_stack([V2.confirm([f['fused_raw'][q] for f in rows]) for q in range(2)])
                np.testing.assert_equal(confirm,np.array([f['fused'] for f in rows]))
                by_pitch[knob] = runs(rows,timing[0])
                totals[knob+'_events'] += len(by_pitch[knob])
                totals[knob+'_frames'] += int(confirm.any(1).sum())
            tof_conf = np.column_stack([V2.confirm([f['tof_strong'][q] for f in frames]) for q in range(2)])
            by_pitch['tof_only'] = runs([dict(f,fused=t.tolist(),rgb=[False,False]) for f,t in zip(frames,tof_conf)],timing[0])
            totals['tof_only_events'] += len(by_pitch['tof_only'])
            totals['tof_only_frames'] += int(tof_conf.any(1).sum())
            totals['ready_frames'] += len(frames)
            totals['sessions'] += 1
            sid = name[17:23]
            if by_pitch['original']:
                indices = sorted({i for e in by_pitch['original'] for i in (e['indices'][0],e['indices'][len(e['indices'])//2],e['indices'][-1])})
                context_sheet(cam,folder,indices,sid,timing)
            sessions.append(dict(batch=batch,session=name,cls=cls,scene=left['scene'],ready=len(frames),**by_pitch))
            if sid=='073635':
                voters = {i for e in by_pitch['pitch0'] for i in e['voters']['rgb']['BODY']}
                masks = {key: corrected_side_masks(cam,name,voters,key=='pitch0') for key in ('original','pitch0')}
            if sid in ('032738','054233'):
                band,ix = next(iter(by_pitch['original'][0]['voters']['rgb'].items()))
                if model is None:
                    import onnxruntime as ort
                    export = WORK/'dav2s-mobile-20261012/dav2s_686x518.onnx'
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 4
                    model = ort.InferenceSession(str(export),options,providers=['CPUExecutionProvider'])
                corridor[sid] = original_corridor_masks(cam,batch,name,band,ix,model)
    output = dict(units='Ready-frame V6 any-band alarms, episode split at a quiet row, a skipped ToF index or >0.5 s. Duration excludes sampling interval after last alarm.',
                  scope='All quiet and side clips from the five consumed, paired V6 batches. Clip labels proxy nuisance alarms; physical path occupancy is unmeasured.',
                  calibration='Changed camera-from-ToF pitch from 14.5 to 0 degrees; yaw and roll fixed. Development sensitivity, not a measured replacement extrinsic.',
                  paired_inputs={'original':str(ORIGINAL),'pitch0':str(PITCHFIX/'v6')},
                  totals=dict(totals),sessions=sessions,positive_controls=controls,h6_body_evidence=masks,
                  original_corridor_evidence=corridor, visual_adjudication_by_session=VISUAL_REVIEW)
    (OUT/'paired-real-v6.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    print(dict(totals),flush=True)


if __name__ == '__main__':
    main()

"""Import and replay new phone sessions with frozen joint weights/cuts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import numpy as np
import real_phone_imu_replay as R
from real_contract_v5 import adapter_v5

ROOT = R.ROOT
OUT = ROOT / 'artifacts.local/work/phone-new-capture-20261012'
ADB = None  # import stage receives the host's adb executable explicitly
SERIAL = None
PACKAGE = 'com.linnan.blindassist'
MANIFEST = ROOT / 'artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012/real_manifest.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def adb(args, **kwargs):
    return subprocess.run([ADB, '-s', SERIAL, 'exec-out', 'run-as', PACKAGE, *args],
                          check=True, timeout=300, **kwargs)


def import_batch():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((OUT/'phone_inventory.json').read_text())
    sessions = [d['session_id'] for d in inventory if not d.get('scene_note') and
                d['session_id'] >= 'capture-20261012T0659' and d['status'] != 'RECORDING']
    if not sessions:
        raise ValueError('No completed new sessions')
    if not all(re.fullmatch(r'capture-\d{8}T\d{9}Z-[0-9a-f]{8}', n) for n in sessions):
        raise ValueError('Unexpected capture identity')
    archive = OUT/'phone_original.tar'
    if archive.exists():
        raise FileExistsError('Preserve previous import')
    started = time.monotonic()
    with archive.open('wb') as f:
        adb(['tar', '-C', 'no_backup/phone-capture', '-cf', '-', *sessions], stdout=f)
    raw = OUT/'raw'
    raw.mkdir(exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(raw, filter='data')
    # Compare every transferred file with an independent phone-side checksum.
    checks = adb(['sh', '-c', 'cd no_backup/phone-capture && find ' + ' '.join(sessions) +
                  ' -type f -exec sha256sum {} +'], stdout=subprocess.PIPE).stdout
    (OUT/'phone_sha256.txt').write_bytes(checks)
    count = 0
    for line in checks.decode().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip('*')
        if sha(raw/name) != digest:
            raise ValueError('Transfer mismatch: '+name)
        count += 1
    save(OUT/'import_receipt.json', dict(sessions=sessions, files=count, bytes=archive.stat().st_size,
         archive_sha256=sha(archive), every_phone_sha_equal=True, seconds=time.monotonic()-started,
         serial=SERIAL, package=PACKAGE, source='App private no_backup/phone-capture; exact tar bytes',
         selection='All completed blank-note sessions since 14:59 local; noted integration tests excluded.'))
    print('IMPORTED', len(sessions), count, archive.stat().st_size, flush=True)


def prepare():
    from PIL import Image, ImageOps, ImageDraw
    (OUT/'inputs').mkdir(exist_ok=True)
    (OUT/'contact_sheets').mkdir(exist_ok=True)
    rotation = np.asarray(R.load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    cal = np.load(R.P.OUT/'instrument_arrays.npz')
    noise = R.P.read('instrument_parameters.json')['noise']
    rows = []
    for folder in sorted((OUT/'raw').iterdir()):
        manifest = json.loads((folder/'manifest.json').read_text())
        rgb = list(R.lines(folder/'rgb.jsonl'))
        sheet = Image.new('RGB', (6*260, 245), 'white')
        draw = ImageDraw.Draw(sheet)
        chosen = np.linspace(0, len(rgb)-1, 6).astype(int) if rgb else []
        for j, i in enumerate(chosen):
            item = rgb[i]
            file = item.get('file') or item.get('file_name') or item.get('path')
            if file is None:
                raise ValueError('Missing RGB file field: '+str(item.keys()))
            im = Image.open(folder/file).convert('RGB')
            # Existing accepted display mapping, applied only to the sheet.
            im = ImageOps.mirror(im.rotate(-90, expand=True))
            im.thumbnail((260, 215))
            sheet.paste(im, (j*260, 25))
            draw.text((j*260+3, 5), f'{folder.name[17:23]} index={i}', fill='black')
        sheet.save(OUT/'contact_sheets'/(folder.name+'.jpg'))
        d = R.load(folder, rotation)
        z = adapter_v5(d['H'], d['A'], cal['bias'].reshape(64,16), noise, None, True, False)
        np.savez_compressed(OUT/'inputs'/(folder.name+'.npz'), **d, z=z)
        pitch, roll = float(np.nanmean(d['pitch'])), float(np.nanmean(d['roll']))
        fit = R.floor_fit(d)
        if fit and (abs(fit[1]-pitch)>15 or not .5<=fit[0]<=2 or abs(fit[2])>20):
            fit = None
        arms = {'imu': [pitch,roll,0.]}
        if fit:
            arms.update(imu_h=[pitch,roll,R.SIM_FLOOR_M-fit[0]],
                        floor_h=[fit[1],fit[2],R.SIM_FLOOR_M-fit[0]])
        rows.append(dict(session=folder.name,manifest_counts={k:manifest[k] for k in ('rgb_count','tof_count','imu_count')},
             loaded_cnh=len(z), pitch_mean=pitch, pitch_sd=float(np.nanstd(d['pitch'])),
             roll_mean=roll, floor_fit=list(fit) if fit else None, arms=arms,
             scene='UNKNOWN until image review; no query truth', note=manifest.get('scene_note')))
    save(OUT/'prepared.json', rows)
    print('PREPARED', len(rows), sum(r['loaded_cnh'] for r in rows), flush=True)


def replay():
    import torch
    from local_signed_bias_adapter import ContractReplay
    torch.set_num_threads(4)
    dest = OUT/'joint_replay'; dest.mkdir(exist_ok=True)
    started = time.monotonic()
    engine = ContractReplay(str(MANIFEST), device='cuda')
    reports = []
    receipt = dict(manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST), device='cuda',
                   gpu=torch.cuda.get_device_name(), cuts_frozen=True, development=True, status='running')
    save(dest/'receipt.json', receipt)
    try:
        for row in json.loads((OUT/'prepared.json').read_text()):
            d = np.load(OUT/'inputs'/(row['session']+'.npz'))
            distance = np.where(np.isin(d['S'],(5,6,9,10)), d['D'], np.nan).reshape(-1,8,8)[:,::-1,::-1]
            for arm, pose in row['arms'].items():
                if time.monotonic()-started > 550:
                    raise RuntimeError('600s GPU budget margin reached')
                target = dest/(row['session']+'_'+arm+'.npz')
                meta = target.with_suffix('.json')
                if target.exists() and meta.exists():
                    reports.append(json.loads(meta.read_text())); continue
                engine.engine.configure_geometry(*pose)
                result = engine.infer(d['z'], d['seq'], d['ms'])
                np.savez_compressed(target, **result, seq=d['seq'],ms=d['ms'])
                summary = dict(session=row['session'],arm=arm,pose=pose,**R.summarize(result,distance))
                ready = result['ready']
                summary['strong_count'] = (result['S_grade'][ready]==2).sum(0).tolist()
                summary['light_count'] = (result['S_grade'][ready]==1).sum(0).tolist()
                save(meta, summary); reports.append(summary)
                print(row['session'],arm,summary['ready'],summary['strong_count'],summary['light_count'],flush=True)
    finally:
        engine.close(); torch.cuda.empty_cache()
        receipt.update(seconds=time.monotonic()-started, status='released', completed_cells=len(reports))
        save(dest/'receipt.json',receipt)
        save(dest/'summary.json',reports)


def readout():
    """Frozen quieter readout and range coverage from cached predictions."""
    dest = OUT/'joint_replay'
    quieter = ROOT/'artifacts.local/work/cnh-noise-ablation-dev-20261012/light_joint_v2/real_manifest_quieter_light.json'
    tau = json.loads(quieter.read_text())['cuts']['s_tau']
    tables = []
    for row in json.loads((dest/'summary.json').read_text()):
        pred = np.load(dest/(row['session']+'_'+row['arm']+'.npz'))
        d = np.load(OUT/'inputs'/(row['session']+'.npz'))
        ready = pred['ready']
        assert len(ready)==len(d['seq']) and np.array_equal(pred['seq'],d['seq'])
        assert row['strong_count']==(pred['S_grade'][ready]==2).sum(0).tolist()
        assert row['light_count']==(pred['S_grade'][ready]==1).sum(0).tolist()
        original = pred['S_grade']
        candidate = np.where(original==2,2,np.where((pred['s_score'].astype(float)>=tau)&
                    ready[:,None]&pred['evidence_present'][:,None],1,0)).astype(np.int8)
        assert np.array_equal(original==2,candidate==2)
        distance = np.where(np.isin(d['S'],(5,6,9,10)),d['D'],np.nan).reshape(-1,8,8)[:,::-1,::-1]
        # Unknown firmware ranges must not silently join the >=2m band.
        central = np.where(np.isfinite(distance[:,:,2:6]) & (distance[:,:,2:6]>0),distance[:,:,2:6],np.inf)
        near = central.min(axis=(1,2))
        bands = {'<1.2m':near<1200,'1.2-2m':(near>=1200)&(near<2000),
                 '>=2m':np.isfinite(near)&(near>=2000),'UNKNOWN':~np.isfinite(near)}
        coverage = {k:dict(ready=int((mask&ready).sum()),
                         any_notice=int(((original>0).any(1)&mask&ready).sum())) for k,mask in bands.items()}
        row['alert_by_centre_range']=coverage
        save(dest/(row['session']+'_'+row['arm']+'.json'),row)
        tables.append(dict(session=row['session'],arm=row['arm'],ready=int(ready.sum()),
                           original_strong=row['strong_count'], original_light=row['light_count'],
                           quieter_light=(candidate[ready]==1).sum(0).tolist(),
                           range_coverage=coverage,quieter_tau=tau))
    save(dest/'summary.json',[json.loads((dest/(r['session']+'_'+r['arm']+'.json')).read_text()) for r in tables])
    save(OUT/'cached_readout.json',dict(quieter_manifest=str(quieter),quieter_sha256=sha(quieter),
           original_strong_all_equal=True,selected_on_new_data=False,rows=tables))
    print('READOUT_VERIFIED',len(tables),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['import','prepare','replay','readout'])
    ap.add_argument('--adb');ap.add_argument('--serial');a=ap.parse_args()
    if a.stage=='import':
        ADB=a.adb or shutil.which('adb.exe');SERIAL=a.serial
        if not ADB or not SERIAL:ap.error('import requires --adb executable and --serial device identity')
    {'import':import_batch,'prepare':prepare,'replay':replay,'readout':readout}[a.stage]()

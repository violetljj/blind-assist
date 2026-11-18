"""Export local paired review clips from frozen inputs, without distributing source imagery."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import independent_low_replay_20261118 as frozen


ROOT = frozen.ROOT / 'artifacts.local/work/paired-low-sites-20261118'
DEST = ROOT / 'clips'
W, H, FPS = 800, 520, 10
FONT = Path('C:/Windows/Fonts/msyh.ttc')
FONTS = [ImageFont.truetype(str(FONT), n) for n in (17, 20, 23)]
NAMES = {'sem80': '仅阈值', 'geom': '几何保护'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def label_for(hits):
    if hits is None:
        return '缺少可评分观测'
    return '     '.join(name + ('：报警' if hits[key] else '：静默') for key, name in NAMES.items())


def panel(title, phase, frame, elapsed, message, masks):
    img = cv2.imread(str(frame['rgb']))
    if img is None:
        raise FileNotFoundError(frame['rgb'])
    scale = min((W-24)/img.shape[1], (H-105)/img.shape[0])
    size = (round(img.shape[1]*scale), round(img.shape[0]*scale))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    for mask in masks:
        m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(mask)
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
        lines, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, lines, -1, (30, 225, 245), 2)
    canvas = Image.new('RGB', (W,H), '#15222b')
    canvas.paste(Image.fromarray(img[:,:,::-1]), ((W-size[0])//2, 46+(H-105-size[1])//2))
    draw = ImageDraw.Draw(canvas)
    draw.text((15,7), title, font=FONTS[2], fill='white')
    draw.text((15,H-55), f'{phase}   +{elapsed:.2f} s   ·   帧 {frame["i"]}', font=FONTS[0], fill='#cbd9dc')
    draw.text((15,H-31), message, font=FONTS[1], fill='white')
    return canvas.tobytes()


def write_phase(pipe, title, phase, frames, statuses, masks):
    start = frames[0]['t']
    for j, frame in enumerate(frames):
        elapsed = frame['t'] - start
        frame_bytes = panel(title, phase, frame, elapsed, statuses.get(frame['i'], '接近过程，当前采样点未计分'),
                            masks.get(frame['i'], []))
        # Show observed timing. Give the final sample a short hold for reading the result.
        end = frames[j+1]['t']-start if j+1<len(frames) else elapsed+.6
        repeat = max(1, round(end*FPS)-round(elapsed*FPS))
        for _ in range(repeat):
            pipe.write(frame_bytes)


def export(manifest, labels_path, scores, ffmpeg, encounter_map):
    labels = load(labels_path)
    score = scores[sha(labels_path)]
    assert score['manifest_sha256'] == sha(manifest)
    _, rows = frozen.inputs(manifest.resolve())
    by_key = {(r['seq'],r['i']):r for r in rows}
    windows = {w['id']:w for w in labels['windows']}
    scored_windows = {w['id']:w for w in score['windows']}
    cases = {(c['object_id'],c['encounter_id']):c for c in score['cases']}
    groups = defaultdict(list)
    for c in labels['cases']:
        case_id = c['encounter_id']
        groups[encounter_map.get(case_id, c.get('physical_encounter_id', case_id))].append(c)
    entries = []
    for enc, members in groups.items():
        case = members[0]
        assert all(c['site_id']==case['site_id'] and c['route_id']==case['route_id'] for c in members)
        window_ids = list(dict.fromkeys(w for c in members for w in c['window_ids']))
        scored_members = [cases[c['object_id'],c['encounter_id']] for c in members]
        obs = [o for c in members for o in c['observations']]
        seq = obs[0]['seq']
        assert all(o['seq']==seq for o in obs)
        event = [by_key[(seq,o['i'])] for o in obs]
        first, last = min(r['t'] for r in event), max(r['t'] for r in event)
        event = [r for r in rows if r['seq']==seq and first<=r['t']<=last]
        control_ids = {(r['seq'],r['i']) for wid in window_ids for r in windows[wid]['frames']}
        lead_in = []
        for r in reversed([r for r in rows if r['seq']==seq and first-3<=r['t']<first]):
            next_time = lead_in[-1]['t'] if lead_in else first
            if next_time-r['t']>.7 or (seq,r['i']) in control_ids:
                break
            lead_in.append(r)
        event = list(reversed(lead_in)) + event
        masks, hit_rows = defaultdict(list), defaultdict(list)
        for o in obs:
            if o.get('near'):
                masks[o['i']].append(labels_path.parent/o['reference_footprint'])
        for scored in scored_members:
            for o in scored['observations']:
                if o.get('hits'):
                    hit_rows[o['i']].append(o['hits'])
        statuses = {i:label_for({k:any(entry[k]['ANY'] for entry in hits) for k in NAMES})
                    for i,hits in hit_rows.items()}
        name = enc.replace('/', '-').replace('\\','-') + '.mp4'
        cat = '挡轮块' if 'wheel' in case['category'].lower() else '矮台'
        place = case['site_id']
        title = f'{place}  ·  {cat}'
        data = dict(clip=name, site=place, objects=[c['object_id'] for c in members], encounter=enc,
                    category=case['category'], height=case['height_band'], history=case['history'],
                    labels_sha256=score['labels_sha256'], source_sha256=score['manifest_sha256'],
                    source_sequence=seq, target_frames=[r['i'] for r in event],
                    target_times_s=[r['t'] for r in event], object_hits={s['object_id']:s['hits'] for s in scored_members},
                    hits={a:{b:any(s['hits'][a][b] for s in scored_members) for b in ('LOW','ANY')}
                          for a in ('baseline','sem80','geom')}, windows=[])
        with tempfile.TemporaryDirectory(dir=DEST) as td:
            clip = Path(td)/'clip.mp4'
            args = [ffmpeg,'-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264',
                    '-threads','2','-preset','veryfast','-crf','21','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(clip)]
            with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
                try:
                    write_phase(proc.stdin,title,'接近目标',event,statuses,masks)
                    for wid in window_ids:
                        w, scored_w = windows[wid], scored_windows[wid]
                        frames = sorted([by_key[(o['seq'],o['i'])] for o in w['frames']], key=lambda r:r['t'])
                        status = {o['i']:(label_for(o['alarm']) if o.get('alarm') and
                                  all(isinstance(o['alarm'].get(k),bool) for k in NAMES) else '当前时刻覆盖不足')
                                  for o in scored_w['rows']}
                        for r in frames:
                            status.setdefault(r['i'],'当前时刻覆盖不足')
                        write_phase(proc.stdin,title,'同路线空路' if scored_w['complete'] else '同路观测・覆盖不足',
                                    frames,status,{})
                        data['windows'].append(dict(id=wid, complete=scored_w['complete'],
                            eligible_samples=scored_w['eligible_rows'], observed_s=scored_w['longest_observed_s'],
                            frames=[r['i'] for r in frames], times_s=[r['t'] for r in frames],
                            alarms=scored_w['alarm_windows']))
                finally:
                    proc.stdin.close()
                if proc.wait():
                    raise RuntimeError(f'Clip encoding failed for {enc}')
            cap = cv2.VideoCapture(str(clip))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            try:
                for i in (0,count//2,count-1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, sample = cap.read()
                    if not ok or sample.shape[:2]!=(H,W):
                        raise RuntimeError(f'Clip decode failed for {enc} at {i}')
            finally:
                cap.release()
            os.replace(clip,DEST/name)
        data['clip_sha256'] = sha(DEST/name)
        entries.append(data)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs=2, action='append', type=Path, required=True, metavar=('MANIFEST','LABELS'))
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--encounter-map', type=Path, help='Object-specific IDs sharing a documented walking pass')
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error('Provide the path to ffmpeg with --ffmpeg')
    DEST.mkdir(parents=True, exist_ok=True)
    scores = {}
    for p in (ROOT/'scoring').glob('*.json'):
        s = load(p)
        if 'labels_sha256' in s:
            scores[s['labels_sha256']] = s
    encounter_map = load(args.encounter_map) if args.encounter_map else {}
    entries = [e for source,labels in args.pair for e in export(source,labels,scores,args.ffmpeg,encounter_map)]
    assert len({e['clip'] for e in entries})==len(entries)
    cards = []
    for e in entries:
        name = html.escape(e['clip'],quote=True)
        card = html.escape(e['site'] + ' · ' + e['encounter'])
        cards.append(f'<article><h2>{card}</h2><video preload="metadata" playsinline controls src="{name}"></video></article>')
    (DEST/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>近距障碍配对片段</title><style>body{max-width:920px;'
        'margin:2rem auto;padding:0 1rem;font:1rem/1.6 system-ui;color:#15222b}h2{font-size:1.1rem}video{width:100%;'
        'max-width:800px;background:#15222b}article{margin:2rem 0}</style><h1>近距障碍配对片段</h1>'
        '<p>每段先播放接近目标，再播放同路线空路。黄色轮廓标出盲审确认的近距实物面积。按原始采样时刻呈现，'
        '每个采样画面停留到下一次观测。画面下方分别显示两种方法是否在目标上触发提醒，空路段显示任一带报警。'
        '“覆盖不足”的时刻不计误报。原始数据来源和许可见实验报告。</p>' + ''.join(cards) + '</html>',encoding='utf-8')
    (DEST/'pairs.json').write_text(json.dumps(entries,ensure_ascii=False,indent=1),encoding='utf-8')
    print(len(entries),'paired clips in',DEST)


if __name__=='__main__':
    main()

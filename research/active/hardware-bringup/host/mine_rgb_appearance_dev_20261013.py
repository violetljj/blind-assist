"""Describe appearance coverage in already-consumed RGB; no model or labels.

Distances are low-resolution, unregistered image differences, not semantic
novelty, detector scores, or sensor synchronization evidence.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts.local/work/data-potential-dev-20261013'
OUT = BASE / 'deeper/rgb_appearance'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def quantiles(values):
    return dict(zip(['p10', 'p50', 'p90', 'max'],
                    np.quantile(values, [.1, .5, .9, 1]).tolist())) if values else None


def thumbnail(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L').resize((80, 60), Image.Resampling.BOX),
                          dtype=np.float32)


def display(path):
    with Image.open(path) as image:
        # Accepted mount display: clockwise 90 degrees, then horizontal mirror.
        upright = ImageOps.mirror(image.convert('RGB').transpose(Image.Transpose.ROTATE_270))
        upright.thumbnail((175, 210))
        return upright.copy()


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = read(BASE / 'local_mining/per_session.json')
    manifest = BASE / 'local_mining/uncached_unique_rgb.jsonl'
    extra = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x]
    assert len(extra) == 5407 and len(sessions) == 18
    save('selection_before_read.json', {
        'sessions': [s['session_key'] for s in sessions], 'expected_raw': 6746,
        'expected_uncached': 5407, 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'selection': 'All indexed RGB in the 18 consumed sessions; no outcome filtering.',
        'thumbnail': '80x60 grayscale BOX; raw MAE and per-frame median-centered MAE.',
        'matching': 'nearest Android receipt time among cached RGB from the same source and boot',
        'contact_sheet_selection': 'per session centered-MAE p50/p90 representatives and largest receipt gap, stable index ties; if no cached neighbor, first/middle/last uncached observations; deduplicate',
        'limits': 'Unregistered appearance only; no semantic novelty, new labels, pose, inference, or synchronization claim.',
        'cpu_command_wall_budget_s': 180, 'gpu_s': 0, 'downloads': 0})
    uncached = {(r['session_key'], r['raw_index_one_based']) for r in extra}
    output = []; summaries = []; selected = []; raw_count = 0; cached_count = 0
    for session in sessions:
        if time.perf_counter() - start > 150:
            raise TimeoutError('Reserve 30 seconds for evidence output within budget')
        key = session['session_key']
        folder = ROOT / 'artifacts.local/work/phone-new-capture-20261012/raw' / session['session']
        rows = [json.loads(x) for x in (folder / 'rgb.jsonl').read_text(encoding='utf-8').splitlines() if x]
        thumbs = [thumbnail(folder / row['file']) for row in rows]
        cached = [i for i, r in enumerate(rows) if (key, r['index']) not in uncached]
        groups = {}
        for i in cached:
            r = rows[i]; ident = (r['source_id'], r['boot_id'])
            groups.setdefault(ident, []).append(i)
        for indices in groups.values():
            indices.sort(key=lambda i: rows[i]['received_elapsed_ns'])
        local = []
        for i, row in enumerate(rows):
            if (key, row['index']) not in uncached:
                continue
            t = row['received_elapsed_ns']; a = thumbs[i]
            indices = groups.get((row['source_id'], row['boot_id']), [])
            record = dict(session_key=key, raw_index_one_based=row['index'],
                          file=(folder / row['file']).relative_to(ROOT).as_posix(),
                          received_elapsed_ns=t, sha256=row['sha256'],
                          gray_mean=float(a.mean()), gray_std=float(a.std()),
                          label_status='UNLABELED_CONSUMED_DEVELOPMENT', nearest_cached=None)
            if indices:
                times = [rows[j]['received_elapsed_ns'] for j in indices]
                p = bisect.bisect_left(times, t)
                choices = indices[max(0, p-1):min(len(indices), p+1)]
                j = min(choices, key=lambda j: (abs(rows[j]['received_elapsed_ns']-t), j))
                b = thumbs[j]
                record.update(nearest_cached=dict(raw_index_one_based=rows[j]['index'],
                    file=(folder / rows[j]['file']).relative_to(ROOT).as_posix(),
                    signed_receipt_delta_ms=(t-rows[j]['received_elapsed_ns'])/1e6),
                    receipt_gap_ms=abs(t-rows[j]['received_elapsed_ns'])/1e6,
                    gray_mae_0_255=float(np.abs(a-b).mean()),
                    centered_gray_mae_0_255=float(np.abs((a-np.median(a))-(b-np.median(b))).mean()))
            local.append(record)
        comparable = [r for r in local if r['nearest_cached'] is not None]
        summary = dict(session_key=key, scene_name_weak=session['scene_name_weak'],
            raw=len(rows), cached=len(cached), uncached=len(local), comparable=len(comparable),
            centered_gray_mae=quantiles([r['centered_gray_mae_0_255'] for r in comparable]),
            receipt_gap_ms=quantiles([r['receipt_gap_ms'] for r in comparable]))
        summaries.append(summary)
        if comparable:
            ranked = sorted(comparable, key=lambda r:(r['centered_gray_mae_0_255'], r['raw_index_one_based']))
            picks = [(ranked[round((len(ranked)-1)*q)], f'appearance_p{round(q*100)}') for q in [.5,.9]]
            picks.append((max(comparable, key=lambda r:(r['receipt_gap_ms'], -r['raw_index_one_based'])), 'largest_receipt_gap'))
        else:
            picks = [(local[round((len(local)-1)*q)], f'no_cached_neighbor_time_p{round(q*100)}') for q in [0,.5,1]] if local else []
        seen = set()
        for rec, rule in picks:
            if rec['raw_index_one_based'] not in seen:
                selected.append(dict(rec, selection_rule=rule)); seen.add(rec['raw_index_one_based'])
        output.extend(local); raw_count += len(rows); cached_count += len(cached)
    assert raw_count == 6746 and cached_count == 1339 and len(output) == 5407
    (OUT/'per_uncached_rgb.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in output), encoding='utf-8')
    save('per_session.json', summaries); save('review_candidates.json', selected)
    for offset in range(0, len(selected), 9):
        panel = Image.new('RGB', (1080, 810), 'white'); draw = ImageDraw.Draw(panel)
        for n, rec in enumerate(selected[offset:offset+9]):
            x=(n%3)*360; y=(n//3)*270
            neighbor = rec['nearest_cached']
            cached_label = neighbor['raw_index_one_based'] if neighbor else 'NONE'
            draw.text((x+4,y+4), f"{rec['session_key']} raw {rec['raw_index_one_based']} / cached {cached_label}", fill='black')
            detail = f"d={rec['centered_gray_mae_0_255']:.1f} | {rec['receipt_gap_ms']:.1f}ms" if neighbor else 'No same-source/boot cached RGB'
            draw.text((x+4,y+20), f"{rec['selection_rule']} | {detail}", fill='black')
            panel.paste(display(ROOT/rec['file']), (x+4,y+43))
            if neighbor:
                panel.paste(display(ROOT/neighbor['file']), (x+182,y+43))
        panel.save(OUT/f'review_{offset//9+1:02d}.jpg', quality=88)
    comparable=[r for r in output if r['nearest_cached'] is not None]
    result=dict(raw=raw_count, cached=cached_count, uncached=len(output), comparable=len(comparable),
        no_same_boot_cached_neighbor=len(output)-len(comparable),
        centered_gray_mae=quantiles([r['centered_gray_mae_0_255'] for r in comparable]),
        receipt_gap_ms=quantiles([r['receipt_gap_ms'] for r in comparable]),
        review_candidates=len(selected), contact_sheets=(len(selected)+8)//9,
        wall_s=time.perf_counter()-start,
        limits='All pixels read for coverage, but thumbnail differences are unregistered and can reflect motion, lighting, blur or texture. Ranking is for inspection, not labels, semantic independence or detector gain.')
    save('summary.json', result); print(json.dumps(result))


if __name__ == '__main__':
    main()

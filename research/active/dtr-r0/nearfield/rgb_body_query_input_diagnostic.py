"""Full-frame input loss on consumed SANPO-Real train anchors, not detection.

Inference inputs/crops never use labels. Human panoptic masks are evaluator-only
for projected pole support statistics; these are not metric obstacle thickness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def source_frames(repo):
    source = repo / 'artifacts.local/work/sanpo-coverage-20260927'
    data = repo / 'artifacts.local/datasets/sanpo-real-coverage-20260927'
    proposal = json.loads((source / 'download_proposal.json').read_text('utf-8-sig'))
    receipt = json.loads((source / 'acquisition_receipt.json').read_text('utf-8-sig'))
    hashes = {r['path'].replace('\\', '/'): r['sha256'] for r in receipt['files']}
    rows = []
    for row in proposal['frames']:
        base = f"{row['session']}/{row['camera']}/left"
        relative = f"{base}/video_frames/{row['frame']:06d}.png"
        rows.append({**row, 'rgb_path': data / relative,
                     'mask_path': data / f"{base}/segmentation_masks/{row['frame']:06d}.png",
                     'description_path': data / row['session'] / 'description.json',
                     'rgb_sha256': hashes[relative],
                     'mask_sha256': hashes[f"{base}/segmentation_masks/{row['frame']:06d}.png"],
                     'description_sha256': hashes[f"{row['session']}/description.json"]})
    return rows


def detail(gray):
    gray = np.asarray(gray, dtype=np.float32)
    # Mean absolute first differences on the same native pixel lattice.
    return float((np.abs(np.diff(gray, axis=0)).mean() +
                  np.abs(np.diff(gray, axis=1)).mean()) / 2)


def run(repo, output, budget_s):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    source = source_frames(repo)
    frames, objects, observations = [], [], []
    first_by_session = {}
    for row in source:
        if time.perf_counter() - start >= budget_s:
            raise TimeoutError('CPU diagnostic budget reached')
        for kind in ('rgb', 'mask', 'description'):
            if sha(row[f'{kind}_path']) != row[f'{kind}_sha256']:
                raise ValueError(f'Changed source {kind}: {row["session"]} {row["frame"]}')
        image = Image.open(row['rgb_path']).convert('RGB')
        mask = np.asarray(Image.open(row['mask_path']))
        w, h = image.size
        if mask.shape != (h, w, 3):
            raise ValueError('Native mask/RGB mismatch')
        gray = np.asarray(image.convert('L'))
        native_detail = detail(gray)
        record = {k: row[k] for k in ('session', 'camera', 'frame', 'rgb_sha256')}
        record.update(native_width=w, native_height=h, native_detail=native_detail)
        versions = [image]
        for tw, th in ((128, 72), (256, 144), (640, 360)):
            low = image.resize((tw, th), Image.Resampling.BILINEAR)
            reconstructed = low.resize((w, h), Image.Resampling.BILINEAR)
            detail_low = detail(np.asarray(reconstructed.convert('L')))
            record[f'detail_ratio_{tw}'] = detail_low / native_detail if native_detail else None
            if tw in (128, 640):
                versions.append(reconstructed)
        frames.append(record)
        first_by_session.setdefault(row['session'], versions)
        observations.append(dict(session=row['session'], camera=row['camera'], frame=row['frame'],
                                 rgb=str(row['rgb_path'].relative_to(repo)), rgb_sha256=row['rgb_sha256'],
                                 description=str(row['description_path'].relative_to(repo)),
                                 description_sha256=row['description_sha256']))
        instances = mask[..., 1].astype(np.int32) * 256 + mask[..., 2]
        pole = mask[..., 0] == 24
        for iid in np.unique(instances[pole]):
            ys, xs = np.nonzero(pole & (instances == iid))
            area = len(xs)
            bh, bw = int(ys.max()-ys.min()+1), int(xs.max()-xs.min()+1)
            average_row_support = area / bh
            objects.append(dict(session=row['session'], camera=row['camera'], frame=row['frame'],
                                instance=int(iid), native_area=area, bbox_width=bw, bbox_height=bh,
                                identity_status='UNKNOWN_ZERO_ID' if iid == 0 else 'POSITIVE_ID_MASK_GROUP',
                                native_mean_row_support_px=average_row_support,
                                input128_mean_row_support_px=average_row_support*128/w,
                                input640_mean_row_support_px=average_row_support*640/w))
        print(f'{len(frames)}/{len(source)} image diagnostic', flush=True)
    summary = dict(frames=len(frames), sessions=len(first_by_session),
                   native_shapes=sorted({(r['native_width'], r['native_height']) for r in frames}),
                   pole_class_id_group_frames=len(objects),
                   zero_id_group_frames=sum(r['instance'] == 0 for r in objects),
                   positive_id_group_frames=sum(r['instance'] > 0 for r in objects),
                   pole_input128_support_lt1=sum(r['input128_mean_row_support_px'] < 1 for r in objects),
                   pole_input128_support_lt2=sum(r['input128_mean_row_support_px'] < 2 for r in objects),
                   pole_positive_unique_session_camera_ids=len({(r['session'], r['camera'], r['instance']) for r in objects if r['instance'] > 0}),
                   source_role='already consumed SANPO-Real official train Development; human anchor frames',
                   detail_definition='mean absolute gray first difference after full-frame down/up bilinear resize on native grid',
                   pole_definition='human class/id mask area/bbox_height times input_width/native_width; not physical thickness; zero ID retained with unknown identity',
                   crops='fixed middle-third image region, identical all versions; no GT/model-selected crops',
                   inference_run=False, event_evaluation='NOT_RUN',
                   cpu_reason='TASK_NOT_GPU_SUITABLE', seconds=time.perf_counter()-start)
    for tw in (128, 256, 640):
        values = [r[f'detail_ratio_{tw}'] for r in frames if r[f'detail_ratio_{tw}'] is not None]
        summary[f'median_detail_ratio_{tw}'] = float(np.median(values)) if values else None
    write(output / 'input_diagnostic.json', dict(summary=summary, frames=frames, poles=objects))
    write(output / 'observations.json', observations)
    # One fixed frame per session, fixed center crop. No outcome-based selection.
    sheet = Image.new('RGB', (1440, 300*len(first_by_session)), '#111111')
    draw = ImageDraw.Draw(sheet)
    for j, (sid, versions) in enumerate(first_by_session.items()):
        for k, version in enumerate(versions):
            w, h = version.size
            crop = version.crop((w//3, h//3, 2*w//3, 2*h//3))
            sheet.paste(crop.resize((480, 270)), (480*k, 300*j+30))
            label = ('Native RGB', '128x72 round-trip', '640x360 round-trip')[k]
            draw.text((480*k+8, 300*j+7), f'{sid[:8]} | {label}', fill='white')
    sheet.save(output / 'fixed_crop_comparison.png')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=600)
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.budget_s)

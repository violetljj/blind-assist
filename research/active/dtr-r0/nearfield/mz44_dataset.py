"""Reuse the verified native pipeline, then independently admit far relation groups."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(capture, output, runtime, helper):
    loader = importlib.util.spec_from_file_location('retained_mz42_dataset', helper)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    module.build(capture, output, runtime)
    native = read(output / 'result.json')
    spec = read(capture / 'source/spec.json')
    rows = []
    for row, case in zip(native['records'], spec['cases']):
        assert row['frame_id'] == case['name']
        rows.append(dict(row, range_variant=case['range_variant'], expected_events=case['expected_events'],
                         far_intent_matches=row['event_truth'] == [bool(v) for v in case['expected_events']],
                         near_leakage=bool(row['event_truth'][0] or row['event_truth'][2])))
    groups = []
    for family in dict.fromkeys(row['family'] for row in rows):
        group = [row for row in rows if row['family'] == family]
        assert len(group) == 4
        accepted = all(row['source_valid'] and row['far_intent_matches'] for row in group)
        groups.append(dict(family=family, attempted_frames=4, admitted=accepted,
            intent_matching_frames=sum(row['far_intent_matches'] for row in group),
            near_leakage_frames=sum(row['near_leakage'] for row in group)))
    truth = np.array([row['event_truth'] for row in rows])
    with Image.open(output / 'contact-sheet.jpg') as original:
        preview = original.convert('RGB')
    draw = ImageDraw.Draw(preview)
    for i, row in enumerate(rows):
        x, y = i % 4 * 320, i // 4 * 410
        draw.rectangle((x, y, x+319, y+28), fill='#171c20')
        draw.text((x+5, y+3), row['family'] + ' / ' + row['range_variant'], fill='white')
        draw.rectangle((x, y+395, x+319, y+409), fill='#171c20')
        draw.text((x+5, y+395), 'BN/BF/HN/HF: ' + str([int(v) for v in row['event_truth']]), fill='white')
    preview.save(output / 'far-contact-sheet.jpg', quality=94)
    result = dict(status='PASS', attempted_frames=20, source_valid_frames=native['source_valid_frames'],
        unknown_event_bits=native['unknown_bits'], groups=groups, admitted_pair_groups=sum(g['admitted'] for g in groups),
        event_order=['BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR'],
        positive_frames_by_query=truth.sum(0).tolist(), negative_frames_by_query=(~truth).sum(0).tolist(),
        near_leakage_frames=sum(row['near_leakage'] for row in rows),
        native_support_pixels_by_query=np.array([row['event_counts'] for row in rows]).sum(0).tolist(), records=rows,
        visual_review='PENDING', training_steps=0, model_inference_frames=0,
        semantics='Actual visible native events; far intent/pair admission is separate; no invalid-to-CLEAR conversion')
    path = output / 'range-result.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    receipt = dict(status='PASS', code_sha256=sha(__file__), shared_helper_sha256=sha(helper),
        inputs={str(p): sha(p) for p in (capture/'source/spec.json', output/'receipt.json', output/'result.json')},
        outputs={name: sha(output/name) for name in ('range-result.json', 'far-contact-sheet.jpg')})
    (output/'range-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'admitted_pair_groups', 'positive_frames_by_query', 'near_leakage_frames')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'output', 'runtime', 'helper'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    build(args.capture.resolve(), args.output.resolve(), args.runtime.resolve(), args.helper.resolve())

"""Label and package the bounded MZ42 render; native surfaces, never mesh bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(capture, output, runtime):
    sys.path.insert(0, str(runtime / 'research/active/dtr-r0/nearfield'))
    sys.path.insert(0, str(Path(__file__).parent / 'labels-deps'))
    from body_query_collection_labels import verify_capture, floor_acceptance
    from body_query_labels import labels
    from multizone64_observation import native_events
    output.mkdir(parents=True, exist_ok=False)
    spec, world, inputs, probes = verify_capture(capture)
    assert len(spec['cases']) == 20
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    expected = {'HEAD_ONLY': [0, 1], 'BODY_ONLY': [1, 0], 'BOTH': [1, 1], 'CLEAR': [0, 0]}
    records, truths, knowns = [], [], []
    board = Image.new('RGB', (4 * 320, 5 * 410), '#171c20')
    draw = ImageDraw.Draw(board)
    with torch.inference_mode():
        for index, (case, row) in enumerate(zip(spec['cases'], world['rows'])):
            path = capture / f'evaluator/native/{index:04d}.npy'
            depth = np.load(path, allow_pickle=False)
            tensor = torch.from_numpy(depth).cuda()
            native = labels(tensor, case['camera'], case['floor_z_m'])
            actual = native['near'].cpu().tolist()
            np.testing.assert_array_equal(native['support'].cpu().numpy(), np.load(capture / row['mask_path'], allow_pickle=False))
            assert actual == row['body_head_visible_targets']
            events = native_events(tensor[None], crop=False)
            truth = events['events'][0].cpu().tolist()
            floor = floor_acceptance(case, probes)
            observed = bool(events['observation_valid'][0])
            intent_matches = actual == expected[case['variant_id']]
            frame_ok = floor['accepted'] and observed
            records.append(dict(index=index, frame_id=case['name'], family=case['condition']['family'],
                variant=case['variant_id'], intended_relation=expected[case['variant_id']], actual_body_head=actual,
                event_truth=truth, event_counts=events['counts'][0].cpu().tolist(),
                floor=floor, observation_valid=observed, intent_matches=intent_matches,
                source_valid=frame_ok, rgb_sha256=row['rgb_sha256'], native_sha256=row['native_sha256'],
                rgb=f'model/sample/{index:04d}.png', native=f'evaluator/native/{index:04d}.npy'))
            truths.append(truth)
            knowns.append([frame_ok] * 4)
            with Image.open(capture / f'model/sample/{index:04d}.png') as source:
                rgb = source.convert('RGB')
            x, y = index % 4 * 320, index // 4 * 410
            board.paste(rgb.resize((320, 180)), (x, y + 30))
            # Native axial range preview, fixed0..5m scale; display only, not labels.
            shade = np.uint8(np.clip(1. - depth / 5., 0., 1.) * 255)
            depth_rgb = np.stack((shade, shade, shade), -1)
            support = native['support'].cpu().numpy()
            depth_rgb[support[0] == 1] = [80, 180, 255]
            depth_rgb[support[1] == 1] = [255, 120, 80]
            board.paste(Image.fromarray(depth_rgb).resize((320, 180)), (x, y + 215))
            draw.text((x + 5, y + 3), f"{case['condition']['family']} / {case['variant_id']}", fill='white')
            draw.text((x + 5, y + 395), f'Native B/H: {actual}   intent: {intent_matches}', fill='white')
    groups = []
    for family in dict.fromkeys(row['family'] for row in records):
        group = [row for row in records if row['family'] == family]
        assert len(group) == 4 and {row['variant'] for row in group} == set(expected)
        accepted = all(row['source_valid'] and row['intent_matches'] for row in group)
        groups.append(dict(family=family, attempted_frames=4, source_valid_frames=sum(row['source_valid'] for row in group),
                           intent_matching_frames=sum(row['intent_matches'] for row in group), complete_pair_group=accepted))
    # Observed labels stay valid even when design intent differs; pair admission is separate.
    known = np.array(knowns, bool)
    truth = np.array(truths, bool)
    np.savez_compressed(output / 'labels.npz', frame_ids=np.array([row['frame_id'] for row in records]),
                        truth=truth, known=known, labels=np.where(known, truth.astype(np.int8), -1))
    board.save(output / 'contact-sheet.jpg', quality=94)
    result = dict(status='PASS', attempted_frames=20, source_valid_frames=int(known.all(1).sum()),
        unknown_bits=int((~known).sum()), groups=groups, records=records,
        complete_pair_groups=sum(group['complete_pair_group'] for group in groups),
        label_authority='Visible native3D surfaces; >=3pixels/event; intent and bounds not labels',
        visual_review='PENDING; inspect RGB/native previews before accepting diversity/holes',
        source_role='Controlled synthetic Development, shared prior site/background, no model outputs',
        training_steps=0, model_inference_frames=0)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    inputs.update({str(path): sha(path) for path in (Path(__file__).parent / 'labels-deps').glob('*.py')})
    receipt = dict(status='PASS', inputs=inputs, code_sha256=sha(__file__), backend='CUDA native geometry; CPU image preview',
                   outputs={path.name: sha(path) for path in output.iterdir() if path.is_file()},
                   training_steps=0, model_inference_frames=0)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', groups=groups, source_valid_frames=result['source_valid_frames'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('capture', 'output', 'runtime'):
        parser.add_argument('--' + argument, type=Path, required=True)
    args = parser.parse_args()
    build(args.capture.resolve(), args.output.resolve(), args.runtime.resolve())

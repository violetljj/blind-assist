"""Label four rendered rod interventions using the unchanged MZ42 native helpers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(capture, output, runtime, dependencies):
    sys.path.insert(0, str(runtime / 'research/active/dtr-r0/nearfield'))
    sys.path.insert(0, str(dependencies))
    from body_query_collection_labels import verify_capture, floor_acceptance
    from body_query_labels import labels
    from multizone64_observation import native_events
    spec, world, inputs, probes = verify_capture(capture)
    assert len(spec['cases']) == 4 and len(world['rows']) == 4
    assert torch.cuda.is_available()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records = []
    board = Image.new('RGB', (1280, 410), '#171c20')
    draw = ImageDraw.Draw(board)
    with torch.inference_mode():
        for i, (case, row) in enumerate(zip(spec['cases'], world['rows'])):
            assert len(case['objects']) == 1 and case['objects'][0]['name'] == 'adjustable_cross_member'
            depth = np.load(capture / f'evaluator/native/{i:04d}.npy', allow_pickle=False)
            tensor = torch.from_numpy(depth).cuda()
            native = labels(tensor, case['camera'], case['floor_z_m'])
            support = native['support'].cpu().numpy()
            np.testing.assert_array_equal(support, np.load(capture / row['mask_path'], allow_pickle=False))
            actual = native['near'].cpu().tolist()
            assert actual == row['body_head_visible_targets']
            events = native_events(tensor[None], crop=False)
            truth = events['events'][0].cpu().tolist()
            floor = floor_acceptance(case, probes)
            observed = bool(events['observation_valid'][0])
            records.append(dict(index=i, frame_id=case['name'], original_case_id=case['name'],
                family='oblique_rod', variant=case['variant_id'], range_variant=case['range_variant'],
                actual_body_head=actual, event_truth=truth, event_counts=events['counts'][0].cpu().tolist(),
                floor=floor, observation_valid=observed, source_valid=bool(floor['accepted'] and observed),
                expected_events=case['expected_events'], intent_matches=truth == [bool(v) for v in case['expected_events']],
                rgb_sha256=row['rgb_sha256'], native_sha256=row['native_sha256'],
                rgb=f'model/sample/{i:04d}.png', native=f'evaluator/native/{i:04d}.npy'))
            with Image.open(capture / f'model/sample/{i:04d}.png') as source:
                board.paste(source.convert('RGB').resize((320, 180)), (i * 320, 30))
            shade = np.uint8(np.clip(1. - depth / 5., 0., 1.) * 255)
            colored = np.stack((shade, shade, shade), -1)
            colored[support[0] == 1] = [80, 180, 255]
            colored[support[1] == 1] = [255, 120, 80]
            board.paste(Image.fromarray(colored).resize((320, 180)), (i * 320, 215))
            draw.text((i * 320 + 5, 3), 'Unsupported rod / ' + case['range_variant'], fill='white')
            draw.text((i * 320 + 5, 395), 'BN/BF/HN/HF: ' + str([int(v) for v in truth]), fill='white')
    truth = np.array([r['event_truth'] for r in records], bool)
    known = np.array([[r['source_valid']] * 4 for r in records], bool)
    np.savez_compressed(output / 'labels.npz', frame_ids=np.array([r['frame_id'] for r in records]),
                        truth=truth, known=known, labels=np.where(known, truth.astype(np.int8), -1))
    board.save(output / 'contact-sheet.jpg', quality=94)
    result = dict(status='PASS', attempted_frames=4, source_valid_frames=int(known.all(1).sum()),
        unknown_bits=int((~known).sum()), records=records, positive_frames_by_query=truth.sum(0).tolist(),
        near_leakage_frames=int(truth[:, [0, 2]].any(1).sum()), intent_matching_frames=sum(r['intent_matches'] for r in records),
        label_authority='Unchanged native visible surfaces, >=3 pixels/event; expectation is not truth',
        visual_review='PENDING', training_steps=0, model_inference_frames=0)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    inputs.update({str(p): sha(p) for p in dependencies.glob('*.py')})
    receipt = dict(status='PASS', inputs=inputs, code_sha256=sha(__file__),
                   outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                   backend='CUDA native geometry; no learned model execution')
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ('status', 'source_valid_frames', 'positive_frames_by_query', 'unknown_bits')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'output', 'runtime', 'dependencies'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    build(args.capture.resolve(), args.output.resolve(), args.runtime.resolve(), args.dependencies.resolve())

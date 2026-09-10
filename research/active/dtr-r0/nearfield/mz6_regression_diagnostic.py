"""Zero-fit, consumed MZ5 regression attribution; no decision rule selection."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from mz5_fixed_ensemble import EVENTS, load, metric, read, sha, write


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    work = root / 'artifacts.local/work'
    allowed = (work / 'mz6-regression-20260910').resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('Use a fresh child of the canonical MZ6 diagnostic root')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = dict(index=work/'body-query-5000-20260909/dataset-v1/index.json',
        mz1=work/'mz1-tiny-fusion-20260910/run-v1/predictions.npz',
        mz1_receipt=work/'mz1-tiny-fusion-20260910/run-v1/receipt.json',
        mz5=work/'mz5-fixed-ensemble-20260910/run-v1/predictions.npz',
        mz5_receipt=work/'mz5-fixed-ensemble-20260910/run-v1/receipt.json',
        mz5_result=work/'mz5-fixed-ensemble-20260910/run-v1/result.json',
        depth=work/'mz0-clean-20260910/run-v1/depth.npz',
        depth_receipt=work/'mz0-clean-20260910/run-v1/depth-receipt.json',
        source=Path(__file__))
    hashes = {k: sha(p) for k, p in paths.items()}
    write(output/'protocol.json', dict(scope='Consumed controlled EVAL1500 diagnostic',
        selection='All exact regressions against MZ1, plus all event TP losses/gains',
        categories='Both branches wrong; RGB correct but overruled; ToF correct but overruled',
        margins='Raw logits and descriptive absolute-margin bins <=0.25, <=0.5, <=1; no tuning',
        observation='All regression RGB images, saved generic packets and evaluator native support',
        limits='Branch logit failure does not prove missing information; center support is approximate',
        input_sha256=hashes, training_steps=0, model_forward_passes=0))
    try:
        r1, r5 = read(paths['mz1_receipt']), read(paths['mz5_receipt'])
        assert hashes['mz1'] == r1['predictions_sha256']
        assert hashes['mz5'] == r5['output_sha256']['predictions.npz']
        assert hashes['index'] == r5['input_sha256']['index']
        assert hashes['depth'] == read(paths['depth_receipt'])['depth_sha256']
        allrows = read(paths['index'])['frames']
        rows = [r for r in allrows if r['source_role'] == 'EVAL_ONLY']
        a, b, d = load(paths['mz1']), load(paths['mz5']), load(paths['depth'])
        ev = a['role'] == 'EVAL_ONLY'
        truth = b['truth']
        np.testing.assert_array_equal(b['frame_id'], [r['frame_id'] for r in rows])
        np.testing.assert_array_equal(truth, a['truth'][ev])
        np.testing.assert_array_equal(truth, d['D_full_events'])
        np.testing.assert_array_equal(b['original_alerts'], a['original_alerts'][ev])
        rgb, tof, old = (a[k+'_logits'][ev] for k in ('RGB_ONLY','TOF_ONLY','FUSION'))
        ens = (rgb + tof) * .5
        np.testing.assert_array_equal(ens, b['ENSEMBLE_logits'])
        p, q = ens >= 0, old >= 0
        correct, prior = (p == truth).all(1), (q == truth).all(1)
        lost, gained = np.flatnonzero(prior & ~correct), np.flatnonzero(correct & ~prior)
        expected = read(paths['mz5_result'])
        assert metric(p, truth) == expected['aggregate']['ENSEMBLE']
        assert b['frame_id'][lost].tolist() == expected['comparisons']['FUSION']['lost_frame_ids']
        assert (len(lost), len(gained)) == (27, 52)

        def bit(i, k):
            rc, tc = bool((rgb[i,k] >= 0) == truth[i,k]), bool((tof[i,k] >= 0) == truth[i,k])
            category = ('BOTH_BRANCHES_WRONG' if not rc and not tc else
                        'RGB_CORRECT_OVERRULED' if rc else 'TOF_CORRECT_OVERRULED')
            assert not (rc and tc)  # A positive-weight mean cannot lose agreement.
            return dict(event=EVENTS[k], truth=bool(truth[i,k]), category=category,
                RGB=float(rgb[i,k]), ToF=float(tof[i,k]), MZ1=float(old[i,k]), MZ5=float(ens[i,k]),
                native_full_pixels=int(d['D_full_counts'][i,k]),
                native_crop_pixels=int(d['D_crop_counts'][i,k]),
                zone_center_support=bool(d['z8_multi_surface_events'][i,k]))

        records = []
        observation_hashes = {}
        for i in lost:
            row = rows[i]
            for key, hash_key in [('rgb_file','rgb_sha256'), ('native_file','native_sha256')]:
                path = Path(row[key]); digest = sha(path)
                assert digest == row[hash_key]
                observation_hashes[str(path)] = digest
            packet_valid = d['z8_multi_surface_valid'][i]
            records.append(dict(frame_id=int(b['frame_id'][i]),
                metadata={k:row[k] for k in ('condition','declared_range','family','region_id','name','rgb_file')},
                truth=truth[i].tolist(), logits={k:v[i].tolist() for k,v in
                    dict(RGB=rgb,ToF=tof,MZ1=old,MZ5=ens).items()},
                errors=[bit(i,k) for k in range(4) if p[i,k] != truth[i,k]],
                valid_zones=int(packet_valid.any(1).sum()), two_return_zones=int(packet_valid.all(1).sum()),
                packet_range_m=np.where(packet_valid,d['z8_multi_surface_range'][i],0).tolist(),
                packet_valid=packet_valid.tolist()))
        write(output/'regressions.json', records)
        transitions = {}
        for k, event in enumerate(EVENTS):
            losses = np.flatnonzero(truth[:,k] & q[:,k] & ~p[:,k])
            gains = np.flatnonzero(truth[:,k] & ~q[:,k] & p[:,k])
            details = [dict(frame_id=int(b['frame_id'][i]), **bit(i,k)) for i in losses]
            transitions[event] = dict(tp_losses=len(losses),tp_gains=len(gains),
                gain_frame_ids=b['frame_id'][gains].tolist(), losses=details,
                loss_categories=dict(Counter(x['category'] for x in details)))
        errors = [x for row in records for x in row['errors']]
        summary = dict(status='PASS', frames=1500, exact_gains=len(gained), exact_losses=len(lost),
            regression_rows_by_condition=dict(Counter(r['metadata']['condition'] for r in records)),
            regression_rows_by_range=dict(Counter(r['metadata']['declared_range'] for r in records)),
            regression_error_bits=len(errors), categories=dict(Counter(x['category'] for x in errors)),
            regression_fn=sum(x['truth'] for x in errors), regression_fp=sum(not x['truth'] for x in errors),
            abs_margin_quantiles=dict(zip(['min','p25','p50','p75','max'],
                np.quantile([abs(x['MZ5']) for x in errors],[0,.25,.5,.75,1]).tolist())),
            abs_margin_counts={str(t):sum(abs(x['MZ5'])<=t for x in errors) for t in (.25,.5,1)},
            error_native_crop_supported=sum(x['native_crop_pixels']>=3 for x in errors),
            error_zone_center_supported=sum(x['zone_center_support'] for x in errors),
            event_transitions=transitions,
            nonnegative_control_exact=dict(MZ1=int(prior[[r['condition'] in ('BODY_ONLY','HEAD_ONLY','BOTH') for r in rows]].sum()),
                MZ5=int(correct[[r['condition'] in ('BODY_ONLY','HEAD_ONLY','BOTH') for r in rows]].sum())),
            truth_not_in_predictor=True, decision_rule_changed=False)
        write(output/'result.json',summary)
        # Original RGB thumbnails with raw output values, for human visual diagnosis.
        for page, start in enumerate(range(0,len(records),9)):
            sheet = Image.new('RGB',(1200,3*320),'#eeeeee'); draw = ImageDraw.Draw(sheet)
            for n, rec in enumerate(records[start:start+9]):
                x,y=(n%3)*400,(n//3)*320
                with Image.open(rec['metadata']['rgb_file']) as im:
                    sheet.paste(im.convert('RGB').resize((400,225)),(x,y))
                lines=[f"{rec['frame_id']} {rec['metadata']['condition']} {rec['metadata']['declared_range']}"]
                for error in rec['errors']:
                    lines.append(f"{error['event']} R{error['RGB']:+.2f} T{error['ToF']:+.2f}")
                    lines.append(f"old{error['MZ1']:+.2f} mean{error['MZ5']:+.2f} crop{error['native_crop_pixels']}")
                draw.multiline_text((x+5,y+230),'\n'.join(lines),fill='black',spacing=3)
            sheet.save(output/f'regressions-{page+1}.jpg')
        # Independently expressed scalar checks of all 6000 event signs and row changes.
        scalar_loss = scalar_gain = 0
        for i in range(len(rows)):
            pred=[(float(rgb[i,k])+float(tof[i,k]))/2 >= 0 for k in range(4)]
            assert pred == p[i].tolist()
            pc, bc=pred==truth[i].tolist(),q[i].tolist()==truth[i].tolist()
            scalar_loss += bc and not pc; scalar_gain += pc and not bc
        assert (scalar_loss,scalar_gain)==(len(lost),len(gained))
        assert hashes == {k:sha(p) for k,p in paths.items()}
        write(output/'receipt.json',dict(status='PASS',input_sha256=hashes,observation_sha256=observation_hashes,
            output_sha256={p.name:sha(p) for p in output.iterdir()},
            scalar_sign_checks=6000, original_alert_parity=1500, training_steps=0,model_forward_passes=0,
            backend='CPU',placement_reason='TASK_NOT_GPU_SUITABLE',seconds=time.perf_counter()-started))
        print({k:v for k,v in summary.items() if k!='event_transitions'})
        print({k:{n:v for n,v in x.items() if n not in ('losses','gain_frame_ids')} for k,x in transitions.items()})
    except Exception as exc:
        write(output/'failure.json',dict(error=repr(exc),input_sha256=hashes))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.output)

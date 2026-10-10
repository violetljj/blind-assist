"""Join public input metadata and hash payloads without opening eval predictions."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np


def seal(root):
    started = time.perf_counter()
    root = Path(root).resolve()
    read = lambda p: json.loads((root / p).read_text(encoding="utf-8"))
    cache = {}
    def digest(p):
        p = str(p)
        if p not in cache:
            cache[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        return cache[p]
    def binding(p, expected=None):
        h = digest(p)
        if expected:
            assert h == expected, p
        return dict(path=str(p), sha256=h)
    def stats(v):
        return dict(n=len(v), median_s=float(np.median(v)), p95_s=float(np.quantile(v,.95)), max_s=float(max(v))) if v else dict(n=0)
    refs = read('reference_roster.json')['rows']
    syn = {x['source_id']: x for x in read('synthesis_manifest.json')['frames']}
    predictions = {}
    for model in ['dav2','unidepth']:
        predictions[model] = {}
        for role in ['train','cal','sealed_eval']:
            for x in read(f'rgb_inference/{model}/{role}/predictions.json')['rows']:
                predictions[model][x['source_id']] = x
    frames, summaries = [], {}
    for role in ['train','cal','eval']:
        rr = [x for x in refs if x['role']==role]
        counts = {b:Counter() for b in ['near','mid','far']}
        rgbdt, tofdt, posedelta, widths = [],[],[],[]
        for x in rr:
            s = syn[x['source_id']]
            for q in x['queries']:
                counts[['near','mid','far'][int(q['name'][-1])]][q['state']] += 1
            rgbdt.append(abs(x['rgb_timestamp_s']-x['grid_timestamp_s']))
            if x['highres_timestamp_s'] is not None:
                tofdt.append(abs(x['highres_timestamp_s']-x['grid_timestamp_s']))
            trajectory = np.loadtxt(x['trajectory_path'])[:,0]
            posedelta.append(float(np.min(abs(trajectory-x['grid_timestamp_s']))))
            bracket = s['grid_pose_bracket']
            if 'left_s' in bracket:
                widths.append(bracket['right_s']-bracket['left_s'])
            arms = [dict(a, **binding(a['path'],a['sha256'])) for a in s['arms']]
            frame = {k:x[k] for k in ['source_id','capture','visit_id','role','window_id','window_frame','grid_timestamp_s','rgb_timestamp_s','highres_timestamp_s','color_K','native_K','highres_K']}
            frame.update(rgb=binding(x['rgb_path'],x['rgb_sha256']), native_depth=binding(x['native_depth_path'],x['native_depth_sha256']),reference=binding(x['reference_path'],x['reference_sha256']),trajectory=binding(x['trajectory_path']),faro=binding(x['highres_depth_path']) if x['highres_depth_path'] else None,cnh=arms,source_available=s['source_available'])
            frame['pose'] = dict(binding(arms[0]['path']), keys=['grid_pose','rgb_pose'], grid_bracket=bracket, rgb_bracket=s['rgb_pose_bracket'],source_bracket=s['source_pose_bracket'])
            frame['raw_rgb_predictions'] = {m:binding(predictions[m][x['source_id']]['path'],predictions[m][x['source_id']]['sha256']) for m in predictions}
            frames.append(frame)
        summaries[role] = dict(visits=sorted(set(x['visit_id'] for x in rr)),frames=len(rr),queries=len(rr)*27,reference_denominators=counts,rgb_abs_residual=stats(rgbdt),faro_matched_abs_residual=stats(tofdt),trajectory_nearest_grid_abs_residual=stats(posedelta),trajectory_interpolation_bracket_width=stats(widths),faro_matched_frames=len(tofdt),faro_registered_frames=sum(syn[x['source_id']]['source_available'] for x in rr),grid_pose_unavailable=sum(syn[x['source_id']]['grid_pose_bracket'].get('unavailable',False) for x in rr))
    for a,b in [('train','cal'),('train','eval'),('cal','eval')]:
        assert not set(summaries[a]['visits']) & set(summaries[b]['visits'])
    assert len(frames)==384 and len({x['source_id'] for x in frames})==384
    manifest=dict(version=1,status='SEALED',frames=frames,splits=summaries,receipts={p:binding(root/p) for p in ['PLAN.json','frozen_readout_seal.json','admission_seal.json','candidate_gate_records.json','license_source_receipts.json','network_progress.json']},eval_method_metrics_computed=False)
    def write(name,d):
        (root/name).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    write('dataset_manifest_v1.json',manifest)
    receipt=dict(status='PASS',splits=summaries,hashed_unique_files=len(cache),manifest_sha256=digest(root/'dataset_manifest_v1.json'),eval_method_metrics_computed=False,eval_payload_access='bytes for SHA256 only; no predicted arrays decoded',elapsed_s=time.perf_counter()-started)
    write('dataset_integrity.json',receipt)
    write('eval_integrity_receipt.json',dict(status='SEALED_INTEGRITY_PASS',**summaries['eval'],manifest_sha256=receipt['manifest_sha256'],eval_method_metrics_computed=False))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    import sys
    seal(sys.argv[1])

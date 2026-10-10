"""Seal v1.1 public inputs; eval arrays are not decoded for method evaluation."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def main(root, old):
    started = time.perf_counter()
    root, old = Path(root).resolve(), Path(old).resolve()
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    cache = {}
    def bind(path, expected=None):
        path = str(path)
        if path not in cache:
            cache[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if expected:
            assert cache[path] == expected, path
        return dict(path=path, sha256=cache[path])
    def distribution(values):
        return dict(n=len(values), minimum=float(min(values)), p10=float(np.quantile(values,.1)), median=float(np.median(values)), p90=float(np.quantile(values,.9)), maximum=float(max(values))) if values else dict(n=0)
    baseline = read(old/'dataset_manifest_v1.json')
    original = {f['source_id']:f for f in baseline['frames']}
    synthesis = read(root/'synthesis_manifest.json')
    subray_coverage = {(f['frame_id'],a['arm'],str(a['repeat'])):a['coverage_mean'] for f in synthesis['frames'] for a in f['arms']}
    surfaces = {str(s['capture']):s for s in read(root/'surface_manifest.json')['entries']}
    rows = list(csv.DictReader((root/'input_coverage_gate.csv').open()))
    primary = {r['frame_id']:r for r in rows if r['arm']=='faro_rho030_ambient1' and r['K']=='0'}
    frames = []
    for s in synthesis['frames']:
        frame = dict(original[s['source_id']])
        assert frame['role']==s['role'] and frame['visit_id']==s['visit_id']
        frame['v1_single_frame_provenance'] = {key:frame.pop(key) for key in ['faro','highres_timestamp_s','highres_K'] if key in frame}
        frame['geometry_source'] = 'fused_surface rendered at grid pose; v1_single_frame_provenance is historical only'
        frame['cnh'] = [dict(a,**bind(a['path'],a['sha256'])) for a in s['arms']]
        frame['rendered_depth'] = bind(s['geometry_path'])
        surf = surfaces.get(str(s['capture']))
        frame['fused_surface'] = bind(surf['path'],surf['sha256']) if surf else None
        frame['pose'] = dict(bind(s['geometry_path']),keys=['grid_pose','rgb_pose'],grid_bracket=s['grid_pose_bracket'],rgb_bracket=s['rgb_pose_bracket'])
        frame['source_available'] = s['source_available']
        frame['v1_1_input_quality'] = primary[s['frame_id']]
        frames.append(frame)
    assert len(frames)==384 and len({f['source_id'] for f in frames})==384
    summaries = []
    for role in ['all','train','cal','eval']:
        for arm in sorted({r['arm'] for r in rows}):
            for k in ['0','1']:
                rr = [r for r in rows if r['arm']==arm and r['K']==k and (role=='all' or r['role']==role)]
                count = len(rr)
                joint = sum(int(r['joint_pass_zones'])>=52 for r in rr)
                geom = sum(int(r['geometry_pass_zones'])>=52 for r in rr)
                summaries.append(dict(role=role,arm=arm,K=int(k),frames=count,joint_80zone_frames=joint,joint_frame_fraction=joint/count if count else None,geometry_80zone_frames=geom,geometry_frame_fraction=geom/count if count else None,joint_zone_fraction_distribution=distribution([int(r['joint_pass_zones'])/64 for r in rr]),geometry_zone_fraction_distribution=distribution([int(r['geometry_pass_zones'])/64 for r in rr]),subray_coverage_distribution=distribution([subray_coverage[r['frame_id'],arm,k] for r in rr])))
    main_gate = next(s for s in summaries if s['role']=='all' and s['arm']=='faro_rho030_ambient1' and s['K']==0)
    inherited = ['visits','frames','queries','reference_denominators','rgb_abs_residual','trajectory_nearest_grid_abs_residual','trajectory_interpolation_bracket_width','grid_pose_unavailable']
    splits = {role:dict({key:v[key] for key in inherited},frame_ids=[f['source_id'] for f in frames if f['role']==role],fused_surface_available_frames=sum(bool(f['source_available']) for f in frames if f['role']==role)) for role,v in baseline['splits'].items()}
    manifest = dict(version='1.1',parent_manifest=bind(old/'dataset_manifest_v1.json'),frames=frames,splits=splits,receipts={name:bind(root/name) for name in ['PLAN.json','frozen_readout_seal.json','source_manifest.json','surface_manifest.json','residual_fit.json','input_coverage_gate.csv']},eval_query_metrics_computed=False)
    def write(name,data):
        (root/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    write('dataset_manifest_v1_1.json',manifest)
    write('coverage_summary.json',dict(rows=summaries,primary_gate=dict(main_gate,PASS=main_gate['joint_80zone_frames']>=308),input_validity_only=True))
    write('eval_integrity_receipt.json',dict(status='SEALED',frames=128,queries=3456,visits=splits['eval']['visits'],manifest=bind(root/'dataset_manifest_v1_1.json'),input_coverage=[s for s in summaries if s['role']=='eval'],query_metrics_computed=False,model_alarm_rates_computed=False))
    receipt=dict(status='PASS',frames=len(frames),queries=len(frames)*27,manifest=bind(root/'dataset_manifest_v1_1.json'),unique_files_hashed=len(cache),eval_payloads='raw byte hashes only; input quality metadata only; no method array evaluation',elapsed_s=time.perf_counter()-started)
    write('dataset_integrity.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--old',required=True)
    args=parser.parse_args()
    main(args.root,args.old)

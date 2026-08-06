"""Fail-closed six-metric S1 screen without promotion-gate arithmetic."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import cv2,numpy as np
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from evaluate_dav2_model_variant_gate_r0 import depth_metrics,summarize_depth,_task_summary
from evaluate_metric3d_clearance_field_a0 import clearance_field,intrinsics_matrix,tum_depth_metres,BANDS,HORIZONS_M
from prepare_bonn_rgbd_metric_depth_manifest import normalize_depth_image

def main():
 p=argparse.ArgumentParser(); p.add_argument('--roster',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--candidate-depth',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 roster=json.loads(a.roster.read_text(encoding='utf-8')); depths=np.load(a.candidate_depth,mmap_mode='r'); rows=[]; dm=[]
 for i,r in enumerate(roster['rows']):
  path=a.source_root/str(r['sequence_root'])/str(r['depth_path']); raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED); truth=tum_depth_metres(normalize_depth_image(raw,path)); candidate=np.asarray(depths[i],dtype=np.float32); metric=depth_metrics(candidate,truth);dm.append(metric); rows.append({'sequence_id':r['sequence_id'],'timestamp':r['timestamp'],'sensor':clearance_field(truth,intrinsics_matrix(r)),'candidate':clearance_field(candidate,intrinsics_matrix(r))})
 ds=summarize_depth(dm); task=_task_summary(rows,'candidate'); false_block=known=clear_truth=0
 for row in rows:
  t,c=row['sensor'],row['candidate']
  if t.get('status')!='VALID' or c.get('status')!='VALID': continue
  for band in BANDS:
   for h in HORIZONS_M:
    tv=t['bands'][band]['occupied_by_horizon'][str(h)];cv=c['bands'][band]['occupied_by_horizon'][str(h)]
    if tv is None or cv is None: continue
    known+=1; clear_truth+=int(not bool(tv)); false_block+=int(not bool(tv) and bool(cv))
 values={'scale_aligned_absrel':ds['frame_median_scale_aligned_abs_rel_median'],'camera_height_mae_m':task['camera_height_mae_m'],'clearance_mae_m':task['clearance_mae_m'],'collision_agreement':task['collision_agreement'],'false_clear':task['false_clear_rate_all_known_decisions'],'false_block':false_block/clear_truth if clear_truth else None}
 finite=all(isinstance(v,(int,float)) and math.isfinite(float(v)) for v in values.values()); out={'schema':'blindassist_clearance_student_mobile_s1_fast_screen','metrics':values,'known_collision_decisions':known,'terminal':'S1_A_GEOMETRY_PASS' if finite else 'S1_A_NOT_SUPPORTED_UNDEFINED_GEOMETRY','s1_b_authorized':False,'qnn_profile_authorized':False,'production_model_replacement_authorized':False}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(out,indent=2,sort_keys=True));
 if not finite: raise SystemExit(2)
if __name__=='__main__':main()

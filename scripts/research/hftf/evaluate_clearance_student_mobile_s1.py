"""Extract the six frozen S1 fast-screen metrics from a gate result."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from clearance_student_mobile_s1 import require_finite_metrics

def main():
 p=argparse.ArgumentParser(); p.add_argument('--report',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); r=json.loads(a.report.read_text(encoding='utf-8')); task=r['candidate']['task']; depth=r['candidate']['depth']
 known=int(task['known_collision_decisions']); fc=int(task['false_clear_count']); occupied=int(task['truth_occupied_decisions']); false_block_count=max(0, known-int(round(float(task['collision_agreement'])*known))-fc); clear_truth=known-occupied
 metrics={'scale_aligned_absrel':float(depth['frame_median_scale_aligned_abs_rel_median']),'camera_height_mae_m':float(task['camera_height_mae_m']),'clearance_mae_m':float(task['clearance_mae_m']),'collision_agreement':float(task['collision_agreement']),'false_clear':float(task['false_clear_rate_all_known_decisions']),'false_block':false_block_count/clear_truth if clear_truth else float('nan')}
 require_finite_metrics(metrics); out={'schema':'blindassist_clearance_student_mobile_s1_fast_screen','metrics':metrics,'terminal':'S1_A_FAST_SCREEN_COMPLETE_DEVELOPMENT_ONLY','qnn_profile_authorized':False,'production_model_replacement_authorized':False}
 text=json.dumps(out,indent=2,sort_keys=True)+'\n'; print(text,end='');
 if a.output: a.output.write_text(text,encoding='utf-8')
if __name__=='__main__': main()

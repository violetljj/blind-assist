"""Materialize a frozen 120-frame aligned S1 depth cache."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from clearance_student_mobile_s1 import ClearanceStudentMobileS1, normalize_bgr_batch
from evaluate_dav2_model_variant_gate_r0 import sha256_file

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--roster',type=Path,required=True); p.add_argument('--source-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 roster=json.loads(a.roster.read_text(encoding='utf-8')); rows=roster['rows']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model=ClearanceStudentMobileS1(False).to(device); model.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=True),strict=True); model.eval(); out=[]
 with torch.inference_mode():
  for r in rows:
   path=a.source_root/str(r['sequence_root'])/str(r['rgb_path']); bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
   if bgr is None: raise OSError(path)
   x=normalize_bgr_batch([torch.from_numpy(bgr.transpose(2,0,1).copy())]).to(device)
   depth=model(x,(480,640))['metric_depth'][0].float().cpu().numpy(); out.append(depth.astype(np.float16))
 a.output.parent.mkdir(parents=True,exist_ok=True); np.save(a.output,np.stack(out))
 receipt={'schema':'blindassist_clearance_student_mobile_s1_gate_cache','frames':len(out),'shape':[len(out),480,640],'checkpoint_sha256':sha256_file(a.checkpoint),'roster_sha256':sha256_file(a.roster),'depth_path':str(a.output.resolve()),'depth_sha256':sha256_file(a.output),'terminal':'S1_A_GATE_CACHE_COMPLETE_DEVELOPMENT_ONLY'}
 a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8'); print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()

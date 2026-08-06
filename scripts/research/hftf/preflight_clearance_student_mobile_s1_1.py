"""Fail-closed S1.1 E0 preflight; never opens the consumed 120-frame gate."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest().upper()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,required=True); p.add_argument('--encoder-weights',type=Path,required=True); p.add_argument('--geometry-manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    protocol=json.loads(a.protocol.read_text(encoding='utf-8')); geometry=json.loads(a.geometry_manifest.read_text(encoding='utf-8'))
    if protocol.get('protocol_id')!='clearance-student-mobile-s1-1': raise ValueError('S1.1 protocol mismatch')
    if not a.encoder_weights.is_file(): raise FileNotFoundError('pretrained MobileNetV3-Large weights are required')
    expected=geometry.get('protocol_sha256'); actual_protocol=sha256(a.protocol)
    if expected!=actual_protocol: raise ValueError('geometry targets are not bound to this protocol')
    if geometry.get('consumed_120_frame_cohort_opened') is not False: raise ValueError('consumed cohort firewall failed')
    coverage=geometry.get('coverage',{}); required=('ground_plane','camera_height','clearance_any_band')
    invalid=[k for k in required if not isinstance(coverage.get(k),(int,float)) or not math.isfinite(float(coverage[k])) or float(coverage[k])<=0.0]
    terminal='S1_1_E0_PREFLIGHT_PASS' if not invalid else 'S1_1_E0_PREFLIGHT_NOT_SUPPORTED'
    out={'schema':'blindassist_clearance_student_mobile_s1_1_preflight','protocol_sha256':actual_protocol,'encoder_weights':{'path':str(a.encoder_weights.resolve()),'sha256':sha256(a.encoder_weights)},'geometry_manifest_sha256':sha256(a.geometry_manifest),'coverage':coverage,'invalid_coverage':invalid,'consumed_120_frame_cohort_opened':False,'terminal':terminal,'e0_training_authorized':not invalid,'e1_authorized':False,'qnn_profile_authorized':False,'production_model_replacement_authorized':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(out,indent=2,sort_keys=True))
    if invalid: raise SystemExit(2)
if __name__=='__main__': main()

"""S1.1 E0 sanity trainer on the A4 train/validation stream only."""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from clearance_student_mobile_s1_1 import ClearanceStudentMobileS11,S11Config,confidence_masked_geometry_loss,feature_distillation_loss,normalize_bgr_batch,parameter_count,encoder_binding_digest

def truth_paths(r):
 rgb=Path(str(r['rgb_path'])); root,stem=rgb.parent.parent,rgb.stem
 return root/'lowres_depth'/f'{stem}.png',root/'confidence'/f'{stem}.png'
def run_epoch(model,records,ids,targets,teacher,features,device,batch_size,opt=None,scaler=None):
 model.train(opt is not None); vals=[]
 for s in range(0,len(ids),batch_size):
  batch=ids[s:s+batch_size]; images=[];depths=[];confs=[];ground=[];heights=[];clear=[];gvalid=[];cvalid=[]
  for i in batch:
   r=records[i]; rgb=cv2.imread(str(r['rgb_path']),cv2.IMREAD_COLOR); dp,cp=truth_paths(r); d=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED); c=cv2.imread(str(cp),cv2.IMREAD_UNCHANGED)
   images.append(torch.from_numpy(rgb.transpose(2,0,1).copy())); depths.append(d.astype(np.float32)/1000.0);confs.append(c);ground.append(targets['plane'][i]);heights.append(targets['camera_height_m'][i]);clear.append(targets['clearance_m'][i]);gvalid.append(targets['ground_valid'][i]);cvalid.append(targets['clearance_valid'][i])
  x=normalize_bgr_batch(images).to(device); truth=torch.from_numpy(np.stack(depths)).to(device); conf=torch.from_numpy(np.stack(confs)).to(device); tg=torch.from_numpy(np.stack(ground)).to(device);th=torch.from_numpy(np.stack(heights)).to(device);tc=torch.from_numpy(np.stack(clear)).to(device);gv=torch.from_numpy(np.stack(gvalid)).to(device).bool();cv=torch.from_numpy(np.stack(cvalid)).to(device).bool()
  if opt: opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
   pred=model(x,(192,256)); loss,parts=confidence_masked_geometry_loss(pred,truth,conf,teacher=None)
   gl=F.smooth_l1_loss(pred['ground_plane'][gv],tg[gv]) if gv.any() else loss*0
   hl=F.smooth_l1_loss(pred['camera_height'][gv],th[gv]) if gv.any() else loss*0
   cmask=cv & torch.isfinite(tc); cl=F.smooth_l1_loss(pred['clearance'][cmask],tc[cmask]) if cmask.any() else loss*0
   loss=loss+0.05*gl+0.05*hl+0.05*cl; parts.update(ground=gl,height=hl,clearance=cl)
  if opt:
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update()
  vals.append(float(loss.detach().cpu()))
 return float(np.mean(vals))
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--teacher-manifest',type=Path,required=True);p.add_argument('--geometry-targets',type=Path,required=True);p.add_argument('--teacher-features',type=Path);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--epochs',type=int,default=2);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--encoder-weights',type=Path,required=True);a=p.parse_args()
 if a.output_root.exists(): raise FileExistsError(a.output_root)
 pre=json.loads(a.preflight.read_text());
 if pre.get('terminal')!='S1_1_E0_PREFLIGHT_PASS' or not pre.get('e0_training_authorized'): raise ValueError('E0 preflight did not pass')
 manifest=json.loads(a.teacher_manifest.read_text());records=manifest['records'];targets={k:v for k,v in np.load(a.geometry_targets,allow_pickle=False).items()};features=np.load(a.teacher_features,mmap_mode='r') if a.teacher_features else None
 seed=20260806;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=ClearanceStudentMobileS11(pretrained=True,config=S11Config(decoder_channels=256)).to(device)
 if not m.encoder_binding['encoder_unchanged']: raise ValueError('encoder binding failed')
 opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-2);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');train=[i for i,r in enumerate(records) if r['role']=='train'];val=[i for i,r in enumerate(records) if r['role']=='validation'];a.output_root.mkdir(parents=True);history=[];started=time.perf_counter()
 for e in range(a.epochs):
  tr=run_epoch(m,records,[train[j] for j in np.random.default_rng(seed+e).permutation(len(train))],targets,None,features,device,a.batch_size,opt,scaler)
  with torch.inference_mode(): va=run_epoch(m,records,val,targets,None,features,device,a.batch_size)
  row={'epoch':e+1,'train_total':tr,'validation_total':va};history.append(row);print(json.dumps(row),flush=True)
 with torch.inference_mode():
  sample=[]; finite_geometry=0
  for i in val[:min(64,len(val))]:
   r=records[i]; rgb=cv2.imread(str(r['rgb_path']),cv2.IMREAD_COLOR); x=normalize_bgr_batch([torch.from_numpy(rgb.transpose(2,0,1).copy())]).to(device); pred=m(x,(192,256)); sample.append(pred['metric_depth'].float().detach().cpu().numpy()); finite_geometry += int(all(torch.isfinite(pred[k]).all().item() for k in ('metric_depth','ground_plane','camera_height','clearance')))
  depths=np.concatenate([v.reshape(-1) for v in sample]) if sample else np.asarray([],dtype=np.float32)
  quantiles={str(q):float(np.quantile(depths,q)) for q in (0.01,0.5,0.99)} if depths.size else {}
  saturation_fraction=float(np.mean((depths<=0.251)|(depths>=5.99))) if depths.size else None
  geometry_coverage=float(finite_geometry/len(sample)) if sample else None
 result={'schema':'blindassist_clearance_student_mobile_s1_1_e0_result','terminal':'S1_1_E0_TRAINING_COMPLETE_DEVELOPMENT_ONLY','parameter_count':parameter_count(m),'encoder_binding':m.encoder_binding,'preflight_sha256':str(a.preflight.resolve()),'history':history,'training_seconds':time.perf_counter()-started,'e0_diagnostics':{'depth_quantiles_m':quantiles,'depth_saturation_fraction':saturation_fraction,'finite_geometry_output_coverage':geometry_coverage,'diagnostic_sample_frames':len(sample)},'e1_authorized':False,'qnn_profile_authorized':False}
 (a.output_root/'e0_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

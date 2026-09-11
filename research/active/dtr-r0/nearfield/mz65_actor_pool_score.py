"""Frozen MZ65 A/B/R engineering scorer. CUDA geometry only, never a model.

Run only after all three registered captures and native world verifications.
--runs is the directory containing A/capture-v1, B/capture-v1, R/capture-v1.
The separately supplied visual review must bind all sixteen generated panels;
this scorer emits numerical gates plus NEEDS_ACTUAL_VISUAL_REVIEW, never invents
that review. No tolerance is estimated from B: A/R defines every residual bound.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from PIL import Image, ImageDraw

T=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def eq(a,b):return bool(np.array_equal(a,b,equal_nan=True))
def edge(mask):
 p=np.pad(mask,1,constant_values=False)
 neighbors=np.stack([p[y:y+360,x:x+640] for y in range(3) for x in range(3)])
 return neighbors.any(0)&~neighbors.all(0)
def errors(a,b,mask):
 v=np.abs(a.astype(np.float64)-b.astype(np.float64))[mask]
 if not v.size:return None
 return dict(mae=float(v.mean()),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),maximum=float(v.max()),changed_values=int(np.count_nonzero(v)))
def bounded(candidate,repeat):
 return candidate is not None and repeat is not None and all(candidate[k]<=repeat[k] for k in candidate)
def performance(receipts):
 wall={a:float(r['wall_elapsed_s'])-float(r['map_load_s']) for a,r in receipts.items()}
 prep={a:sum(float(x['prepare_s']) for x in r['frame_timings']) for a,r in receipts.items()}
 drift=abs(wall['A']-wall['R']);saving=min(wall['A'],wall['R'])-wall['B']
 return dict(map_excluded_s=wall,prepare_s=prep,baseline_drift_s=drift,saving_over_faster_baseline_s=saving,
             faster_than_both=wall['B']<min(wall['A'],wall['R']),beyond_baseline_drift=saving>drift,
             preparation_reduced=prep['B']<min(prep['A'],prep['R']))

def run(args):
 sys.path.insert(0,str(T/'labels-deps'))
 sys.path.insert(0,str(T/'runtime-v1/research/active/dtr-r0/nearfield'))
 import torch
 from body_query_collection_labels import verify_capture,floor_acceptance
 from multizone64_observation import geometry,native_events
 from mz8_attribution import query_membership
 from mz9_contributors import reconstruct
 assert torch.cuda.is_available(),'Frozen geometry kernels require CUDA; no silent backend rewrite'
 out=args.output.resolve();root=Path(args.artifacts).resolve(strict=True)
 assert out.is_relative_to(root) and out!=root and not out.exists()
 out.mkdir(parents=True);started=time.perf_counter();inputs={};result={}
 def bind(p):inputs[str(p.resolve())]=sha(p)
 try:
  freeze=read(T/'input-manifest.json')
  for rel,digest in freeze['task_files'].items():assert sha(T/rel)==digest,rel;bind(T/rel)
  bind(T/'input-manifest.json')
  original=read(T/'original-selected-labels.json');selection=read(T/'selection.json')['rows']
  expected_ids=[x['frame_id'] for x in selection]
  specs={};worlds={};receipts={};arrays={};summaries={};floors={};controlled={}
  g=geometry('cuda');ray=g['rays'];factor=g['radial_factor']
  for arm in ('A','B','R'):
   cap=(args.runs/arm/'capture-v1').resolve(strict=True)
   spec,world,bound,probes=verify_capture(cap);inputs.update(bound)
   assert sha(cap/'source/spec.json')==sha(T/'specs-v1'/f'{arm}.json')
   assert [c['name'] for c in spec['cases']]==expected_ids
   r=read(cap/'receipt.json');assert len(r['frame_timings'])==len(r['object_preparation'])==16
   assert r['actor_pool']['enabled']==(arm=='B') and r['actor_pool']['cleanup_complete'] and r['actor_pool']['slots_remaining']==0
   for name,digest in read(cap/'launch.json')['source_hashes'].items():assert sha(T/'runtime-v1/research/active/dtr-r0/nearfield'/name)==digest
   assert len(r['view_readiness'])==16 and [x['index'] for x in r['view_readiness']]==list(range(16))
   assert all(x['status']=='READY' and not x.get('error') and x['settling_unchanged'] for x in r['view_readiness'])
   assert r['settling_policy']==dict(ticks=32,first_use_ticks=64,interval_s=0.,readiness_gate='UNCHANGED_NATIVE_ASSET_SHADER_STREAMING')
   frames=[];fchecks=[]
   for i,c in enumerate(spec['cases']):
    depthpath=cap/f'evaluator/native/{i:04d}.npy';rgbpath=cap/f'model/sample/{i:04d}.png';supportpath=cap/f'evaluator/world_support/{i:04d}.npy'
    for p in (depthpath,rgbpath,supportpath):bind(p)
    d=np.load(depthpath,allow_pickle=False);support=np.load(supportpath,allow_pickle=False)
    assert d.shape==(360,640) and d.dtype==np.dtype('<f4') and np.isfinite(d).all() and (d>=0).all() and (d<100).all()
    assert support.shape==(2,360,640) and support.dtype==np.int8 and np.isin(support,[-1,0,1]).all()
    with Image.open(rgbpath) as im:assert im.size==(640,360);rgb=np.array(im.convert('RGB'))
    td=torch.from_numpy(d).cuda()[None];native=native_events(td,crop=False);packet=reconstruct(td)
    valid=torch.isfinite(td)&(td>0)&(td<100)&(td.double()*factor<=4.)
    points=td.double()[...,None]*ray;points[...,2]+=1.7
    masks=(query_membership(points)&valid[...,None])[0].cpu().numpy()
    vm=valid[0].cpu().numpy();counts=masks.sum((0,1));events=counts>=3
    assert eq(counts,native['counts'][0].cpu().numpy()) and bool(native['observation_valid'][0])
    # Original realized query bits are immutable input evidence, never forced intent.
    original_bits_match=eq(events,np.asarray(original[i]['event_truth'],dtype=bool))
    fcheck=floor_acceptance(c,probes);assert fcheck['accepted'];fchecks.append(fcheck)
    cell_counts=masks.reshape(45,8,80,8,4).sum((1,3)).astype(np.uint8)
    valid_counts=vm.reshape(45,8,80,8).sum((1,3)).astype(np.uint8)
    assert np.array_equal((support<0),np.broadcast_to(~((d>0)&np.isfinite(d)&(d<100)),support.shape))
    p={k:v[0].cpu().numpy() for k,v in packet.items() if isinstance(v,torch.Tensor)}
    frames.append(dict(depth=d,rgb=rgb,support=support,events=events,counts=counts,masks=masks,original_bits_match=original_bits_match,full_valid=vm,native_valid=(d>0)&(d<100),cell_counts=cell_counts,valid_counts=valid_counts,**p))
   specs[arm]=spec;worlds[arm]=world;receipts[arm]=r;arrays[arm]=frames;floors[arm]=fchecks
   controlled[arm]=r['controlled_objects']
   summaries[arm]=dict(frames=16,event_positives=np.stack([f['events'] for f in frames]).sum(0).tolist(),unknown_native_pixels=sum(int((f['support'][0]<0).sum()) for f in frames),unknown_fullframe_cells=sum(int((f['valid_counts']==0).sum()) for f in frames),actor_pool=r['actor_pool'],frame_timings=r['frame_timings'],object_preparation=r['object_preparation'],view_readiness=r['view_readiness'],pair_exports=r['pair_exports'],pair_drain_s=r['pair_drain_s'])
  assert controlled['A']==controlled['B']==controlled['R'],'Native target transforms or actor bounds changed'
  rows=[];gates=dict(exact_geometry=True,residual_bounds=True,paired_target_invariance=True,active_pool_flags=True,configured_profile_unchanged=True,all_readiness_READY=True,actual_render_counts_equal=True,original_source_bits_preserved=True,baseline_A_R_identity=True,nonempty_A_roi=True)
  npz={};previews=[]
  for i,identity in enumerate(selection):
   a,b,r=(arrays[x][i] for x in ('A','B','R'))
   fixed=a['masks'].any(-1);boundary=edge(fixed)
   gates['nonempty_A_roi'] &= bool(fixed.any() and boundary.any())
   masks={'full':np.ones((360,640),dtype=bool),'A_native_query_target':fixed,'A_native_query_boundary_3x3':boundary}
   checks={k:dict(A_B=eq(a[k],b[k]),A_R=eq(a[k],r[k])) for k in ('support','events','counts','masks','full_valid','native_valid','cell_counts','valid_counts','range_m','valid','source_counts','query_counts','cell_known_counts','selected_bins','selected_bin_counts')}
   gates['exact_geometry'] &= all(v['A_B'] and v['A_R'] for v in checks.values())
   gates['baseline_A_R_identity'] &= all(v['A_R'] for v in checks.values())
   gates['original_source_bits_preserved'] &= a['original_bits_match'] and b['original_bits_match'] and r['original_bits_match']
   residuals={}
   for modality in ('rgb','depth'):
    residuals[modality]={}
    for roi,m in masks.items():
     ab=errors(a[modality],b[modality],m);ar=errors(a[modality],r[modality],m)
     passed=bounded(ab,ar);gates['residual_bounds'] &= passed
     residuals[modality][roi]=dict(B_vs_A=ab,R_vs_A=ar,within_repeat_bound=passed,pixels=int(m.sum()))
   renders=[receipts[x]['frame_timings'][i]['rgb_render_calls'] for x in ('A','B','R')]
   gates['actual_render_counts_equal'] &= len(set(renders))==1
   for arm in ('A','B','R'):
    prep=receipts[arm]['object_preparation'][i]
    gates['active_pool_flags'] &= prep['flags_verified'] is True and prep['active']==len(specs[arm]['cases'][i]['objects'])
    for k in ('events','counts','range_m','valid','cell_counts','valid_counts'):npz[f'{arm}/{i}/{k}']=arrays[arm][i][k]
   rows.append(dict(**identity,exact_checks=checks,residuals=residuals,rgb_render_calls_A_B_R_order=renders))
   # All 16 fixed IDs receive a review panel; no outcome-driven image selection.
   board=Image.new('RGB',(1920,784),'white');draw=ImageDraw.Draw(board)
   for j,arm in enumerate(('A','B','R')):
    image=arrays[arm][i]['rgb'];overlay=image.copy();overlay[boundary]=[0,255,255]
    board.paste(Image.fromarray(image),(j*640,28));board.paste(Image.fromarray(overlay),(j*640,410))
    draw.text((j*640+5,5),arm+' original RGB',fill='black');draw.text((j*640+5,389),'Fixed A native query boundary (cyan)',fill='black')
   path=out/f'preview-{i:02d}.png';board.save(path);previews.append(dict(index=i,frame_id=identity['frame_id'],path=path.name,sha256=sha(path)))
  pairs=[]
  for i in range(0,16,2):
   assert selection[i]['pair_id']==selection[i+1]['pair_id']
   for arm in ('A','B','R'):
    x,y=arrays[arm][i:i+2];target=x['masks'].any(-1)|y['masks'].any(-1)
    same=eq(x['masks'],y['masks']) and eq(x['cell_counts'],y['cell_counts']) and eq(x['depth'][target],y['depth'][target])
    gates['paired_target_invariance'] &= same
    pairs.append(dict(arm=arm,pair_id=selection[i]['pair_id'],event_masks_counts_target_depth_exact=same,background_valid_may_change=True))
  perf=performance(receipts)
  gates.update({k:perf[k] for k in ('faster_than_both','beyond_baseline_drift','preparation_reduced')})
  result=dict(status='PASS',numeric_admission=all(gates.values()),preservation_evaluability='EVALUABLE' if gates['baseline_A_R_identity'] else 'NOT_EVALUABLE_BASELINE_INSTABILITY',visual_admission='NEEDS_ACTUAL_VISUAL_REVIEW',gates=gates,performance=perf,arms=summaries,rows=rows,pairs=pairs,previews=previews,unknown_is_not_clear=True,threshold_adaptation=False,training_steps=0,model_inference_frames=0,geometry_backend='CUDA',device=torch.cuda.get_device_name(),scope='Consumed controlled Development engineering A/B/R only; native labels are evaluator-only. RGB target ROI is the A native query-positive surface, not whole-object segmentation.',seconds=time.perf_counter()-started)
  np.savez_compressed(out/'thin.npz',**npz);write(out/'result.json',result)
  write(out/'receipt.json',dict(status='PASS',input_hashes=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},numeric_admission=result['numeric_admission'],visual_review_pending=True,code_sha256=sha(Path(__file__))))
  print(json.dumps(dict(status='PASS',numeric_admission=result['numeric_admission'],gates=gates,performance=perf)))
 except BaseException as e:
  write(out/'receipt.json',dict(status='FAIL',error=repr(e),input_hashes=inputs,seconds=time.perf_counter()-started,code_sha256=sha(Path(__file__))))
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);run(p.parse_args())

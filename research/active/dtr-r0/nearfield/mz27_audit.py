"""Independent MZ27 saved-descriptor and source-isolation reductions; no fit."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rms(a,b):return np.sqrt(np.sum((a.astype(float)-b.astype(float))**2,axis=-1)/a.shape[-1])
def sample(x,grid):
 # NumPy four-neighbor implementation, independently from Torch grid_sample.
 h,w=x.shape[-2:];xx=((grid[:,0].astype(float)+1)*w-1)/2;yy=((grid[:,1].astype(float)+1)*h-1)/2
 x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);dx=xx-x0;dy=yy-y0
 out=np.zeros((len(x),len(grid),x.shape[1]),float)
 for ox,oy,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
  xi=x0+ox;yi=y0+oy;valid=(xi>=0)&(xi<w)&(yi>=0)&(yi<h)
  out[:,valid]+=x[:,:,yi[valid],xi[valid]].transpose(0,2,1)*weight[None,valid,None]
 return out


def main(root,run):
 receipt=read(run/'receipt.json');assert receipt['status']=='PASS' and receipt['training_steps']==0
 for name,digest in receipt['outputs'].items():assert sha(run/name)==digest,name
 result=read(run/'result.json');start=read(run/'start.json');assert result['protocol_sha256']==sha(root/'research/active/dtr-r0/nearfield/MZ27_FEATURE_SEPARATION_PROTOCOL_20260910.md')
 for path,digest in result['inputs'].items():assert sha(Path(path))==digest,path
 work=root/'artifacts.local/work';old=np.load(work/'mz8-attribution-20260910/cache-v5/observations.npz')
 new=np.load(work/'mz15-shared-support-20260910/cache-v1/observations.npz')
 oldlabels=np.load(work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz');newlabels=np.load(work/'mz15-shared-support-20260910/cache-v1/evaluator.npz')
 known=np.concatenate([oldlabels['cell_known_counts']>0,newlabels['known']])
 ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
 good=valid&np.isfinite(ranges)&(ranges>0)&(ranges<=4);p=np.concatenate([np.where(good,ranges,0)/4,good.astype(np.float32)],axis=-1).astype(np.float32)
 source=read(work/'body-query-5000-20260909/dataset-v1/index.json')['frames'];placed=read(work/'mz15-shared-support-20260910/cache-v1/selected.json')
 train=np.unique(np.load(work/'mz20-rank-objective-20260910/run-v1/batches.npy'))
 def identity(i):
  if i<3500:
   r=source[int(old['old_index'][i])];return r.get('site_id'),r.get('group_id')
  if i>=3700:
   r=placed[i-3700];return r.get('site'),r.get('group')
  return None,None
 bank=[int(i) for i in train if all(identity(int(i)))];vectors=np.load(run/'descriptors.npz')
 anchors=result['anchors'];assert len(anchors)==result['unique_anchor_locations']
 assert len({(a['global_id'],a['zone'],a['cell']) for a in anchors})==len(anchors)
 requests=[m for a in anchors for m in a['memberships']];assert len(requests)==len(start['anchor_requests'])==result['anchor_requests']
 actual=sorted((m['kind'],m['cohort'],m['global_id'],m['query']) for m in requests)
 expect=sorted((m['kind'],m['cohort'],m['global_id'],m['query']) for m in start['anchor_requests']);assert actual==expect
 assert sum(m['kind']=='residual_false_addition' for m in requests)==3
 assert sum(m['kind']=='negative_tail' for m in requests)==32
 assert sum(m['kind']=='far_true_witness' for m in requests)+len(result['missing_witnesses'])==75
 counts={};comparisons=0;max_distance_error=0.;scored=[]
 for a in anchors:
  counts[a['status']]=counts.get(a['status'],0)+1;i,z,c=a['global_id'],a['zone'],a['cell'];sid,gid=identity(i)
  assert a['identity']['site']==sid and a['identity']['group']==gid
  assert a['known']==bool(known[i,z,c])
  if not sid or not gid:assert a['status']=='UNKNOWN_IDENTITY';continue
  eligible=np.array([j for j in bank if j!=i and identity(j)[0]!=sid and identity(j)[1]!=gid])
  d=np.sqrt(((p[eligible,z].astype(float)-p[i,z])**2).sum(1));order=np.lexsort((eligible,d))[:128];refs=eligible[order]
  np.testing.assert_array_equal(refs,[r['global_id'] for r in a['references']]);assert len(refs)==128
  labs=known[refs,z,c];np.testing.assert_array_equal(labs,[r['known'] for r in a['references']])
  np.testing.assert_allclose(d[order],[r['packet_distance'] for r in a['references']],atol=1e-12,rtol=0)
  count=[int((~labs).sum()),int(labs.sum())];assert count==a['reference_class_counts']
  for rr in a['references']:assert (rr['identity']['site'],rr['identity']['group'])==identity(rr['global_id'])
  if min(count)<5:assert a['status']=='INSUFFICIENT_CLASS_SUPPORT';continue
  assert a['status']=='SCORED';scored.append(a)
  np.testing.assert_array_equal(vectors[f"anchor{a['anchor_id']}/packet4"],p[np.r_[i,refs],z])
  for name,res in a['retrieval'].items():
   v=vectors[f"anchor{a['anchor_id']}/{name}"];dist=rms(v[1:],v[:1]);rank=np.lexsort((refs,dist))
   max_distance_error=max(max_distance_error,float(np.max(np.abs(dist-res['distances']))));np.testing.assert_allclose(dist,res['distances'],atol=1e-12,rtol=0)
   np.testing.assert_array_equal(rank[:5],res['top5_reference_indices']);assert bool(labs[rank[:5]].sum()>=3)==res['prediction']
   assert res['agreement']==(res['prediction']==a['known']);comparisons+=len(dist)
 assert counts==result['status_counts']
 # Recompute source/kind summaries without importing the runner's implementation.
 for name,s in result['summaries'].items():
  chosen=[a for a in scored if any((('kind/'+m['kind'])==name if name.startswith('kind/') else m['cohort']==name) for m in a['memberships'])]
  for desc,entry in s['descriptors'].items():
   assert entry['anchors']==len(chosen);rates=[]
   for label in [False,True]:
    group=[a for a in chosen if a['known']==label];assert entry['label_counts'][str(int(label))]==len(group)
    val=sum(a['retrieval'][desc]['prediction']==label for a in group)/len(group) if group else None
    assert entry['agreement'][str(int(label))]==val
    if val is not None:rates.append(val)
   assert entry['balanced_agreement']==(sum(rates)/2 if len(rates)==2 else None)
  for pair,entry in s['paired'].items():
   lhs,rhs=pair.split('_vs_');flags=[(a['retrieval'][lhs]['agreement'],a['retrieval'][rhs]['agreement']) for a in chosen]
   assert entry==dict(n=len(flags),improved=sum(x and not y for x,y in flags),worsened=sum(y and not x for x,y in flags),both_correct=sum(x and y for x,y in flags),both_wrong=sum(not x and not y for x,y in flags))
 # Small selected descriptor reconstruction: no model inference or gradient work.
 maps=np.load(work/'mz16-visual-detail-20260910/cache-v2/dense_HIGH_DETAIL.npy',mmap_mode='r');ids=np.load(work/'mz16-visual-detail-20260910/cache-v2/ids.npy');lookup={int(g):k for k,g in enumerate(ids)}
 norm=np.load(work/'mz9-source-supervision-20260910/run-v1/normalization.npz')
 import torch
 grid=torch.load(work/'mz26-deterministic-convergence-20260910/run-v1/availability-step4800.pt',map_location='cpu',weights_only=True)['grid'].numpy()
 sample_error=0.;sample_frames=0
 selected=[a for a in scored if any(m['kind']=='residual_false_addition' for m in a['memberships'])]+scored[:3]
 selected=list({a['anchor_id']:a for a in selected}.values())
 for a in selected:
  ii=[a['global_id']]+[r['global_id'] for r in a['references'][:5]];x=(np.array(maps[[lookup[i] for i in ii]])-norm['mean'])/norm['std']
  offset=np.array([[dx,dy] for dy in [-1,0,1] for dx in [-1,0,1]],dtype=np.float32)*np.float32(2/28)
  patch=sample(x,grid[a['zone'],a['cell']][None]+offset)
  for name,value in [('raw64',patch[:,4]),('raw3x3_576',patch.reshape(len(ii),-1))]:
   saved=vectors[f"anchor{a['anchor_id']}/{name}"][:len(ii)];err=float(np.max(np.abs(value-saved)));sample_error=max(sample_error,err)
   np.testing.assert_allclose(value,saved,atol=2e-5,rtol=1e-5)
  sample_frames+=len(ii)
 audit=dict(status='PASS',training_steps=0,anchor_requests=len(requests),unique_locations=len(anchors),status_counts=counts,
  reference_distance_comparisons=comparisons,distance_max_abs_error=max_distance_error,independent_raw_descriptor_frames=sample_frames,raw_sample_max_abs_error=sample_error,
  verified_input_hashes=len(result['inputs']),receipt_sha256=sha(run/'receipt.json'),code_sha256=sha(Path(__file__)),
  backend='CPU: saved scalar/identity/descriptor reductions; selected NumPy bilinear check, no neural inference',reason='TASK_NOT_GPU_SUITABLE')
 (run/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.root,a.run)

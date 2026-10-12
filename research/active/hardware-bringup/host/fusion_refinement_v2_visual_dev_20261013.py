"""Diagnostic pixel overlays, not new labels or metric ground truth."""
import json
from pathlib import Path
import cv2
import numpy as np
from atom_camera import AtomCamera
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts.local/work/fusion-refinement-dev-20261013'
OUT=ROOT/'artifacts.local/work/fusion-refinement-v2-dev-20261013'

def main():
    data=json.loads((OUT/'combined_readout.json').read_text())
    manifest=json.loads((OLD/'cache_manifest.json').read_text())
    cam=AtomCamera();K=cam.K_upright;w,h=cam.upright_size
    Rct=np.array(json.loads((X.OUT/'camera_tof_extrinsic.json').read_text())['rotation'])
    vv,uu=np.mgrid[:h,:w]
    rays=np.stack(((uu-K[0,2])/K[0,0],(vv-K[1,2])/K[1,1],np.ones_like(uu)),-1).reshape(-1,3)
    sheet=[];details=[]
    for key,i,arm in [('073020',79,'mode_replace'),('073020',81,'mode_replace')]:
        s=data['sessions'][key]
        c=next(r for r in manifest['sessions'][key]['frames'] if r['i']==i)
        raw=F.RAW/s['session']; index=list(F.lines(raw/'rgb.jsonl'))
        img=cam.undistort_upright(cv2.imread(str(raw/index[c['rgb_index']]['file'])))
        p=np.load(OLD/c['cache']);disp,valid=p['disp'],p['valid']
        cr=rays@(F.rot(c['pitch'],c['roll'])@Rct.T).T
        panels=[];entry=dict(session=key,i=i,arm=arm,pixel_counts={})
        for name in ('raw','baseline',arm):
            pic=img.copy();title=key+' i'+str(i)+' '+name
            if name!='raw':
                r=next(r for r in s['arms'][name] if r['i']==i)
                a,b=r['fit'];inv=(disp.ravel()-b)/a
                good=valid.ravel()&(inv>1e-3);pts=np.full_like(cr,np.nan);pts[good]=cr[good]/inv[good,None]
                counts=[]
                for (lo,hi),color in zip(F.BOXES.values(),[(0,255,255),(255,180,0)]):
                    inside=good&(np.abs(pts[:,0])<=.30)&(pts[:,1]>=lo)&(pts[:,1]<=hi)&(pts[:,2]>=.30)&(pts[:,2]<=2.)
                    counts.append(int(inside.sum()))
                    mask=inside.reshape(h,w)
                    pic[mask]=(pic[mask]*.4+np.array(color)*.6).astype(np.uint8)
                assert counts==r['px']
                entry['pixel_counts'][name]=counts
                title+=' H/B '+str(counts)
            header=np.zeros((45,w,3),np.uint8)
            cv2.putText(header,title,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            panels.append(np.vstack((header,pic)))
        sheet.append(np.hstack(panels));details.append(entry)
    cv2.imwrite(str(OUT/'mode_failure_overlay.jpg'),np.vstack(sheet))
    (OUT/'visual_receipt.json').write_text(json.dumps(details,indent=1),encoding='utf8')

if __name__=='__main__':main()

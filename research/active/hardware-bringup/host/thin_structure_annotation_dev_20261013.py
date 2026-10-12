"""Outcome-independent RGB sampling for assistant-reviewed weak line labels."""
import json
from pathlib import Path
import cv2
import numpy as np
from atom_camera import AtomCamera
import real_rgb_query_feasibility_dev_20261012 as F

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts.local/work/fusion-refinement-dev-20261013'
OUT=ROOT/'artifacts.local/work/thin-structure-dev-20261013'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OLD/'cache_manifest.json').read_text())
    cam=AtomCamera();samples=[]
    for key in ('073554','073635','073740','073821','070601','073020'):
        s=manifest['sessions'][key];fr=s['frames'];rgb=list(F.lines(F.RAW/s['session']/'rgb.jsonl'))
        picks=np.rint(np.linspace(.05,.95,6)*(len(fr)-1)).astype(int)
        panels=[]
        for order,j in enumerate(picks):
            r=fr[j];file=F.RAW/s['session']/rgb[r['rgb_index']]['file']
            image=cam.undistort_upright(cv2.imread(str(file)))
            cv2.imwrite(str(OUT/f'raw_{key}_{r["i"]:04d}.jpg'),image)
            overlay=image.copy()
            for xx in range(0,480,80):cv2.line(overlay,(xx,0),(xx,639),(150,150,150),1)
            for yy in range(0,640,80):cv2.line(overlay,(0,yy),(479,yy),(150,150,150),1)
            image=cv2.addWeighted(image,.8,overlay,.2,0)
            for xx in range(80,480,80):cv2.putText(image,str(xx),(xx+2,18),0,.42,(0,0,255),1)
            for yy in range(80,640,80):cv2.putText(image,str(yy),(2,yy-3),0,.42,(0,0,255),1)
            header=np.zeros((30,480,3),np.uint8)
            cv2.putText(header,f'{order}: {key} tof_i={r["i"]}',(8,21),0,.6,(255,255,255),1)
            panels.append(np.vstack((header,image)))
            samples.append(dict(session=key,session_name=s['session'],i=r['i'],order=order,rgb_index=r['rgb_index'],file=str(file.relative_to(ROOT)),label_status='pending'))
        cv2.imwrite(str(OUT/f'sheet_{key}.jpg'),np.vstack((np.hstack(panels[:3]),np.hstack(panels[3:]))))
    (OUT/'annotation_samples.json').write_text(json.dumps(dict(sampling='six uniform ready-frame positions .05..95, before detector output',samples=samples),indent=1),encoding='utf8')

if __name__=='__main__':main()

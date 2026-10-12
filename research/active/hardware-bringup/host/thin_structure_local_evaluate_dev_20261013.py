"""Join label-blind tracking/geometry with frozen assistant weak localization labels."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from thin_structure_evaluate_dev_20261013 import matches
from atom_camera import AtomCamera
import real_rgb_query_feasibility_dev_20261012 as F

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts.local/work/thin-structure-dev-20261013'
OUT=ROOT/'artifacts.local/work/thin-local-geometry-dev-20261013'


def read(path):
    return {(r['key'],r['i']):r for r in map(json.loads,path.read_text().splitlines())}


def main():
    sources={'global':read(OLD/'detection_ridge_frames.jsonl'),'local':read(OUT/'local_frames.jsonl')}
    if (OUT/'camera_clock_frames.jsonl').exists():sources['camera_clock']=read(OUT/'camera_clock_frames.jsonl')
    if (OUT/'camera_chain_frames.jsonl').exists():sources['camera_chain']=read(OUT/'camera_chain_frames.jsonl')
    geo=read(OUT/'geometry_frames.jsonl')
    for key,original in sources['global'].items():
        for source in [*sources.values(),geo]:
            assert key in source
            cs=source[key]['candidates'];assert len(cs)==len(original['candidates'])
            for a,b in zip(cs,original['candidates']):
                assert all(a[k]==b[k] for k in ('x1','y1','x2','y2','score'))
    labels=json.loads((OLD/'annotations.json').read_text());rows=[];panels={}
    for s in labels['samples']:
        key=(s['session'],s['i']);original=sources['global'][key]['candidates']
        indices=[j for j,c in enumerate(original) if s['segment'] and matches(s['segment'],c)]
        row=dict(session=s['session'],i=s['i'],label_status=s['label_status'],localized=bool(indices),arms={},target_geometry=[])
        for name,frames in sources.items():
            cs=frames[key]['candidates']
            row['arms'][name]=dict(stable_target=any(cs[j]['stable'] for j in indices),target_max_hits=max((cs[j]['hits'] for j in indices),default=0),all_stable=sum(c['stable'] for c in cs))
        for j in indices:row['target_geometry'].append(dict(candidate_index=j,geometry=geo[key]['candidates'][j]['geometry']))
        row['geometry']={}
        for arm in ('baseline','robust_replace'):
            valid_indices=[j for j in indices if geo[key]['candidates'][j]['geometry'][arm]['status']=='AVAILABLE']
            inside=[j for j in valid_indices if any(geo[key]['candidates'][j]['geometry'][arm]['query'])]
            row['geometry'][arm]=dict(available=bool(valid_indices),inside=bool(inside),stable_inside={name:any(frames[key]['candidates'][j]['stable'] for j in inside) for name,frames in sources.items()})
        rows.append(row)
        im=cv2.imread(str(OLD/f'raw_{key[0]}_{key[1]:04d}.jpg'))
        name=list(sources)[-1];cs=sources[name][key]['candidates']
        for j,c in enumerate(cs):
            color=(80,80,80)
            if c['stable']:color=(0,180,255)
            if j in indices:color=(0,255,0)
            cv2.line(im,(round(c['x1']),round(c['y1'])),(round(c['x2']),round(c['y2'])),color,1)
        if s['segment']:
            x,y,xx,yy=s['segment'];cv2.line(im,(x,y),(xx,yy),(255,0,255),2)
        h=np.zeros((40,480,3),np.uint8)
        message=f'{key[0]}/{key[1]} hits '+ '/'.join(str(row['arms'][n]['target_max_hits']) for n in sources)
        cv2.putText(h,message,(5,25),0,.5,(255,255,255),1)
        panels.setdefault(key[0],[]).append(np.vstack((h,im)))
    summary={}
    for key in ['all']+list(panels):
        rr=[r for r in rows if key=='all' or r['session']==key]
        targets=[r for r in rr if r['label_status']=='visible_segment']
        distractors=[r for r in rr if r['label_status']=='no_target_rod']
        summary[key]=dict(n=len(targets),localized=sum(r['localized'] for r in targets),stable={n:sum(r['arms'][n]['stable_target'] for r in targets) for n in sources},distractor_frames=len(distractors),distractor_stable={n:sum(r['arms'][n]['all_stable'] for r in distractors) for n in sources})
        summary[key]['target_geometry']={arm:dict(available=sum(r['geometry'][arm]['available'] for r in targets),inside=sum(r['geometry'][arm]['inside'] for r in targets),stable_inside={name:sum(r['geometry'][arm]['stable_inside'][name] for r in targets) for name in sources}) for arm in ('baseline','robust_replace')}
    for key,pp in panels.items():cv2.imwrite(str(OUT/f'tracking_{key}.jpg'),np.vstack((np.hstack(pp[:3]),np.hstack(pp[3:]))))
    all_frames={}
    for name,frames in sources.items():
        all_frames[name]={}
        for (session,i),r in frames.items():
            out=all_frames[name].setdefault(session,dict(frames=0,stable=0,stable_inside_candidates=0,stable_inside_frames=0))
            accepted=sum(c['stable'] and any(geo[session,i]['candidates'][j]['geometry']['robust_replace']['query']) for j,c in enumerate(r['candidates']))
            out['frames']+=1;out['stable']+=sum(c['stable'] for c in r['candidates']);out['stable_inside_candidates']+=accepted;out['stable_inside_frames']+=accepted>0
    result=dict(summary=summary,rows=rows,all_frames=all_frames,checks='1677 same keys and exact candidate geometry/score/count across arms',annotation_sha256=hashlib.sha256((OLD/'annotations.json').read_bytes()).hexdigest())
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
    if 'camera_chain' in sources:
        # Result-driven visual audit only: never fed back into inference.
        cam=AtomCamera();tiles=[];traces=[]
        frames=list(sources['camera_chain'].values())
        for sample_i in (22,45):
            s=next(s for s in labels['samples'] if s['session']=='073635' and s['i']==sample_i)
            r=sources['camera_chain']['073635',sample_i]
            cs=[c for c in r['candidates'] if matches(s['segment'],c)]
            chosen=max(cs,key=lambda c:c['hits']);tid=chosen['track_id'];seen=set();chain=[]
            for f in reversed(frames):
                if f['key']!='073635' or f['i']>sample_i:continue
                cc=next((c for c in f['candidates'] if c['track_id']==tid),None)
                if cc and f['rgb_t_ns'] not in seen:
                    seen.add(f['rgb_t_ns']);chain.append((f,cc))
                if len(chain)==3:break
            for f,c in reversed(chain):
                rgb=list(F.lines(F.RAW/f['session']/'rgb.jsonl'))
                im=cam.undistort_upright(cv2.imread(str(F.RAW/f['session']/rgb[f['rgb_index']]['file'])))
                cv2.line(im,(round(c['x1']),round(c['y1'])),(round(c['x2']),round(c['y2'])),(0,255,0),2)
                h=np.zeros((40,480,3),np.uint8);cv2.putText(h,f'i={f["i"]} track={tid} hits={c["hits"]}',(5,24),0,.6,(255,255,255),1)
                tiles.append(np.vstack((h,im)));traces.append(dict(sample_i=sample_i,i=f['i'],rgb_t_ns=f['rgb_t_ns'],candidate=c))
        cv2.imwrite(str(OUT/'side_track_audit.jpg'),np.vstack((np.hstack(tiles[:3]),np.hstack(tiles[3:]))))
        (OUT/'side_track_audit.json').write_text(json.dumps(traces,indent=2))


if __name__=='__main__':main()

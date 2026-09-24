"""No-capture Street200 authored-region inventory and conservative clearance screen."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from cnh_route_native_clearance import swept_query_bounds

PROTOCOL=dict(grid_m=2.,travel_m=4.,camera_height_above_support_m=1.6,pitch_deg=0.,
    lateral_offsets_m=[0.,-.3,.3,0.],native_margin_m=.15,observation_envelope_radius_m=8.,
    support='Native pavement AABB upper surface 0..0.6m; sampled support every1m on all lateral paths; height variation<=0.1m',
    supported_poses='Cardinal fixed heading only; no yaw/pitch sway, no floor mesh or visibility admission',
    site_identity='Authored regions are not independent sites. One existing street block; no cross-split visible-instance closure.',
    collection='NOT_RUN',benchmark_eligible=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regions(recipe):
    result=[dict(id='intersection-0',category='intersection',low=[-15.,-15.],high=[15.,15.],
                 headings=[0,90,180,270],authority='Authored orthogonal road junction at origin; diagnostic 30m window')]
    for graph in recipe['graphs']:
        x,y,_=graph['location_m']
        if graph['label']=='BA sidewalk':
            parameters={p['name']:p['value'] for p in graph.get('parameters',[])}
            long=parameters['Width']/100;short=parameters['Length']/100
            yaw=int(graph.get('yaw',0));dx,dy=(long/2,short/2) if yaw==0 else (short/2,long/2)
            result.append(dict(id=f'sidewalk-{sum(r["category"]=="sidewalk" for r in result)}',category='sidewalk',
                low=[x-dx,y-dy],high=[x+dx,y+dy],headings=[yaw,(yaw+180)%360],authority='Authored pavement rectangle'))
        elif graph['label']=='BA plaza':
            result.append(dict(id='plaza-0',category='plaza',low=[x-25,y-25],high=[x+25,y+25],
                headings=[0,90,180,270],authority='Authored plaza graph centre; fixed50m diagnostic window, not certified boundary'))
    return result


def clips_at(x,y,z,yaw):
    angle=math.radians(yaw);forward=np.array([round(math.cos(angle)),round(math.sin(angle))]);right=np.array([-forward[1],forward[0]])
    clips=[]
    for name,offset in zip(('centre','boundary','outside','removed'),PROTOCOL['lateral_offsets_m']):
        poses=[]
        for travel in (0.,4.):
            xy=np.array([x,y])+offset*right+travel*forward
            poses.append(dict(x=float(xy[0]),y=float(xy[1]),z=z,pitch=0.,yaw=yaw,roll=0.))
        clips.append(dict(id=name,trajectory_model='piecewise_linear_fixed_orientation',poses=poses))
    return clips


def query_envelope(clips):
    boxes=[b for c in clips for b in swept_query_bounds(c['poses'][0],c['poses'][1])]
    return np.min([b['min_m'] for b in boxes],axis=0),np.max([b['max_m'] for b in boxes],axis=0)


def overlaps(low,high,lows,highs):return np.all(highs>=low,axis=1)&np.all(lows<=high,axis=1)


def inventory(capture,recipe_path,output):
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal or json.loads(Path(journal).read_text(encoding='utf-8-sig')).get('state')!='running':
        raise RuntimeError('Governed research-ue diagnostic required')
    started=time.monotonic();capture=Path(capture)
    clearance=json.loads((capture/'clearance.json').read_text());spec=json.loads((capture/'source/spec.json').read_text(encoding='utf-8-sig'))
    if not clearance['loaded_world_coverage_complete'] or clearance['required_missing']:
        raise ValueError('Existing loaded-world bounds are unresolved')
    if sha(spec['map_file'])!=spec['map_sha256']:raise ValueError('Map differs from audited source')
    intervention=json.loads((capture/'native-material-intervention.json').read_text())
    if intervention.get('compiled_verification')!='PASS':raise ValueError('Static material intervention not verified')
    entities=clearance['candidates'][0]['manifest']['native_entities']
    if any(not e['bounds_conservative'] or not e['deformation_bounded'] for e in entities):raise ValueError('Unbounded native geometry')
    if clearance['candidates'][1]['manifest']['native_entities']!=entities:raise ValueError('Two receipts disagree on inventory')
    lows=np.asarray([e['bounds_min_m'] for e in entities]);highs=np.asarray([e['bounds_max_m'] for e in entities])
    support=np.array([0<=e['bounds_max_m'][2]<=.6 and any(('Sidewalk' in m or 'AsphaltDark' in m) for m in e['material_paths']) for e in entities])
    sl,sh=lows[support],highs[support]
    recipe=json.loads(Path(recipe_path).read_text());areas=regions(recipe);rows=[];seen=set()
    def support_height(xy):
        mask=np.all(sl[:,:2]<=xy+1e-6,axis=1)&np.all(sh[:,:2]>=xy-1e-6,axis=1)
        return float(sh[mask,2].max()) if mask.any() else None
    for area in areas:
        for x in np.arange(math.ceil(area['low'][0]/2)*2,area['high'][0]+1e-6,2):
            for y in np.arange(math.ceil(area['low'][1]/2)*2,area['high'][1]+1e-6,2):
                for yaw in area['headings']:
                    key=(float(x),float(y),yaw)
                    if key in seen:continue
                    seen.add(key);row=dict(id=len(rows),region=area['id'],category=area['category'],x=float(x),y=float(y),yaw=yaw)
                    rows.append(row)
                    angle=math.radians(yaw);f=np.array([round(math.cos(angle)),round(math.sin(angle))]);r=np.array([-f[1],f[0]])
                    points=[np.array([x,y])+travel*f+side*r for travel in range(5) for side in (-.3,0.,.3)]
                    if any(np.any(p<area['low']) or np.any(p>area['high']) for p in points):
                        row['status']='PATH_OUTSIDE_AUTHORED_REGION';continue
                    heights=[support_height(p) for p in points]
                    if any(h is None for h in heights) or max(heights)-min(heights)>.100001:
                        row['status']='SUPPORT_NOT_ESTABLISHED';continue
                    z=max(heights)+1.6;clips=clips_at(x,y,z,yaw);low,high=query_envelope(clips)
                    hit=np.flatnonzero(overlaps(low,high,lows,highs))
                    row.update(z=z,query_min=low.tolist(),query_max=high.tolist(),conflict_count=len(hit),
                        first_conflict_ids=[entities[i]['id'] for i in hit[:5]],
                        status='CLEAR_LOADED_WORLD_ONLY' if not len(hit) else 'BOUNDS_CONFLICT')
                    endpoints=np.array([[x,y],np.array([x,y])+4*f])
                    row['local_8m_envelope']=[(endpoints.min(axis=0)-8).tolist(),(endpoints.max(axis=0)+8).tolist()]
    # Deterministic greedy local packing is a lower-bound diagnostic, not maximal
    # site count and not cross-split scene independence. Touching boxes conflict.
    packed=[]
    for row in rows:
        if row['status']!='CLEAR_LOADED_WORLD_ONLY':continue
        lo,hi=map(np.asarray,row['local_8m_envelope'])
        if any(np.all(np.asarray(p['local_8m_envelope'][1])>=lo)&np.all(np.asarray(p['local_8m_envelope'][0])<=hi) for p in packed):continue
        packed.append(row)
    summary=[]
    for category in ('sidewalk','intersection','plaza','alley'):
        subset=[r for r in rows if r['category']==category];clear=[r for r in subset if r['status']=='CLEAR_LOADED_WORLD_ONLY']
        summary.append(dict(category=category,authored_regions=sum(a['category']==category for a in areas),
            enumerated_candidates=len(subset),status_counts=dict(Counter(r['status'] for r in subset)),
            clear_candidates=len(clear),authored_regions_with_clear_candidate=len(set(r['region'] for r in clear)),
            greedily_nonoverlapping_local_envelopes=sum(r['category']==category for r in packed),
            certified_independent_cross_split_sites=0,independence_status='NOT_ESTABLISHED' if subset else 'NO_SUBSTANTIATED_REGION'))
    result=dict(schema='cnh-street-site-inventory-v1',status='INSUFFICIENT_SOURCE_DIVERSITY_FOR_FOUR_ENVIRONMENTS',
        benchmark_eligible=False,collection='NOT_RUN',new_ue_processes=0,protocol=PROTOCOL,
        native_entities=len(entities),support_boxes=int(support.sum()),source_street_blocks=1,
        categories=summary,regions=areas,candidates=rows,packed_candidate_ids=[r['id'] for r in packed],
        limitations=['AABB prefilter failure does not prove collision or exhaustive absence of paths.',
            'Passing fixed pitch/cardinal path does not establish jittered trajectory, precise labels or ground mesh support.',
            'Nonoverlapping8m boxes do not prove no shared distant visible physical instances.',
            'One authored plaza and junction cannot be subdivided and renamed to establish split independence.',
            'No authored alley found; not an exhaustive topological proof that no narrow gap exists.'],
        hashes={str(p):sha(p) for p in (capture/'clearance.json',capture/'source/spec.json',capture/'native-material-intervention.json',Path(recipe_path),Path(__file__))},
        backend='CPU_TASK_NOT_GPU_SUITABLE_BOUNDED_AABB_METADATA',wall_s=time.monotonic()-started)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise FileExistsError(output/'result.json')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    plot(result,lows,highs,output/'site-map.png')
    print(json.dumps(summary))


def plot(result,lows,highs,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(14,9))
    obstacles=(highs[:,2]>.8)&(lows[:,2]<2.4)
    boxes=[Rectangle(lo[:2],*(hi[:2]-lo[:2])) for lo,hi in zip(lows[obstacles],highs[obstacles])]
    ax.add_collection(PatchCollection(boxes,facecolor='#ccd2d7',edgecolor='none',alpha=.35))
    colors={'sidewalk':'#3c83bd','intersection':'#d77a22','plaza':'#9571b8'}
    for a in result['regions']:
        lo=np.array(a['low']);hi=np.array(a['high'])
        ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,edgecolor=colors[a['category']],linewidth=1.5))
        ax.text(*((lo+hi)/2),a['id'],fontsize=8,ha='center',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    unique={}
    for r in result['candidates']:
        key=(r['x'],r['y']);unique[key]=unique.get(key,False) or r['status']=='CLEAR_LOADED_WORLD_ONLY'
    for status,color,label in ((False,'#d88888','No accepted heading on fixed grid'),(True,'#237c50','At least one 15cm-clear fixed heading')):
        points=np.asarray([k for k,v in unique.items() if v==status])
        if len(points):ax.scatter(points[:,0],points[:,1],s=5,c=color,label=label)
    ax.set(xlim=(-110,110),ylim=(-65,85),xlabel='UE world X (m)',ylabel='UE world Y (m)',
        title='Street200V7: authored regions + no-capture clearance prescreen\nGreen points are NOT independent sites; no authored alley')
    ax.set_aspect('equal');ax.legend(loc='lower left',fontsize=9);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inventory(a.capture,a.recipe,a.output)

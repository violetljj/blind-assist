"""Deterministic alley authoring plans from real loaded mesh bounds.

Physical map identity is separate from layout identity. Background dressing is
outside a conservative pedestrian swept envelope, never a hidden positive label.
No random rejection changes the assigned partition or overwrites a prior map.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import random
import re
from pathlib import Path


def validate(config):
    if config.get('schema') != 'cnh-street-alley-generator-v1':
        raise ValueError('Alley generator schema required')
    if config.get('namespace') != '/Game/BAResearchAlley':
        raise ValueError('Dedicated new-map namespace required')
    sites = config.get('sites', [])
    if not 1 <= len(sites) <= 24:
        raise ValueError('Bounded batch of 1..24 sites required')
    ids = [s['site_id'] for s in sites]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]{2,63}', s) for s in ids):
        raise ValueError('Unique safe site IDs required')
    if config.get('clearance_margin_m') != .15:
        raise ValueError('Original 15 cm background clearance retained')
    for site in sites:
        if site['topology'] not in ('straight', 'L', 'T'):
            raise ValueError('Unsupported alley topology')
        if not 1.5 <= site['width_m'] <= 4. or not 12. <= site['length_m'] <= 48.:
            raise ValueError('Width1.5..4m and length12..48m required')
        if not 12. <= site.get('branch_length_m', 16.) <= 32.:
            raise ValueError('Branch length12..32m required')
        if site['proposed_split'] not in ('train', 'dev', 'test'):
            raise ValueError('Explicit prospective physical-site assignment required')
        if type(site['seed']) is not int or not 6 <= site['clutter_count'] <= 32:
            raise ValueError('Integer seed and bounded dressing count required')
        for key, value in site.get('clutter_scales', {}).items():
            if key not in site['clutter_assets'] or not isinstance(value, (int, float)) or not math.isfinite(value) or not .05 <= value <= 2.:
                raise ValueError('Explicit uniform dressing scale .05..2 required')


def box_gap(a, b):
    """Euclidean separation of closed AABBs (zero on contact/intersection)."""
    return math.sqrt(sum(max(a[0][i]-b[1][i], b[0][i]-a[1][i], 0.)**2 for i in range(3)))


def transformed_bounds(bounds, yaw, scale=(1., 1., 1.)):
    angle = math.radians(yaw)
    c, s = math.cos(angle), math.sin(angle)
    points = []
    for x, y, z in itertools.product(*zip(*bounds)):
        x, y, z = x*scale[0], y*scale[1], z*scale[2]
        points.append((c*x-s*y, s*x+c*y, z))
    return [[min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]]


def placed(name, asset, bounds, low, yaw=0., scale=(1., 1., 1.), material=None, role='clutter'):
    rotated = transformed_bounds(bounds, yaw, scale)
    high = [low[i]+rotated[1][i]-rotated[0][i] for i in range(3)]
    return dict(name=name, asset=asset, location_m=[low[i]-rotated[0][i] for i in range(3)],
                yaw=yaw, scale=list(scale), material=material, role=role, planned_bounds_m=[list(low), high])


def network(site):
    length, branch, width = site['length_m'], site.get('branch_length_m', 16.), site['width_m']
    segments = [([0., 0.], [length, 0.])]
    endpoints = [([-width/2, 0.], 0), ([length+width/2, 0.], 0)]
    if site['topology'] != 'straight':
        joint = length if site['topology']=='L' else length*.57
        segments.append(([joint, 0.], [joint, branch]))
        endpoints.append(([joint, branch+width/2], 1))
        if site['topology']=='L':
            endpoints.pop(1)
    rectangles = []
    protected, paths = [], []
    for i, (a, b) in enumerate(segments):
        rectangles.append([min(a[0],b[0])-width/2,min(a[1],b[1])-width/2,
                           max(a[0],b[0])+width/2,max(a[1],b[1])+width/2])
        low = [min(a[0],b[0])-.6, min(a[1],b[1])-.6, 0.]
        high = [max(a[0],b[0])+.6, max(a[1],b[1])+.6, 2.]
        protected.append(dict(name=f'pedestrian_swept_{i}', low=low, high=high))
        heading = 0. if b[0]!=a[0] else 90.
        paths.append(dict(segment_id=i, start_m=a, end_m=b, camera_height_m=1.6, heading_degrees=heading,
            rule='Fixed-heading endpoint tracks inside this segment; actual frame queries must remain within protected volume'))
        # Recessed service bays keep bulky dressing visible even at 1.5m width.
        for fraction, side in ((.25, -1), (.72, 1)):
            x, y = [a[k]+fraction*(b[k]-a[k]) for k in range(2)]
            if heading==0:
                edge=side*width/2
                rectangles.append([x-1.8,min(edge,edge+side*1.8),x+1.8,max(edge,edge+side*1.8)])
            else:
                edge=x+side*width/2
                rectangles.append([min(edge,edge+side*1.8),y-1.8,max(edge,edge+side*1.8),y+1.8])
    return rectangles, protected, endpoints, paths


def union_cells(rectangles):
    xs=sorted({round(r[i],6) for r in rectangles for i in (0,2)})
    ys=sorted({round(r[i],6) for r in rectangles for i in (1,3)})
    cells={}
    for ix in range(len(xs)-1):
        for iy in range(len(ys)-1):
            x,y=(xs[ix]+xs[ix+1])/2,(ys[iy]+ys[iy+1])/2
            if any(a<x<c and b<y<d for a,b,c,d in rectangles):
                cells[ix,iy]=[xs[ix],ys[iy],xs[ix+1],ys[iy+1]]
    return cells


def boundary(cells, endpoints, width):
    pieces={}
    for (ix,iy),(x0,y0,x1,y1) in cells.items():
        edges=[((ix-1,iy),1,x0,y0,y1,1),((ix+1,iy),1,x1,y0,y1,-1),
               ((ix,iy-1),0,y0,x0,x1,1),((ix,iy+1),0,y1,x0,x1,-1)]
        for neighbour,axis,constant,start,end,inward in edges:
            if neighbour in cells:continue
            # axis is tangent coordinate; each entrance is an open end cap.
            if any(normal==1-axis and abs(point[1-axis]-constant)<1e-5 and
                   start>=point[axis]-width/2-1e-5 and end<=point[axis]+width/2+1e-5
                   for point,normal in endpoints):continue
            pieces.setdefault((axis,constant,inward),[]).append((start,end))
    merged=[]
    for (axis,constant,inward),intervals in sorted(pieces.items()):
        for start,end in sorted(intervals):
            if merged and merged[-1][:3]==[axis,constant,inward] and abs(merged[-1][4]-start)<1e-5:
                merged[-1][4]=end
            else:merged.append([axis,constant,inward,start,end])
    return merged


def generate_site(site, bounds, config):
    rectangles,protected,endpoints,paths=network(site)
    cells=union_cells(rectangles)
    edges=boundary(cells,endpoints,site['width_m'])
    rows=[]
    wall,paving=bounds['wall'],bounds['paving']
    module_width=wall[1][1]-wall[0][1]
    wall_height=wall[1][2]-wall[0][2]
    if min(module_width,wall_height)<=0:raise ValueError('Nonempty native wall bounds required')
    for ei,(axis,const,inward,start,end) in enumerate(edges):
        yaw=90. if axis==0 else 0.
        thickness=wall[1][0]-wall[0][0]
        for mi in range(math.ceil((end-start)/module_width)):
            span=min(module_width,end-start-mi*module_width)
            for storey in range(2):
                low=[0.,0.,storey*wall_height]
                low[axis]=start+mi*module_width
                low[1-axis]=const-thickness if inward==1 else const
                rows.append(placed(f'wall_{ei}_{mi}_{storey}','wall',wall,low,yaw,(1.,span/module_width,1.),site['wall_material'],'wall'))
    dx,dy=[paving[1][i]-paving[0][i] for i in (0,1)]
    for ci,(x0,y0,x1,y1) in enumerate(cells.values()):
        for ix in range(math.ceil(x1-x0)):
            for iy in range(math.ceil(y1-y0)):
                sx,sy=min(1.,x1-x0-ix),min(1.,y1-y0-iy)
                rows.append(placed(f'paving_{ci}_{ix}_{iy}','paving',paving,[x0+ix,y0+iy,-(paving[1][2]-paving[0][2])],
                    scale=(sx/dx,sy/dy,1.),material=site['floor_material'],role='support'))
    rows.append(placed('outer_ground','paving',paving,[-1000.,-1000.,-(paving[1][2]-paving[0][2])-.03],
                       scale=(2000/dx,2000/dy,1.),material='__neutral__',role='support'))
    rng=random.Random(site['seed'])
    keys=list(site['clutter_assets'])
    if not keys or any(k not in bounds for k in keys):raise ValueError('All dressing meshes must have actual bounds')
    rng.shuffle(keys)
    candidates=[]
    for edge in edges:
        axis,const,inward,start,end=edge
        if end-start<.6:continue
        for fraction in (.25,.5,.75):
            candidates.append((axis,const,inward,start+(end-start)*fraction,start,end))
    rng.shuffle(candidates)
    clutter=[]
    walls=[r for r in rows if r['role']=='wall']
    for attempt,(axis,const,inward,centre,start,end) in enumerate(candidates):
        if len(clutter)>=site['clutter_count']:break
        key=keys[len(clutter)%len(keys)]
        scale=(site.get('clutter_scales',{}).get(key,1.),)*3
        # Orient the narrow horizontal dimension normal to the wall, including
        # roof-source pipes whose native long axis differs from facade modules.
        options=(0.,90.)
        yaw=min(options,key=lambda angle: transformed_bounds(bounds[key],angle)[1][1-axis]-transformed_bounds(bounds[key],angle)[0][1-axis])
        yaw+=180. if inward==-1 else 0.
        rotated=transformed_bounds(bounds[key],yaw,scale)
        size=[rotated[1][i]-rotated[0][i] for i in range(3)]
        if size[axis]>end-start-.15:continue
        for lift in ([0.,2.3] if key in ('pipe','breaker_box','hvac') else [0.]):
            low=[0.,0.,lift]
            low[axis]=min(max(centre-size[axis]/2,start+.05),end-.05-size[axis])
            low[1-axis]=const+.04 if inward==1 else const-.04-size[1-axis]
            row=placed(f'clutter_{len(clutter):02d}_{key}',key,bounds[key],low,yaw,scale)
            box=row['planned_bounds_m']
            # Entire footprint must be supported by the union of walkable cells.
            corners=itertools.product((box[0][0],box[1][0]),(box[0][1],box[1][1]))
            if not all(any(a-1e-6<=x<=c+1e-6 and b-1e-6<=y<=d+1e-6 for a,b,c,d in cells.values()) for x,y in corners):continue
            if min(box_gap(box,[p['low'],p['high']]) for p in protected)<.15-1e-6:continue
            if any(box_gap(box,r['planned_bounds_m'])<.05 for r in clutter):continue
            if any(box_gap(box,r['planned_bounds_m'])<.02 for r in walls):continue
            row['placement']='elevated_service_equipment' if lift else 'service_bay_or_wallside'
            row['clearance_m']=min(box_gap(box,[p['low'],p['high']]) for p in protected)
            clutter.append(row)
            break
    if len(clutter)<site['clutter_count']:
        raise ValueError(f"{site['site_id']}: could place only {len(clutter)}/{site['clutter_count']} dressing assets; no silent omission")
    if len({r['asset'] for r in clutter})<3:raise ValueError('At least three dressing asset types per site required')
    rows.extend(clutter)
    structural=[r for r in rows if r['role']!='support']
    for row in structural:
        if min(box_gap(row['planned_bounds_m'],[p['low'],p['high']]) for p in protected)<.15-1e-5:
            raise ValueError('Native wall/dressing conflicts with protected swept volume: '+row['name'])
    extent=[min(r[0] for r in rectangles),min(r[1] for r in rectangles),max(r[2] for r in rectangles),max(r[3] for r in rectangles)]
    cx,cy=(extent[0]+extent[2])/2,(extent[1]+extent[3])/2
    h=max((extent[2]-extent[0]+8)/(2*math.tan(math.radians(35))),
          (extent[3]-extent[1]+8)/(2*math.tan(math.radians(35))*720/1280))+6
    views=[dict(name='overview',position_m=[cx,cy,h],yaw=90.,pitch=-90.),
           dict(name='entry',position_m=[.6,0.,1.6],yaw=0.,pitch=0.),
           dict(name='interior',position_m=[site['length_m']*.45,0.,1.6],yaw=0.,pitch=0.)]
    geometry_signature=hashlib.sha256(json.dumps(dict(rectangles=rectangles,rows=[{k:r[k] for k in ('asset','location_m','scale','yaw')} for r in rows if r['role']=='wall']),sort_keys=True).encode()).hexdigest()
    return dict(**site,map_asset=config['namespace']+'/'+site['site_id'],asset_root=config['namespace']+'/'+site['site_id']+'_Assets',
        rows=rows,protected_volumes=protected,views=views,clearance_margin_m=.15,paths=paths,
        floor_rectangles=rectangles,geometry_signature=geometry_signature,
        clutter_counts={k:sum(r['asset']==k for r in clutter) for k in sorted(set(r['asset'] for r in clutter))},
        planned_min_clearance_m=min(box_gap(r['planned_bounds_m'],[p['low'],p['high']]) for r in structural for p in protected),
        authority='AUTHORING_ASSIGNMENT_NOT_BENCHMARK_ADMISSION',background_assets_shared=True)


def generate(config, bounds):
    validate(config)
    for key,pair in bounds.items():
        if len(pair)!=2 or any(len(v)!=3 for v in pair) or not all(math.isfinite(x) for v in pair for x in v):
            raise ValueError('Finite measured mesh bounds required: '+key)
    sites=[generate_site(site,bounds,config) for site in config['sites']]
    signatures=[s['geometry_signature'] for s in sites]
    if len(set(signatures))!=len(signatures):raise ValueError('Duplicate physical geometry across site IDs rejected')
    return dict(schema='cnh-street-alley-plan-v1',sites=sites,benchmark_eligible=False,
        partition_rule='A physical map/site ID belongs to one prospective partition; layouts may not cross it',
        independence_limit='Distinct authored maps and geometry; shared background assets and procedural distribution are not independent real-world sources',
        protected_definition='All centreline segments plus lateral0.6m, height0..2m, and separate15cm geometry margin; supports excluded')


def check_registry(sites, registry):
    """Cross-run physical identity guard, independent of chosen map/file names."""
    if registry.get('schema')!='cnh-street-alley-site-registry-v1':
        raise ValueError('Unknown persistent site registry schema')
    for site in sites:
        for old in registry['sites']:
            if old['site_id']==site['site_id'] or old['map_asset']==site['map_asset']:
                raise ValueError('Previously authored physical site cannot be reassigned or overwritten: '+site['site_id'])
            if old['geometry_signature']==site['geometry_signature']:
                raise ValueError('Previously authored geometry cannot acquire a new identity or partition: '+site['site_id'])


def register_saved_site(path,site,geometry_receipt,map_sha256):
    """Append only successfully saved, measured maps; single-editor controller."""
    path=Path(path)
    registry=json.loads(path.read_text()) if path.exists() else dict(schema='cnh-street-alley-site-registry-v1',sites=[])
    check_registry([site],registry)
    if geometry_receipt.get('status')!='SAVED_AUTHORING_GEOMETRY_REQUIRES_VISUAL_REVIEW' or not geometry_receipt.get('clearance_checks') or not all(r['passed'] for r in geometry_receipt['clearance_checks']):
        raise ValueError('Saved map with measured clearance required before registry append')
    registry['sites'].append({k:site[k] for k in ('site_id','map_asset','geometry_signature','proposed_split','topology')}|
        dict(map_sha256=map_sha256,authority='PROSPECTIVE_PARTITION_NOT_BENCHMARK_ADMISSION'))
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.writing.json')
    if temp.exists():raise FileExistsError('Unresolved prior registry write: '+str(temp))
    temp.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)
    return registry

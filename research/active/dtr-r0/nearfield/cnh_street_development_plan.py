"""Freeze a six-layout Development starter; does not admit formal benchmark data."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from cnh_route_source_compare_adapter import DEVELOPMENT_SCOPE
from cnh_route_source_launch import validate


SITES=[('sidewalk',-40.,-10.,0),('intersection',-4.,0.,0),
       ('plaza',38.,28.,180),('sidewalk',40.,-10.,0),
       ('intersection',0.,-4.,90),('plaza',64.,30.,0)]


def freeze(template, inventory, output):
    template=Path(template);inventory=Path(inventory);output=Path(output)
    source=json.loads(template.read_text(encoding='utf-8-sig'))
    screen=json.loads(inventory.read_text(encoding='utf-8-sig'))
    layouts=[]
    for index,(category,x,y,yaw) in enumerate(SITES):
        rows=[r for r in screen['candidates'] if (r['category'],r['x'],r['y'],r['yaw'])==(category,x,y,yaw)]
        if len(rows)!=1 or rows[0]['status']!='CLEAR_LOADED_WORLD_ONLY':
            raise ValueError('Frozen starter site failed conservative screen')
        z=rows[0]['z'];angle=math.radians(yaw)
        f=(round(math.cos(angle)),round(math.sin(angle)));r=(-f[1],f[0])
        base=copy.deepcopy(source['layouts'][index%2]);base.pop('candidates',None)
        base.update(layout_id=f'street-dev-{index:02d}',physical_site_id='Street200V7-single-street-block',
                    environment_category=category,inventory_candidate_id=rows[0]['id'])
        original_camera=base['clips'][0]['poses'][0]
        base['camera']=dict(x=x,y=y,z=z,pitch=0.,yaw=yaw,roll=0.)
        for clip in base['clips']:
            # Controlled counterfactuals retain identical camera trajectories.
            clip['poses']=[dict(base['camera'],x=x+.1*i*f[0],y=y+.1*i*f[1]) for i in range(40)]
            for obj in clip['insertions']:
                lateral=obj['center_m'][1]-original_camera['y']
                distance=4.25 if obj['id']==1 else 4.5
                obj['center_m']=[x+distance*f[0]+lateral*r[0],y+distance*f[1]+lateral*r[1],
                                 z+obj['center_m'][2]-original_camera['z']]
                obj['rotation_deg']['yaw']+=yaw
        layouts.append(base)
    output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for shard in range(1):
        spec=copy.deepcopy(source)
        spec.update(schema='cnh-street-development-pilot-v1',scope=DEVELOPMENT_SCOPE,scene_layer=DEVELOPMENT_SCOPE,
            data_role='Development',layouts=layouts,nominal_sample_interval_s=.1,render_recipe='STATIC_SPATIAL_V1',
            temporal_scope='40_SETTLED_STATIC_WORLD_POSES_NOT_REALTIME_OR_DYNAMIC_VIDEO',
            quality_policy='RECORD_PER_LAYOUT_PENDING_OR_QUARANTINED_NO_FORMAL_ADMISSION',
            provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (template,inventory)},
            limits=dict(total_starter_layouts=6,batch_frames=960,timeout_s=3600),
            disclosure=['One shared Street200 block; no independent train/test split.',
                'Frozen native WPO/PDO zero; vegetation static; SceneDepth omits non-depth-writing glass.',
                'Two reused insertion asset families, three environment categories; no alley or full384 claim.',
                'Energy, label accuracy, asset isolation and geometry are checked on collected data; failed units quarantined.'])
        validate(spec)
        path=output/f'street-development-batch-{shard:02d}.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(spec,indent=2),encoding='utf-8');paths.append(str(path))
    return paths


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('template','inventory','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(freeze(a.template,a.inventory,a.output)))

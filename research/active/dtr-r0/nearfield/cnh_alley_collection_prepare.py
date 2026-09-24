"""Freeze six train/dev authoring-map acquisition specs; no test map capture."""
from copy import deepcopy
import json
from pathlib import Path

def insertion(identifier, bounds, x, y, hidden=False):
    lo,hi=bounds
    extent=[float(b)-float(a) for a,b in zip(lo,hi)]
    if len(extent)!=3 or any(v<=0 for v in extent):raise ValueError('Measured nondegenerate mesh bounds required')
    # Explicit controlled shape: narrowest 1.8m lane has wall at +/-0.9m.
    # y=.78 plus .06 halfwidth leaves .06m native-wall gap; prefilter is authoritative.
    size=[.24,.12,1.2 if identifier==1 else 1.0]
    return dict(id=identifier,center_m=[x,y,size[2]/2],scale=[size[i]/extent[i] for i in range(3)],
        rotation_deg=dict(pitch=0.,yaw=0.,roll=0.),hidden=hidden)

def make_spec(template, site, manifest, assets, catalog=None):
    split=site['proposed_split']
    if split not in ('train','dev'):raise ValueError('Test maps reserved, no capture permitted')
    if catalog is None:
        catalog=json.loads(Path(__file__).with_name('cnh_street_alley_assets.json').read_text(encoding='utf-8'))
    keys=site.get('insert_assets')
    if not isinstance(keys,list) or len(keys)!=2 or len(set(keys))!=2 or len(assets)!=2:
        raise ValueError('Two distinct site-declared inserted asset families required')
    declarations=[catalog['insert_assets'][key] for key in keys]
    for actual,declared in zip(assets,declarations):
        if (declared.get('role')!='CONTROLLED_INSERT' or declared.get('proposed_split')!=split or
                actual.get('source')!=declared.get('asset_path')):
            raise ValueError('Inserted asset differs from site declaration or partition')
    if len({row['family_id'] for row in declarations})!=2 or len({row['source_family_root'] for row in declarations})!=2:
        raise ValueError('Inserted assets share a semantic or source family')
    target,distractor=assets
    poses=[dict(x=round(2.+i*.1,8),y=0.,z=1.6,pitch=0.,yaw=0.,roll=0.) for i in range(40)]
    clips=[]
    for name,y in [('centre',0.),('boundary',.6),('outside',.78),('removed',0.)]:
        clips.append(dict(id=name,trajectory_model='piecewise_linear_fixed_orientation',poses=deepcopy(poses),
            insertions=[insertion(1,target['bounds_m'],7.3,y,name=='removed'),
                        insertion(254,distractor['bounds_m'],7.6,-.78)]))
    spec=deepcopy(template)
    for key in ('source_snapshot_sha256','provenance','diagnostic','candidate_seed'):spec.pop(key,None)
    spec.update(schema='cnh-alley-development-capture-v1',scope='ALLEY_DEVELOPMENT_PILOT_NOT_BENCHMARK',
        scene_layer='ALLEY_DEVELOPMENT_PILOT_NOT_BENCHMARK',benchmark_eligible=False,
        data_role='Development',map_asset=site['map_asset'],map_file=manifest['map_file'],
        map_sha256=manifest['map_sha256'],alley_manifest={k:v for k,v in manifest.items() if k not in ('map_file','map_sha256')},
        native_material_policy='ALLEY_FROZEN_STATIC_COMPILED',nominal_sample_interval_s=.1,
        render_recipe='STATIC_SPATIAL_V1',transport_policy='NATIVE_SEVEN_ASYNC_V1',
        rgb_exposure_policy='ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2',rgb_probe_ev100=-2.,
        assets=[dict(mesh_asset=row['source'],id=i) for row,i in zip(assets,(1,254))],
        layouts=[dict(layout_id=site['site_id']+'-worker-00',physical_site_id=manifest['physical_site_id'],
            environment_category='alley',camera=poses[0],clips=clips)],limits=dict(frames=160,layouts=1,timeout_s=600),
        cohort='alley-worker-development-r3',normal_crossmachine_mixing='NOT_ADMITTED',
        disclosure=['Authored map Development acquisition; proposed split labels are not formal test access.',
            'Controlled insertions explicitly nonuniform-scaled to 0.24x0.12m footprint; source family preserved.',
            'One fixed layout per map; nominal static-world 10Hz poses, no dynamic video claim.',
            'One settled RGB probe selects a fixed per-layout exposure bias; RGB-only recipe revision.',
            'Sample geometry only; failed layout quarantined without threshold tuning or quality retry.'])
    return spec

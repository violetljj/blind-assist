"""Materialize a supported-scene preview against a pinned City map."""
import argparse
import json
import sys
from pathlib import Path
from make_city_obstacle_suite import REPO, read, sha, under_artifacts
sys.path.insert(0,str(REPO/'research/active/dtr-r0/nearfield'))
from contextual_scene import preview, collection


def make(map_spec, output, full=False):
    map_spec,output=map(under_artifacts,(map_spec,output))
    template=read(map_spec)
    map_file=under_artifacts(template['map_file'])
    if sha(map_file)!=template['map_sha256']:
        raise ValueError('Map differs from pinned source')
    result=(collection if full else preview)(template)
    content=next(p for p in map_file.parents if p.name=='Content')
    assets={o['material_asset'] for c in result['cases'] for o in c['objects']}
    hashes={a:sha(content/(a.removeprefix('/Game/')+'.uasset')) for a in sorted(assets)}
    result['provenance']=dict(map_spec_sha256=sha(map_spec),material_sha256=hashes,
        sources_sha256={n:sha(REPO/'research/active/dtr-r0/nearfield'/n)
                        for n in (('contextual_scene.py','city_pcg_capture.py','contextual_sampling.py',
                                   'contextual_geometry.py','contact_retina_spec.py') if full else
                                  ('contextual_scene.py','city_pcg_capture.py'))})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),frames=len(result['cases']),sha256=sha(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map-spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collection-128',action='store_true')
    a=p.parse_args();make(a.map_spec,a.output,a.collection_128)

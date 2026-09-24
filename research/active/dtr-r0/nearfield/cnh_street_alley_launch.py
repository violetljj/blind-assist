"""Governed one-editor launcher for bounded three-map authoring, no dataset run."""
import argparse
import json
from pathlib import Path
import shutil
import cnh_route_insert_launch as common

HERE=Path(__file__).parent


def launch(args):
    from cnh_street_alley_build import validate
    originals=(common.validate_spec,common.run_owned,common.finalize)
    def run_owned(command,env,out,timeout):
        extras=('cnh_street_alley_build.py','cnh_street_alley_preview.py','cnh_street_alley_plan.py','cnh_street_alley_assets.json')
        for name in extras:shutil.copy2(HERE/name,out/'source'/name)
        command=[('-ExecCmds=py '+(out/'source/cnh_street_alley_build.py').as_posix()) if x.startswith('-ExecCmds=py ') else x for x in command]
        env=dict(env,BA_CNH_ALLEY_SPEC=env['BA_CNH_INSERT_SPEC'],BA_CNH_ALLEY_OUTPUT=str(out))
        receipt=json.loads((out/'launch.json').read_text());receipt['command']=command
        receipt['source_hashes'].update({name:common.file_hash(out/'source'/name) for name in extras})
        common.write(out/'launch.json',receipt)
        return originals[1](command,env,out,timeout)
    def finalize(out):
        from PIL import Image
        result=json.loads((out/'preview-receipt.json').read_text())
        if result['status']!='PASS_MULTI_MAP_AUTHORING_PREVIEW_ONLY':raise RuntimeError(str(result))
        expected=len(json.loads((out/'source/spec.json').read_text())['sites'])
        if len(result['maps'])!=expected:raise RuntimeError('Every declared map must complete')
        for item in result['maps']:
            if len(item['views'])!=3:raise RuntimeError('Three actual views per map required')
            for view in item['views']:
                with Image.open(view['path']) as image:
                    image.load()
                    if image.size!=(1280,720):raise RuntimeError('Preview dimensions differ')
        return dict(status='PASS_AUTHORING_REQUIRES_VISUAL_REVIEW',benchmark_eligible=False,
            formal_split_assignment='NONE',dataset_admission='NOT_RUN',maps=result['maps'])
    common.validate_spec=validate;common.run_owned=run_owned;common.finalize=finalize
    try:return common.launch(args)
    finally:common.validate_spec,common.run_owned,common.finalize=originals


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ('project','engine','plugin','spec','output','result'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=600)
    print(json.dumps(launch(parser.parse_args())))

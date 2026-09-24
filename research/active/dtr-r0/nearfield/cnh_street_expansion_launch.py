"""Governed process lifecycle for first independent concept-A authoring map."""
import argparse
import json
from pathlib import Path
import shutil
import cnh_route_insert_launch as common
from cnh_street_expansion_build import validate

HERE=Path(__file__).parent


def launch(args):
    originals=(common.validate_spec,common.run_owned,common.finalize)
    def run_owned(command,env,out,timeout):
        extras=('cnh_street_expansion_build.py','cnh_street_expansion_preview.py')
        for name in extras:shutil.copy2(HERE/name,out/'source'/name)
        command=[('-ExecCmds=py '+(out/'source/cnh_street_expansion_build.py').as_posix()) if x.startswith('-ExecCmds=py ') else x for x in command]
        env=dict(env,BA_CNH_EXPANSION_SPEC=env['BA_CNH_INSERT_SPEC'],BA_CNH_EXPANSION_OUTPUT=str(out))
        receipt=json.loads((out/'launch.json').read_text());receipt['command']=command
        receipt['source_hashes'].update({name:common.file_hash(out/'source'/name) for name in extras})
        common.write(out/'launch.json',receipt)
        return originals[1](command,env,out,timeout)
    def finalize(out):
        from PIL import Image
        receipt=json.loads((out/'preview-receipt.json').read_text())
        if receipt['status']!='PASS_AUTHORING_PREVIEW_ONLY':raise RuntimeError(str(receipt))
        if len(receipt['views'])!=2:raise RuntimeError('Missing preview views')
        for view in receipt['views']:
            with Image.open(view['path']) as image:
                image.load()
                if image.size!=(1280,720):raise RuntimeError('Preview dimensions differ')
        return dict(status='PASS_AUTHORING_PREVIEW_REQUIRES_VISUAL_REVIEW',benchmark_eligible=False,
            new_data_admission='NOT_RUN',views=receipt['views'],build_receipt=str(out/'build/build-receipt.json'))
    common.validate_spec=validate;common.run_owned=run_owned;common.finalize=finalize
    try:return common.launch(args)
    finally:common.validate_spec,common.run_owned,common.finalize=originals


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('project','engine','plugin','spec','output','result'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout',type=float,default=600)
    print(json.dumps(launch(p.parse_args())))

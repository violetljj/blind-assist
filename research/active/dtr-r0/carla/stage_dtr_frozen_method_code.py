"""Stage a separate method-code snapshot without reverting concurrent changes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]


def sha(data):return hashlib.sha256(data).hexdigest().upper()


def stage(output,revision):
    output=output.resolve()
    if not output.is_relative_to((REPO/'artifacts.local').resolve()):raise ValueError('Artifact routing')
    original=subprocess.check_output(['git','rev-parse',revision],cwd=REPO).decode().strip()
    closure=json.loads((HERE/'dtr_carla_c44_x96_dropout_survival_protocol.json').read_text())
    frozen=closure['c44_x96_preregistration']['frozen_component_sha256']
    names=subprocess.check_output(['git','ls-files','research/active/dtr-r0/carla'],cwd=REPO).decode().splitlines()
    payloads={};records=[]
    for name in names:
        path=REPO/name
        if path.suffix not in ('.py','.json'):continue
        data=path.read_bytes();current=sha(data);source='current_working_file'
        if path.name in frozen and current!=frozen[path.name].upper():
            data=subprocess.check_output(['git','show',original+':'+name],cwd=REPO)
            if sha(data)!=frozen[path.name].upper():raise ValueError('Historical bytes do not match '+name)
            source=original
        payloads[name]=data
        records.append({'path':name,'sha256':sha(data),'working_sha256':current,'source':source})
    output.mkdir(parents=True,exist_ok=False)
    for name,data in payloads.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest={'scope':'ISOLATED_METHOD_CODE_NO_SOURCE_RECAPTURE','historical_revision':original,
              'generator_sha256':sha(Path(__file__).read_bytes()),'files':records}
    (output/'method-code-provenance.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'files':len(records),'historical_restorations':[r['path'] for r in records if r['source']!='current_working_file']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--revision',required=True)
    args=parser.parse_args();stage(args.output,args.revision)

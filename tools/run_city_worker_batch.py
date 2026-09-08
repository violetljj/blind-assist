"""Worker artifact-owned batch entry: input spec plus a fresh output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--spec', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--timeout', type=float, default=1800)
parser.add_argument('--production', action='store_true',
                    help='Use asynchronous canonical RGB/depth export without reference copies or 720p appearance')
args = parser.parse_args()
source = Path(__file__).resolve().parents[1]
config_path = source.parent/'worker-config.json'
config = json.loads(config_path.read_text())
original = args.spec.resolve(strict=True)
spec = json.loads(original.read_text(encoding='utf-8-sig'))
require = lambda condition: condition or (_ for _ in ()).throw(ValueError('Unsupported map or existing output'))
require(spec['map_asset'] == config['map_asset'])
require(not args.output.exists())
spec['map_file'] = config['map_file']
changes = ['map_file remapped to worker']
if args.production:
    spec.update(pair_export_mode='native_async', export_appearance=False)
    changes.append('production export: native_async, export_appearance=false')
require(hashlib.sha256(Path(spec['map_file']).read_bytes()).hexdigest() == spec['map_sha256'])
adapted = args.output.parent/(args.output.name+'-worker-spec.json')
spec['worker_input_provenance'] = dict(original_path=str(original), original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                                    source_commit=config['source_commit'], changes=changes)
adapted.parent.mkdir(parents=True, exist_ok=True)
with adapted.open('x') as stream:
    json.dump(spec, stream, indent=2)
commands = [
    [sys.executable,'-B',str(source/'tools/run_city_pcg_capture.py'),'--project',config['project'],
     '--plugin',config['plugin'],'--engine',config['engine'],'--spec',str(adapted),
     '--output',str(args.output),'--timeout',str(args.timeout)],
    [sys.executable,'-B',str(source/'tools/verify_city_pcg_world.py'),'--capture',str(args.output)],
    [sys.executable,'-B',str(source/'tools/summarize_city_groups.py'),'--capture',str(args.output)],
]
for command in commands:
    subprocess.run(command, cwd=source, check=True)
print(json.dumps(dict(status='PASS',capture=str(args.output),config=str(config_path))))

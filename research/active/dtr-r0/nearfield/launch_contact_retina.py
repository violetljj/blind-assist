"""Launch owned offscreen Unreal capture of NF-G7 fixed prescribed trajectory capture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(repo / 'tools'), str(repo / 'research/active/dtr-r0/unreal')]
    from run_obstacle_research import engine_root
    from street_process_lifecycle import TaskProcessTree
    artifact_root = (repo / 'artifacts.local').resolve()
    spec, out = args.spec.resolve(), args.output.resolve()
    if not spec.is_relative_to(artifact_root) or not spec.is_file():
        parser.error('Spec must be an existing file under artifacts.local')
    if not out.is_relative_to(artifact_root) or out.exists():
        parser.error('Use a new output under artifacts.local')
    out.mkdir(parents=True)
    project = repo / 'artifacts.local/unreal/BlindAssistStreetLab'
    env = dict(os.environ, BA_NEARFIELD_SPEC=str(spec), BA_NEARFIELD_OUTPUT=str(out))
    env['UE-LocalDataCachePath'] = str(project / 'DerivedDataCache')
    startup = subprocess.STARTUPINFO() if os.name == 'nt' else None
    if startup is not None:
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    proc = subprocess.Popen([
        str(engine_root() / 'Engine/Binaries/Win64/UnrealEditor.exe'),
        str(project / 'BlindAssistStreetLab.uproject'),
        '-ExecCmds=py ' + Path(__file__).with_name('capture_contact_retina.py').as_posix(),
        '-RenderOffscreen', '-unattended', '-nosound', '-nop4', '-NoSplash', '-ddc=NoShared',
        '-abslog=' + str(out / 'editor.log')], env=env, startupinfo=startup,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    tree = TaskProcessTree(proc, owner=str(out))
    try:
        code = tree.wait(timeout=1800)
        receipt = json.loads((out / 'receipt.json').read_text(encoding='utf-8'))
        if code or receipt['status'] != 'PASS':
            raise RuntimeError('Capture failed; editor log and receipt retained')
        print(json.dumps(receipt, indent=2))
    finally:
        release = tree.cleanup()
        (out / 'process-release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
        if not release['released']:
            raise RuntimeError('Owned process release incomplete; inspect process-release.json')


if __name__ == '__main__':
    main()

"""Owned, non-rendering UE metadata inspection using the established worker runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(config_path, script, output):
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(config['runtime']) / 'tools'))
    from ue_native_capture import run_owned, write
    from run_city_pcg_capture import cache_service_port
    before = {path: sha(path) for path in (config['project'], config['map_file'], str(script))}
    env = dict(os.environ, BA_MZ42_METADATA=str(output / 'metadata.json'))
    env['UE-LocalDataCachePath'] = config['ddc']
    port = cache_service_port(Path(config['ddc']))
    command = [str(Path(config['engine']) / 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'),
               config['project'], '-run=pythonscript', '-script=' + str(script), '-NullRHI',
               '-unattended', '-nosound', '-nop4', '-NoSplash', '-ddc=NoShared',
               '-ini:Engine:[Zen.AutoLaunch]:DesiredPort=' + str(port),
               '-abslog=' + str(output / 'editor.log'), '-EnablePlugins=PythonScriptPlugin',
               '-DisablePlugins=CLionSourceCodeAccess,VisualStudioCodeSourceCodeAccess',
               '-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
               '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    write(output / 'launch.json', dict(command=command, inputs=before, zen_port=port))
    try:
        run_owned(command, env, output, 900.)
        metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
        if metadata['status'] != 'PASS':
            raise RuntimeError('Metadata failed')
        if any(sha(path) != digest for path, digest in before.items()):
            raise RuntimeError('Source changed during metadata probe')
        write(output / 'completion.json', dict(status='PASS', inputs=before,
              outputs={path.name: sha(path) for path in output.iterdir() if path.is_file()},
              rendering_frames=0, training_steps=0, model_inference_frames=0))
    except BaseException as error:
        write(output / 'failure.json', dict(status='FAIL', error=str(error), inputs=before))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('config', 'script', 'output'):
        parser.add_argument('--' + argument, type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.script, args.output)

"""Build the optional UE depth exporter in a fresh artifacts.local directory.

Requires the engine's supported MSVC, Windows SDK, and .NET Framework SDK.
This command never installs a plugin into the shared project.
"""
import argparse
import os
from pathlib import Path
import shutil

from ue_native_capture import UE_SOURCE, file_hash, owned_output, run_owned, write


def build(args):
    from run_obstacle_research import engine_root

    engine = engine_root(args.engine)
    source = UE_SOURCE / 'native_capture/BlindAssistCapture'
    if not (source / 'BlindAssistCapture.uplugin').is_file():
        raise FileNotFoundError('Native capture plugin source is missing')
    out = owned_output(args.output)
    staged = out / 'source/BlindAssistCapture'
    # UAT may generate Config/FilterPlugin.ini: keep that mutation in the copy.
    shutil.copytree(source, staged)
    source_hashes = {
        p.relative_to(staged).as_posix(): file_hash(p)
        for p in sorted(staged.rglob('*')) if p.is_file()
    }
    env = dict(os.environ, uebp_LogFolder=str(out / 'uat-logs'))
    command = [str(engine / 'Engine/Build/BatchFiles/RunUAT.bat'), 'BuildPlugin',
               '-Plugin=' + str(staged / 'BlindAssistCapture.uplugin'),
               '-Package=' + str(out / 'package'), '-HostPlatforms=Win64',
               '-NoTargetPlatforms', '-Unattended', '-UTF8Output']
    write(out / 'launch.json', dict(command=command, engine=str(engine),
                                   source_hashes=source_hashes))
    try:
        run_owned(command, env, out, args.timeout)
        plugin = out / 'package/BlindAssistCapture.uplugin'
        binaries = plugin.parent / 'Binaries'
        if not plugin.is_file() or not (binaries / 'Win64/UnrealEditor-BlindAssistCapture.dll').is_file():
            raise RuntimeError('Build finished without the expected plugin package and DLL')
        write(out / 'build-receipt.json', dict(
            status='PASS', engine=str(engine), plugin=str(plugin),
            source_hashes=source_hashes, plugin_sha256=file_hash(plugin),
            binary_hashes={p.relative_to(plugin.parent).as_posix(): file_hash(p)
                           for p in sorted(binaries.rglob('*')) if p.is_file()}))
    except BaseException as exc:
        write(out / 'build-receipt.json', dict(status='FAIL', error=str(exc),
                                              source_hashes=source_hashes))
        raise
    print(str(plugin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory strictly under artifacts.local')
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--timeout', type=float, default=1200)
    build(parser.parse_args())


if __name__ == '__main__':
    main()

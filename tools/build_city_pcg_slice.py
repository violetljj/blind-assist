"""Build a new task-owned City PCG slice without replacing existing maps."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil

from run_city_pcg_capture import artifact_file
from run_obstacle_research import engine_root
from ue_native_capture import REPO, file_hash, owned_output, run_owned, write


def build(args):
    project = artifact_file(args.project, 'Project')
    if project.suffix.lower() != '.uproject':
        raise ValueError('Project must be a .uproject descriptor')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    if not re.fullmatch(r'BAResearchSlice/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*', args.map_name):
        raise ValueError('Map name must be BAResearchSlice/ followed by alphanumeric/underscore segments')
    content = project.parent / 'Content'
    if not content.is_dir() or content.resolve() != content.absolute():
        raise ValueError('Project Content must be a physical directory, not a junction')
    target = content.joinpath(*args.map_name.split('/')).with_suffix('.umap')
    if target.exists():
        raise FileExistsError('Refuse overwrite existing map: ' + str(target))
    if target.resolve() != target.absolute():
        raise ValueError('Target map cannot pass through a junction')
    source = REPO / 'research/active/dtr-r0/nearfield/city_pcg_build.py'
    if not source.is_file():
        raise FileNotFoundError(source)
    before = file_hash(project)
    graph_sources = {}
    if getattr(args, 'layout', None):
        layout_data = json.loads(artifact_file(args.layout, 'Layout').read_text(encoding='utf-8-sig'))
        if layout_data.get('base_map'):
            base_map=layout_data['base_map']
            if not re.fullmatch(r'/Game/BAResearchSlice/[A-Za-z0-9_/]+',base_map):
                raise ValueError('Base map must be a BA research map')
            base_file=artifact_file(project.parent / 'Content' / (base_map[len('/Game/'):] + '.umap'), 'Base map')
            graph_sources[str(base_file)]=file_hash(base_file)
        for row in layout_data['graphs']:
            path = row['asset']
            if not path.startswith('/CitySamplePCG/'):
                raise ValueError('Layout graphs must come from CitySamplePCG')
            source_path = (project.parent / 'Plugins/Experimental/CitySamplePCG/Content' / (path[len('/CitySamplePCG/'):] + '.uasset')).resolve(strict=True)
            if not source_path.is_relative_to(project.parent):
                raise ValueError('Graph source escapes project')
            graph_sources[str(source_path)] = file_hash(source_path)
    engine = engine_root(getattr(args, 'engine', None))
    out = owned_output(args.output)
    try:
        snapshot = out / 'source'
        snapshot.mkdir()
        script = snapshot / source.name
        shutil.copy2(source, script)
        temp = out / 'temp'
        temp.mkdir()
        env = dict(os.environ, BA_CITY_OUT=str(out), BA_CITY_MAP='/Game/' + args.map_name,
                   TEMP=str(temp), TMP=str(temp), PYTHONDONTWRITEBYTECODE='1')
        if getattr(args, 'layout', None):
            layout = artifact_file(args.layout, 'Layout')
            shutil.copy2(layout, snapshot / 'layout.json')
            env['BA_CITY_LAYOUT'] = str(snapshot / 'layout.json')
        env['UE-LocalDataCachePath'] = str(project.parent / 'DerivedDataCache')
        command = [str(engine / 'Engine/Binaries/Win64/UnrealEditor.exe'), str(project),
                   '-ExecCmds=py ' + script.as_posix(), '-RenderOffscreen', '-unattended',
                   '-nosound', '-nop4', '-NoSplash', '-NoDefaultMaps', '-ddc=NoShared',
                   '-abslog=' + str(out / 'editor.log'),
                   '-EnablePlugins=CitySamplePCG,PythonScriptPlugin',
                   '-DisablePlugins=CLionSourceCodeAccess,VisualStudioCodeSourceCodeAccess',
                   '-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
                   '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
        write(out / 'launch.json', dict(command=command, project=str(project),
              project_sha256_before=before, map_asset=env['BA_CITY_MAP'], map_file=str(target),
              builder_sha256=file_hash(script), launcher_sha256=file_hash(Path(__file__))))
        # Recheck immediately before starting the builder after snapshot preparation.
        if target.exists():
            raise FileExistsError('Target map appeared before launch')
        run_owned(command, env, out, args.timeout)
        receipt = json.loads((out / 'build-receipt.json').read_text(encoding='utf-8'))
        if receipt.get('status') != 'PASS':
            raise RuntimeError('UE slice build did not pass')
        declared = Path(receipt['map_file'])
        if not declared.is_absolute() or declared.resolve() != target.resolve():
            raise ValueError('Builder map_file differs from planned target')
        if not target.is_file():
            raise FileNotFoundError('Builder did not save expected map')
        after = file_hash(project)
        if after != before:
            raise RuntimeError('Source project descriptor changed')
        source_after = {path: file_hash(Path(path)) for path in graph_sources}
        if source_after != graph_sources:
            raise RuntimeError('Official source graph changed')
        result = dict(status='PASS', map_file=str(target), map_sha256=file_hash(target),
                      project_sha256_before=before, project_sha256_after=after,
                      source_unchanged=True, official_graph_hashes=graph_sources)
        write(out / 'completion.json', result)
    except BaseException as exc:
        after = file_hash(project) if project.is_file() else None
        write(out / 'completion.json', dict(status='FAIL', error=str(exc), map_file=str(target),
              map_sha256=file_hash(target) if target.is_file() else None,
              project_sha256_before=before, project_sha256_after=after, source_unchanged=before == after))
        raise
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map-name', default='BAResearchSlice/PlazaV1')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--layout', type=Path)
    build(parser.parse_args())

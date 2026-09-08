"""Read-only on-disk City PCG package closure; callable inside a live editor."""
import json
from pathlib import Path
import time
import traceback


def export_dependencies(out, roots):
    """Write a fresh manifest and return it without saving assets or quitting UE."""
    import unreal as u

    out = Path(out)
    manifest = out / 'dependency-manifest.json'
    if manifest.exists():
        raise FileExistsError('Refuse overwrite dependency-manifest.json')
    roots = list(roots)
    result = dict(status='RUNNING', roots=roots, packages=[], files=[], warnings=[], engine_resources=[],
                  scope='On-disk hard and soft package closure, including editor-only references; runtime string loads are not guaranteed covered')
    started = time.monotonic()
    try:
        if not roots or any(not isinstance(p, str) or not p.startswith('/') or
                            any(part in ('', '.', '..') for part in p[1:].split('/')) for p in roots):
            raise ValueError('Roots must be nonempty absolute Unreal package paths')
        project = Path(u.Paths.project_dir()).resolve()
        registry = u.AssetRegistryHelpers.get_asset_registry()
        registry.search_all_assets(True)
        options = u.AssetRegistryDependencyOptions()
        for key in ('include_soft_package_references', 'include_hard_package_references',
                    'include_game_package_references', 'include_editor_only_package_references'):
            options.set_editor_property(key, True)
        for key in ('include_searchable_names', 'include_soft_management_references', 'include_hard_management_references'):
            options.set_editor_property(key, False)
        engine = Path(u.Paths.engine_dir()).resolve()
        engine_mounts = {p.stem for p in (engine / 'Plugins').rglob('*.uplugin')} | {'Engine', 'Script'}
        local = {'Game': Path('Content'), 'CitySamplePCG': Path('Plugins/Experimental/CitySamplePCG/Content')}
        queue, seen, file_paths = list(roots), set(), set()
        while queue:
            package = queue.pop()
            if package in seen:
                continue
            seen.add(package)
            parts = package.strip('/').split('/')
            if not package.startswith('/') or any(part in ('', '.', '..') for part in parts):
                raise ValueError('Invalid dependency package path: ' + package)
            mount = parts[0]
            if mount not in local:
                if mount in engine_mounts:
                    result['engine_resources'].append(package)
                else:
                    result['warnings'].append('Unmatched mount: ' + package)
                continue
            base = project / local[mount] / Path(*parts[1:])
            payloads = [base.with_suffix(ext) for ext in ('.uasset', '.umap', '.uexp', '.ubulk', '.uptnl')
                        if base.with_suffix(ext).is_file()]
            if not any(p.suffix in ('.uasset', '.umap') for p in payloads):
                result['warnings'].append('Missing local package file: ' + package)
            deps = registry.get_dependencies(package, options)
            if deps is None:
                result['warnings'].append('AssetRegistry has no dependency result: ' + package)
                deps = []
            dependencies = sorted(str(d) for d in deps)
            result['packages'].append(dict(package=package, dependencies=dependencies,
                files=[p.relative_to(project).as_posix() for p in payloads]))
            for path in payloads:
                if str(path) not in file_paths:
                    file_paths.add(str(path))
                    result['files'].append(dict(relative_path=path.relative_to(project).as_posix(),
                                               local_path=str(path), bytes=path.stat().st_size))
            queue.extend(dependencies)
        result['packages'].sort(key=lambda row: row['package'])
        result['files'].sort(key=lambda row: row['relative_path'])
        result['engine_resources'].sort()
        result.update(status='PASS' if not result['warnings'] else 'INCOMPLETE', project=str(project),
                      file_count=len(result['files']), total_bytes=sum(row['bytes'] for row in result['files']))
    except Exception:
        result.update(status='FAIL', error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic() - started
    out.mkdir(parents=True, exist_ok=True)
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result

"""Rebuild isolated Atom camera library with native first-EOI completion.

Known target: M5Stack 3.3.8, ESP-IDF 735507283d, ESP32-S3 qio_opi.
Never modifies the SDK or operates a device. Run from the repository root.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import subprocess
import tarfile
import urllib.request

COMMIT = '202df95d7b1dc72e9303ad78f47b8dc9f339e6a1'
ARCHIVE_SHA256 = '2c0c8b7c522d2e694291e5e712247ccff23110858c369acc01b0ea3129318938'
PATCH_SHA256 = 'bfc435afdc6bed71e637694cfdaa947aa6e9ad0ef85dcc4cef374758291af282'
PARSER_SHA256 = '3a46e73d8f3160ada611379681c14209685f16ac14feda38d0d9ac82ab36f6d2'
SDK_LIBRARY_SHA256 = '9f25728226978bdf68fe92df6c6fe9cc6283efdc237a4c1e6927b3d13713992a'
SDK_CONFIG_SHA256 = 'f7f482eccb2ab90187e18f51d475779d89f5dd8912504362d4f0373c936bb10f'
PROBE_KEYS = ('sizeof_camera_config', 'sizeof_camera_fb', 'sizeof_sensor',
              'config_sccb_offset', 'fb_timestamp_offset', 'sensor_set_reg_offset', 'sensor_set_pll_offset')

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def patch_sha(path):
    # Git enforces LF for research files; accept the same patch from a Windows
    # checkout without making its identity depend on physical line endings.
    return hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()

def run(args):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    require(result.returncode == 0, result.stdout + result.stderr)
    return result.stdout

def response_file(path, args):
    path.write_text('\n'.join('"' + str(x).replace('\\', '/').replace('"', '\\"') + '"' for x in args), encoding='utf-8')
    return '@' + str(path)

def symbols(nm, library):
    return sorted({line.split()[-1] for line in run([nm, '-g', '--defined-only', library]).splitlines() if len(line.split()) == 3})

def apply_patch(original, patch):
    """Apply our pinned one-file unified patch only after every context matches."""
    lines = original.splitlines(True)
    hunks = patch.splitlines(True)
    result, cursor, i = [], 0, 2
    require(hunks[0].startswith('--- ') and hunks[1].startswith('+++ '), 'Invalid patch header')
    while i < len(hunks):
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n?', hunks[i])
        require(match is not None, 'Invalid hunk header')
        start = int(match[1]) - 1
        require(start >= cursor, 'Overlapping patch hunks')
        result.extend(lines[cursor:start]); cursor = start; i += 1
        old_count = new_count = 0
        while i < len(hunks) and not hunks[i].startswith('@@ '):
            kind, text = hunks[i][0], hunks[i][1:]
            require(kind in ' +-', 'Unsupported patch line')
            if kind != '+':
                require(cursor < len(lines) and lines[cursor] == text, f'Patch context mismatch at line {cursor + 1}')
                cursor += 1; old_count += 1
            if kind != '-': result.append(text); new_count += 1
            i += 1
        require(old_count == int(match[2] or 1) and new_count == int(match[4] or 1), 'Patch hunk length mismatch')
    result.extend(lines[cursor:])
    return ''.join(result)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--repo-root', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = (args.repo_root or (here.parents[3] if here.name == 'host' else Path.cwd())).resolve()
    artifacts = (repo / 'artifacts.local').resolve()
    output = args.output_dir.resolve()
    require(output != artifacts and output.is_relative_to(artifacts), 'Output must be below this checkout artifacts.local.')
    require((repo / 'AGENTS.md').is_file(), 'Invalid repository root; pass --repo-root.')
    config = json.loads((artifacts / 'hardware-bringup/local-config.json').read_text(encoding='utf-8-sig'))
    sdk = (Path(config['arduino_data']) / 'packages/m5stack/tools/esp32s3-libs/3.3.8').resolve()
    tools = sdk.parent.parent / 'esp-x32/2601/bin'
    tool = lambda name: tools / ('xtensa-esp32s3-elf-' + name + '.exe')
    versions = (sdk / 'versions.txt').read_text(encoding='utf-8')
    require('esp-idf: v5.5.4 735507283d' in versions and 'lib-builder: master 14a0af9' in versions, 'Unverified SDK version.')
    patch_path, parser_path = here / 'camera_early_eoi.patch', here / 'jpeg_first_eoi.h'
    original = sdk / 'lib/libespressif__esp32-camera.a'
    require(patch_sha(patch_path) == PATCH_SHA256 and sha(parser_path) == PARSER_SHA256, 'Camera patch/parser differ from verified candidate.')
    require(sha(original) == SDK_LIBRARY_SHA256 and sha(sdk / 'qio_opi/include/sdkconfig.h') == SDK_CONFIG_SHA256, 'Unverified installed camera library/configuration.')
    sdk_files = [original, sdk / 'versions.txt', sdk / 'qio_opi/include/sdkconfig.h']
    sdk_files += sorted((sdk / 'flags').glob('*')) + sorted((sdk / 'include/espressif__esp32-camera').rglob('*.h'))
    inputs = dict(schema=1, upstream_commit=COMMIT, archive_sha256=ARCHIVE_SHA256,
                  recipe_sha256=sha(Path(__file__)), patch_sha256=patch_sha(patch_path), parser_sha256=sha(parser_path),
                  sdk=str(sdk), sdk_sha256={str(p.relative_to(sdk)): sha(p) for p in sdk_files},
                  compiler_sha256=sha(tool('gcc')), compiler_cpp_sha256=sha(tool('g++')),
                  compiler_version=run([tool('gcc'), '--version']))
    library = output / 'libespressif__esp32-camera.a'
    manifest_path = output / 'rebuild-manifest.json'
    if library.exists():
        require(manifest_path.exists(), 'Library has no manifest; use a fresh output directory.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        require(manifest['inputs'] == inputs, 'Build inputs changed; use a fresh output directory.')
        require(sha(library) == manifest['library_sha256'], 'Cached library changed.')
        require(all(sha(output / p) == digest for p, digest in manifest['source_sha256'].items()), 'Cached source changed.')
        print(f'Reused verified library: {library}')
        return
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'esp32-camera.tar.gz'
    if not archive.exists():
        request = urllib.request.Request(f'https://codeload.github.com/espressif/esp32-camera/tar.gz/{COMMIT}', headers={'User-Agent': 'BlindAssist-camera-rebuild'})
        archive.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    require(sha(archive) == ARCHIVE_SHA256, 'Pinned source archive hash mismatch.')
    source = output / 'source'
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile(): continue
            relative = PurePosixPath(member.name).relative_to('esp32-camera-' + COMMIT)
            require('..' not in relative.parts, 'Invalid archive path.')
            path = source / str(relative); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    public_headers = ('driver/include/esp_camera.h', 'driver/include/sensor.h')
    require(all((source / f).read_bytes() == (sdk / 'include/espressif__esp32-camera' / f).read_bytes() for f in public_headers), 'Camera public headers differ from installed SDK.')
    cam_hal = source / 'driver/cam_hal.c'
    original_source = cam_hal.read_text(encoding='utf-8')
    (output / 'cam_hal.upstream.c').write_text(original_source, encoding='utf-8')
    cam_hal.write_text(apply_patch(original_source, patch_path.read_text(encoding='utf-8')), encoding='utf-8')
    (source / 'driver/jpeg_first_eoi.h').write_bytes(parser_path.read_bytes())
    members = run([tool('ar'), 't', original]).splitlines()
    sources = [p for p in source.rglob('*') if p.suffix in ('.c', '.cpp') and p.name + '.obj' in members and (p.name != 'll_cam.c' or '/esp32s3/' in p.as_posix())]
    require(sorted(p.name + '.obj' for p in sources) == sorted(members), 'Camera library source/member mismatch.')
    includes = [source / d for d in ('driver/include', 'conversions/include', 'conversions/private_include', 'driver/private_include', 'sensors/private_include', 'target/private_include')]
    sdk_includes = [sdk / 'include' / p for p in re.findall(r'-iwithprefixbefore\s+(\S+)', (sdk / 'flags/includes').read_text())]
    def flags(cpp=False, installed=False):
        return ['-I' + str(sdk / 'qio_opi/include'), *['-I' + str(p) for p in ([] if installed else includes) + sdk_includes], '@' + str(sdk / 'flags/defines'), '-Os', *shlex.split((sdk / 'flags' / ('cpp_flags' if cpp else 'c_flags')).read_text())]
    probe = output / 'abi_probe.c'
    probe.write_text('#include "esp_camera.h"\n#include <stddef.h>\nconst unsigned probe[]={sizeof(camera_config_t),sizeof(camera_fb_t),sizeof(sensor_t),offsetof(camera_config_t,sccb_i2c_port),offsetof(camera_fb_t,timestamp),offsetof(sensor_t,set_reg),offsetof(sensor_t,set_pll)};\n')
    probes = {}
    for label in ('installed', 'rebuilt'):
        obj, raw = output / (label + '.o'), output / (label + '.bin')
        run([tool('gcc'), response_file(output / (label + '.rsp'), flags(installed=label == 'installed')), '-c', probe, '-o', obj])
        run([tool('objcopy'), '-O', 'binary', '--only-section=.rodata.probe', obj, raw])
        probes[label] = dict(zip(PROBE_KEYS, struct.unpack('<7I', raw.read_bytes())))
    require(probes['installed'] == probes['rebuilt'], 'Camera public ABI differs.')
    objects = output / 'objects'; objects.mkdir(exist_ok=True)
    def compile_one(path):
        cpp = path.suffix == '.cpp'; obj = objects / (path.name + '.obj')
        command = [tool('g++' if cpp else 'gcc'), response_file(objects / (path.name + '.rsp'), flags(cpp)), '-c', path, '-o', obj]
        run(command); return obj, list(map(str, command))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        built = list(pool.map(compile_one, sources))
    pending = output / 'camera.pending.a'
    require(not pending.exists(), 'Incomplete previous archive exists; use a fresh output directory.')
    run([tool('ar'), response_file(output / 'archive.rsp', ['rcs', pending, *[x[0] for x in built]])])
    require(sorted(run([tool('ar'), 't', pending]).splitlines()) == sorted(members), 'Built member mismatch.')
    exported = symbols(tool('nm'), pending)
    require(exported == symbols(tool('nm'), original), 'Camera public symbol set changed.')
    pending.rename(library)
    manifest = dict(inputs=inputs, library=str(library), library_sha256=sha(library), members=len(members),
                    defined_symbols=len(exported), abi_probes=probes, public_headers_equal=list(public_headers),
                    source_sha256={str(p.relative_to(output)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'commands.json').write_text(json.dumps([x[1] for x in built], indent=2), encoding='utf-8')
    print(f'Built verified library: {library}\nSHA256: {manifest["library_sha256"]}\nNo SDK changes or device operations performed.')

if __name__ == '__main__': main()

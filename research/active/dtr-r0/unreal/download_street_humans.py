"""Fetch a bounded, pinned Microsoft Rocketbox subset for the local UE street lab."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'microsoft/Microsoft-Rocketbox'
REVISION = '0943055db6ec570bcef9f2c8b41c9e5467c808f9'
HUMANS = ['Female_Adult_01', 'Female_Adult_02', 'Male_Adult_01', 'Male_Adult_02']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1',
                                     headers={'User-Agent': 'BlindAssist-local-asset-import'})
    tree = json.load(urllib.request.urlopen(request))
    commit = tree['sha']
    selected = []
    for entry in tree['tree']:
        path = entry['path']
        if entry['type'] != 'blob':
            continue
        wanted = path in ['LICENSE.md', 'README.md']
        for human in HUMANS:
            base = f'Assets/Avatars/Adults/{human}/'
            wanted |= path == base + f'Export/{human}.fbx'
            wanted |= path.startswith(base + 'Textures/') and path.endswith(('_color.tga', '_normal.tga'))
        wanted |= path in [f'Assets/Animations/all_animations_max_motextr_xy/{sex}_walk_neutral.max.fbx'
                           for sex in ['f', 'm']]
        if wanted:
            selected.append(entry)

    def fetch(entry):
        path = entry['path']
        target = args.output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}'
        if not target.exists() or target.stat().st_size != entry['size']:
            urllib.request.urlretrieve(url, target)
        content = target.read_bytes()
        if len(content) != entry['size']:
            raise RuntimeError(f'Truncated download: {path}')
        return {'path': path, 'url': url, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, selected))
    receipt = {'source': f'https://github.com/{REPO}', 'commit': commit,
               'license': 'MIT (see downloaded LICENSE.md)', 'humans': HUMANS, 'files': files}
    (args.output / 'sources.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(files), 'bytes': sum(x['size'] for x in files), 'commit': commit}))


if __name__ == '__main__':
    main()

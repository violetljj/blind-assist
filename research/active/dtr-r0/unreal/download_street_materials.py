"""Download a small, attributed CC0 PBR palette from the official Poly Haven API."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[4] / 'artifacts.local/unreal/asset-downloads/materials'
PALETTE = ['cobblestone_pavement', 'brick_wall_001', 'plastered_wall_02', 'wood_planks']
PROPS = ['painted_wooden_bench', 'planter_box_02']


def read_json(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


def download(asset):
    info = read_json(f'https://api.polyhaven.com/files/{asset}')
    target = ROOT / asset
    target.mkdir(parents=True, exist_ok=True)
    result = {'asset':asset, 'source':f'https://polyhaven.com/a/{asset}',
              'license':'CC0', 'license_source':'https://polyhaven.com/license', 'maps':{}}
    for role, candidates in {'color':('diff','Diffuse'), 'normal':('nor_dx',), 'roughness':('rough','Rough')}.items():
        key = next(k for k in candidates if k in info)
        formats = info[key]['2k']
        item = formats.get('jpg') or formats.get('png')
        path = target / item['url'].split('/')[-1]
        if not path.exists():
            response = requests.get(item['url'], timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        result['maps'][role] = {'path':str(path), 'url':item['url'],
                                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(f'Downloaded {asset}', flush=True)
    return result


def download_prop(asset):
    info = read_json(f'https://api.polyhaven.com/files/{asset}')
    target = ROOT.parent / asset
    target.mkdir(parents=True, exist_ok=True)
    items = {f'{asset}_2k.fbx':info['fbx']['2k']['fbx']}
    for relative, item in info['gltf']['2k']['gltf']['include'].items():
        if relative.endswith('.jpg'):
            items[relative] = item
    for relative, item in items.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            response = requests.get(item['url'], timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        if hashlib.md5(path.read_bytes()).hexdigest() != item['md5']:
            raise ValueError(f'Provider checksum mismatch: {path}')
    (target/'polyhaven-files.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (target/'SOURCE_LICENSE.txt').write_text(f'https://polyhaven.com/a/{asset}\nCC0\nhttps://polyhaven.com/license\n',encoding='utf-8')
    print(f'Downloaded {asset}',flush=True)


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, PALETTE))
    (ROOT/'sources.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download_prop, PROPS))

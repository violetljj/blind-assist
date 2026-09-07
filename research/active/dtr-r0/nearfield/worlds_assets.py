"""Download and verify the bounded G14 Poly Haven surface set (CPU network/I/O)."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

ASSETS = {'concrete_pavement_02': 180, 'leafy_grass': 200, 'shrub_01': None, 'terrazzo_tiles': 200}
ROLES = {'color': 'Diffuse', 'normal': 'nor_dx', 'rough': 'Rough'}


def download(destination):
    import requests
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for asset, tile_cm in ASSETS.items():
        api_url = 'https://api.polyhaven.com/files/' + asset
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        metadata = response.json()
        (destination / (asset + '-provider.json')).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        roles = dict(ROLES)
        if asset == 'shrub_01':
            roles.update(normal='nor_gl', alpha='Alpha', mesh='fbx')
        for role, provider_role in roles.items():
            extension = 'fbx' if role == 'mesh' else 'png' if role == 'alpha' else 'jpg'
            entry = metadata[provider_role]['2k'][extension]
            url = entry['url']
            if urlparse(url).hostname != 'dl.polyhaven.org':
                raise ValueError('Unexpected download host: ' + url)
            relative = asset + '/' + Path(urlparse(url).path).name
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = target.read_bytes() if target.exists() else b''
            if hashlib.md5(data).hexdigest() != entry['md5']:
                response = requests.get(url, timeout=180)
                response.raise_for_status()
                data = response.content
                if hashlib.md5(data).hexdigest() != entry['md5'] or len(data) != entry['size']:
                    raise ValueError('Provider integrity mismatch: ' + url)
                target.write_bytes(data)
            if len(data) != entry['size']:
                raise ValueError('Provider size mismatch: ' + url)
            records.append(dict(asset=asset, role=role, path=relative, api_url=api_url,
                                url=url, provider_md5=entry['md5'], bytes=len(data),
                                sha256=hashlib.sha256(data).hexdigest(), tile_cm=tile_cm))
    manifest = dict(schema='g14-realism-assets-v1', status='PASS', provider='Poly Haven',
                    license='CC0', license_url='https://polyhaven.com/license',
                    resolution='2k', files=records, displacement=False,
                    scale_assumptions={'terrazzo_tiles': '200 cm authored repeat; physical provider width not verified'},
                    vegetation_claim='leafy_grass is a flat textured ground layer, not 3D vegetation')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[4] / 'artifacts.local/unreal/g14-realism-assets')
    args = parser.parse_args()
    result = download(args.destination)
    print(json.dumps({'status': result['status'], 'files': len(result['files']),
                      'bytes': sum(f['bytes'] for f in result['files'])}))

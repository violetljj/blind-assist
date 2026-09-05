"""Download the attributed CC0 environment used by StreetLabV3."""
import hashlib
import json
from pathlib import Path
import requests


def main():
    asset = 'overcast_industrial_courtyard'
    root = Path(__file__).resolve().parents[4] / 'artifacts.local/unreal/asset-downloads/environment'
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'BlindAssist-UE-Asset-Downloader/1.0'
    response = session.get(f'https://api.polyhaven.com/files/{asset}', timeout=45)
    response.raise_for_status()
    metadata = response.json()
    item = metadata['hdri']['2k']['hdr']
    path = root / item['url'].rsplit('/', 1)[1]
    if not path.exists():
        response = session.get(item['url'], timeout=120)
        response.raise_for_status()
        payload = response.content
        if hashlib.md5(payload).hexdigest() != item['md5']:
            raise ValueError('Environment download differs from provider checksum')
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.md5(payload).hexdigest() != item['md5']:
        raise ValueError('Existing environment differs from provider checksum')
    receipt = {'asset': asset, 'path': str(path), 'source': f'https://polyhaven.com/a/{asset}',
               'license': 'CC0', 'license_source': 'https://polyhaven.com/license',
               'url': item['url'], 'provider_md5': item['md5'],
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    (root / 'source.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    from download_street_materials import download_prop
    download_prop('outdoor_table_chair_set_01')


if __name__ == '__main__':
    main()

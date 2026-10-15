"""Bounded read-only source probe; never executes downloaded author scripts."""
from pathlib import Path
import argparse, hashlib, json, time, urllib.request, urllib.error, urllib.parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    url = ('https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/'
           'yfc5578_psu_edu/IQB7trduxN_rT4sHdsU-EyoFAVKw1kG8nTksqoLPlq7eMfw'
           '?e=Ajwakl&download=1')
    started = time.monotonic()
    receipt = {'url': url, 'requested_bytes': '0-65535', 'max_body_bytes': 65536}
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-65535',
                                               'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read(65536)
            receipt.update(status=response.status,
                           final_host=urllib.parse.urlsplit(response.url).hostname,
                           headers={k:v for k,v in response.headers.items()
                                    if k.lower() in {'content-length', 'content-type', 'date'}},
                           bytes_read=len(data),
                           sha256=hashlib.sha256(data).hexdigest())
            (args.out / 'sharepoint_probe.bin').write_bytes(data)
            receipt['prefix_utf8'] = data[:1500].decode('utf-8', errors='replace')
    except Exception as error:
        receipt['error'] = repr(error)
        receipt['bytes_read'] = 0
    receipt['command_work_wall_s'] = time.monotonic() - started
    (args.out / 'sharepoint_receipt.json').write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

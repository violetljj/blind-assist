"""Fetch four public checkpoint files without credentials; cap total at 50 MB."""
import hashlib
import json
import requests
from pathlib import Path

root=Path('artifacts.local/work/assistive-thesis-screen-20261015/thin_structure')
out=root/'cityscapes_b0'
out.mkdir(parents=True,exist_ok=True)
old=json.loads((root/'download_receipt.json').read_text(encoding='utf8')) if (root/'download_receipt.json').exists() else {'files':[],'bytes':0}
receipt=old['files']
total=old['bytes']

def get(url,name):
    global total
    if (out/name).exists() and any(r['file']==name and r['sha256']==hashlib.sha256((out/name).read_bytes()).hexdigest() for r in receipt):
        return (out/name).read_bytes()
    # requests supplies its CA bundle; the first urllib attempt had a TLS EOF.
    with requests.get(url,headers={'User-Agent':'BlindAssist-Explore/1.0'},timeout=30,stream=True) as response:
        response.raise_for_status()
        chunks=[]
        for chunk in response.iter_content(1024*1024):
            total+=len(chunk)
            if total>50000000: raise RuntimeError('50 MB checkpoint download cap reached')
            chunks.append(chunk)
        data=b''.join(chunks)
    if total>50000000: raise RuntimeError('50 MB checkpoint download cap reached')
    (out/name).write_bytes(data)
    receipt.append({'url':url,'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (root/'download_receipt.json').write_text(json.dumps({'bytes':total,'files':receipt},indent=2),encoding='utf8')
    return data

model='nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
meta=json.loads(get(f'https://huggingface.co/api/models/{model}','model_metadata.json'))
revision=meta['sha']
names={v['rfilename'] for v in meta['siblings']}
weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
for name in ['config.json','preprocessor_config.json','README.md',weight]:
    get(f'https://huggingface.co/{model}/resolve/{revision}/{name}',name)
print(json.dumps({'revision':revision,'weight':weight,'bytes':total,'folder':str(out)}))

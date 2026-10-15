"""Fetch small official source assets; never fetch test labels or full image archives."""
from pathlib import Path
import hashlib, json, time, urllib.request, zipfile

ROOT = Path(__file__).resolve().parents[4] / 'artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa'
ROOT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'products.json': 'https://raw.githubusercontent.com/Accessibility-Research-Collective-UCI/image-quality-vlm-chi26/main/data/study-2-annotated-dataset.json',
    'vqa_annotations.zip': 'https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip',
    'quality_annotations.zip': 'https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip',
}
def main():
    start=time.monotonic(); receipts=[]
    for name,url in SOURCES.items():
        path=ROOT/name
        if not path.exists():
            with urllib.request.urlopen(url,timeout=35) as response:
                data=response.read(50_000_001)
                if len(data)>50_000_000: raise RuntimeError('single source byte budget exceeded')
                path.write_bytes(data)
        receipts.append(dict(name=name,url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                print(name, z.namelist())
                for member in z.namelist():
                    # Explicit allowlist of public training and validation only.
                    if Path(member).name.lower() in {'train.json','val.json'}:
                        out=ROOT/name.removesuffix('.zip')/Path(member).name
                        out.parent.mkdir(exist_ok=True)
                        out.write_bytes(z.read(member))
    record={'sources':receipts,'download_wall_seconds':time.monotonic()-start,'bytes':sum(r['bytes'] for r in receipts)}
    (ROOT/'download_receipts.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
    products=json.loads((ROOT/'products.json').read_text(encoding='utf-8'))
    print(type(products).__name__,len(products))
    print(json.dumps(products[0] if isinstance(products,list) else list(products.items())[:1],indent=2))
if __name__=='__main__': main()

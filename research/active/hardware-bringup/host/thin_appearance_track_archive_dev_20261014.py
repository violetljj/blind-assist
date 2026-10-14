"""Lossless task-owned partial evidence archive; deletion uses native PowerShell."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('path');a=ap.parse_args();p=Path(a.path).resolve()
    assert p.parent.name in ('partial_slow','h8_three_arm_partial')
    rawhash=hashlib.sha256(p.read_bytes()).hexdigest();out=p.with_suffix(p.suffix+'.gz')
    with p.open('rb') as src,gzip.open(out,'wb',compresslevel=6) as dst:shutil.copyfileobj(src,dst)
    with gzip.open(out,'rb') as src:uncompressedhash=hashlib.file_digest(src,'sha256').hexdigest()
    assert rawhash==uncompressedhash
    receipt=p.parent/'archive_manifest.json'
    records=json.loads(receipt.read_text()) if receipt.exists() else []
    records.append(dict(raw_file=p.name,raw_bytes=p.stat().st_size,raw_sha256=rawhash,
                        gzip_file=out.name,gzip_bytes=out.stat().st_size,gzip_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                        decompressed_sha256=uncompressedhash,status='LOSSLESS_VERIFIED'))
    receipt.write_text(json.dumps(records,indent=2));print(json.dumps(records[-1]))

if __name__=='__main__':main()

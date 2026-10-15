"""Download all official Seen test ZIP members without decoding GT or RGB."""
import argparse, hashlib, json, struct, zlib
from pathlib import Path
from full_plsp_prepare import drive_form, fetch_chunks, ChunkReader, dump, AGD_ID

def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    entries=sorted(json.loads(a.index.read_text()),key=lambda e:e['offset'])
    selected=[e for e in entries if e['name'].startswith('AGD20K/Seen/testset/') and e['size']]
    start=selected[0]['offset'];end=next(e['offset'] for e in entries if e['offset']>selected[-1]['offset'])
    assert end-start<=2*1024**3
    assert all(e['name'].startswith('AGD20K/Seen/testset/') for e in entries if start<=e['offset']<end)
    dump(a.out/'test_archive_manifest.json',{'entries':selected,'start':start,'end':end,'bytes':end-start,'selection':'complete official Seen test only','GT_decoded':False})
    url,params=drive_form(AGD_ID)
    jobs,new=fetch_chunks(url,params,start,end,6702534286,a.out/'chunks',a.workers)
    reader=ChunkReader(a.out/'chunks',jobs);records=[]
    for e in selected:
        fields=struct.unpack('<4s5H3L2H',reader.read(e['offset'],30));assert fields[0]==b'PK\x03\x04'
        nl,xl=fields[-2:];assert reader.read(e['offset']+30,nl).decode()==e['name']
        packed=reader.read(e['offset']+30+nl+xl,e['compressed_size'])
        b=zlib.decompress(packed,-15) if fields[3]==8 else packed
        assert len(b)==e['size'] and f'{zlib.crc32(b):08x}'==e['crc32']
        p=a.data/e['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        records.append({'name':e['name'],'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
    (a.out/'test_content_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')
    rgb=[r for r in records if '/egocentric/' in r['name']];gt={r['name'] for r in records if '/GT/' in r['name']}
    paired=sum(r['name'].replace('/egocentric/','/GT/').replace('.jpg','.png') in gt for r in rgb)
    dump(a.out/'terminal.json',{'status':'COMPLETE','rgb_files':len(rgb),'GT_files':len(gt),'paired_rows':paired,'missing_GT_rows':len(rgb)-paired,'unique_rgb_sha256':len({r['sha256'] for r in rgb}),'download_bytes':new,'GT_decoded':False,'RGB_decoded':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);main(p.parse_args())

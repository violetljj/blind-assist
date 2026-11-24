"""Lock a sequence-disjoint Oxford timestamp cohort and fetch only its verified cam0 frames."""
import binascii
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
WORK = ROOT / "artifacts.local/work"
SRC = ROOT / "artifacts.local/downloads/oxford-spires-20261117"
OX = WORK / "paired-low-sites-20261118/oxford"
OUT = WORK / "egogeom-omni-v0/math_theory" / HERE.parent.name / HERE.name
OLD = WORK / "shallow-alerts-20261118/controls/oxford_valid_camera.json"
REV = "03f4382308333aa70c3253f12acd3fbf0c7c4a15"
SITES = ("blenheim-palace", "christ-church", "keble-college", "observatory-quarter")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def persist(path, data, frozen=False):
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    if frozen and path.exists():
        if json.loads(path.read_text(encoding="utf8")) != data:
            raise ValueError(f"Existing frozen identity differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(payload, encoding="utf8")
    tmp.replace(path)


def lock():
    excluded = sorted({r["seq"] for r in json.loads(OLD.read_text())["rows"]})
    rows, inputs = [], {}
    for idx in sorted((OX / "zips").glob("*.json")):
        seq = idx.stem
        if seq in excluded:
            continue
        site, = [s for s in SITES if s in seq]
        trajectory = SRC / "seq" / seq / "gt-tum.txt"
        if not (SRC / "map" / (site+".pcd")).is_file() or not trajectory.is_file():
            raise FileNotFoundError(seq)
        with trajectory.open() as f:
            first = last = next(f)
            for last in f:
                pass
        pack = json.loads(idx.read_text())
        records = pack["entries"]
        stamps = [float(Path(e[0]).stem) for e in records]
        lo = max(float(first.split()[0]), min(stamps))
        hi = min(float(last.split()[0]), max(stamps))
        assert hi > lo
        for q in (.125,.375,.625,.875):
            t = lo+(hi-lo)*q
            i = min(range(len(stamps)), key=lambda i: abs(stamps[i]-t))
            assert lo <= stamps[i] <= hi
            rows.append(dict(seq=seq, site=site, fraction=q, t=stamps[i], goal_t=t,
                             entry=records[i], zip_size=pack["size"]))
        inputs[seq] = dict(index=sha(idx), trajectory=sha(trajectory))
    assert len(rows) == len({(r["seq"],r["entry"][0]) for r in rows}) == 36
    plan = dict(rows=rows,excluded=excluded,revision=REV,inputs=inputs,
                original_panel=sha(OLD),protocol=sha(HERE/"RECORD.md"))
    persist(OUT/"cohort.json", plan, frozen=True)
    return rows


def fetch(row):
    seq = row["seq"]
    name,start,end,crc,compressed,size,method = row["entry"]
    dest = OUT / "original" / seq / name
    def valid(buf):
        return len(buf)==size and binascii.crc32(buf)==crc
    if dest.exists():
        if not valid(dest.read_bytes()):
            raise ValueError(f"Cached frame has a different CRC: {name}")
    else:
        local = OX/"original"/seq/name
        if local.exists():
            data=local.read_bytes()
        else:
            url = f"https://huggingface.co/datasets/ori-drs/oxford_spires_dataset/resolve/{REV}/sequences/{seq}/raw/images.zip"
            for attempt in range(4):
                try:
                    with requests.get(url,params={"m14_range":f"{start}-{end}-{attempt}"},
                                      headers={"Range":f"bytes={start}-{end-1}","Accept-Encoding":"identity"},
                                      stream=True, timeout=(12,30)) as response:
                        expected=f"bytes {start}-{end-1}/{row['zip_size']}"
                        if response.status_code!=206 or response.headers.get("Content-Range")!=expected:
                            raise IOError(f"Unexpected range for {name}: HTTP {response.status_code}")
                        buf=response.raw.read(end-start+1)
                    if len(buf)!=end-start or buf[:4]!=b"PK\x03\x04":
                        raise IOError("Incomplete ZIP local record")
                    n,e=struct.unpack_from("<HH",buf,26)
                    if buf[30:30+n].decode()!="cam0/"+name or method not in (0,8):
                        raise ValueError("ZIP member mismatch")
                    raw=buf[30+n+e:30+n+e+compressed]
                    data=zlib.decompress(raw,-15) if method==8 else raw
                    if not valid(data):
                        raise IOError("ZIP data does not match the signed-off index")
                    break
                except (OSError, requests.RequestException) as exc:
                    if attempt==3:
                        raise
                    time.sleep(.5*(attempt+1))
        if not valid(data):
            raise ValueError(name)
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix(".partial")
        tmp.write_bytes(data)
        tmp.replace(dest)
    return {"seq":seq,"name":name,"sha256":sha(dest),"bytes":size}


if __name__=="__main__":
    rows=lock()
    with ThreadPoolExecutor(max_workers=5) as pool:
        files=list(pool.map(fetch,rows))
    persist(OUT/"input_receipt.json", dict(files=files,cohort=sha(OUT/"cohort.json")),frozen=True)
    print(len(files),"verified cam0 images from",len({r["seq"] for r in rows}),"sequences")

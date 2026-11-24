"""Fetch only IMU and past GrayLeftFront pairs for the inherited three local cohorts."""
import csv
import hashlib
import json
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / "artifacts.local" / "work"
DATA = WORK / "lamar-independent-heading-dev-20261116-codex"
OUT = WORK / "egogeom-omni-v0" / "math_theory" / HERE.name
sys.path.insert(0, str(DATA/"hololens"))
from scout_hololens import RemoteZip, fetch
from fetch_hololens_native_cohort import read_archive_record, verify_crc


def pairs(session, rows):
    root = DATA/"hololens/CAB"
    with np.load(root/"prepared"/(session+".npz"), allow_pickle=False) as f:
        targets = f["depth_timestamp_us"][rows]
    raw = root/"raw"/session
    with (raw/"images.txt").open() as f:
        stamps = np.array([int(row[0]) for row in csv.reader(f, skipinitialspace=True)
                           if len(row)>1 and row[1]=="hetlf"])
    if not np.all(np.diff(stamps)>0):
        raise ValueError("unsorted images")
    selected = {}
    for row, t in zip(rows, targets):
        ij = stamps.searchsorted([t-100000, t], side="right")-1
        if ij.min() < 0:
            continue
        a,b = stamps[ij]
        if t-b<=50000 and 65000<=b-a<=135000:
            selected[str(row)] = [int(a),int(b)]
    return selected


def fetch_source(native):
    source = native.stem
    folder = OUT/"lamar_inputs"/source
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(native) as src:
        requested = pairs(source, src["depth_rows"])
    stamps = sorted({t for p in requested.values() for t in p})
    members = {f"{source}/raw_data/images/hetlf/{t}.jpg":str(t)+".jpg" for t in stamps}
    members[source+"/imu.txt"]="imu.txt"
    remote = f"raw/CAB/sessions/{source}.zip"
    with RemoteZip(remote) as r, zipfile.ZipFile(r) as z:
        expected = json.loads((DATA/"hololens/CAB/raw"/source/"source.json").read_text())
        assert expected["size"]==r.size and expected["etag"]==r.etag
        index = sorted(z.infolist(),key=lambda i:i.header_offset)
        last = {a.filename:b.header_offset for a,b in zip(index,index[1:])}
        last[index[-1].filename]=z.start_dir
        manifest = {}
        groups = []
        for info in index:
            if info.filename not in members:
                continue
            filename = members[info.filename]
            path = folder/filename
            if path.exists():
                data = path.read_bytes()
                verify_crc(data,info)
                manifest[filename] = hashlib.sha256(data).hexdigest()
                continue
            if groups and info.header_offset-groups[-1][-1][1]<=80000 and last[info.filename]-groups[-1][0][0].header_offset<=2000000:
                groups[-1].append((info,last[info.filename]))
            else:
                groups.append([(info,last[info.filename])])
        local = threading.local()
        def recover(group):
            if not hasattr(local,"http"):
                local.http = requests.Session()
            start = group[0][0].header_offset
            end = group[-1][1]
            block = fetch(local.http,r.url,start,end-1,r.size)
            out = {}
            for info,stop in group:
                payload = read_archive_record(block[info.header_offset-start:stop-start],info)
                verify_crc(payload,info)
                path = folder/members[info.filename]
                path.write_bytes(payload)
                out[path.name] = hashlib.sha256(payload).hexdigest()
            return out
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(recover,groups):
                manifest.update(result)
        if set(manifest)!=set(members.values()):
            raise ValueError("Missing source members")
        record=dict(source=source, url=r.url, etag=r.etag, total_bytes=r.size,
                    pairs=requested, sha256=manifest, archive_member_crc_verified=True)
    (folder/"manifest.json").write_text(json.dumps(record,indent=1)+"\n")
    print(source,len(requested),"causal pairs",len(stamps),"gray images",flush=True)


if __name__=="__main__":
    files = sorted((DATA/"hololens/CAB/native_depth").glob("*.npz"))
    assert len(files)==3
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch_source,files))

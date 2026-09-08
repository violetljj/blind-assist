"""Read-only completed scout receipt audit; never grants scene admission."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(capture):
    receipt = capture/'receipt.json'; release = capture/'process-release.json'
    if not receipt.is_file() or not release.is_file():
        return dict(status='PENDING_OR_INCOMPLETE', capture=str(capture))
    r, released = read(receipt), read(release)
    if r['status'] != 'PASS' or not released.get('released') or released.get('survivors'):
        return dict(status='CAPTURE_NOT_RELEASED_OR_FAILED', capture=str(capture))
    specfile = capture/'source/spec.json'; spec = read(specfile)
    if sha(specfile) != r['spec_sha256'] or not r['source_unchanged']:
        raise ValueError('Source spec integrity failure: '+str(capture))
    payload = read(capture/'payload-hashes.json')
    for path, digest in payload.items():
        target = (capture/path).resolve()
        if not target.is_relative_to(capture.resolve()) or sha(target) != digest:
            raise ValueError('Payload integrity failure: '+path)
    cases = {c['name']: c for c in spec['cases']}
    floor = []
    for probe in r.get('native_floor_probes', []):
        c = cases[probe['case']]
        floor.append(dict(case=probe['case'], hit=probe['hit'], point_m=probe.get('point_m'),
            camera_z_m=c['camera']['z'], eye_above_hit_m=c['camera']['z']-probe['point_m'][2] if probe['hit'] else None,
            scope='Vertical first hit; not a walkability certificate or automatic camera correction'))
    hp = capture/'evaluator/hlod-membership.json'
    hlod = read(hp) if hp.is_file() else {}
    proxies = hlod.get('hlod', [])
    return dict(status='SCOUT_VERIFIED_SOURCE_NOT_ADMITTED', capture=str(capture),
        frame_count=r['frame_count'], payload_files_verified=len(payload),
        receipt_sha256=sha(receipt), spec_sha256=sha(specfile), payload_manifest_sha256=sha(capture/'payload-hashes.json'),
        floor_probes=floor, hlod=dict(status=hlod.get('status','MISSING'), schema=hlod.get('schema'),
            proxies=len(proxies), unresolved=hlod.get('unresolved_hlod_count'),
            errors=dict(Counter(p.get('error','') for p in proxies if p.get('status') != 'VERIFIED'))),
        pending=['Final camera/fixture native quartets', 'Actual route and obstruction review',
            'Complete visible-background identity and cross-split checks'], model_observations=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scouting', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    if not a.output.resolve().is_relative_to(root): raise ValueError('Canonical artifact output required')
    rows={f'big_candidate_{i:02d}':inspect(a.scouting/f'big_candidate_{i:02d}-v4') for i in range(1,8)}
    result=dict(status='SOURCE_ADMISSION_PENDING', runner_sha256=sha(Path(__file__)),regions=rows,
        scope='Read-only snapshot of completed source work; no process, capture, training or admission mutation')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:dict(status=v['status'],frames=v.get('frame_count'),hlod=v.get('hlod')) for k,v in rows.items()}))

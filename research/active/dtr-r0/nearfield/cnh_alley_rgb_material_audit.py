"""Check actual six-layout RGB material use against the declared test-only probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(captures: list[Path], probe_path: Path) -> dict:
    probe_path = probe_path.resolve()
    probe = json.loads(probe_path.read_text())
    if probe.get('status') != 'PASS_ZERO_CROSS_SPLIT_EDITOR_USED_TEXTURES':
        raise ValueError('Declared test material probe did not complete')
    probe_assets = {row['source_mesh']: row for row in probe['assets']}
    used = {'train': set(), 'dev': set(), 'test': set()}
    entries = []
    for capture in captures:
        capture = capture.resolve()
        receipt_path = capture / 'format-receipt.json'
        receipt = json.loads(receipt_path.read_text())
        spec = json.loads((capture / 'source/spec.json').read_text(encoding='utf-8-sig'))
        split = spec['alley_manifest']['proposed_split']
        if (split not in ('train', 'dev') or
                receipt.get('status') != 'RGB_NUMERIC_PASS_VISUAL_REVIEW_REQUIRED' or
                receipt.get('frame_count') != 160 or
                receipt.get('rgb_insert_material_policy') != 'ALLEY_DERIVED_MFPD_OFF_V1' or
                len(receipt.get('derived_insert_materials', {})) != 2):
            raise ValueError(f'Capture does not have complete MFPD-off RGB receipt: {capture}')
        assets = []
        for asset in spec['assets']:
            material = receipt['derived_insert_materials'][str(asset['id'])]
            source = material['source_mesh']
            material_receipt = Path(material['path']).resolve()
            if (source != asset['mesh_asset'] or
                    material_receipt != capture / f'derived-{asset["id"]}.json' or
                    sha(material_receipt) != material['sha256']):
                raise ValueError('Actual derived material receipt identity differs')
            candidate = probe_assets.get(source)
            actual = set(material['derived_editor_used_textures'])
            if (candidate is None or candidate['split'] != split or
                    material['source_material'] != candidate['source_material'] or
                    actual != set(candidate['derived_after_used_textures'])):
                raise ValueError('Actual derived material used textures differ from probe')
            used[split].update(actual)
            assets.append(dict(id=asset['id'],source_mesh=source,
                source_material=material['source_material'],
                derived_material=material['derived_material'],
                used_textures=sorted(actual),
                removed_used_textures=material['removed_editor_used_textures'],
                material_receipt_sha256=material['sha256']))
        entries.append(dict(split=split,layout_id=spec['layouts'][0]['layout_id'],
            capture=str(capture),capture_receipt_sha256=sha(receipt_path),assets=assets))
    if len(entries) != 6 or {split:sum(row['split']==split for row in entries)
                            for split in ('train','dev')} != {'train':3,'dev':3} or len({e['layout_id'] for e in entries})!=6:
        raise ValueError('Exactly three train and three dev layouts required')
    for row in probe['assets']:
        if row['split']=='test':
            used['test'].update(row['derived_after_used_textures'])
    intersections={f'{a}_vs_{b}':sorted(used[a]&used[b])
                   for a,b in (('train','dev'),('train','test'),('dev','test'))}
    return dict(schema='cnh-alley-six-rgb-actual-used-texture-audit-v1',
        status=('PASS_ZERO_USED_TEXTURE_INTERSECTIONS' if all(not x for x in intersections.values())
                else 'FAIL_SHARED_USED_TEXTURES'),
        benchmark_eligible=False,test_authority='UNSAVED_DERIVED_MATERIAL_PROBE_ONLY_NO_TEST_CAPTURE',
        test_probe=dict(path=str(probe_path),sha256=sha(probe_path)),
        capture_count=len(entries),captures=entries,
        used_textures_by_split={key:sorted(value) for key,value in used.items()},
        intersections=intersections)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,action='append',required=True)
    parser.add_argument('--test-probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.capture,args.test_probe)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],capture_count=result['capture_count'],
                          intersections={key:len(value) for key,value in result['intersections'].items()})))


if __name__=='__main__':
    main()

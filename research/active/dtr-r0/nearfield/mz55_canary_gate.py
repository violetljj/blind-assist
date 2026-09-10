"""Admit only the frozen64 MZ55 canary with native and explicit visual evidence."""
import argparse
from pathlib import Path
from collections import Counter
from mz48_prepare import read, sha, write
from data_lightweight import CompactSource


def gate(manifest, packages, review, output):
    output.mkdir(parents=True,exist_ok=False)
    m=read(manifest);assert m['total_frames']==2560 and m['canary_frames']==64
    inputs={str(p):sha(p) for p in (manifest,packages,review)}
    specs={};expected=[]
    for s in m['shards']:
        if s['canary']:
            path=Path(s['path']);assert sha(path)==s['sha256'];inputs[str(path)]=s['sha256']
            specs[s['sha256']]=s
            expected.extend(c['name'] for c in read(path)['cases'])
    requested=read(packages)
    assert len(requested['archives'])==2
    records=[];pairs=[];shards=[]
    for ref in requested['archives']:
        path=Path(ref['path']);assert sha(path)==ref['sha256'];inputs[str(path)]=ref['sha256']
        with CompactSource(path) as source:
            def member(name):
                data=source.read_bytes(name)
                import hashlib,json
                assert hashlib.sha256(data).hexdigest()==source.entries[name]['sha256']
                return json.loads(data)
            meta=member('evaluator/metadata.json');assert meta['schema']=='mz55-training-source-v1'
            binding=member('evaluator/source-bindings.json')
            assert binding['source_spec_sha256'] in specs
            shards.append(specs[binding['source_spec_sha256']]['shard_id'])
            records.extend(meta['records']);pairs.extend(meta['pairs'])
            release=member('evaluator/capture-process-release.json')
            assert release['released'] and not release['survivors']
    assert len(set(shards))==2 and len(records)==len(set(r['frame_id'] for r in records))==64
    assert {r['frame_id'] for r in records}==set(expected) and len(pairs)==32
    visual=read(review);assert visual['manifest_sha256']==sha(manifest)
    vrows=visual['rows'];assert len(vrows)==64 and {r['frame_id'] for r in vrows}==set(expected)
    # Reviewer binds every actually inspected original RGB + native array. No
    # generic PASS string can stand in for the required64-frame review.
    by_id={r['frame_id']:r for r in records}
    for r in vrows:
        src=by_id[r['frame_id']]
        assert r['rgb_sha256']==src['rgb_sha256'] and r['native_sha256']==src['native_sha256']
        assert isinstance(r['visual_note'],str) and r['visual_note'].strip()
    counts=Counter((r['family'],r['relation']) for r in records if r['intent_matches'])
    represented={f:{relation:counts[f,relation] for relation in m['relations']} for f in m['families'][:-1]}
    checks=dict(source_health=all(r['source_valid'] and r['floor']['accepted'] for r in records),
        native_pairs=all(p['target_dictionary_equal'] and p['actual_target_receipt_equal'] and p['native_pair_invariant']
                         and not any(p['changed_event_mask_pixels']) and p['event_depth_max_abs_m']==0 for p in pairs),
        intent_min=sum(r['intent_matches'] for r in records)>=56,
        every_new_family_relation=all(n>0 for d in represented.values() for n in d.values()),
        all64_visual=visual['status']=='PASS' and all(r['accepted'] and r['rgb_native_agreement'] and
            r['openings_backface_material_checked'] for r in vrows))
    result=dict(status='PASS' if all(checks.values()) else 'STOP_NO_EXPANSION',checks=checks,
        manifest_sha256=sha(manifest),frames=64,source_valid_frames=sum(r['source_valid'] for r in records),
        intent_matching_frames=sum(r['intent_matches'] for r in records),family_relation_matches=represented,
        all_actual_labels_retained=True,new_captures=0,allowed_next_frames=2496 if all(checks.values()) else 0,
        inputs=inputs,code_sha256=sha(Path(__file__)))
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS_CHECK_EXECUTED',admission=result['status'],inputs=inputs,
        code_sha256=sha(Path(__file__)),outputs={'result.json':sha(output/'result.json')}))
    print(result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','packages','review','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();gate(a.manifest,a.packages,a.review,a.output)

"""Separate consumed-source diagnostic for visible HEAD evidence in an intact assembly.

Preserves failed full-extent admission. Does not relabel occluded/UNKNOWN rays.
No model access or recapture. Reuses hash-bound native labels unchanged.
"""
import argparse
import copy
import math
from pathlib import Path
import numpy as np
from body_query_collection_labels import read, write, sha
from contact_retina_spec import BODY_BOXES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    original=a.run/'admission-v1';receipt=read(original/'admission.json')
    assert receipt['status']=='NOT_EVALUABLE'
    for file,key in [('frame_metadata.json','frame_metadata_sha256'),('evaluator_truth.npz','truth_sha256')]:
        assert sha(original/file)==receipt[key]
    # Every original native admission binding remains authoritative.
    for table in receipt['bindings'].values():
        for path,digest in table.items():assert sha(Path(path))==digest
    rows=copy.deepcopy(read(original/'frame_metadata.json'));cache={}
    low,high=BODY_BOXES[1];focal=320/math.tan(math.radians(50))
    for row in rows:
        region=row['region_id'];i=row['capture_index'];root=a.run/region/'capture'
        if region not in cache:cache[region]=read(root/'evaluator/target-raycheck.json')['rows']
        matches=[x for x in cache[region] if x['sample_index']==i and x['target_id']=='adjustable_cross_member'];assert len(matches)==1
        ray=matches[0]
        native_path=root/f'evaluator/native/{i:04d}.npy';isolated_path=root/f'evaluator/isolated/adjustable_cross_member/{i:04d}.npy'
        assert sha(native_path)==row['native_sha256'] and sha(isolated_path)==row['isolated_sha256']
        native=np.load(native_path,allow_pickle=False);isolated=np.load(isolated_path,allow_pickle=False)
        witnesses=[]
        for sample in ray['rows']:
            if sample['status']!='MATCH' or sample['component_path']!=ray['source_component']:continue
            u,v=sample['pixel'];x=float(native[v,u]);ix=float(isolated[v,u])
            if not math.isfinite(x) or not math.isfinite(ix) or abs(x-ix)>.03:continue
            y=(u-319.5)*x/focal;z=1.7-(v-179.5)*x/focal
            if not (high[0]<=x<=high[0]+3 and low[1]<=y<=high[1] and low[2]<=z<=high[2]):continue
            endpoint='near' if x<high[0]+1.5 else 'far'
            if endpoint==row['endpoint']:witnesses.append(dict(pixel=[u,v],xyz_floor_relative_m=[x,y,z],native_component=sample['component_path']))
        row['original_full_extent_reasons']=row['reasons']
        row['original_full_extent_pair_reasons']=row['pair_reasons']
        row['original_full_extent_admitted']=row['admitted']
        row['reasons']=[x for x in row['reasons'] if x not in ('TARGET_NATIVE_IDENTITY_UNKNOWN','TARGET_VISIBLE_EXTENT_OCCLUDED')]
        if len({tuple(x['pixel']) for x in witnesses})<3:row['reasons'].append('LESS_THAN_THREE_IDENTITY_MATCHED_VISIBLE_HEAD_RANGE_PIXELS')
        row['visible_head_witnesses']=witnesses
    pairs={}
    for row in rows:pairs.setdefault(row['pair_id'],[]).append(row)
    assert len(pairs)==60 and len(rows)==120
    for members in pairs.values():
        reasons=sorted({reason for row in members for reason in row['reasons']})
        for row in members:row.update(pair_reasons=reasons,admitted=not reasons)
    coverage={region:sum(r['admitted'] and r['endpoint']=='near' and r['region_id']==region for r in rows) for region in cache}
    status='PASS' if all(v>=24 for v in coverage.values()) else 'NOT_EVALUABLE'
    write(a.output/'frame_metadata.json',rows)
    (a.output/'evaluator_truth.npz').write_bytes((original/'evaluator_truth.npz').read_bytes())
    write(a.output/'admission.json',dict(status=status,planned_pairs=60,region_coverage=coverage,
          admitted_pairs=dict(background=sum(coverage.values())),original_primary_status='NOT_EVALUABLE',
          mode='SEPARATE_CONSUMED_SOURCE_VISIBLE_HEAD_DIAGNOSTIC',
          original_admission_sha256=sha(original/'admission.json'),source_sha256=sha(__file__),
          frame_metadata_sha256=sha(a.output/'frame_metadata.json'),truth_sha256=sha(a.output/'evaluator_truth.npz'),
          training_steps=0,inference_frames=0,source_counts_unchanged=True,
          policy='Off-corridor or unmatched rays remain UNKNOWN/occluded; no full-extent or size claim'))
    print(status,coverage)


if __name__=='__main__':main()

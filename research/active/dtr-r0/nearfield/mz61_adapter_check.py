"""CPU-only MZ61 adapter check; no native files, rendering or model execution."""
import argparse
import ast
import copy
import difflib
import json
from pathlib import Path
import zipfile

import numpy as np
import mz55_dataset as old
import mz61_dataset as new
from mz61_source_contract import admit_spec, geometry_metadata, read, sha, MZ55_DATASET_SHA, source_binding_fields


def run(root, output):
    assert not output.exists()
    task = root/'artifacts.local/work/mz61-geometry-source-20260911'
    previous = root/'artifacts.local/work/mz55-diverse-mesh-source-20260911'
    manifest = task/'spec-v2/manifest.json'; specs = task/'spec-v2/shards'
    files = [Path(__file__), Path(new.__file__), Path(old.__file__),
             Path(__file__).with_name('mz61_source_contract.py'), Path(__file__).with_name('mz61_fullframe.py'),
             manifest, previous/'source-index.json', previous/'spec-v1/manifest.json']
    inputs = {str(p):sha(p) for p in files}
    assert sha(old.__file__) == MZ55_DATASET_SHA
    assert sha(previous/'source-index.json') == '84848bc92ea077b9bbc651f6d080eedb9cdc97205c5ddc9f1033a1961032e176'
    assert sha(previous/'spec-v1/manifest.json') == '97a13ecb7d4cc9cf58c6ae7062bb70d699b913788332a14e29360ba2f059cd43'
    trees = [ast.parse(Path(mod.__file__).read_text()) for mod in (old, new)]
    kernel_checks = {}
    for name in ('rotation', 'actual_bounds', 'event_masks'):
        functions = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
        assert ast.dump(functions[0]) == ast.dump(functions[1])
        kernel_checks[name] = 'AST_IDENTICAL_TO_MZ55'
    # Read only small metadata members from immutable bound old archives.
    old_index, prior = read(previous/'source-index.json'), read(previous/'spec-v1/manifest.json')
    old_frames, old_targets, max_old_error = 0, 0, 0.
    for shard in old_index['shards'][:2]:
        archive = previous/shard['archive']['path']
        assert sha(archive) == shard['archive']['sha256']; inputs[str(archive)] = sha(archive)
        declared = next(s for s in prior['shards'] if s['shard_id'] == shard['shard_id'])
        path = previous/'spec-v1/shards'/(shard['shard_id']+'.json')
        assert sha(path) == declared['sha256']; inputs[str(path)] = sha(path)
        with zipfile.ZipFile(archive) as z:
            metadata = json.loads(z.read('evaluator/metadata.json'))
        cases = read(path)['cases']
        assert len(metadata['records']) == len(cases) == 32
        for row, case in zip(metadata['records'], cases):
            assert row['frame_id'] == case['name']; old_frames += 1
            for actor in row['target_actors']:
                lo, hi = new.actual_bounds(actor, case['camera'], case['floor_z_m'])
                original_lo, original_hi = old.actual_bounds(actor, case['camera'], case['floor_z_m'])
                np.testing.assert_array_equal(lo, original_lo); np.testing.assert_array_equal(hi, original_hi)
                expected = next(b for b in case['placement_bounds'] if b['name'] == actor['name'])
                for actual, key in ((lo, 'camera_aligned_min_m'), (hi, 'camera_aligned_max_m')):
                    np.testing.assert_allclose(actual, expected[key], rtol=0, atol=1e-5)
                    max_old_error = max(max_old_error, float(np.max(np.abs(actual-expected[key]))))
                old_targets += 1
    new_frames, new_targets, new_max = 0, 0, 0.
    shard_rows, rejected = [], []
    for shard in read(manifest)['shards']:
        path = specs/(shard['shard_id']+'.json'); spec = read(path)
        contract = admit_spec(manifest, specs, spec); inputs.update(contract['inputs'])
        bindings = source_binding_fields(contract, path)
        assert bindings['original_spec_sha256'] == bindings['captured_spec_sha256'] == shard['sha256']
        shard_rows.append(dict(shard=shard['shard_id'], frames=len(spec['cases']),
                               preview_frames=sum(c['canary'] or c['native_audit_sample'] for c in spec['cases'])))
        if not shard['canary']: continue
        remapped = copy.deepcopy(spec); remapped['map_file'] = 'HOST_REMAP_ONLY'
        assert admit_spec(manifest, specs, remapped)['shard_id'] == shard['shard_id']
        for key, value in [('training_role', 'TRAIN_CANDIDATE_INVALID'), ('floor_z_m', -999.)]:
            changed = copy.deepcopy(spec); changed['cases'][0][key] = value
            try: admit_spec(manifest, specs, changed)
            except AssertionError: rejected.append(shard['shard_id']+':'+key)
            else: raise AssertionError('Mutated capture spec was accepted')
        for case in spec['cases']:
            assert set(geometry_metadata(case)) == {'geometry_id','geometry_recipe_id','geometry_joint_factors','split_unit','site_replica'}
            new_frames += 1
            for obj in case['objects']:
                if not obj['target_part']: continue
                if 'mesh_asset' in obj:
                    asset = prior['assets'][case['condition']['family']]
                    bounds = dict(min=(np.array(asset['bounds_min_m'])*100).tolist(), max=(np.array(asset['bounds_max_m'])*100).tolist())
                else:
                    bounds = dict(min=[-50.]*3, max=[50.]*3)
                # Predicted receipt fields, explicitly not an observed UE actor.
                actor = dict(mesh_local_bounds_cm=bounds, scale=obj.get('scale',obj.get('size_m')),
                             rotation_deg=obj['rotation_deg'], actor_origin_m=obj['center_m'])
                lo, hi = new.actual_bounds(actor, case['camera'], case['floor_z_m'])
                expected = next(b for b in case['placement_bounds'] if b['name'] == obj['name'])
                for actual, key in ((lo,'camera_aligned_min_m'),(hi,'camera_aligned_max_m')):
                    np.testing.assert_allclose(actual, expected[key], rtol=0, atol=1e-5)
                    new_max = max(new_max,float(np.max(np.abs(actual-expected[key]))))
                new_targets += 1
    assert old_frames == new_frames == 64 and len(rejected) == 4
    # Empty-preview shards are valid; no empty image is saved by the adapter.
    output.mkdir(parents=True, exist_ok=False)
    diff = ''.join(difflib.unified_diff(Path(old.__file__).read_text().splitlines(True),Path(new.__file__).read_text().splitlines(True),fromfile='mz55_dataset.py',tofile='mz61_dataset.py'))
    (output/'dataset-diff.patch').write_text(diff,encoding='utf-8')
    result = dict(status='PASS_METADATA_COMPATIBILITY_ONLY', kernels=kernel_checks,
                  frozen_manifest_frames=4096, old_observed_receipt_frames=old_frames, old_observed_targets=old_targets,
                  old_bounds_max_abs_error_m=max_old_error, new_canary_spec_frames=new_frames,
                  new_configured_targets=new_targets, configured_bounds_max_abs_error_m=new_max,
                  mutated_spec_rejections=rejected, map_file_only_remap_accepted=True, shards=shard_rows,
                  fullframe_core_sha256='5fb19e543e1b6c841a2c61c8a4e4ee4c918985a3883d1c061a2ce673423aeaa3',
                  fullframe='MZ52 exact core reused; MZ61 wrapper retains all attempted rows and known flags',
                  angular_auxiliary='Existing mz48_auxiliary.py has no frame-count or role restriction; reuse unchanged after source verification',
                  limitations=['No MZ61 native rendering or labels exist in this check',
                               'Configured bounds use frozen asset AABBs; real capture actor/bounds checks remain mandatory',
                               'No packet recomputation: copied packet/truth loop retained, integration awaits actual capture'],
                  backend='CPU metadata only', native_files_read=0, rgb_decodes=0, captures=0, model_inference_frames=0,training_steps=0)
    old.write(output/'result.json',result)
    old.write(output/'receipt.json',dict(status='PASS', inputs=inputs, outputs={p.name:sha(p) for p in output.iterdir()},scope=result['status']))
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.output.resolve())

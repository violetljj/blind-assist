"""Inspect downloaded source only; toy indexing fixture is not a model result."""
from pathlib import Path
import argparse, hashlib, json, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    tree = json.loads((a.root / 'author-tree.json').read_text(encoding='utf-8-sig'))
    files = [x for x in tree['tree'] if x['type'] == 'blob']
    method = (a.root / 'method.py').read_text(encoding='utf-8-sig')
    generation = (a.root / 'gennerate_qa.py').read_text(encoding='utf-8-sig')
    camera = (a.root / 'CameraPathPlayer.cs').read_text(encoding='utf-8-sig')
    source_match = ('chosen_rel = final_idx[order_rel]' in method
                    and 'retrieved_indices = subset_indices[chosen_rel]' in method
                    and 'subset_paths[i]) for i in order_rel' in method)
    # Exactly mirrors the two index paths, with fixed illustrative candidates.
    subset_indices = list(range(10))
    subset_paths = [f'frame_{i}.jpg' for i in subset_indices]
    final_idx = [7, 4, 9, 6]
    order_rel = sorted(range(len(final_idx)), key=lambda j: subset_indices[final_idx[j]])[:3]
    chosen_rel = [final_idx[i] for i in order_rel]
    logged_indices = [subset_indices[i] for i in chosen_rel]
    actually_selected_paths = [subset_paths[i] for i in order_rel]
    expected_paths = [subset_paths[i] for i in chosen_rel]
    payloads = []
    for name in ['author-README.md','author-tree.json','gennerate_qa.py',
                 'CameraPathPlayer.cs','evaluation.py','method.py','Task.jpg',
                 'sharepoint_probe.bin']:
        data = (a.root / name).read_bytes()
        payloads.append({'file': name, 'bytes': len(data),
                         'sha256': hashlib.sha256(data).hexdigest()})
    result = {
        'evidence': 'SOURCE_CONTRACT_ONLY; synthetic index fixture; MODEL_NOT_RUN',
        'repository': 'https://github.com/sding11/ObjChangeVR',
        'repository_tree_sha': tree['sha'],
        'github_blob_count': len(files),
        'github_tree_entries': len(tree['tree']),
        'github_csv_files': [x['path'] for x in files if x['path'].endswith('.csv')],
        'runtime_qa_rows': 0, 'runtime_paired_raw_frames': 0,
        'author_figure_examples_inspected': 1,
        'native_object_visibility_field_in_inspected_camera_schema': False,
        'is_disappear_definition_is_global_phase': 'int isDisappear = isAfterPhase ? 1 : 0;' in camera,
        'generator_requests_4_3_3_class_balance': '4 disappearing objects, 3 always present objects and 3' in generation,
        'qa_alignment_uses_Index': 'gt_df[gt_df["Index"] == index_value]' in generation,
        'indexing_fixture': {
            'source_pattern_confirmed': source_match,
            'candidate_indices': final_idx, 'order_rel': order_rel,
            'logged_indices': logged_indices, 'actual_sent_paths': actually_selected_paths,
            'expected_paths': expected_paths,
            'mismatched_paths': sum(x != y for x,y in zip(actually_selected_paths,expected_paths)),
            'denominator': 3,
        },
        'downloaded_payload_bytes': sum(x['bytes'] for x in payloads),
        'payloads': payloads,
    }
    (a.root / 'source_contract.json').write_text(json.dumps(result, indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='payloads'}, indent=2))


if __name__ == '__main__':
    main()

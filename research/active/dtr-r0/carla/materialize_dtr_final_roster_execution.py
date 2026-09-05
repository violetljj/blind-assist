"""Versioned R1 source preparation; never changes the frozen roster or probes.

Output is an inspectable candidate, not permission to capture or open outcomes.
Source-only audit must close all semantic gaps before a capture seal is issued.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import materialize_dtr_final_visual_shell_probe as shell
import dtr_final_s10_temporal_design as temporal


def materialize(group_id):
    roster = shell.base.read_json(shell.base.ROSTER_PROTOCOL)
    group = next(g for g in roster['source_design']['seed_groups'] if g['group_id'] == group_id)
    value = shell.materialize()
    value['cohort_id'] = 'DTR_FINAL_RECKONING_R1_' + group_id
    value['capture']['seed'] = group['capture_seed']
    value['objective'] = 'R1 source candidate pending complete source audit and execution seal'
    design = temporal.design()
    library = value['trajectory_library']
    library['fr_r1_alias_cross'] = shell.base.trajectory(1.7, 1.5, [(0.,0.,-1.),(3.,0.,0.)])
    value['scenarios'][5]['asset_trajectories']['c8_l01_alias'] = 'fr_r1_alias_cross'
    for key in ('wearer', 'target', 'shell'):
        library['fr_r1_s10_' + key] = copy.deepcopy(design[key])
    for cell in (value['scenarios'][9], value['scenarios'][11]):
        cell['wearer_trajectory'] = 'fr_r1_s10_wearer'
        cell['asset_trajectories']['c8_l01_target'] = 'fr_r1_s10_target'
        cell['issued_plan'] = shell.base.plan_for('fr_r1_s10_wearer', design['wearer'], 10)
    value['scenarios'][9]['asset_trajectories']['c8_l01_c16_shell_02'] = 'fr_r1_s10_shell'
    value['occlusion_contracts'][1]['planned_occlusion_window_s'] = design['full_window_s']
    # Reference captures are evaluator-only auxiliaries, never extra examples.
    pairs = copy.deepcopy(value.pop('final_visual_shell_probe')['pairs'])
    pairs[1]['window_s'] = design['full_window_s']
    value.pop('final_reckoning_source_probe')
    value['source_disjoint_contract'] = {
        'group_id': group_id, 'seed': group['capture_seed'],
        'prior_probe_pixels_reusable': False,
        'capture_authorized': False,
        'reason': 'COMPLETE_SOURCE_AUDIT_AND_EXECUTION_SEAL_REQUIRED',
    }
    strata = {}
    for cell in value['scenarios'][:10]:
        episode = cell['episode_id']
        score_end = 3.5 if episode in ('ep_01', 'ep_09') else 6.0
        value['evaluation_contract']['score_window_end_seconds'][episode] = score_end
        target = cell['layout_id'] + '_target'
        item = {'episode_id': episode, 'target_asset': target, 'score_start_s': .1, 'score_end_s': score_end,
                'minimum_trackable_pixels': 185,
                'expected_responsible_assets': cell['expected_responsible_assets'],
                'issued_plan_receipt': shell.base.c2.build_plan_receipt(cell['issued_plan'])}
        if episode == 'ep_02': item['removal_windows'] = [[12]]
        if episode == 'ep_03': item['removal_windows'] = [[12,13], [17,18,19], list(range(23,29))]
        if episode == 'ep_06': item['secondary_asset'] = cell['layout_id'] + '_alias'
        strata[cell['final_reckoning_stratum_id']] = item
        cell['twin_role'] = 'r1_main_source'
    annex = {
        'schema': 'dtr-final-roster-execution-candidate-v1',
        'status': 'PREPIXEL_CANDIDATE_NOT_CAPTURE_SEALED',
        'group': group, 'roster_sha256': shell.base.c2.sha256_file(shell.base.ROSTER_PROTOCOL),
        'main_episode_ids': [c['episode_id'] for c in value['scenarios'][:10]],
        'auxiliary_reference_ids': ['ep_11', 'ep_12'], 'reference_pairs': pairs,
        'strata': strata, 's10_analytic': temporal.analytic_receipt(),
        'measured_collision_credential_gate': 'DETECTOR_LEDGER_GATE_REQUIRED_BEFORE_INTERVENTION',
        'algorithm_or_fit_or_final_outcomes_opened': False,
        'capture_authorized': False,
        'startup_policy': 'Uniformly exclude sample 0 from source visibility and every method metric; retain raw payload.',
        'source_failure_policy': 'NOT_EVALUABLE_NO_SCENE_REPAIR_OR_RETRY_AFTER_DURABLE_FRAMES',
    }
    value['claim_boundary'] = [
        'Candidate source geometry only; missing gates cannot be waived.',
        'Ten main episodes; two reference rasters are evaluator-only and excluded from every method.',
        'Frozen arm bytes, seeds, and event definitions remain unchanged.',
    ]
    shell.base.c2.validate_protocol(value)
    return value, annex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=['FIT_ONLY','FINAL_A','FINAL_B'])
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    protocol, annex = materialize(args.group)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for name, value in [('protocol.json', protocol), ('annex.json', annex)]:
        with (args.output_root/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
    print(json.dumps({'status': annex['status'], 'path': str(args.output_root.resolve())}))


if __name__ == '__main__':
    main()

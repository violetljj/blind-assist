"""Independent displayed-table audit of the final robustness Development report."""
import argparse
import json
import re
import time
from pathlib import Path
from audit_sync_fusion_robust_dev import load,sha

LABELS={'native_perturbed':'半循环','faro_rho015_ambient1':'rho .15','faro_rho060_ambient1':'rho .6',
        'faro_rho030_ambient3':'ambient ×3','faro_rho030_ambient1':'过门 FARO','faro_rho030_ambient10':'ambient ×10'}
BANDS=('0.3-0.8m','0.8-1.5m','1.5-3m')

def ci(value):
    return 'NA' if value['lo'] is None else f"[{value['lo']:.2f}, {value['hi']:.2f}]"

def ratio(value):
    return 'N/E' if value is None else f'{100*value:.1f}%'

def ratio_ci(value):
    return 'NA' if value['lo'] is None else f"[{100*value['lo']:.1f}%, {100*value['hi']:.1f}%]"

def run(root,report_path):
    started=time.monotonic();s=load(root/'summary.json');a=load(root/'independent_audit.json');addon=load(root/'independent_addon_audit.json')
    assert a['status']==addon['status']=='PASS'
    assert a['features_sha256']==s['features_sha256'] and a['oof_sha256']==s['oof_sha256']
    assert addon['original_summary_sha256']==sha(root/'summary.json')
    text=report_path.read_text('utf8');lines=text.splitlines()
    M={(r['fold'],r['arm'],r['band'],r['model']):r for r in s['metrics']}
    P={(r['fold'],r['arm'],r['band'],r['model'],r['baseline']):r for r in s['paired']}
    original_count={r['model']:r for r in a['original_count_descriptor_candidates']}
    for r in s['candidates']:
        passed=False if r['model']=='A' else original_count[r['model']]['original_count_candidate']
        descriptor='是（比例N/E）' if passed and r['retention'] is None else '是' if passed else '否' if r['model']!='A' else 'A对照'
        expected=f"| {r['model']} | {r['FARO_near_net']:+d} | {r['stress_near_net']:+d} | {r['native_middle_net']:+d} | {ratio(r['retention'])} | {descriptor} |"
        assert expected in lines,('candidate',expected)
    assert '空约束' in text and '保留率N/E' in text and 'False' in text
    focus_rows=0
    for arm,b in (('faro_rho030_ambient1',0),('faro_rho030_ambient10',0),('native_perturbed',1)):
        for model in 'ABCD':
            r,x=M[-1,arm,b,model],P[-1,arm,b,model,'rgb']
            expected=f"| {LABELS[arm]} / {BANDS[b]} | {model} | {r['W']}/{r['POS']} | {r['F']}/{r['FREE']} | {x['rescue']}/{x['loss']} | {ci(x['net_pp_ci'])} |"
            assert expected in lines,('focus',expected);focus_rows+=1
    full_rows,single_rows=0,0
    for f in (-1,0,1,2,3,4):
        marker='### 合计' if f==-1 else '### 折 '+str(f);start=lines.index(marker)
        stops=[i for i in range(start+1,len(lines)) if lines[i].startswith('### ') or lines[i].startswith('## ')]
        section=lines[start:min(stops) if stops else len(lines)]
        actual=[x for x in section if x.startswith('| ') and len(x.split('|'))==10 and x.split('|')[3].strip() in ('A','B','C','D')]
        assert len(actual)==72,(f,len(actual))
        for arm in LABELS:
            for b in range(3):
                for model in 'ABCD':
                    r=M[f,arm,b,model];pp=[P[f,arm,b,model,base] for base in ('rgb','tof')]
                    pairs=[f"{x['rescue']}/{x['loss']}={x['net']:+d} {ci(x['net_pp_ci'])}" for x in pp]
                    free='; '.join(f"{x['FREE_added']}/{x['FREE_removed']}" for x in pp)
                    expected=f"| {LABELS[arm]} | {BANDS[b]} | {model} | {r['W']}/{r['POS']} {ci(r['W_rate_ci'])} | {r['F']}/{r['FREE']} {ci(r['F_rate_ci'])} | {pairs[0]} | {pairs[1]} | {free} |"
                    assert expected in section,('full',f,arm,b,model);full_rows+=1
                cells=[]
                for model in ('rgb','tof'):
                    r=M[f,arm,b,model];cells.append(f"{r['W']}/{r['POS']}, {r['F']}/{r['FREE']}")
                expected=f"| {LABELS[arm]} | {BANDS[b]} | {' | '.join(cells)} |"
                assert expected in section;single_rows+=1
    assert full_rows==432 and single_rows==108
    for r in s['retention']:
        fold='合计' if r['fold']==-1 else str(r['fold'])
        expected=f"| {fold} | {r['model']} | {r['best_single']} | {r['A_net']} | {r['model_net']} | {ratio(r['retention'])} | {ratio_ci(r['retention_ci'])} | {r['retention_ci']['valid']} |"
        assert expected in lines,('retention',expected)
    evidence=load(root/'evidence_summary.json')
    for r in evidence['prior_metrics']:
        fold='合计' if r['fold']==-1 else str(r['fold'])
        expected=f"| {fold} | {LABELS[r['arm']]} | {BANDS[r['band']]} | {r['W']}/{r['POS']} | {r['F']}/{r['FREE']} |"
        assert expected in lines,('prior',expected)
    E={(r['fold'],r['arm'],r['band'],r['model'],r['variant'],r['category']):r for r in evidence['evidence_split']}
    categories=('both','tof_only','rgb_only','neither')
    for arm in LABELS:
        for band in range(3):
            for model in 'ABCD':
                rr=[E[-1,arm,band,model,'raw_mean_margin',category] for category in categories]
                expected=f"| {LABELS[arm]} | {BANDS[band]} | {model} | {'/'.join(str(r['W']) for r in rr)} | {'/'.join(str(r['F']) for r in rr)} |"
                assert expected in lines,('evidence',expected)
                assert sum(r['W'] for r in rr)==M[-1,arm,band,model]['W']
                assert sum(r['F'] for r in rr)==M[-1,arm,band,model]['F']
    for arm,band in (('faro_rho030_ambient1',0),('faro_rho030_ambient10',0),('native_perturbed',1)):
        for category in categories:
            r=E[-1,arm,band,'D','raw_mean_margin',category]
            expected=f"| {LABELS[arm]} / {BANDS[band]} | {category} | {r['W']} | {r['F']} | {r['rescue_rgb']}/{r['loss_rgb']}={r['rescue_rgb']-r['loss_rgb']:+d} | {r['rescue_tof']}/{r['loss_tof']}={r['rescue_tof']-r['loss_tof']:+d} |"
            assert expected in lines,('evidence focus',expected)
    assert f"共有{evidence['bothK_missing_but_mean_supported']}条query" in text
    assert '统一最低证据合同' in text and 'K0/K1余量均值并传播missing' in text
    assert '更严格敏感性' in text
    checked_links=[]
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        if target.startswith(('https://','http://')):
            continue
        local=report_path.parent/target.split('#')[0]
        assert local.exists(),('broken report link',target)
        checked_links.append(target)
    receipt=load(report_path.with_name('SYNC_FUSION_ROBUST_RECEIPTS_DEV_20261011.json'))
    assert receipt['report_sha256']==sha(report_path)
    assert receipt['PLAN_sha256']==sha(root/'PLAN.json')
    assert receipt['audit']==a and receipt['addon']['audit']==addon
    assert load(report_path.with_name('SYNC_FUSION_ROBUST_INDEPENDENT_AUDIT_DEV_20261011.json'))==a
    assert load(report_path.with_name('SYNC_FUSION_ROBUST_EVIDENCE_INDEPENDENT_AUDIT_DEV_20261011.json'))==addon
    for name,digest in receipt['outputs'].items():
        assert sha(root/name)==digest,('receipt hash',name)
    assert receipt['resource']['CPU_command_wall_conservative_s']<=1800 and receipt['resource']['GPU_s']<=300
    result=dict(status='PASS',report_sha256=sha(report_path),summary_sha256=sha(root/'summary.json'),
                candidate_rows=4,focus_rows=focus_rows,full_fusion_rows=full_rows,baseline_rows=single_rows,retention_rows=24,
                prior_rows=108,raw_mean_margin_split_rows=72,evidence_focus_rows=12,checked_report_links=checked_links,
                receipt_output_hashes=len(receipt['outputs']),audit_copies_exact=True,
                CPU_command_wall_s=time.monotonic()-started,audit_source_sha256=sha(__file__),
                check='Every displayed original-model metric/paired/CI cell and original-count clarification agrees with independently audited immutable summary')
    (root/'independent_report_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();run(args.root.resolve(),args.report.resolve())

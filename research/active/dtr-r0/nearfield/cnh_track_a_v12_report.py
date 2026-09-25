"""Concise v1.2 pilot receipt; detailed gate values remain in machine JSON."""
import argparse
import json
from pathlib import Path


def run(folder, output):
    folder=Path(folder)
    result=json.loads((folder/'result.json').read_text())
    audit_path=folder/'readonly-audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
    saved_units=[json.loads(p.read_text()) for p in sorted(folder.glob('unit[0-9][0-9].json'))]
    units=[u for u in saved_units if 'checks' in u]
    repo=Path(__file__).resolve().parents[4]
    wall=json.loads((repo/'artifacts.local/evidence/cnh-track-a-v12-wall-20260926-v1/result.json').read_text())
    rows=['# Track A v1.2 试采结果 — 2026-09-26','',
          f"状态：**{result['status']}**。协议冻结于`f5911d2f`，新种子族、32配置计划、原G0–G5门槛和5cm边界带；未修改旧v1.1失败。",'',
          '|单位|划分|几何帧|主帧 N|边界帧|合格组合 / 12|多对象帧 / 门槛|G2失败项|',
          '|---:|---|---:|---:|---:|---:|---|---|']
    for u in units:
        c=u['checks'];m=c.get('multi',{})
        rows.append(f"|{u['unit']}|{u['split']}|{c['total_frames']}|{c['N_main']}|{c['boundary_frames']}|{c.get('eligible_combinations','—')}|{m.get('actual','—')} / {m.get('required','—')}|{', '.join(c['failures']) or 'PASS'}|")
    rows.extend(['',f"完成单位文件{len(units)}/12；完整几何帧{sum(u['checks']['total_frames'] for u in units)}。部分配置与停止原因以原始回执为准；不把部分单位算作完整单位。",'',
                 '计数门槛逐单位使用ceil(旧值×N/160)，配置下限与组合12/20不缩放。以下保留逐位计数/相关性和按划分的失败差距；全部余量分布、配置来源与拒绝日志见JSON。',''])
    for u in saved_units:
        if 'checks' not in u:
            rows.extend([f"单位{u['unit']}仅保存{len(u['configs'])}/32配置、{12*len(u['configs'])}帧；尚无完整单位G2判定。",''])
    for u in units:
        c=u['checks']
        hb=[round(v,4) for v in c.get('HB_disagreement',[])]
        phi={k:round(v,4) if v is not None else None for k,v in c.get('phi',{}).items()}
        rows.extend([f"单位{u['unit']}：六查询正数 / N={c.get('positive_counts')} / {c['N_main']}；H/B不一致={hb}；phi={phi}。",''])
    rows.extend(['## 门槛和停止回执',''])
    for gate in ('G0','G1','G2'):
        value=(audit or result).get(gate,'NOT_RUN')
        state=('PASS' if value.get('pass_gate') else 'FAIL / PARTIAL') if isinstance(value,dict) else value
        rows.append(f'- {gate}: {state}。')
    for key in ('unit','config','attempts','rejections','rejection_rate','wall_s','G3_sim','G4','G5','B0_B1'):
        if key in result:rows.append(f'- {key}: {result[key]}。')
    split_checks=result.get('G2',{}).get('split_checks',{}) if isinstance(result.get('G2'),dict) else {}
    for split,c in split_checks.items():
        rows.append(f"- {split}: N={c.get('N_main')}，合格组合{c.get('eligible_combinations')}/20；失败项：{', '.join(c.get('failures',[])) or '无'}。")
    if audit:
        rows.extend(['',f"只读验收{audit['wall_s']:.3f}秒：G0未通过整批完成要求，错误仅为{audit['G0']['errors']}，未发现已有记录损坏。G1在{audit['G1']['nonempty_configs']}个非空配置上重复为0，近重复单位对为0。G2三个完整单位通过，所有split均NOT_EVALUABLE_PARTIAL。",'',
                     '验收对全部帧检查margin→标签/边界/贡献者，对每配置首/锚点/末帧重算精确几何；所有非空配置重算精确体素。回执completed_units字段列的是出现文件的0–3，实际完成仅0–2。'])
    rows.extend(['',
        '停止点为单位3/配置9：H011、B100的两个窄物体在固定轨迹下须同时贡献≥6主帧。速度1.00053m/s、主帧偏航24.69°–28.41°；平移部分横向漂移约8.36–9.52cm/步。32候选均为HiGHS infeasible，不是2秒超时。当前模板与保守横向/距离约束没有可行解；未证明横漂是唯一原因，也不证明任意真实场景不可构造。',
        '共138候选、33拒绝（23.91%），生成206.125秒。保存105配置/1260几何帧，包含3完整单位及1部分单位；传感帧0。G3–G5、4audit读出、可观测性b/c均NOT_RUN。未换轨迹、未进入第三次计划。'])
    rows.extend(['',
        '## 读出与前置复现','',
        '[旧数据完整扫描表](CNH_SCAN_DEVELOPMENT_RESULTS_20260926.md)：H3/zero S2 AP0.663231、TP183/FP8；raw→H3/zero正确方差S2 AP0.660945、TP190/FP5，分母408正/2472负。旧R0逐位对账通过。Street1920为描述性复现，不用于新协议选参。','',
        f"静态墙相对L1：未补偿{wall['uncompensated']['relative_L1']:.6f} → r²补偿{wall['r2']['relative_L1']:.6f}；总计数{wall['uncompensated']['accumulated_total_counts']:.3f} → {wall['r2']['accumulated_total_counts']:.3f}，当前帧{wall['r2']['current_total_counts']:.3f}。仅历史逐点施加增益，方差按增益平方传播；总能量更近但空间L1没有改善。",'',
        '[G1根因](CNH_TRACK_A_V12_ROOTCAUSE_20260926.md)：旧三对seed不同，连续顶点相差3–8mm；固定尾部边界构造在1cm体素碰撞。新版本采用按用途派生seed和接受前全局精确去重，不将换seed等同于去重。','',
        '## 范围与局限','',
        '本轮是有停止规则的受控模拟工程试采，不是实机、泛化或信息上限证据。旧Development扫描结果重复使用且标签退化。未执行项保持NOT_RUN；未放量、不采City/test、不启动UE/RGB或硬件。若生成/验收失败，只否定本次构造，不能推出门槛与32配置数学不相容。下一次计划须用户决定。','',
        f"产物：`{folder.as_posix()}`；协议：[v1.2](CNH_NEARFIELD_ACCEPTANCE_PLAN_V1_2_20260925.md)。"])
    Path(output).write_text('\n'.join(rows)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)

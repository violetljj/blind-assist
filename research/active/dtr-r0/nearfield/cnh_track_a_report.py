"""Report frozen pilot outputs without fitting, selecting or altering cohorts."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def run(folder, report):
    result=json.loads((folder/'result.json').read_text())
    audit=json.loads((folder/'readonly-audit.json').read_text())
    if result.get('completed_geometry_units')!=12:raise ValueError('Partial pilot requires explicit partial report')
    units=[json.loads(p.read_text()) for p in sorted(folder.glob('unit[0-9][0-9].json'))]
    n=sum(u['checks']['N_main'] for u in units);boundary=sum(u['checks']['boundary_frames'] for u in units)
    failures=collections.Counter(f for u in units for f in u['checks']['failures'])
    lines=['# Track A v1.1 工程试采结果 — 2026-09-25', '',
      '**停止结论：12个单位的几何/标签已生成，G1精确指纹及G2未通过；传感器采集、G5及B0/B1评估均NOT_RUN。没有放量，也没有按结果调整配额或种子。**', '',
      '阶段1规则先提交f61b1789；视场选择及v1.1先提交79ff1be1；生成器在首次运行前提交c65cb1c4。后续只读验收补充身份闭环及精确体素，不修改生成数据。', '',
      '证据：`artifacts.local/evidence/cnh-track-a-pilot12-20260925-v1/`中的manifest、unit00–11、result、readonly-audit。所有数字为受控Development；不替代硬件标定或自然泛化。', '',
      '## 阶段1与协议变化', '',
      'M0=0°，M1按冻结规则由−15°改为−10°：[1,2)m的min(HEAD,BODY)姿态覆盖中位数0.848145→0.913289，差0.065145≥0.05。完整表含7安装角、各横向盒、3距离段、HEAD-above及p10/p90，见[视场结果](CNH_TRACK_A_FOV_RESULTS_20260925.md)。图保存在FOV证据目录coverage.png。产品角度未定。', '',
      '相对v1.0：Python代UE；6/2/4独立划分；每配置12帧5Hz、第8帧标签优先锚点；完整世界固定、逐帧标签与5cm边界；主集非边界且历史≥4帧；计数随N缩放；三维H3重投影；遮挡感知可观测性定义；calib拟合、audit不调参；硬件/多目标标量推迟。取消旧HEAD诊断。用户后续授权修正B1测试：不要求真实移动后的光子期望恒等，只测试几何、计数搬运及因果性。', '',
      '## 完成量与门槛', '',
      f'12单位×20配置×12帧=2880几何帧；主子集{n}帧，边界{boundary}帧；另有非边界但历史不足帧。主集与边界/全帧分母不混。传感观测0帧，SNR三档与两安装合成0帧。', '',
      '|门槛|状态|实际检查|','|---|---|---|',
      f"|G0|{'PASS' if audit['G0']['pass_gate'] else 'FAIL'}|{audit['G0']['frame_count']}帧身份/有限几何/坐标/标签余量/贡献ID/见证闭环，错误{len(audit['G0']['errors'])}|",
      f"|G1 几何指纹|{'PASS' if audit['G1']['pass_gate'] else 'FAIL'}|{audit['G1']['nonempty_configs']}非空相关配置，精确1cm三角面–体素SAT；单位内外重复{len(audit['G1']['exact_duplicates'])}；近重复失败{len(audit['G1']['near_failed'])}|",
      '|G2|FAIL|逐单位及split结果如下；不移除失败布局|',
      '|G3 模拟/SNR|NOT_RUN|G2失败，未合成|',
      '|G3 硬件/标量|DEFERRED_PHASE2|未连接设备|',
      '|G4|NOT_RUN|没有多臂传感观测可核对|',
      '|G5|NOT_RUN|数据门槛失败，不训练|',
      '|B0/B1、可观测性b/c|NOT_RUN|没有传感/首命中可见性采样，不伪造分层或AP|','',
      'G1指纹检查的是锚点Q下HEAD/BODY相关表面；低矮背景/墙体身份另外记录，不用远景差异证明独立。初始生成回执的5mm采样指纹只作筛查；最终G1以只读精确SAT回执为准。全部均是程序化几何替身，没有验证真实类别外观/光学分布。', '',
      f"精确1cm指纹重复对(unit,config,unit,config)：`{json.dumps(audit['G1']['exact_duplicates'])}`。这表示冻结分辨率下几何指纹相同，不声称浮点顶点逐元素相同；布局级≥90%近重复0。仍按原严格门槛FAIL，不以随机种子不同豁免。", '',
      f"候选总数{result['attempts']}，拒绝{result['rejections']}，拒绝率{result['rejection_rate']:.4%}；32次候选上限未增加。类别对象计数：`{json.dumps(audit['category_objects'],ensure_ascii=False)}`。", '',
      '|单位|split|主帧N|边界帧|六位正例数 Lh/Lb/Ch/Cb/Rh/Rb|合格组合数(需12)|多对象正帧/门槛|G2失败|',
      '|---:|---|---:|---:|---|---:|---|---|']
    margin_receipt={}
    for u in units:
        c=u['checks'];multi=c['multi']
        lines.append(f"|{u['unit']}|{u['split']}|{c['N_main']}|{c['boundary_frames']}|{'/'.join(map(str,c['positive_counts']))}|{c['eligible_combinations']}|{multi['actual']}/{multi['required']}|{', '.join(c['failures']) or 'PASS'}|")
        mm=np.concatenate([np.abs(np.array(x['margins'])).min(axis=(1,2)) for x in u['configs']])
        margin_receipt[str(u['unit'])]=dict(frame_min_abs_margin_quantiles=np.quantile(mm,[0,.1,.25,.5,.75,.9,1]).tolist())
    lines += ['', '组合数只计算在至少两个配置出现的组合；原始不同组合更多，不可混作合格数。', '', '### 比例及相关性', '',
      '|单位|H/B不一致率 左/中/右|phi LH-LB / CH-CB / RH-RB / LH-CH / RH-CH / LB-CB / RB-CB|', '|---:|---|---|']
    for u in units:
        c=u['checks'];lines.append(f"|{u['unit']}|{' / '.join(f'{v:.4f}' for v in c['HB_disagreement'])}|{' / '.join('NA' if v is None else f'{v:.4f}' for v in c['phi'].values())}|")
    lines += ['', '### 各单位换算后的计数门槛', '', '统一ceil(旧计数×N/160)；配置数下限不变，组合种类数12/20不缩放。', '',
      '|单位|N|旧8→|旧16→|旧24→|旧32→|', '|---:|---:|---:|---:|---:|---:|']
    for u in units:
        m=u['checks']['N_main'];lines.append(f"|{u['unit']}|{m}|"+'|'.join(str(int(np.ceil(v*m/160))) for v in (8,16,24,32))+'|')
    lines += ['', '### 逐帧最近边界余量分布', '', '每帧取全部对象×查询的min(|m|)，单位m；完整原始有符号m保存在unit文件。', '', '|单位|p10|median|p90|', '|---:|---:|---:|---:|']
    for u in units:
        q=margin_receipt[str(u['unit'])]['frame_min_abs_margin_quantiles'];lines.append(f"|{u['unit']}|{q[1]:.5f}|{q[3]:.5f}|{q[5]:.5f}|")
    lines += ['', '### 完整标签组合分布', '']
    for u in units:
        c=u['checks'];lines += [f"单位{u['unit']}（组合:帧数/配置数）：", '', '; '.join(f"{k}:{v['frames']}/{len(v['configs'])}" for k,v in c['combinations'].items()),'']
    lines += ['### split汇总', '', '|split|N|合格组合(需20)|失败项|','|---|---:|---:|---|']
    for split,c in result['G2']['split_checks'].items():lines.append(f"|{split}|{c['N_main']}|{c['eligible_combinations']}|{', '.join(c['failures']) or 'PASS'}|")
    lines += ['', '## 单测及实现限制', '',
      '最终17项合成单测全部通过，0.477秒。包括圆柱网格弦误差、薄杆/接触/分离、三角形内部margin最优点、零面积接触见证、K1/零运动恒等、90°旋转、纯平移落点/质量、部分视场W、因果性、拒绝raw128、calib偏置隔离、精确体素与无噪H3逐区能量解析一致。', '',
      '独立合成静态墙2m→1.8m、两帧历史的B1-R相对当前帧L1残差0.589036；当前计数33864.506，累积27724.557。该无噪声诊断包含r^-2与H3离散搬运偏差，仅描述、不设恒等门槛，也不解释为模型故障或开发集AP。原始参数与回执见FOV证据目录b1-static-wall-diagnostic.json。', '',
      'B1-R只用H3和位姿；W表示历史视场/距离覆盖，不推断遮挡解除。H3角/径向离散及r^-2响应变化使它不等价旧raw128读出。离线遮挡面积可观测性、SNR设档、完整传感批处理、G5适配尚未进入集成运行；不把已写协议称为已实现/已验证。', '',
      '## 耗时与下一步', '', '|单位|几何生成秒|每几何帧秒|', '|---:|---:|---:|']
    for u in units:lines.append(f"|{u['unit']}|{u['wall_s']:.3f}|{u['wall_s']/240:.4f}|")
    total=sum(u['wall_s'] for u in units);size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    lines += ['',f"几何生成累计{total:.2f}s，流程总墙钟{result['wall_s']:.2f}s；只读审计{audit['wall_s']:.2f}s。报告前持久产物{size/2**20:.2f}MiB。每传感帧耗时NOT_RUN。",
      f"按同配置数量线性外推192单位**仅几何**约{total/12*192/3600:.2f}小时；不包括射线、面积可见性、SNR合成、训练及I/O变化，不能作为完整放量预算。未启动96/32/64。", '',
      '64独立audit的±.05 AP只是配对SD=.20的t近似规划；SD=.10/.30时半宽约.025/.075。当前4audit且G2未过，不足以支持可靠方差估计；后续放量仍需新方案审阅。', '',
      f"逐单位失败计数：`{json.dumps(dict(failures),ensure_ascii=False)}`。锚点标签优先不保证完整主子集满足多对象共现和组合重复配额；这是当前生成计划的失败，不能归因传感噪声或学习器。", '',
      '建议下一版先在几何层按整段轨迹的有效停留长度预分配组合/距离与多对象同时侵入时长，再冻结新生成计划。保留本版FAIL，不改5cm、不换seed、不删困难帧、不恢复当前试采。没有HEAD可观测性b/c结果，不能据此决定相机主导。', '',
      '提交记录见最终回复。持久场景/标签/失败/哈希全部保留，任务进程完成后释放；无GPU训练、UE、实物或远程付费资源。', '', '## 本轮改动文件（18个）', '',
      '- docs/CURRENT_DECISION.md', '- experiments/index.jsonl', '- research/knowledge/decision/index.json', '- research/active/dtr-r0/CURRENT.md',
      '- 以下14个均在 research/active/dtr-r0/nearfield/：', '',
      '```text', 'CNH_NEARFIELD_ACCEPTANCE_PLAN_20260925.md', 'CNH_NEARFIELD_ACCEPTANCE_PLAN_V1_1_20260925.md',
      'CNH_TRACK_A_FOV_PROTOCOL_20260925.md', 'CNH_TRACK_A_FOV_RESULTS_20260925.md', 'CNH_TRACK_A_PILOT12_RESULTS_20260925.md',
      'cnh_track_a_fov.py', 'cnh_track_a_generate.py', 'cnh_track_a_geometry.py', 'cnh_track_a_readout.py', 'cnh_track_a_report.py', 'cnh_track_a_validate.py',
      'test_cnh_track_a_geometry.py', 'test_cnh_track_a_readout.py', 'test_cnh_track_a_validate.py', '```']
    (folder/'margin-summary.json').write_text(json.dumps(margin_receipt,indent=2)+'\n',encoding='utf-8')
    completion=dict(status='STOP_G1_G2_FAIL',geometry_units=12,geometry_frames=2880,main_frames=n,boundary_frames=boundary,
        sensor_frames=0,G0=audit['G0']['pass_gate'],G1=audit['G1']['pass_gate'],G2=False,G3_sim='NOT_RUN',G4='NOT_RUN',G5='NOT_RUN',B0_B1='NOT_RUN',
        no_scale_up=True,no_test_city_ue_hardware=True,original_result_preserved=True,
        unit_input_hashes=audit['input_hashes'],synthetic_tests=17)
    (folder/'completion.json').write_text(json.dumps(completion,indent=2)+'\n',encoding='utf-8')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.input,a.report)

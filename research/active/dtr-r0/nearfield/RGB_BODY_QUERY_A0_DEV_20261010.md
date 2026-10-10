# RGB 几何主干与受约束残差：A0 Development 检验

2026-10-10，EXPLORE／已消费 Development，继承 `167181b5`。用户授权按讨论结论执行。

A0 完成三个 δ、三条 capture 的全部比较，未获得跨 capture 的有利修正信号。40777060 主结果只救回 0／1／2 个正见证，40777065 损失 1／1／2 个；arkit16 明显退化。没有因未满足双判据便自动判负：本次负信号来自第二条追加 capture 的实测正见证下降，以及辅助曲线各 δ 均存在低于 affine 的区段。未提高残差训练优先级；A0 失败不能否定重新训练残差。

同时已补充 15 个未被现有模型 train/cal 或迁移 eval 消费的官方 train 缓存环境、240 帧、6480 固定网格查询，新增严格 sampled FREE 30 个／10 环境，合法负校准候选从原 2 增到 32／12 环境。补参考之后完成冻结分数的按距离带校准备选探针，实测阈值漂移更大，当前不采用。冻结 log-affine 为当前 RGB 几何代表；重新训练残差仍保留为未来不同机制，尚未证明无价值。

## 机制、口径和预算

A0 以报告中的冻结原 train log-affine 为主干，在 log-Z 上加旧 depth/ray 的有界修正：`log Z = log Z_affine + clip(mu_old_depth_ray − log Z_affine, ±δ)`，δ 事先固定为 .05、.1、.2。无深度 geometry 臂不作主干。本次不训练、不运行模型推理、不下载。三个 δ 同时报告，同 δ 跨 capture 比较，不按 capture 挑最佳；最佳 δ 也只能是 Development 调参。两模型有效 mask 48 帧均完全一致；δ=0 精确退化为 affine，没有借 mask 或默认输入范围回退改变可用域。q01/q99 回退消融为可选诊断，本次未运行。

主结果沿用三折留一，阈值仅使用另两条 capture 的严格 sampled FREE；从经验观察分数选最接近且不超过 5% 的可达支持率，全部并列分数保留。每 fold 先保存阈值再评价留出 capture。四臂每折 cal 实际为 23/467、17/352、16/325；5% 是探索设计选择，不是认证 FPR。query 分数为有效域第 16 大米制 interval margin，正见证为至少 16 条已知正射线获支持；UNKNOWN 不补见证、不转 FREE。

事后辅助按留出 capture 自身 FREE 对齐同折 affine 实际成本：arkit16 5/105、40777060 12/220、40777065 10/247。affine 主点保持固定，A0 用仅 FREE 分数允许的最小有限阈值，不查询正例择阈值；遇并列分数报告实际能达到的计数，不硬凑。另保存全 0..N FREE 成本预算曲线，每臂在相同预算下选择最低 FREE 允许阈值，因此 affine 在同 FREE 数内也可能获得更多见证。曲线是事后描述，不是主结果或统计非劣证明。

双判据是双方认可的强信号；Claude 将其作为必要门槛，Codex 仅视为充分条件。明显增加见证但小幅增加 FREE 支持时必须展示收益／成本供权衡。本次未出现两个追加 capture 同时受益的这种分支。

本轮预算 CPU wall 1800s：A0 600s、负参考准备 600s、核验与整合 600s；GPU 0、新下载 0。目标为本报告、缓存分数／参考、逐查询配对、focused 核验、路线与 Git 交付；对应支线预算达到或完成即停止。旧配方、CNH 和其他 run 的预算／停止规则不变。

## A0 主结果：全部九组

| capture | 臂 | 正见证 / POS | FREE 支持 / FREE | 正见证救／损 vs affine | FREE 消除／新增 vs affine | UNKNOWN 支持 / UNKNOWN |
| --- | --- | ---: | ---: | --- | --- | ---: |
| arkit16 | affine | 103/228 | 5/105 | — | — | 74/99 |
| arkit16 | A0 δ=.05 | 88/228 | 4/105 | 0/15 | 1/0 | 74/99 |
| arkit16 | A0 δ=.1 | 71/228 | 2/105 | 0/32 | 3/0 | 74/99 |
| arkit16 | A0 δ=.2 | 62/228 | 1/105 | 1/42 | 4/0 | 74/99 |
| arkit_40777060 | affine | 120/202 | 12/220 | — | — | 9/10 |
| arkit_40777060 | A0 δ=.05 | 120/202 | 12/220 | 0/0 | 0/0 | 9/10 |
| arkit_40777060 | A0 δ=.1 | 121/202 | 12/220 | 1/0 | 0/0 | 10/10 |
| arkit_40777060 | A0 δ=.2 | 122/202 | 12/220 | 2/0 | 0/0 | 10/10 |
| arkit_40777065 | affine | 131/179 | 10/247 | — | — | 3/6 |
| arkit_40777065 | A0 δ=.05 | 130/179 | 10/247 | 0/1 | 0/0 | 3/6 |
| arkit_40777065 | A0 δ=.1 | 130/179 | 11/247 | 0/1 | 0/1 | 4/6 |
| arkit_40777065 | A0 δ=.2 | 129/179 | 11/247 | 0/2 | 0/1 | 4/6 |

## 事后等成本与完整曲线

| capture | 臂 | 对齐固定 affine 主点后的正见证 | 实际 FREE | 事后正见证救／损 vs 主点 | 完整曲线低于 affine 的预算数 | 0..主点成本内低于 affine 的预算数 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| arkit16 | affine | 103/228 | 5/105 | — | — | — |
| arkit16 | A0 δ=.05 | 107/228 | 5/105 | 9/5 | 14 | 6 |
| arkit16 | A0 δ=.1 | 96/228 | 5/105 | 8/15 | 14 | 5 |
| arkit16 | A0 δ=.2 | 93/228 | 5/105 | 14/24 | 15 | 6 |
| arkit_40777060 | affine | 120/202 | 12/220 | — | — | — |
| arkit_40777060 | A0 δ=.05 | 125/202 | 12/220 | 5/0 | 14 | 12 |
| arkit_40777060 | A0 δ=.1 | 125/202 | 12/220 | 5/0 | 28 | 10 |
| arkit_40777060 | A0 δ=.2 | 125/202 | 12/220 | 5/0 | 24 | 12 |
| arkit_40777065 | affine | 131/179 | 10/247 | — | — | — |
| arkit_40777065 | A0 δ=.05 | 130/179 | 10/247 | 0/1 | 14 | 10 |
| arkit_40777065 | A0 δ=.1 | 130/179 | 10/247 | 0/1 | 16 | 8 |
| arkit_40777065 | A0 δ=.2 | 128/179 | 10/247 | 0/3 | 10 | 6 |

相同 FREE 成本不等于相同阈值。若 affine 也使用仅 FREE 允许的最低有限阈值，其 5／12／10 FREE 成本点见证为 **113／122／132**，而非固定主点的 103／120／131。因此 arkit16 的 A0 .05 事后 107 不能写成曲线优于 affine；40777060 的事后 125 对完整曲线 affine 122 只有 +3，40777065 仍低于 affine 132。各 δ 都有低成本和完整曲线退化区段。

图：[完整事后曲线与主协议工作点](../../../../artifacts.local/work/rgb-body-query-a0-dev-20261010/repair-2/a0_curves.png)。曲线仅按 held FREE 选阈值；叉号是主协议工作点。

## 0.3–0.8m 单列

| capture | 臂 | 正见证 / POS | FREE 支持 / FREE | 主结果救／损 vs affine |
| --- | --- | ---: | ---: | --- |
| arkit16 | affine | 0/44 | 0/96 | — |
| arkit16 | A0 δ=.05 | 0/44 | 0/96 | 0/0 |
| arkit16 | A0 δ=.1 | 0/44 | 0/96 | 0/0 |
| arkit16 | A0 δ=.2 | 0/44 | 0/96 | 0/0 |
| arkit_40777060 | affine | 0/11 | 0/133 | — |
| arkit_40777060 | A0 δ=.05 | 0/11 | 0/133 | 0/0 |
| arkit_40777060 | A0 δ=.1 | 0/11 | 0/133 | 0/0 |
| arkit_40777060 | A0 δ=.2 | 0/11 | 0/133 | 0/0 |
| arkit_40777065 | affine | 0/0 | 0/144 | — |
| arkit_40777065 | A0 δ=.05 | 0/0 | 0/144 | 0/0 |
| arkit_40777065 | A0 δ=.1 | 0/0 | 0/144 | 0/0 |
| arkit_40777065 | A0 δ=.2 | 0/0 | 0/144 | 0/0 |

这一带主协议四臂全无正见证；40777060 有 11 个 POS 均未见证，40777065 POS 为 0，不能把零损／零救解释成近场有效。其余两带以及事后各带均在 summary.csv 和 query_pairs.csv 完整保存。

## 补参考后：冻结按距离带校准备选

[负参考准备报告](RGB_BODY_QUERY_NEGATIVE_REFERENCE_DEV_20261010.md)保存来源选择、全部 8424 查询与独立几何核验。新增近／中／远带 FREE 为 6／19／5，分别覆盖 1／9／3 环境；原 train 2 个 FREE 仅诊断、不混入 cal。新增参考尚无预测分数，未拿旧分数冒充新参考评分。

补参考准备完成后，使用现有三条已消费 capture 的冻结 query 分数按公开 Z 带分别三折校准。每带阈值仍仅来自另外两条 capture 同带严格 FREE，目标最接近且不超过 5%，并列保留；不是对新增 15 环境的评价。全部五臂、三 capture、三带和配对均保存。

| capture | 臂 | 带条件正见证 / POS | 带条件 FREE / FREE | 相对自身全局阈值见证救／损 | 相对同带 affine 见证救／损 | 近带正见证 / POS | 近带 FREE / FREE |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| arkit16 | affine | 94/228 | 78/105 | 59/68 | — | 8/44 | 69/96 |
| arkit16 | A0 δ=.05 | 74/228 | 74/105 | 50/64 | 2/22 | 8/44 | 66/96 |
| arkit16 | A0 δ=.1 | 68/228 | 74/105 | 56/59 | 5/31 | 9/44 | 65/96 |
| arkit16 | A0 δ=.2 | 55/228 | 54/105 | 49/56 | 4/43 | 7/44 | 47/96 |
| arkit16 | 旧 depth/ray 点 | 19/228 | 4/105 | 0/8 | 19/94 | 0/44 | 4/96 |
| arkit_40777060 | affine | 0/202 | 0/220 | 0/120 | — | 0/11 | 0/133 |
| arkit_40777060 | A0 δ=.05 | 0/202 | 0/220 | 0/120 | 0/0 | 0/11 | 0/133 |
| arkit_40777060 | A0 δ=.1 | 3/202 | 0/220 | 0/118 | 3/0 | 0/11 | 0/133 |
| arkit_40777060 | A0 δ=.2 | 4/202 | 1/220 | 0/118 | 4/0 | 0/11 | 0/133 |
| arkit_40777060 | 旧 depth/ray 点 | 31/202 | 8/220 | 0/36 | 31/0 | 0/11 | 5/133 |
| arkit_40777065 | affine | 6/179 | 1/247 | 0/125 | — | 0/0 | 0/144 |
| arkit_40777065 | A0 δ=.05 | 1/179 | 1/247 | 0/129 | 1/6 | 0/0 | 0/144 |
| arkit_40777065 | A0 δ=.1 | 12/179 | 2/247 | 0/118 | 12/6 | 0/0 | 0/144 |
| arkit_40777065 | A0 δ=.2 | 0/179 | 3/247 | 0/129 | 0/6 | 0/0 | 2/144 |
| arkit_40777065 | 旧 depth/ray 点 | 33/179 | 20/247 | 0/41 | 31/4 | 0/0 | 13/144 |

简单带条件校准没有恢复有利迁移取舍。arkit16 affine FREE 从 5/105 增至 78/105，两个追加 capture affine 见证降至 0/202、6/179；不是“更细校准必然更好”。旧 depth/ray 也丢见证且后两条 FREE 为 8/220、20/247。部分 A0 对已坍缩的带条件 affine 有救回，但远低于原全局 affine，不能择弱对照宣布成功。新增负参考尚薄且来源不同，下一先验证其冻结预测和跨环境／距离带工作点，当前不将这个备选升级。

## 核验、失败与下一决定

A0 focused 核验：48 帧有效 mask、2592 原分数精确重现、5184 独立排序／见证阈值等价、3 原 affine fold 精确重现均通过；独立 102821 项阈值／决定／配对比较、7776 配对行、2300 曲线行复算 PASS。负参考 8424 查询独立 XYZ 和严格状态复算、312 native 几何与身份、34 负索引重读通过，元数据修复保留前后 SHA。带条件探针 45 阈值、6480 查询决定、11664 配对及 168831 项独立比较通过。没有重跑无关 Android 或旧训练。

两次 A0 实现故障分别为把相同 FREE 数误认为相同见证的断言、曲线饱和端 NumPy 标量进入 JSON；失败 receipt 保留，修正后成功。修复在同一 run 的 repair-1／repair-2 保存，不改变主协议、δ、capture 或科学判据，前两次实测 10.817s 加最终 7.110s，合计 17.928/600s。负参考 payload 前语法／默认 Python 故障及 manifest 元数据修复均留证，保守 54/600s；整合／核验未逐项全计时，保守按 600s 全额记，总预算保守 **674/1800 CPU wall s**，GPU／训练／模型推理／下载 0。任务进程已结束，无常驻资源。

当前采用 affine 作为 RGB 几何代表，保留本次所有负结果及完整救／损成本。A0 不提高训练优先级，重新训练有界 log 残差仍是未来不同机制；本次未训练。新增 cal32 可以继续评分，但 5% 只有 1 个可支持查询的计数尺度，近带 6 个且单环境使条件校准尤其不稳。优先补充近场跨环境负覆盖并评价新参考上的冻结预测，再决定有界残差或其他校准形式。

这些结果均限于 Development。A0 裁剪为绝对距离训练的旧模型，不能代替真正残差训练；已消费同家族 capture、相关帧／格／ray 不构成独立确认。严格 sampled FREE 不证明整盒或身体空间空闲，未建立实机、安全、完整身体、细障碍或步行收益。不能由带条件失败定位单一原因或排除其他校准方法。

实现：[A0](rgb_body_query_bounded_residual_proxy.py)、[负参考](rgb_body_query_negative_reference_prepare.py)、[距离带备选](rgb_body_query_distance_conditioned_probe.py)。

复算使用 `config/local.toml` 的 `dtr_r0_python`（不要用缺 NumPy 的默认 Python），令 PowerShell `$researchPython` 指向该配置值。输出必须是尚未存在 plan 的 ignored 目录；既有目录含故障时通过 `--prior-attempt` 继承已用 CPU wall，不能覆盖或重置预算。

```powershell
& $researchPython research/active/dtr-r0/nearfield/rgb_body_query_bounded_residual_proxy.py --repo . --output artifacts.local/work/<new-unused-a0-output> --cpu-budget-s 600
& $researchPython research/active/dtr-r0/nearfield/rgb_body_query_negative_reference_prepare.py --repo . --output artifacts.local/work/<new-unused-reference-output> --budget-s 600
& $researchPython research/active/dtr-r0/nearfield/rgb_body_query_distance_conditioned_probe.py --repo . --a0 artifacts.local/work/rgb-body-query-a0-dev-20261010/repair-2 --negative-reference artifacts.local/work/rgb-body-query-negative-reference-dev-20261010 --output artifacts.local/work/<new-unused-conditioned-output> --cpu-budget-s 30
```

持久工件 owner 为本 run：`artifacts.local/work/rgb-body-query-a0-dev-20261010/`，成功主结果在 `repair-2/`，带条件备选在 `distance-conditioned/`；负参考在 `artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/`。旧 payload 不覆盖；plan／失败 receipt／完整分数／逐查询配对／阈值／全曲线／执行源码快照保留。相关前轮：[迁移诊断](RGB_BODY_QUERY_MIGRATION_DIAGNOSTIC_DEV_20261010.md)。

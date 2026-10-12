# RGB 同实际 FREE 排序描述与 ARKit 校准条件门（Development，2026-10-10）

**阶段1出现主交叉，按预封存规则停止阶段2。** fresh6 中远在实际 FREE=0 时，DAV 见证 276、Uni 227（差 −49）；在 FREE=104 时为 1047 对 1046（差 −1）。其余 old6 near、old6 midfar、fresh6 near 三组合计，在全部共同精确可达 FREE 点均为 Uni≥DAV。结果支持 Uni 具有更强的近带排序与不少中远收益，但未满足“四条曲线全域弱支配”的本次门槛。保留 DAV2 raw R0 的 RGB≥0.8m 职责，近带由 ToF 承担。没有进行阶段2新校准、下载或推理。

本次仅复用提交 `dd297b476679488136ec0cca9ea72dd4ffcc854e` 对应确认 run 的 old6+fresh6 缓存，192 帧、5184 query、10368 model-score rows。全部是已消费 Development；curve 的 witness 最大化是事后描述，不能充当选定阈值或新校准迁移证据。旧、新分别汇总，未合并掩盖交叉。

## 冻结定义与完整扫描

PLAN SHA256 `b99005f00e5ea0da90d99f899f310fc3e34ba2860d67ddec46495a613262afdf`，于 2026-10-10 14:03:42 UTC 封存，随后才读取评分。两臂仅 DAV2 Indoor Large raw R0、UniDepth V2 ViT-L raw R0；query/witness 第16顺序统计量、参考、public valid 域、分母全部沿用缓存，没有神经重跑、训练、affine 或读出改变。

每个 capture×near/midfar，以及四个六-capture aggregate 各用一个全局 cutoff。扫描全部有限 query_score 与 POS known_positive_score 的完整 ties，支持条件为 score≥cutoff；包括严格空支持及全有限支持端点，−inf 等 invalid 永不支持。正见证只计 POS 的 known_positive_score，FREE 状态严格为 FREE_ON_SAMPLED_RAYS，UNKNOWN 不当负例。每个精确实际 FREE 数取所有实现该数的 cutoff 中最大 W；主比较仅两模型精确可达数的交集，保留 ties，不插值或随机拆 ties，也不把单 capture 最优加总。

≤整数预算曲线是补充：范围为两臂共同有限 reachable range，列出每臂实际 FREE。若最大 W 并列，选实际 FREE 较少的代表；请求点不可精确达到时明确 NOT_EXACT，不借补充点冒称同数成本。图只画实际可达点，连接、插值、随机拆 tie 均为0。

## 四组合计门槛与全部主交叉

| 组 | POS/FREE/UNKNOWN分母 | 共同精确点 | Uni−DAV最小ΔW | Uni较低实际F连续区间 | 全共同点Uni≥DAV |
| --- | --- | --- | --- | --- | --- |
| old6/near | 225/604/35 | 540 | 0 | [] | True |
| old6/midfar | 1033/237/458 | 230 | 0 | [] | True |
| fresh6/near | 292/515/57 | 474 | 0 | [] | True |
| fresh6/midfar | 1047/107/574 | 103 | -49 | [[0, 0], [104, 104]] | False |

四组中任一实际 FREE 交叉即 STOP；fresh6/midfar 的 F=0 与 F=104 都保留，未限定低成本区间后改写主门槛。两者不能推成“Uni 全部排序都差”，另外三条曲线的全域弱支配同样是本次完整结论。

## 请求实际 FREE 的精确匹配及配对

下表全部 cutoff 都是描述上包络代表，不是选择部署阈值。救/损按 POS witness，FREE 增/减按 query support；同实际 FREE 时净增为0，格身份仍可能不同。

| 组 | 请求F | 状态 | DAV W | Uni W | Uni救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- |
| old6/near | 0 | EXACT | 1 | 4 | 4/1 | 0/0 |
| old6/near | 7 | EXACT | 29 | 99 | 86/16 | 6/6 |
| old6/near | 5 | EXACT | 16 | 91 | 85/10 | 5/5 |
| old6/near | 10 | EXACT | 31 | 101 | 87/17 | 9/9 |
| old6/near | 15 | EXACT | 38 | 152 | 126/12 | 14/14 |
| old6/near | 20 | EXACT | 49 | 161 | 124/12 | 18/18 |
| old6/near | 1 | EXACT | 4 | 14 | 13/3 | 1/1 |
| old6/midfar | 0 | EXACT | 202 | 328 | 151/25 | 0/0 |
| old6/midfar | 7 | EXACT | 541 | 756 | 261/46 | 5/5 |
| old6/midfar | 5 | EXACT | 411 | 660 | 288/39 | 5/5 |
| old6/midfar | 10 | EXACT | 599 | 769 | 229/59 | 7/7 |
| old6/midfar | 15 | EXACT | 667 | 821 | 209/55 | 7/7 |
| old6/midfar | 20 | EXACT | 727 | 834 | 162/55 | 11/11 |
| fresh6/near | 0 | EXACT | 13 | 16 | 7/4 | 0/0 |
| fresh6/near | 7 | EXACT | 35 | 63 | 39/11 | 4/4 |
| fresh6/near | 5 | EXACT | 28 | 61 | 40/7 | 2/2 |
| fresh6/near | 10 | EXACT | 38 | 92 | 62/8 | 7/7 |
| fresh6/near | 15 | EXACT | 42 | 105 | 72/9 | 9/9 |
| fresh6/near | 20 | EXACT | 54 | 132 | 87/9 | 13/13 |
| fresh6/midfar | 0 | EXACT | 276 | 227 | 102/151 | 0/0 |
| fresh6/midfar | 7 | EXACT | 483 | 641 | 271/113 | 4/4 |
| fresh6/midfar | 5 | EXACT | 419 | 606 | 288/101 | 3/3 |
| fresh6/midfar | 10 | EXACT | 577 | 707 | 237/107 | 5/5 |
| fresh6/midfar | 15 | NOT_EXACT | - | - | -/- | -/- |
| fresh6/midfar | 20 | EXACT | 761 | 794 | 135/102 | 8/8 |

## 请求预算 ≤F 补充（实际 FREE 明列）

| 组 | 预算F | DAV实际F/W | Uni实际F/W | Uni救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- |
| old6/near | 0 | 0/1 | 0/4 | 4/1 | 0/0 |
| old6/near | 7 | 7/29 | 7/99 | 86/16 | 6/6 |
| old6/near | 5 | 5/16 | 5/91 | 85/10 | 5/5 |
| old6/near | 10 | 8/31 | 9/101 | 87/17 | 8/7 |
| old6/near | 15 | 15/38 | 15/152 | 126/12 | 14/14 |
| old6/near | 20 | 20/49 | 18/161 | 124/12 | 16/18 |
| old6/near | 1 | 1/4 | 1/14 | 13/3 | 1/1 |
| old6/midfar | 0 | 0/202 | 0/328 | 151/25 | 0/0 |
| old6/midfar | 7 | 7/541 | 7/756 | 261/46 | 5/5 |
| old6/midfar | 5 | 5/411 | 5/660 | 288/39 | 5/5 |
| old6/midfar | 10 | 10/599 | 10/769 | 229/59 | 7/7 |
| old6/midfar | 15 | 15/667 | 15/821 | 209/55 | 7/7 |
| old6/midfar | 20 | 20/727 | 20/834 | 162/55 | 11/11 |
| fresh6/near | 0 | 0/13 | 0/16 | 7/4 | 0/0 |
| fresh6/near | 7 | 6/35 | 7/63 | 39/11 | 4/3 |
| fresh6/near | 5 | 4/28 | 5/61 | 40/7 | 3/2 |
| fresh6/near | 10 | 10/38 | 10/92 | 62/8 | 7/7 |
| fresh6/near | 15 | 15/42 | 12/105 | 72/9 | 8/11 |
| fresh6/near | 20 | 20/54 | 20/132 | 87/9 | 13/13 |
| fresh6/midfar | 0 | 0/276 | 0/227 | 102/151 | 0/0 |
| fresh6/midfar | 7 | 7/483 | 7/641 | 271/113 | 4/4 |
| fresh6/midfar | 5 | 5/419 | 4/606 | 288/101 | 2/3 |
| fresh6/midfar | 10 | 10/577 | 10/707 | 237/107 | 5/5 |
| fresh6/midfar | 15 | 14/683 | 15/764 | 178/97 | 6/5 |
| fresh6/midfar | 20 | 20/761 | 19/794 | 135/102 | 8/9 |

所有整数预算逐点及全部补充交叉见 supplemental_budget_comparison.csv 和 summary.json 的 supplemental_budget_groups；该补充不参与阶段1主门。

四个 aggregate 的精确请求点另有 `requested_exact_query_pairs.csv`（30240行）：逐 query 保留两臂 cutoff、query支持、POS见证、POS救/损与FREE增/减，含原DAV实际F补点。不可达点只在请求表标明，不生成伪精确配对。这份增补只读现有三个curve/score-pair CSV，未读原预测、重扫ties或重新绘图；逐请求点汇总与原配对计数一致，收据为 `requested_exact_query_pairs_receipt.json`。

## 原 DAV 工作点与描述上包络的区别

| 组 | 原DAV实际F | 原DAV W | 同F DAV上包络W | 同F Uni上包络W | 同F状态 |
| --- | --- | --- | --- | --- | --- |
| old6/near | 1 | 2 | 4 | 14 | EXACT |
| old6/midfar | 7 | 514 | 541 | 756 | EXACT |
| fresh6/near | 0 | 3 | 13 | 16 | EXACT |
| fresh6/midfar | 7 | 464 | 483 | 641 | EXACT |

原 DAV cutoff 固定为 0.24403834342956543 m。这些不同见证数来自在相同实际 FREE 数内事后最大化，不能把 DAV 上包络或 Uni 上包络当作原工作点结果。请求表已额外纳入原 DAV 的实际 F（old near 为1，其他已在请求集合）。

## 逐 capture 全部交叉

| capture/带 | 共同精确点 | Uni−DAV最小ΔW | Uni较低实际F连续区间 | 全共同点Uni≥DAV |
| --- | --- | --- | --- | --- |
| arkit16/near | 90 | -3 | [[3, 7], [12, 12], [17, 21], [23, 23], [25, 25], [27, 37]] | False |
| arkit16/midfar | 10 | -14 | [[0, 0]] | False |
| arkit_40777060/near | 131 | -4 | [[0, 22], [26, 38], [40, 62]] | False |
| arkit_40777060/midfar | 85 | -10 | [[0, 2], [4, 32], [34, 34], [37, 53], [55, 75]] | False |
| arkit_40777065/near | 142 | 0 | [] | True |
| arkit_40777065/midfar | 101 | -2 | [[23, 26], [29, 29]] | False |
| new_arkit_41069021/near | 87 | 0 | [] | True |
| new_arkit_41069021/midfar | 34 | 0 | [] | True |
| new_arkit_41069042/near | 60 | -3 | [[30, 31], [33, 40], [42, 42], [45, 48]] | False |
| new_arkit_41069042/midfar | 5 | -2 | [[0, 0]] | False |
| new_arkit_41069048/near | 79 | 0 | [] | True |
| new_arkit_41069048/midfar | 2 | -6 | [[0, 0]] | False |
| confirm_arkit_41125718/near | 108 | 0 | [] | True |
| confirm_arkit_41125718/midfar | 23 | 0 | [] | True |
| confirm_arkit_41125756/near | 92 | -9 | [[0, 5]] | False |
| confirm_arkit_41125756/midfar | 18 | -5 | [[3, 4]] | False |
| confirm_arkit_41142278/near | 35 | -2 | [[13, 13], [16, 18]] | False |
| confirm_arkit_41142278/midfar | 1 | 0 | [] | True |
| confirm_arkit_41159503/near | 61 | 0 | [] | True |
| confirm_arkit_41159503/midfar | 4 | 0 | [] | True |
| confirm_arkit_41159519/near | 91 | -9 | [[0, 0], [3, 5], [67, 81]] | False |
| confirm_arkit_41159519/midfar | 29 | 0 | [] | True |
| confirm_arkit_41159529/near | 123 | 0 | [] | True |
| confirm_arkit_41159529/midfar | 35 | -9 | [[0, 0], [2, 2], [33, 35]] | False |

这些局部交叉完整呈现，但按 PLAN 不额外加入 aggregate 门槛。全部逐 capture 请求点（含不可达状态）见 requested_FREE.csv；全 query 两模型原 score 与 witness score 配对见 query_score_pairs.csv。

## 阶段2预注册规则与未执行范围

阶段2原计划仅在四组合计全域排序占优时启动：以已消费 old6、fresh6 及原48帧ARKit cal 的去重严格sampledFREE，分别按全局5%规则封存两模型切点，再按官方Validation（不足才Training）顺序取六个新visit各16帧、nearPOS≥16。近带要求≥4/6 capture UniW>DAV且净FREE增量≤floor(1%×分母)；中远见证净损≤3%，净FREE增量≤max(3,floor(3%×中远FREE分母))，两条全通过才建议Uni raw、RGB≥0.3m近带辅助。

用户在新eval前将中远FREE从上轮2%改为上述式：约100的分母下floor2%只允许+2个query，容易受小计数波动影响；新式至少允许+3并按3%增长。该改动是本任务的前瞻预注册，不回改dd297b47的失败结论。本次阶段1门未通过，**阶段2 NOT_RUN**：没有拟合ARKit域切点、选新visit、下载新数据或推理；ARKit域校准的迁移能力仍未检验。

## 科学图与可重建数据

全范围四组图、24个 capture×band 面板与低成本放大图均有 PNG/PDF，放大只改坐标范围，曲线相同：

![四组合计全范围](../../../../artifacts.local/work/rgb-matched-free-arkit-cal-dev-20261010/stage1-complete/aggregate_4panels.png)

![实际FREE 0至20放大](../../../../artifacts.local/work/rgb-matched-free-arkit-cal-dev-20261010/report-summary/aggregate_low_FREE_0_20.png)

![逐capture24面板](../../../../artifacts.local/work/rgb-matched-free-arkit-cal-dev-20261010/stage1-complete/capture_24panels.png)

ignored artifacts 位于 canonical artifacts.local/work/rgb-matched-free-arkit-cal-dev-20261010（junction物理根 F:\ba-data\blindassist-artifacts-20260805\work\rgb-matched-free-arkit-cal-dev-20261010）：stage1-complete/all_ties.csv（23683行）、envelope.csv（5747行）、common_actual_FREE.csv（2793行）、at_most_budget_envelope.csv、supplemental_budget_comparison.csv、requested_FREE.csv、requested_FREE_pairs.csv、original_DAV_points.csv、query_score_pairs.csv（5184行）、curves.json、summary.json、terminal.json；report-summary/summary.json 额外明确全部连续交叉区间与最小差值。CSV 可独立重建图，不读模型预测或 native reference。

## 实现、资源与独立复算

实现 [rgb_matched_free_arkit_cal_describe.py](rgb_matched_free_arkit_cal_describe.py)，执行 source SHA `edaf173cbfb890ab1321caf29a25101dde2005689fd259045e0a0d935a867bee`。terminal 保存12份原 score cache路径/SHA和 PLAN SHA。实际完整描述/绘图 wall 4.9550528 s，无 GPU、下载、训练、部署切点。评价与报告保守35/400 CPU command-wall s、独立审计9/250 s、根代理准备/目视/整合/交付保守100/250 s，合计 **144/1200 s**；根代理100 s为覆盖全部相关命令及交付的保守计账，非实测CPU内核时间。新增下载 **0/1GB**，GPU推理 **0/600 s**，没有新GPU分配或任务常驻进程，评价/审计workers均已结束。

运行前 toy whole-tie gap/invalid endpoint/零 FREE 正见证/最小实际成本并列检查发现并修正 FREE 状态拼写与测试预期；独立源码复核同时指出同一状态字段及 budget comparison 补充。首次生产解析因缓存以字符串保存 nonfinite score 产生 TypeError，stage1/terminal.json 与 executed source 完整保留；增加 float 解码后执行 stage1-complete，原缓存未修改。绘图后处理及后续命令设置 MPLCONFIGDIR 到新run/mpl-cache、PYTHONDONTWRITEBYTECODE=1；首次图前已存在的共享 Matplotlib 缓存没有删除。

独立重导与逐query增补审计均 **PASS**：主审核对12份score SHA、10368源score行、23683完整tie行、5747精确上包络/图点、2793同实际FREE点、2954整数预算比较、5908预算上包络行、338请求行、50汇总配对行、5184原始query配对。后增产物只做受影响独立复算，24个精确请求点、30240逐query行、241920布尔字段与192汇总字段全部吻合，没有重做已通过曲线。收据为 `independent_audit.json`、`independent_exact_pairs_audit.json` 与最终独立资源收据；无生产评分函数导入、GPU、下载或训练，累计实测CPU command-wall 6.8199812 s，保守计9/250 s。不因审计改变扫描点、子集、主gate或开启已停止的阶段2。许可证与原模型 URL/revision/SHA 沿用上一确认 run license_receipts.json 指向的已固定收据，本次未下载模型权重、仅本地非商业研究，无权重再分发。

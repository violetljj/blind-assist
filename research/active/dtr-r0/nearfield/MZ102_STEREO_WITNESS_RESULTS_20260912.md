# MZ102：空间跨度过滤未带来可靠性突破

Date2026-09-12. EXPLORE. Frozen execution commit `7e93fd09`.
Decision: `STEREO_SPAN_REJECTION_FAILS_FRESH_RECALL_AND_EVENTS`.
Intended disposition: NEGATIVE_CONTROL for fixed query-local span rejection and
its fixed disparity-perturbation contrast; structured registration is pending
the unrelated evidence-availability blocker documented below. Preserve MZ101 spatial complementarity evidence;
do not promote this filter or alter retained-core/App behavior.

Protocol: [MZ102_STEREO_WITNESS_PROTOCOL_20260912.md](MZ102_STEREO_WITNESS_PROTOCOL_20260912.md).

## 直接结论

这次没有实现所要求的可靠性突破。固定过滤器在旧数据能减少小块误配，但在
新288帧上只减少3个FP，同时损失15个TP、漏掉一个完整小型HEAD障碍事件，
部分首次提醒最多晚1.25秒。不能因为它仍优于纯ToF，就称它改进了现有融合。

失败明确针对“查询区内可见支持跨度>=0.10m才允许立体深度报警”的这一个
配置。短小或碎片化支持既可能是误配，也可能是弱纹理、遮挡或头部运动下仅存
的真实证据。可见跨度不能单独承担可靠性否决权；本轮未扫描或放宽该阈值。

## 同输入主比较：新seed102013

24序列/12几何族、288帧、4Hz摆拍采样；textured/flat成对共享几何。
576个BODY/HEAD查询、222个positive、28个危险区间。新位置、速度、厚度、
纹理与背景图案；部分家族与MZ101相同，不声称新家族或真实环境泛化。

| 方法 | TP | FP | FN | F1 | 完整区间漏检 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整64区ToF | 126 | 8 | 96 | 0.7079 | 2/28 |
| 双目原始读出 | 181 | 15 | 41 | 0.8660 | 2/28 |
| 既有ToF+双目合并 | 200 | 15 | 22 | 0.9153 | 0/28 |
| 主候选：合并+跨度S | 185 | 12 | 37 | 0.8831 | 1/28 |
| 对照：合并+S+视差扰动I | 174 | 10 | 48 | 0.8571 | 1/28 |

S按各BODY/HEAD查询局部建立4邻域、邻接视差差<=1px的组件；以图像水平或
垂直跨度换算到米，不要求最小宽度，也不允许深度轴噪声冒充长度。S+I在S的
合格支持上再要求视差+/-1px端点均在查询区内，并重查组件。+/-1px只是未校准
扰动检查，不是置信区间。主候选始终是S，没有按新结果改选S+I。

ToF分支、SGBM、视场、位姿、区域盒和2on/2off状态完全相同。三种融合结果
满足ToF告警包含、S为旧union子集、S+I为S子集的不变量；这是实现检查，
不是独立性能成果。最终仍是当前前方区域提醒，不是一秒碰撞预测。

## 预定晋级条件

| 条件 | 新数据S结果 | 判定 |
| --- | --- | --- |
| 保住所有ToF TP | 126/126 | 通过 |
| 保留>=90%旧union独有TP | 59/74=79.7% | 失败 |
| 保留>=90%双目新增细杆TP | 33/37=89.2% | 失败 |
| 非ToF承担的额外FP至少减半 | 7到4，减少42.9% | 失败 |
| 完全错误会话不超过ToF | 三者均0 | 通过，但该会话指标无改善机会 |
| 不丢既有union已检事件 | 丢1个 | 失败 |
| 首次正确告警最多晚0.25秒 | 最多晚1.25秒 | 失败 |
| F1不低于既有union | 0.9153到0.8831 | 失败 |

分母使用`old_union AND NOT ToF AND GT`，没有把ToF本来能保住的大量TP混入
新增证据保留率。FP分母同样排除ToF已有FP。S+I的额外FP降至2，但新增TP仅
保住48/74，新增细杆保住31/37，不能因FP更低就晋级。

## 丢失的是哪些能力

| 切片 | ToF TP | 原合并TP | S TP | S+I TP |
| --- | ---: | ---: | ---: | ---: |
| 随头部转动的横杆 | 4 | 15 | 10 | 8 |
| 小型HEAD障碍 | 0 | 13 | 8 | 7 |
| 两种路径内细杆 | 32 | 69 | 65 | 63 |
| 部分遮挡细杆 | 26 | 36 | 36 | 32 |

小型HEAD障碍尺寸为0.12m前向、0.14m横向、0.10m高度。纯ToF两个外观事件
均漏检，双目让旧合并都能检出；S又完整删掉`small_head_flat / HEAD`事件。
头部横杆`head_bar_flat`的首次正确告警晚1.25秒。筛选特别伤害HEAD：

| 部位 | 原合并 TP/FP/FN | S TP/FP/FN | S+I TP/FP/FN |
| --- | --- | --- | --- |
| BODY | 92/10/6 | 90/7/8 | 86/6/12 |
| HEAD | 108/5/16 | 95/5/29 | 88/4/36 |

三种方法都没有完全错误的独立会话；原合并/S/S+I的FP片段为10/8/8，
累计FP查询时间3.75/3.0/2.5秒。BODY与HEAD时间分别累加，不是实际语音秒数。
原合并15个FP中7个有当前支持、8个为保持；S为4+8，S+I为2+8。
当前支持抑制不能自动修复全部保持阶段误报。

排除旧union左截断、GT进入前已报警，并要求双方都检出的共同19个区间，
首次正确告警状态延迟均值为旧union0.1447、S0.2500、S+I0.3816秒。
该成对均值排除了已经完全漏掉的小HEAD事件，必须与事件漏检一起看。

## 旧数据开发回放不能代替新结果

MZ101 consumed replay严格复现原ToF/stereo/union预测，没有阈值搜索：

| 方法 | TP | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: |
| 旧union | 202 | 25 | 22 | 0.8958 |
| S | 195 | 13 | 29 | 0.9028 |
| S+I | 182 | 8 | 42 | 0.8792 |

S消除了旧数据的两个完全错误会话，但新增TP只保留47/54，最大提醒延迟也已
增加1.25秒。配置在开发回放之前冻结，之后不变；新采集没有根据回放结果调参。
因此，旧数据F1微涨并未被用于选择一个新的操作点。

独立只读像素审计进一步区分了7个TP损失：前三个当前真实支持仍在，原方法
借此前错误支持/保持更早启动；其他损失确实包含真实表面被拒绝。
`head_turn_flat_03 / HEAD`的59个支持像素，59个都对应原生可见的同查询
表面，却被跨度规则拒绝，下一帧原告警还依赖它保持。不能把这些损失包装成
“只是删掉了幸运误报”。原生深度仅用于解释，未更改GT、分数或门槛。

## 值得保留的主线事实

新数据再次显示ToF与双目互补：ToF漏掉两个小HEAD事件，双目漏掉无纹理墙面的
BODY/HEAD事件，既有合并覆盖全部28个区间。对比ToF，既有合并新增74TP、7FP，
F1从0.7079到0.9153。这支持继续保留空间观测组合，但不是硬件、产品安全或
跨未知场景的证明，也不能与早期通用Radar任务的分数直接拼接。

下一技术问题应是能否获得与匹配错误直接相关的可观测证据，例如局部匹配代价
是否有唯一解、遮挡边界是否具有一致对应；不是继续把可见长度当可靠性。
这只是由失败导出的待检验方向，本轮没有实现新匹配器、训练模型或后续实验。

## 工程和证据

6项新单位测试通过，覆盖细杆、孤立片段、查询外组件、深度起伏、扰动边界和
UNKNOWN。固定MZ101渲染采集器重新采集288帧，原生bounds/spec一致性和场景
非任务表面不进入查询区断言通过。旧预测精确复现，预测在GT评分前封存。
新采集338.469秒，完整进程生命周期409.0秒；release=true、survivors为空。

新数据CPU SGBM共6.600秒；S与S+I两个对照合计4.854秒(约16.85ms/帧，包含
两种资格计算，不能当单个S耗时)，理由分别为GPU_BACKEND_UNAVAILABLE及
TASK_NOT_GPU_SUITABLE。预测、评分和可视化共25.61秒，不是端到端实时延迟。
录像可解码为288帧/4fps；已查看小HEAD textured/flat对照，后者只留下短小
深度片段，显示了被拒绝的真实信息。

独立最终复核通过采集1441项、fresh/development各37项输出哈希，以及观测输入、
预测封存、计数和主要晋级门；审计见任务目录`final-audit.json`。

保留canonical artifacts中的源、两次比较、审核、图像/视频及预测证据。
本轮复用MZ101专用DDC，没有新增E盘物理数据根；没有存活的UE任务进程。

终态登记未完成：`set-terminal-inheritance`被历史实验
`ase-body-query-pilot-20260909`的输入指纹验证阻断。其
`artifacts.local/downloads/ase-body-query-pilot-20260909/download-receipt.json`
当前缺失；本轮冻结提交时全库校验曾通过。未改写该旧指纹、补造收据、修改验证器
或绕过知识库检查。实验已结束，代码/报告独立交付；旧ledger中的MZ102 active
表示登记未闭合，不表示仍在运行。完整待登记terminal、role和归档row保存于
`artifacts.local/work/mz102-stereo-support-20260912/pending-registration.json`。

- [新场景视频](../../../../artifacts.local/work/mz102-stereo-support-20260912/fresh-v1/comparison.mp4)
- [新场景结果](../../../../artifacts.local/work/mz102-stereo-support-20260912/fresh-v1/summary.json)
- [配对延迟与部位表](../../../../artifacts.local/work/mz102-stereo-support-20260912/descriptive-tables.json)
- [开发集丢失支持审计](../../../../artifacts.local/work/mz102-stereo-support-20260912/dev-lost-support-audit.json)

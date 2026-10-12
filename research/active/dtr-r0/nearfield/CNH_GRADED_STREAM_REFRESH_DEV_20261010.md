# 连续提醒接口、断流重置与固定强档刷新

2026-10-10；EXPLORE，起始 `4653f67a`。用户“继续”执行[0/1帧合并](CNH_GRADED_EPISODE_MERGE_DEV_20261010.md)提出的连续状态与持续风险核对。分级、阈值与episode1保持固定，不改Android。

**时间戳适配器已实现并回放：现有全部36组短片段结果与原episode1逐帧一致，首次提醒和首次强提醒不变；合成30秒强信号在0/12/24秒发通知，轻信号只首报。长期提醒只建立了机制可执行性，现有短片段无法判断12秒刷新是否有用。通知层暂收束，下一回到弱/细目标的剩余漏检与首次证据时刻。**

## 接口核对与候选来源

主Assist链路的RiskEventTracker在任意反馈触发后标记wasAlerted，AssistDecisionKernel用EVENT_ALREADY_ALERTED抑制同事件；基础850/1500ms冷却到期并不会绕过该门控自动刷新。事件通常经3次missing/receding更新释放，曾提醒事件还有1秒再现抑制。因此不能把主App短冷却写成持续风险再次提示能力。见源码[事件状态](../../../../core/assist/src/main/java/com/linnan/blindassist/risk/RiskEventTracker.kt)、[决策门控](../../../../core/assist/src/main/java/com/linnan/blindassist/session/AssistDecisionKernel.kt)。

ToF Demo独立调用[HardwareDemoFeedbackPolicy](../../../../core/assist/src/main/java/com/linnan/blindassist/feedback/HardwareDemoFeedbackPolicy.kt)：持续alert满12秒可发REMINDER，距离缩短300mm且距前次至少1秒可发CLOSER，非alert满1秒释放。Activity用elapsedRealtime处理时钟，100ms watchdog在超过1500ms无结果时清状态并取消输出。基础ToF及可选A+LOCAL都向该策略输出单alert布尔，并未接入CNH强/轻分级。见[调用与watchdog](../../../../app/src/main/java/com/linnan/blindassist/HardwareDemoActivity.kt)。

本轮仅借用Demo的12秒与1500ms常量作为固定离线候选，保留CNH episode1的两次零grade释放；它没有复现Demo的1秒释放、恢复语音或距离接近提示。轻档不周期刷新。未按结果挑850ms、1500ms或其他刷新周期。

## 可执行的连续状态语义

新StreamNotifier只读取当前grade、调用者处理时钟及live/usable状态。首次正grade即发、轻到强即升；同事件最高等级不降，只有当前grade为2且距上次强通知至少12000ms才刷新。grade1不清强计时器，降轻期间不发强，随后真实grade2到来时才可刷新。零grade不发通知，连续两次零grade清事件；允许的一个零grade只是通知记忆。

停止、不可用或显式reset清状态；时钟倒退、两次观测间隔超过1500ms先清事件，再处理当前有效帧。恰1500ms不清。同处理timestamp当作重复样本忽略，不重复累计零grade；同timestamp的不同grade也会被忽略，调用者须提供可区分的新样本。reset后新的正grade立即首报。

该适配器的断流过期在下一次update才执行，不能主动在无输入时取消已经播放的声振。真正持续运行仍需上游watchdog传usable=false或reset，并处理输出取消。时间戳使用处理时钟；这里不提供采集时间到播放时间的延迟测量。

## 固定缓存与独立工程样例

科学缓存沿用baseline/both/head50、3seed、全部cal/validation，各384 scenes×4 replicas×13帧×2query。两方案refresh_none/refresh12共36cells、110592 query streams、216 query summaries。缓存无设备时钟，使用声明的200ms/帧名义代理；13个输出帧跨度2.4秒，两方案均0次周期刷新，输出与父episode1逐值一致。这是接口保留检查，不是刷新收益验证。

head50 validation及时HEAD352/357/351、BODY311/316/320各/384；联合contact通知1137/1196/1144、purepass221/271/257、clear56/89/69不变。强/轻与受提醒clip统计完整保留，新grade或新检测收益均0。

14个工程fixture与科学缓存分开，未拼接或循环科学clip来制造持续障碍：

| 合成输入 | 固定refresh12行为 |
| --- | --- |
| 200ms采样、30秒持续强 | 0秒首报，12与24秒刷新 |
| 同长度持续轻 | 仅0秒首报 |
| 强信号11999/12000/12001ms边界（期间保持新鲜采样） | 12000ms恰触发，仅一次刷新 |
| 首轻后400ms升强 | 400ms立即升级，无需等12秒 |
| 强后1次零／2次零再强 | 1次不重报；2次后新首报 |
| 到12秒先降轻、12001ms再强 | 降轻不发；再强时刷新 |
| 不可用／停止／显式reset／回钟后恢复强 | 清旧事件，当前新强立即首报 |
| 同timestamp重复零grade | 不把重复样本当第二次静默 |
| 观测间隔1500／1501ms后强 | 前者延续事件，后者新首报 |

14样例×两方案共720输入/输出行，详细reason、clock、availability与reset标记均保存。它们验证有限状态机与时钟边界，不是新的contact/pass/clear事件或真实持续风险证据。

## 决定与范围

保留episode1离线成本候选，以及可复用的时间戳/可用性适配器。12秒强刷新为工程候选，短缓存没有使用收益证据；不升级App默认，不追加周期扫描。本轮完成通知层生命周期核对；下一优先按固定head50/both工作点统计剩余HEAD/BODY弱/细障碍的未及时、晚检和窗口无提醒，关联原分数与峰支持定位瓶颈。通知节奏调参暂不优先于漏检与首报。

仅已消费受控模拟Development及独立合成工程样例；不证明长期障碍、物体身份、用户打扰、安全或硬件效果。原M3/5格/L2/body truth/fullbin/480/weak_pass、head50/both工作点及旧stop保留，深度静默证据不足、temporal不默认。没有模型运行、raw重建、科学clip延长或Android派发。载荷 `artifacts.local/work/cnh-graded-stream-refresh-dev-20261010/` 保存PLAN/metrics、两split通知、summary、fixtures、fixture_metrics与收据。

## 验证与交付

主命令2.502s，内部1.343s；audit内部0.953s、Python调用实测1.7404s、完整shell工具墙时2.132s。预算main90＋audit60＋接口核对60＋整合120＝330 CPU command-wall seconds。主运行/定向统计保守20s，审计和接口分别按各自全额60s、整合全额120s，共保守260/330s。GPU/fit/预测/新raw/科学clip延长均0，CPU后端为TASK_NOT_GPU_SUITABLE。没有常驻计算资源，源码与审计保存本目录，载荷完整留复算。

独立审计重算两split共36cells/110592query streams/216summaries，对所有缓存通知逐元素核对父episode1、首次any/首次strong和及时指标、reason计数、强/轻及联合成本；14生命周期fixture两方案720行与3补充API边界逐项核对，共3116170项scalar/array checks、9直接与30继承hash前后一致。只在工程fixture调用主adapter，实际缓存期望/指标独立计算。审计详情见载荷audit/result.json。

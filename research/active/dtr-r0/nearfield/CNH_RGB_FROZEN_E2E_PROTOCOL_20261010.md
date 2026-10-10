# 冻结配置的 HEAD 细障碍端到端评价

2026-10-10。用户授权“做”，执行前一轮明确的冻结比较。状态：FROZEN；本协议先于新评价分数访问，实施身份由执行前 PLAN hashes 固定。证据身份为新受控模拟场景的离线 ToF→通知评价；完整 RGB＋ToF 同事件评价暂不可评价。不得把两条独立子线的数字相加。

## 问题与固定比较

只回答：在相同通知器、校准集限定的同等 clear/pass 通知成本下，HEAD 横向细障碍的及时提醒增加多少？HEAD horizontal 且 size_variant=0 为主终点（cal厚1.3cm、hold厚1.7cm），包含较厚目标的全部 HEAD horizontal 为辅助，标牌边缘和 BODY 单列；不猜 HEAD/BODY 效用权重，不把轻提醒计为零成本。原 M3、原 5 格、原 both 的权重及阈值不修改，不接入 Android 默认。

比较臂：原 M3；原 5 格（M3 raised OR local）；原 both（ordinary_OR 强档＋single 轻档＋score_current c15_p64 追加）；单独成本匹配的 both 工作点。所有臂使用同一每 query 通知器：首正即时发、轻到强即时升级、容许一个静默帧、连续第二个静默帧重置；同帧 HEAD/BODY 通知取最高档计联合次数。事件记忆不代表物体跟踪。

成本匹配工作点始终保留原 5 格。仅非原 5 格提醒使用一个共享标量门控：`margin=max(ordinary-single_theta, score_current-parent_theta)`，仅对原 both 实际非零的候选生效，保留其原强/轻等级。两种未校准 logit margin 的最大值只作为固定排序，不是共同概率或物理置信度。只在新 cal 枚举完整 score ties 的 tau，选择 clear 和 purepass 联合通知次数均不超过原 5 格的最低可行 tau；不读 contact 收益选阈值。通知成本随事件合并未必单调，不以二分假设替代实际计数。`tau=+inf` 为明确回退到原 5 格，不伪装成新收益。原 both 无门控臂完整保留。相同 cal 成本不保证新 holdout 成本相同；报告两轴实际成本和残差，未匹配时不得称同等打扰下获益。

一次强或轻通知均按一次计主成本，强/轻次数分别报告；这是离线通知代理成本，不是用户主观感受。

## 来源与独立性

冻结新解析 AABB 光子模拟 cohort：cal/holdout 各 768 scene，K=2 噪声重复，固定 16 输入帧、f3–15 输出帧；两 split 各两个不同的新背景几何 family、每 family 两个 instance，背景 family 不跨 split；4 种 target family × HEAD/BODY × contact/pass/clear × 8 变体 × 4 背景。背景与 target 几何身份须与继承 train/cal/validation 物理键无交集；不读取保护 480/test。新噪声 seed 不作为新场景的证明。

继承传感器、轨迹、query、体素投影、光子统计和模型；target front 0.65m 及原路径保持以沿用 f3–13 的及时截止。共向/ideal 条件、有限新背景、透明/真实材料和遮挡复杂度未验证。新合成来源不是实机或新真实采集；独立新背景仅支持该受控集合的固定转移比较。

主 seed 固定列表首个 `2026100955`，另外 `2026100956/957` 逐个报告为敏感性，不投票、集成或按结果选择。M3 固定内部五模型沿用原定义，不重训。ordinary 及每高度 score_current HGB 全部加载已冻结模型。RGB affine 参数保留，候选职责限制为 public query 的 0.8m 以上；因本来源没有同步 RGB，RGB 推理和完整融合效果均为 NOT_EVALUABLE，不以 evaluator 深度替代模型输入。

## 指标与决定检查

HEAD 细横杆每 split 16 个 scene × K2，为32个相关模拟事件，独立几何 scene 计16；全部横杆32 scene × K2=64，其他完整 contact family/高度各同量。完整 HEAD/BODY contact 各128 scene × K2=256；实际分母以生成后 evaluator 几何核对为准并原样保留，不按模型分数删场景。K、帧和模型 seed 均不增加独立样本量。配对区间按 scene 重采样，只描述本固定背景集合；另列背景 instance/family 取舍，不能作为普遍新结构泛化。

及时检出只计真实 contact 的 query，另一非接触高度的提醒不能救回此事件。主指标为 HEAD 细横杆及时提醒的配对救回/损失及净变化；同时报告首次通知提前/推迟、晚检、全窗无提醒，BODY、sign_edge、其它 family 和全部 clear/purepass 通知/片段成本。每个主 HEAD 细横杆 scene 的变化是1/16的几何单位；小分母的零损失不代表非劣证书。

不设全部 seed 获胜、零损失或任意百分比门槛。新 holdout 成本两轴不高于原 5 格且 HEAD 横杆有净救回才支持本集合的同成本正收益；成本增加则只报告收益/成本取舍，零/负收益直接保留。不得看 holdout 后调阈值、权重、场景或融合规则。处理实现故障允许保留失败后按原身份恢复；科学结果失败不重跑或换 cohort。

## 执行和停止

载荷：`artifacts.local/work/cnh-frozen-e2e-20261010/`，保持指向 F 盘的 canonical junction。一次记录预算：科学生成＋所有模型推理600 command-wall seconds；CPU评分/校准/评价180s；定向审计180s；准备/整合240s；累计1200s。GPU allocation 最多600s且计入科学段，不另增预算。准备、实际阶段耗时、失败与预算不确定性原样记账；停止时不扩预算。大规模张量优先 GPU，metadata/通知状态和标量成本用 CPU，明确后端与计时。

prepare 先固化来源参数、模型/source hashes、物理身份及输入契约；执行前最终 PLAN 固定全部实现身份和本协议 hash；完成 cal 工作点后写 sealed calibration，再访问 holdout 指标。预算到达、输入不可评价或固定一次比较完成即停。数据源和实现错误保留失败 lineage，禁止覆盖旧收据。输出完整 scores/grades/notifications、cal 阈值与 costs、逐事件配对 ledger、summary/scene CI、terminal/resource-release 和短报告。无新训练、下载、付费算力、UE 启动、硬件采集或保护结果访问。

若完整 RGB＋ToF 不可评价，完成 ToF→通知结果与同步输入清单，主报告明确完整系统缺口；不能将其写成完整系统通过或毕设真实部署效果。

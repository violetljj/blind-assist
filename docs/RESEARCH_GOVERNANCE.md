# BlindAssist 渐进式研究治理

状态：current

当前策略真源：`configs/research_governance_v4.json`

历史 R1/R2/R3 策略分别保留在 `configs/research_governance_v1.json`、
`configs/research_governance_v2.json` 和 `configs/research_governance_v3.json`，
只用于复核绑定对应版本的旧协议和终态；不得用 R4 规则回写旧 receipt。

R4 适用日期：2026-08-01 起的新建或实质修订研究协议

## 目的

本规则解决两种同时存在的风险：

1. 规则过松，看到结果后换数据、换窗口、降门或改写失败；
2. 规则过早写死，在尚未了解数据和实现时就使用最终确证标准，使研究被代理
   指标、偶然实现错误或不成熟的数字门槛卡死。

核心原则是：

> 证据不可追溯改写，研究问题允许通过新版本继续学习。

## 研究风格：Wild Lab 与 Evidence Track

BlindAssist 当前首先是论文、毕业设计和算法原型项目。研究的首要目标是发现
有突破性的表示、机制和算法，不是提前把每个探索分支包装成可部署的助盲产品。
因此，项目采用两个并行但不混淆的研究风格：

| 风格 | 默认用途 | 允许做什么 | 不能声称什么 |
| --- | --- | --- | --- |
| `WILD_LAB` | Discovery、Canary、普通 Thesis Development | 大胆替换抽象、跨数据集合并、Teacher/pseudo-label、合成数据、自监督、未成熟几何和高成本离线模型；可以超过当前 Android、模型大小和默认 YOLO 约束 | 不能把探索结果写成 Confirmation、真实用户安全、生产能力或默认模型授权 |
| `EVIDENCE_TRACK` | 明确启动的 Confirmation、Deployment、claim-critical 评测 | 冻结问题、实现、session split、统计和缺失数据规则；使用独立 anchor/blind、可复核 receipt 和与 claim 匹配的验证 | 不得以较弱的代理证据替代对应的独立事实 |

`WILD_LAB` 不是降低科学要求，而是把科学要求放在正确位置：每个分支仍须有
假设、因果差异、最小判别实验、预算、停止条件和限制说明；但不因缺少最终
安全、设备或用户证据而阻止普通机制研究。`EVIDENCE_TRACK` 也不是所有新工作的
默认入口，只有当用户或协议明确要回答更高等级 claim 时才激活。

两种风格都必须保留以下四条硬线：

1. 不能偷看或泄漏受保护的 confirmation/blind outcome；
2. `UNKNOWN`、缺失标签和冲突证据不得静默当作 negative；
3. source GT、sensor-derived、synthetic、teacher pseudo 和 model consensus 必须分开命名并保留 provenance；
4. 结论的 claim ceiling 必须与实际证据一致。

缺少 safety authority 只关闭安全类 claim，不自动关闭训练、算法比较、机制研究、
论文级结果或研究 demo。

历史 receipt、失败、INVALID 和用过的数据必须保留；但一个 item、sequence、
implementation 或 evidence version 的失败，默认只关闭它实际影响的最小范围，
不得自动扩大成整个算法方向或科学问题永久关闭。

另一个同等重要的原则是：

> 失败必须增加知识；没有形成可检验学习的失败，才是需要纠正的失败。

## 当前原则基线：可证伪、可学习、可进化

本节不是不可违背的宪法。用户提出的原则、Agent 的解释、`AGENTS.md`、领域 current
和本文件本身都有局限，均可被后续证据、成本、论文目标和更好的理论质疑、替换或
删除。它们只代表当前最好判断，不获得永久正确性。

失败不可怕，但失败必须产生可复用的知识。每次失败实验都应得到：验证了什么假设、
排除了什么可能、失败发生在哪些条件下、暴露了什么瓶颈，以及这些结论对后续路线
意味着什么。禁止仅记录“失败”后直接更换参数或开启新的组合。

大模型可以高效率地生成和测试多种数据、公式、信号与算法方案，但不能把无目的穷举
当作研究。每个实验分支必须具备明确假设、理论或经验依据、最小判定实验、评价指标、
资源预算和停止条件。优先进行能够快速区分路线可行性的单变量审计和反事实实验。

项目中的约束、阶段划分、指标门槛、架构设计以及 `AGENTS.md` 等治理文件都允许被
质疑。没有任何规则天然永远正确。认为某项约束不合理时，先形成 evidence-backed
challenge，经明确 review/adoption 后更新 current，不能静默绕过。

已验证失败的方案原则上不得在相同前提下重复测试。只有当数据、输入信号、补偿方式、
系统角色、评价目标或部署条件发生实质变化时才重新开启，并必须说明“本次与上次有
什么不同”。旧方案即使不能作为核心决策信号，也应评估能否转化为辅助特征、质量
门控、不确定性指标、负样本、对照基线、诊断工具或消融组件。

项目不仅要优化算法，也要持续优化研究过程本身。定期复盘哪些实验真正提高了认知，
哪些规范阻碍了有效探索，哪些模块造成了不必要的复杂度。允许提出删除、合并或降级
低价值架构；实际变更仍按正常范围、证据和破坏性操作规则执行，不能只因历史投入而
继续维护。

每轮研究结束必须沉淀：

- 新获得的事实与证据；
- 被否定或被削弱的假设；
- 仍然无法判断的问题；
- 可复用的代码、数据和工具；
- 下一轮最具信息增益的实验；
- 是否需要修改项目治理规则。

我们追求的不是“永远不失败”，也不是“测试过最多方案”，而是让每一次实验都推动
项目模型、算法架构和研究方法共同进化。

## 毕业目标与反官僚约束

BlindAssist 当前首先服务于论文、毕业设计、院内演示和竞赛。论文级机制研究与
产品级安全认证必须分层：

- 论文可以依靠清楚的可证伪问题、可复现机制实验、有效负结果、限制分析和演示闭环；
- 不要求先完成真实用户有效性、独立行走安全认证、全硬件闭环或生产部署；
- 产品级要求不能倒灌，成为 Discovery/Canary/论文机制证据的默认阻塞项；
- positive thesis evidence 也不能反向冒充产品安全结论。

当前优先级明确为：

1. 形成能够支持毕业论文的可证伪问题、方法、实验、结果和限制；
2. 做出与论文主张一致、明确标注边界的演示原型；
3. 在不拖住前两项的前提下积累未来可用的工程证据；
4. 只有用户明确启动产品化后，才进入真实用户、独立行走、设备安全和生产认证。

论文级最小充分包通常包括：清楚的问题、合理 baseline、可复现方法、数据说明、
关键消融/反事实、有效正负结果、失败分析、限制和一个不夸大能力的 demo。它不要求
先证明真实视障用户安全有效。只要不声称产品安全，缺少 deployment evidence 不是
论文机制研究的失败。

默认不为尚未激活的产品认证消耗主线研究预算，也不创建“等待真实用户/安全认证”
作为算法或论文的 blocker。对外展示只需清楚标注研究原型、非独立助行工具，并避免
制造已经验证安全的假象。

任何新增或保留规则都应至少通过一种必要性检验：

```text
INFORMATION_GAIN | REPRODUCIBILITY | CLAIM_INTEGRITY
| EXTERNAL_SAFETY_OR_LEGAL_REQUIREMENT
```

如果一条规则持续增加工作量，却不能提高认知、可复现性、论文可信度或必要风险控制，
默认应质疑并降级为 guardrail/diagnostic，或删除。更严格规则的提出者承担说明其
必要性的责任，而不是让研究者证明为什么不需要它。

## 效率与最小充分严谨度

严谨和效率都是目标。默认追求 `minimum sufficient rigor`：投入的流程、测试和证据
成本应与修改影响面、不可逆性、结论等级和污染风险成比例，而不是对所有动作套同一套
最重流程。

- 纯文字、措辞或不影响结构的说明修改，通常只做针对性的人工、链接或格式检查；
- 协议、validator、数据身份或统计门修改，运行对应的专项合同与恶意反例；
- 公共接口、核心算法、构建链、发布路径或影响面不明时，才扩大到模块或全量回归；
- 可逆、低风险的 Discovery 允许先做最小实验，再补轻量 round 记录；
- 优先复用现有数据、fixture、缓存和工具，并行处理真正独立的工作；
- 允许有理由的工程捷径、近似和代理，但必须标注适用域、误差与退出条件。

捷径不能用于伪造或夸大证据、偷看 confirmation outcome、静默改写旧结果，或制造
未披露且不可逆的安全、权利或数据风险。若一项检查长期耗时却几乎不发现问题，应统计
其发现率和成本，降级为抽样/专项检查或删除。若连续出现同类漏检，再升级对应局部门，
而不是无差别把整个项目加重。

验证选择遵循：

```text
verification scope = smallest suite that covers changed behavior and credible blast radius
```

“不确定影响面”是扩大验证的理由；“以前一直全量跑”不是。

治理工作本身也必须遵守最小影响面：

- 不因命名、文案、receipt key 或未来漂移监控等控制面问题，重复未受影响的算法
  计算、数据生成或科学门；
- 不把非阻断改进升级成下一科学阶段的前置任务；
- 一个阶段只维护一个 current 入口。历史 spec、lock、receipt 和结果用于追溯，
  不再同时充当日常操作面；
- 新增 gate、receipt、lock、状态或文档前，必须说明它提高
  `INFORMATION_GAIN / REPRODUCIBILITY / CLAIM_INTEGRITY /
  EXTERNAL_SAFETY_OR_LEGAL_REQUIREMENT` 中的哪一项；
- 若不能说明，默认进入 backlog、降级为 diagnostic，或删除。

阶段级最小证据包按风险递增，而不是所有研究一律采用 confirmation 配置：

| 阶段 | 默认最小证据包 |
| --- | --- |
| `DISCOVERY` | 问题、来源/访问说明、可复现命令或 notebook、简短结果 |
| `CANARY/DEVELOPMENT` | 问题、实现身份、最小判别实验、专项测试、结果与限制 |
| `CONFIRMATION` | 冻结协议、数据身份、实现/统计 lock、独立 validator、receipt |
| `DEPLOYMENT` | confirmation 包加设备、生命周期、回归、风险和回滚证据 |

领域协议可以增加材料，但必须说明增加它的具体风险；“更严谨”本身不是充分理由。

## 论文优先的默认降级

对硕士论文、毕业设计、演示和比赛原型，新工作默认是可逆
`DEVELOPMENT_STANDARD`；不得仅因存在 formal 模板、旧 one-shot 终态或未来可能投稿，
自动升级成最终 Confirmation。只有用户明确启动最终确认问题、确认数据和结论用途后，
才进入 `CONFIRMATION_STRICT`。

Discovery 默认不分配或消费 fresh holdout。算法早期优先使用已声明的 Development、
consumed、synthetic 或公开可重复数据来找接口错误、验证方向和缩小候选；只有一个
成熟候选确实需要更高等级结论时，才为最终 Confirmation 单独分配独立数据。

小型 label mapping、mask decoder、tensor layout、坐标变换和 schema adapter 必须先在
合成 canary 上覆盖全部合法值、边界值、未知值和预期失败路径。这个 canary 是低成本
实现防线，不是 scientific utility evidence；通过后再进入真实数据候选 utility。

Development 默认允许：

- 在明确标为 Development、burned 或 consumed 的数据上修复、调试和重跑；
- 为每次修复建立新的 run/evidence version，保留旧结果但不把小工程错误扩大为路线终止；
- 一次比较至多 3 个有明确差异和停止条件的候选；
- 在最终选模前做 host 或 device runtime benchmark，作为工程取舍证据；
- 每轮至少形成一个老师可读的结果表、图、失败案例或可运行 demo。

设备 benchmark 分成两个互不替代的角色：

| 类型 | 用途 | 可否参与候选排序 | 默认时点 |
| --- | --- | --- | --- |
| `ALGORITHM_SELECTION_BENCHMARK` | 在固定代表性 harness 中比较候选延迟、内存和总链路成本 | 可以，但只形成 Development 选模证据 | 候选 utility 基本可解释后，可在 formal 选模前执行 |
| `PLATFORM_ENGINEERING_BENCHMARK` | 检查 backend、build、operator、内存、thermal 和测量链可行性 | 不可以；可使用 proxy/synthetic workload | 可与算法早期工作独立、提前执行 |

两类 benchmark 都不产生 Confirmation、产品安全或默认 App 集成 authority。平台工程
失败只阻塞相应 backend/设备路径，不自动否定算法；算法选模 benchmark 也不能替代
效果指标。

Development 的最小交付为：问题、baseline 与 claim ceiling；代码/模型/主配置/数据
manifest 的版本身份；覆盖实际变更的专项 sanity check；结果表/图/demo；限制和下一步。
默认不要求逐文件 SHA、完整 hash chain、一次性 holdout、底层逐行独立全量复算、
activation/freeze/closeout receipt 链。若某个身份、重放或污染风险确实需要其中一项，
只局部增加并写明风险。

路径、schema、decoder、依赖、runner、网络或设备控制在主张指标产出前失败时，记录
轻量 incident，修复后可用同一 Development 数据建立新 evidence version 重跑。若已经
看到主张指标，并据此改算法、阈值、候选或分母，则同一数据只可继续用于 Development；
最终 Confirmation 必须另行明确激活，并使用适当独立的数据。一个输入映射或操作故障
不再默认永久关闭候选、路线或研究问题。

本规则只向前生效。已关闭的 R1、R2-P0、旧 one-shot 证据、receipt 和 consumed 数据
角色保持不可变；可以把它们用于明确标注的 Development regression、rehearsal、
validator 或错误分析，但不能追溯恢复其 fresh/unseen 身份。

## 三档执行配置

新建或实质修订的研究必须选择一个与声明等级和实际风险相称的 profile。profile
决定默认流程负担，不改变阶段能够签署的结论：

| Profile | 默认阶段 | 默认执行方式 | 不默认要求 |
| --- | --- | --- | --- |
| `CANARY_LITE` | Discovery / Canary | 可重复的最小判别实验；按适配度排序数据，满足充分性即停止；确定性元数据自动校验，低风险观察单 Agent 加冻结抽样审计 | one-shot、全量双 Agent、完整 hash chain、穷尽全部可访问数据 |
| `DEVELOPMENT_STANDARD` | Development | 允许在 burned/consumed Development 数据上比较至多 3 个预先说明差异的候选；版本化修复和重跑；可提前做 host/device benchmark；每轮交付表、图或 demo | one-shot、逐文件 SHA、完整 hash chain、底层全量独立复算、把调试失败当科学否定 |
| `CONFIRMATION_STRICT` | Confirmation / Deployment | 仅由用户明确激活；结果访问前冻结问题、数据、实现、统计和缺失处理；独立 validator、完整 receipt；有依据时使用 one-shot | 结果后补门、换算子、缩分母、把 Development 数据包装成独立确认 |

阶段给出的默认映射是：

```text
DISCOVERY/CANARY -> CANARY_LITE
DEVELOPMENT      -> DEVELOPMENT_STANDARD
CONFIRMATION/
DEPLOYMENT       -> CONFIRMATION_STRICT
```

低阶段因真实 outcome 污染、不可逆发布、权利、安全或高成本设备风险，可以升级 profile，
但必须写明具体风险；不能仅因为模板或 validator 已存在就升级。高阶段不得使用更弱
profile。历史协议继续按其绑定 policy 复核，不反向套用 R4。

`CANARY_LITE` 可以建立机器合同，但合同只保留当前问题真正需要的字段和 artifact。
可重跑不表示可改写历史：每个已报告版本仍保留，调试重跑产生新 evidence instance；
只有 `CONFIRMATION_STRICT` 的 outcome access 才默认触发不可回退的新版本边界。

### 风险分层的 Agent 审查

Agent-only 不等于所有 item 都必须双 pass：

- 哈希、路径、schema、帧序和确定性元数据由程序校验；
- 低风险、明显观察由一个 Agent 标注，并按冻结规则抽样给第二 Agent；
- 会改变 terminal、主分母或关键边界的观察，以及 calibration 判定为歧义的观察，
  使用两个互不可见的新上下文；
- 只有影响结论的分歧才启动 fresh 第三 Agent；不确定项局部 `NOT_EVALUABLE`。

同一模型家族的隔离会话只能称 operational isolation，不能称独立人工真值或独立现实
测量。协议必须报告实际抽样比例、分歧率和裁决负担，不能把未复核 item 写成
model consensus。

### 操作故障与科学失败分开

路径、JSON、依赖、runner、网络或设备连接等单次控制面错误，默认只需要轻量
`operational incident receipt`：现象、影响范围、科学 outcome 是否访问、修复或防复发
动作。科学假设失败、重复的重大操作故障，或改变主线判断的事件，才要求完整 failure
learning record。操作 INVALID 关闭受影响 evidence version，不自动消费或否定科学
假设。

### Host 预检按风险触发

以下任务必须使用 guarded host preflight：正式 one-shot 或不可逆 claim、预计超过
15 分钟、高 I/O/内存/设备风险，或轻量 pilot 无法给出运行上界。预计 3–15 分钟的
可逆任务只需轻量 timeout、进度和 scoped output 合同；短小可逆 Canary 可以直接运行。
任何时长一旦出现反复解压、交换、GPU/CPU 明显闲置或无进度，仍须暂停诊断。

## 五阶段证据梯度

| 阶段 | 主要问题 | 允许做什么 | 允许声明什么 | 冻结强度 |
| --- | --- | --- | --- | --- |
| `DISCOVERY` | 数据和现象实际长什么样 | 扫描来源、看分布、验证格式、形成候选 | 找到/未找到候选、数据特征 | `F0` |
| `CANARY` | 机制方向和实现是否基本正确 | 查看输入与输出、修同步/坐标/符号、有限调试 | 机制方向、实现已调通、不可评价 | `F1` |
| `DEVELOPMENT` | 候选实现是否值得进入独立测试 | 在开发集迭代、消融、敏感性分析 | 实现准备好/未准备好进入确证 | `F1` |
| `CONFIRMATION` | 独立数据是否支持预注册命题 | 结果前冻结数据、代码、门槛和统计 | PASS/FAIL/NOT_EVALUABLE | `F2` |
| `DEPLOYMENT` | 设备、用户和风险证据是否支持应用 | 独立设备、回归、最坏场景和安全门 | 仅对应部署边界的通过/失败 | `F3` |

`DISCOVERY`、`CANARY` 和 `DEVELOPMENT` 不是低质量证据，而是不同用途的证据。
它们不能冒充 confirmation，但可以合法地改进问题定义、数据选择和实现。

## 数据能力驱动的三条工作轨道

五阶段描述证据强度；下面三条主轨道描述数据在实际工作中的用途。它们不要求每个
数据源独立回答全部问题：

| 轨道 | 主要用途 | 允许的数据状态 |
| --- | --- | --- |
| `CAPABILITY_DISCOVERY` | 了解自然数据、响应分布和失败模式 | `CONTENT_INSPECTED`、`OUTPUT_INSPECTED` |
| `DEVELOPMENT_DIAGNOSTIC` | 修实现、选诊断、调候选参数 | `OUTPUT_INSPECTED`、`TUNED_ON` |
| `SEALED_EVALUATION` | 在冻结算法和指标后做独立评估 | `SEALED_UNSEEN`；只看过内容时可为 `CONTENT_INSPECTED`，但必须披露筛选依据 |

`EXTERNAL_TRANSFER` 是额外的跨设备、跨场景或跨来源问题，不等同于普通 holdout，
也不是论文机制结果的默认前置条件。

研究顺序默认改为：

```text
低成本发现可获得数据
→ 建立极简能力表
→ 运行少量连续片段
→ 根据真实失败模式确定 Development 问题
→ 在调试前按 session/route/sequence 预留 sealed holdout
→ 冻结算法与指标后评估
```

Discovery 运行前只需写宽松观察清单，不预先规定哪种方法必须获胜。合法发现包括
RCLE 只在部分场景有效、bbox growth 更强、步态/模糊主导误差，或者整条路线不值得
继续。

## 最小操作门与禁止的理想角色门

Discovery 的默认硬门只保留：

- 数据能够合法取得并解码；
- 时间顺序可复算；
- 数据集和 sequence 身份基本明确；
- 已知许可或使用限制被记录；
- 下载、解码和适配成本有界。

固定十秒、同源正负、精确物理闭合率、同时具有 RGB/pose/depth，以及一个来源回答
所有问题，都不是 Discovery 的默认准入条件。它们只有在某个具体命题确实需要时，
才可作为该命题的局部门或诊断。

能力表只允许使用策略真源声明的 10 列 CSV/JSONL。它是工作记录，不是算法运行
许可证；不得为它开发通用数据框架、审批状态机或逐来源长篇准入报告。

## 渐进式冻结

- `F0 EXPLORE`：问题和诊断量可迭代；记录来源、版本和观察，不产生确证结论。
- `F1 INTERFACE`：冻结输入/输出、数据角色和主要机制；实现、诊断与候选参数可在
  development 数据上迭代。
- `F2 CONFIRM`：在 outcome access 前冻结确认集、实现、门槛、统计、缺失处理和
  terminal；结果出现后只能保留原版本并另立新版本。
- `F3 ARCHIVE/DEPLOY`：冻结发布或部署所需的完整 provenance、设备和回归证据。

不再默认要求所有探索任务在第一次真实数据运行前就达到 `F2`。one-shot claim、
byte-exact publication 和独立全量 replay 只在证据价值与成本相称时使用；是否需要
它们必须由协议明确说明，而不是作为所有阶段的默认仪式。

## 约束必须分类

每条规则必须属于以下一类：

- `INVARIANT`：证据诚信、安全、访问控制、权利或不可伪造事实。违反时硬失败。
- `GATE`：某一阶段的晋级判据。只约束声明的阶段和 failure scope。
- `GUARDRAIL`：触发复核、敏感性分析或降级，但不自动判整个实验失败。
- `DIAGNOSTIC`：只帮助理解，不得成为隐藏门槛。
- `ASSUMPTION`：可证伪的建模假设；失败时优先修改假设或代理指标，而非宣布数据
  或研究问题无效。

协议不得把“所有数字”都写成硬门。世界坐标平移速度、运行时、coverage、
aggregate score 等量是否适合作门，必须由它们与科学问题的因果关系决定。
任何 `GATE` 都必须具有可执行的 `metric / operator / threshold / unit`；只有说明文字
而没有可重放判定结构的条目不是 gate。

## 数字阈值的最低说明

任何数值约束都必须写明：

1. `unit`：单位和采样时间尺度；
2. `rationale`：它控制什么风险或混淆；
3. `calibration_source`：物理依据、文献、合成标定或独立开发数据；
4. `sensitivity_plan`：邻近数值是否改变结论；
5. `revision_policy`：何时可在新版本修改，何时禁止因确认集结果而修改。

在 `DISCOVERY/CANARY` 中，上述缺失可以成为 warning 并继续学习；进入
`CONFIRMATION/DEPLOYMENT` 前必须补齐，否则不能声称协议已经冻结。

建议同时保存连续值、探索区间和正式门。例如：

```text
diagnostic: 保存完整 closing-rate 分布
guardrail: 0.025/s <= rate < 0.05/s，仅作弱阳性探索
confirmation gate: rate >= 0.05/s
```

## 数据使用和“用过即烧”

### 已消费数据的主动复用

已消费不等于不可用。对于 Discovery、Canary、Development、回归、消融、失败分析、
论文机制实验和可运行演示，项目默认允许主动复用已消费数据，不得仅因“用过”而阻塞
有信息增益的实验。复用时必须同时记录：此前读取过什么、是否影响当前候选、当前最强
证据角色、独立性的具体维度，以及不能主张什么。

推荐角色包括：

- `PROJECT_CONSUMED_DEVELOPMENT`：项目历史已读取，可用于当前开发评价；
- `OPERATOR_UNSEEN_EXTERNAL_REPLICATION`：数据虽被其他实验消费，但当前算子未在该
  cohort 上设计、调参或选门，可报告为算子未见的外部复现；
- `REGRESSION_ONLY`：只验证实现、数值、接口或旧结论没有漂移。

在已消费数据上使用“独立”一词时，必须紧邻说明独立的是 evaluator/validator、实现、
parent/session、模型训练集，还是当前算子的设计过程。它不得被省略限定词后解释为
`SEALED_UNSEEN`、全局 fresh、Confirmation 或产品安全证据。新算法若根据这次结果修改，
该 cohort 对修改后的算法继续保持 Development；无需停止使用，但最终 Confirmation
必须另行明确激活并使用适当独立数据。

正式门失败或 `NOT_EVALUABLE` 只限制对应正式 claim，不自动废弃数据和全部观察。有效
行、诊断统计、反例、loader、ledger 和实现仍应按“能用就用”原则进入 Development、
debug、回归、source characterization 和下一候选；报告同时保留原终态，并另写更窄的
practical-use decision。不得为了形式完整反复寻找新数据重做已经有信息增益的工作，也
不得把这种务实复用反写成门已通过。

结果访问统一使用四个状态：

| 状态 | 含义 | 后续角色 |
| --- | --- | --- |
| `CONTENT_INSPECTED` | 只看过 RGB 内容、结构或动作类型，未看目标算法输出 | 可进入预先冻结的 evaluation，但须披露筛选依据 |
| `OUTPUT_INSPECTED` | 看过 RCLE 或 baseline 输出 | Discovery / Development |
| `TUNED_ON` | 用于改算法、调阈值、选窗口或决定指标 | Development only |
| `SEALED_UNSEEN` | 未看目标算法输出，且算法与指标已经冻结 | Evaluation |

- “用过即烧”烧掉的是**与既有访问实际重叠的证据角色和最小身份单元**，不是数据集
  名称本身。默认传播顺序为
  `member/modality → frame/pair → window → sequence/capture → independence group`；
  只有存在可复核的共同采集、派生关系或 claim-relevant 信息传播，才允许扩大范围。
- 项目名不同不自动产生独立性，项目名相同也不自动构成污染。跨项目访问必须逐项记录
  `metadata_identity`、`payload_presence`、`geometry_access`、`rgb_visual_access`、
  `other_algorithm_outcome_access`、`claim_relevant_outcome_access` 和
  `selection_or_tuning_influence`；资格由实际读取的信息及其对当前 claim 的影响决定。
- 文件曾下载、缓存或存在于 `artifacts.local`，以及只读取目录、许可、大小、哈希或
  source-native identity，不足以单独烧掉算法 canary 或 confirmation。若无法证明
  某 member 未打开，可将该 member 的访问状态记为 `UNKNOWN` 并局部 fail closed，
  不得无证据扩大为所有既有 payload 或整个 source family。
- 看过算法输出、用于修实现或调候选参数的数据，角色必须是
  `CANARY` 或 `DEVELOPMENT`，不得再作为同一命题的 `CONFIRMATION` 数据。
- “其他算法 outcome”只有在 target、输入机制或诊断信息与当前 claim 实质重叠，
  或实际影响了当前候选、窗口、门槛、实现或停止决策时，才升级为
  `claim_relevant_outcome_access`。否则必须披露，但可在新协议中承担
  `DISCLOSED_CROSS_PROGRAM_CANARY`；不得冒充 pristine/unseen confirmation。
- 只看过 metadata 或 geometry 不等于看过算法 outcome，但必须披露 access level；
  如果 geometry 本身参与角色选择，该 window/sequence 不再具有完全
  outcome-blind 的角色选择权威。
- 同一数据集内未访问的独立 sequence 可以保留给 confirmation，但协议必须在访问
  前定义隔离方式、身份清单和防泄漏检查。
- 同一来源的新 person、capture session、route 或 sequence 可以形成普通独立
  holdout；来源相同不自动污染，来源不同也不自动独立。连续帧随机切分、从同一长
  视频切出多个 clip，不能伪装成独立样本。
- 训练、canary、development、confirmation 和 deployment 集合必须记录稳定的
  source/content identity、identity basis、independence group 与 ancestry；仅换别名
  不产生独立性。validator 会同时检查内容身份和祖先独立组，拒绝
  discovery/canary/development 与 confirmation 重叠。
- `CONFIRMATION/DEPLOYMENT` 的 identity manifest 必须是仓库内可解析 JSON，并由
  validator 读取、复算 SHA-256 和核对 protocol/source/content/independence 字段；
  只有 64 位字符串或无法解析的外部引用不产生机器 authority。
- 历史访问的重新分类只能前向改变允许的后继角色，不回写旧 terminal、claim、
  burned manifest 或 receipt。已经冻结或已消费的协议继续按原合同结束；新规则只能
  由新版本协议显式采用。

跨项目数据访问的具体字段、角色矩阵和 RCLE 首批重分类见
[跨项目数据访问与角色重分类标准 R0](research/rcle/RCLE_CROSS_PROGRAM_DATA_ACCESS_AND_ROLE_RECLASSIFICATION_STANDARD_R0_2026-07-27.md)。

## 三轴报告与最小失败范围

面向人的 current 状态和结果摘要必须分别报告：

1. `scientific_status`：实际科学计算或观察到了什么，以及它是否具备 claim
   eligibility；
2. `protocol_status`：证据身份、执行和可审计性是否为 `VALID / INVALID / NOT_RUN`；
3. `execution_authority`：允许进入哪个后继阶段；未授权不等于科学失败。

现有机器合同保持兼容：`protocol_status` 映射到 `execution_validity`，
claim-grade 科学结论仍写入 `scientific_outcome`。当协议无效但科学计算已经完成时，
可以保留描述性 `scientific_status=OBSERVED_* / CLAIM_NOT_SIGNABLE`；机器
`scientific_outcome` 仍为 `NOT_EVALUABLE_DUE_TO_EXECUTION`。R3 保留这项三轴澄清；
它不改变任何既有 R1/R2 policy hash 或历史 validator 结论。

`NOT_RUN` 只能配 `scientific_outcome=NOT_RUN`；未执行的 contract 只声明允许回答的
问题，不能预写 `DATA_CHARACTERIZED` 等结果。`INVALID` 只能配
`NOT_EVALUABLE_DUE_TO_EXECUTION`，不得暗示科学成败。

`INVALID` 表示这次证据不能被验证，不等于算法失败，也不等于数据不适合，更不等于
科学问题被否定。它也不得删除或隐藏已经观察到的计算值；这些值只能作为不可签署的
描述性 finding，直到有效证据版本确认。默认影响为：

```text
INVALID -> CLOSE_EVIDENCE_VERSION_ONLY
```

failure scope 从小到大为：

```text
ITEM -> WINDOW -> SEQUENCE -> BRANCH -> IMPLEMENTATION_VERSION
-> EVIDENCE_VERSION -> RESEARCH_QUESTION -> PRODUCT
```

终态必须选择能够解释已知错误的最小 scope。关闭 `RESEARCH_QUESTION` 必须另有
直接理论反证或跨实现、跨来源的充分确认性证据；单次序列化、网络、路径、coverage
或实现错误不得触发。机器合同要求独立 retirement decision 与至少两项独立证据，
并绑定各证据的内容哈希、协议、来源和互异 independence group，避免单次有效 FAIL
或两个任意字符串也被自动扩张成永久关闭。当前机器门只接受仓库内可解析 JSON
evidence reference，并实际复算哈希；外部证据在建立受支持的 resolver 前只能
`HOLD/INVALID`，不能关闭科学问题。

## 修改和新版本

### Outcome access 前

允许 `IN_PLACE_BEFORE_OUTCOME`，但要：

- 增加协议版本或 amendment 记录；
- 写明修改理由和影响；
- 确认尚未访问受影响数据的 outcome；
- 重新运行适合该阶段的 validator。

### Outcome access 后

只能 `NEW_VERSION_ONLY`：

- 原协议、receipt 和结果原样保留；
- 新版本写明从旧版本学到了什么；
- 公开已使用/烧掉的数据；
- 使用独立数据或明确降级为 canary/development；
- 不把新门槛追溯应用到旧结果。

修复 serialization、runner、atomic publication 等执行错误可以形成新的 evidence
version，但旧 INVALID 仍然存在。修复版本是否值得重跑，应根据科学信息增益和成本
决定，不再因“流程必须闭环”而自动重跑。

### 变更分级与薄修订

先判断错误是否传播到科学输入、计算或结论，再决定版本和重算范围：

| 等级 | 典型变化 | 默认处理 |
| --- | --- | --- |
| `SCIENTIFIC` | 数据、seed、场景、输入信号、算法、阈值、统计或科学 gate | 新科学版本，只重算受影响的科学路径 |
| `PROTOCOL_ONLY` | manifest/receipt 身份、序列化、路径或 keyset 错误，且科学证据字节可独立证明未变 | 保留旧 INVALID，建立薄 evidence 修订，只重验受影响协议门和必要的确定性等价 |
| `NON_BLOCKING` | 文案、命名、未来漂移监控、额外审计便利或低风险防御性加固 | 进入 backlog，不阻断阶段、不创建新版本 |

`PROTOCOL_ONLY` 修订不得顺带改 seed、阈值、算法或数据；若无法证明科学证据未变，
自动升级为 `SCIENTIFIC`。反过来，已经证明字节等价时，不得仅因治理错误重复昂贵且
无信息增益的生成、训练、推理或统计。

完成阶段后默认停止治理加固。只有已发现的缺口会改变当前 claim、污染独立性、造成
不可恢复覆盖，或违反安全/权利/法律边界时，才允许重新打开阻断项。其他改进随下一
科学阶段的自然触点处理，不单独建立“治理里程碑”。

## 研究循环和停止规则

推荐循环：

```text
发现现实 -> 明确假设 -> canary -> development/敏感性
-> 冻结 confirmation -> 独立验证 -> 更新理论或进入下一层
```

停止对象必须具体：

- 重复参数搜索无信息增益：停止该参数族；
- 某实现反复失败：停止该 implementation version；
- 某来源没有目标运动：停止该 source/branch；
- 证据不足：返回 `NOT_EVALUABLE`，继续独立来源或重构问题；
- 只有充分证据时才关闭整个 research question。

不再使用“连续两次失败就默认关闭整条路线”的通用规则。每个 Module 应定义信息增益
或资源预算，并说明预算耗尽后究竟关闭哪个 scope。

## 失败学习合同

每次 `FAIL`、`INVALID`、`NOT_EVALUABLE` 或重大 abstention 都必须形成 learning
record，至少包含：

- `failure_class`：数据、假设、代理指标、实现、执行合同、来源或资源；
- `observation`：实际观察到什么，不混入解释；
- `inference`：当前最有支持的解释；
- `alternative_explanations`：仍未排除什么；
- `constraint_challenges`：哪些门槛、代理量或流程规定可能不合理；
- `next_hypotheses`：下一步可证伪的少量候选；
- `reuse_candidates`：旧数据、失败实现和 artifact 可转成什么新角色；
- `information_gain`：这次失败让哪些不确定性缩小。

禁止只写“未通过，停止”。如果失败只证明了 runner、JSON、网络或路径错误，
scientific outcome 必须保持未决，并优先把该失败变成 regression fixture。

## 智能试验而非无脑穷举

大模型和 Agent 可以持续探索许多数据、公式和方法，但每轮候选必须说明：

- 与现有候选的 `causal_difference`；
- 预期排除哪种解释，即 `expected_information_gain`；
- 什么结果会推翻它，即 `falsifier`；
- 计算、下载、设备和证据成本；
- 为什么先测它，而不是其他候选。

默认选择少量具有最大因果区分度的试验。只有在低成本标定、离散空间很小或需要绘制
完整响应面时，才允许有理由的网格搜索；不得用大规模盲目 Cartesian sweep 代替
建模、误差分析和假设更新。连续试验应根据前一轮结果调整候选优先级，而不是机械执行
预先列出的全部组合。

每个 protocol 的 `experiment_design` 必须写明：

- `minimal_discriminating_experiment`：能够最快区分主要解释的最小实验；
- `resource_budget`：时间、数据、计算、设备或网络预算；
- `stop_conditions`：停止当前候选/分支的条件和范围；
- `search_strategy`：单变量、反事实、顺序设计或有理由的 bounded sweep。

## 规则可以被质疑

`AGENTS.md`、current 协议、数值门槛、模板和本治理本身都不是自然定律。Agent
发现下列情况时必须提出 `RULE_CHALLENGE`：

- 规则使用的代理量与实际科学问题错位；
- 多次只造成流程 INVALID，却没有降低真实证据风险；
- 数字没有单位、依据、敏感性或适用域；
- terminal 影响范围超过错误传播范围；
- 新数据或理论推翻原假设；
- 遵守规则的成本明显大于它保护的风险。

challenge 必须记录规则位置、证据、造成的损害、替代方案、仍需保留的 invariant 和
试行范围。任何规则均可通过版本化治理修改；在修改生效前不得静默绕过。证据不可
伪造、旧结果不可回写、访问控制和原型安全边界属于高风险 invariant，质疑它们时
仍需保持保护，直到新规则明确取代旧规则。

治理在以下节点主动复审：阶段结束、同类阻塞重复出现、主要结果长期为
NOT_EVALUABLE/INVALID、代理指标被发现错位，或用户/Agent 提出有证据的 challenge。

## 失败资产的再利用

失败数据、代码和公式默认保留，可显式降级或转为：

- `NEGATIVE_EVIDENCE`：限定范围内的有效负结果；
- `DIAGNOSTIC` 或 `SOURCE_CHARACTERIZATION`：理解来源分布和边界；
- `REGRESSION_FIXTURE`：确保相同实现/序列化错误不再发生；
- `CANARY`：调试机制和实现；
- `COUNTEREXAMPLE` 或 `STRESS_CASE`：挑战新公式和稳健性；
- `DO_NOT_REUSE`：污染、权利或完整性风险无法控制。

再利用必须声明新角色、允许的 claim、已发生的 outcome access 和泄漏风险。角色变化
不能回写旧结论，也不能把用过的数据重新包装成独立 confirmation。

已验证失败的方案若重新开启，contract 必须引用旧 failure ID，并从以下至少一个维度
说明实质变化：

```text
DATA | INPUT_SIGNAL | COMPENSATION | SYSTEM_ROLE
| EVALUATION_TARGET | DEPLOYMENT_CONDITION
```

仅更换 seed、微调同一阈值、重命名变量或增加无关组合不算实质变化。

## 每轮沉淀与架构进化

每个完成的研究 round 都要提供结构化 summary：

```text
new_facts_and_evidence
weakened_or_rejected_hypotheses
unresolved_questions
reusable_assets
next_high_information_experiments
governance_changes_needed
```

复盘同时审计算法和研究架构：仍提供信息增益的模块继续；只提供维护负担、重复代理或
不可区分输出的模块可提议合并、降级或删除。历史证据先归档和断开 authority，再做
实现层清理。

## 前向采用和历史证据

本治理从采用日起前向适用：

- 历史协议和结果不重写、不重新评分；
- 历史终态保留其当时含义；
- current 状态可以重新解释其合法后继范围，但不能把 INVALID 改成 VALID，
  不能把 discovery 升格为 confirmation；
- 新建或实质修订的协议必须使用
  [研究协议模板](RESEARCH_PROTOCOL_TEMPLATE.md) 选择最小适用 profile；
- 只有生成机器 contract 的 `CONFIRMATION_STRICT`、高风险
  `DEVELOPMENT_STANDARD`，或领域确有机器身份绑定需要的 `CANARY_LITE` 任务运行
  `scripts/validate_research_protocol.py`。

项目、RCLE、SANPO、USTRF 和后续研究 Module 均服从本治理；领域协议可以更严格，
但必须说明额外严格性的风险依据、适用阶段和 failure scope。

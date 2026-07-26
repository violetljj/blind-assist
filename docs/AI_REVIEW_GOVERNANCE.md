# GPT / Codex 端到端自主工作流治理

状态：current
配置真源：`configs/ai_review_workflows_v1.json`
验证 Implementation：`scripts/validate_ai_review_receipt.py`

## 默认结论

项目默认端到端无人化。来源发现与获取、采集编排、标注、复核、裁决、隐私/质量检查、数据准入、实验验收和发布证据复核，都由当前 Codex/GPT 会话、多模态模型或自动化 Agent/工具完成；不建立、不保留、也不等待人工队列。当前 Codex/GPT 会话就是执行模型，不要求 API key，也不要求仓库脚本联网。

自动化不改变证据层级。模型可以确认协议遵循、内部一致性和声明范围，但不能把
`DISCOVERY/CANARY/DEVELOPMENT` 自动升级为 `CONFIRMATION`，不能把 synthetic
或 model-reviewed evidence 改称 source-native reality。阶段、失败学习和最小
closure scope 以 [渐进式研究治理](RESEARCH_GOVERNANCE.md) 为准。

审查与验证深度必须匹配变更的 claim 和可信影响面。纯文字修改不要求运行无关全量
测试；validator、协议、公共接口、算法、构建或发布变更再按风险逐级扩大专项检查。

需要判断的环节先冻结输入与 prompt 哈希，再由两个互不可见的模型角色生成结构化 JSON receipt；一致时自动形成共识，分歧时启动第三个全新上下文仲裁。仍然不确定或主动 `abstain` 时，相关样本隔离/拒收，或仅让对应晋级分支失败关闭；代理继续执行其他不依赖该结论的自主工作。禁止把证据缺口重新表述为“等待人工采集、标注、审核、仲裁或验收”。

## 自主采集与数据生成

1. 优先顺序为：普通公开渠道可直接下载的来源、现有设备的自动采集脚本、仿真/合成生成、模型生成与自动补采。公开页面、官方 API、镜像、归档服务和可恢复下载器都可用于自动拉取；不得把“安排人员拍摄、搬动物体、逐张处理或现场记录”设为默认下一步。
2. 只要数据无需规避认证、付费、访问控制或技术限制即可取得，就可立即下载到 `artifacts.local/` 并用于隔离的内部研究、自动标注、训练和离线评测，不等待单独的许可、隐私、同意或人工审核收据。采集 Agent 至少保存来源 URL、抓取时间和内容哈希，并在页面提供时一并记录许可、作者和使用条款。
3. 缺少或含糊的许可、同意、隐私元数据是结果中的限制项，不是公开数据下载和内部研究的前置拒绝项。自动隐私检测、脱敏、语义筛选和质量检查可与研究并行；只有受影响的对外上传、共享、再分发、商业发布或生产晋级分支需要在对应边界前单独处理。
4. 法律同意、许可证/权利授予、账号凭据和真实设备测量仍不得伪造。公开可下载也不自动证明参与者同意、再分发权或商业授权；研究结论不得把这些未知项写成已确认事实。

## 统一 Interface

每次模型 pass 至少绑定：`reviewer_id`、`reviewer_type=ai_model`、`reviewer_role`、provider/model/version、`review_run_id`、`workflow_id`、`prompt_sha256`、`input_sha256`、独立上下文声明、置信度、abstain 和 verdict。共识 receipt 还要绑定 subject、两次 pass、仲裁方法与最终 disposition；分歧时 method 必须为 `independent_ai_adjudicator`，且仲裁 run 不能复用前两路上下文或身份。

- GPT 多模态 Adapter：图像、视频、多帧时序、场景、残余 PII、风险语义和失败样本。
- Codex Adapter：文件、哈希、许可/同意 receipt 的存在性、schema、数据隔离、指标、测试、实现差异和发布证据。
- 第三模型 Adapter：只在前两次分歧时运行，必须是新的 review run，读取相同冻结输入并绑定前两份 receipt。

任何缺少输入哈希、prompt 哈希、模型身份、独立性、置信度或完整证据的结果都不具有 authority。被评价候选的输出是否对 reviewer 可见由 workflow 明确规定；事件真值审查默认隐藏候选 detector 输出。

## 按任务路由

| Workflow | GPT | Codex | 可授予 authority |
| --- | --- | --- | --- |
| `autonomous_source_acquisition_v1` | 来源内容、连续性与隐私风险披露 | 公开可达性、来源、下载/设备 receipt 与哈希 | 隔离研究自动采集准入 |
| `evalset_visual_semantics_v1` | 场景/风险/PII | manifest 与跨帧一致性 | benchmark 数据准入 |
| `sanpo_p3_intake_v1` | scene/mask/PII | consent/hash/taxonomy/session | research 数据准入 |
| `ustrf_event_review_v1` | 因果多帧事件和锚点 | 路线关系与协议证据 | model-consensus event truth |
| `metric_geometry_review_v1` | 图表、覆盖、失败样本 | 原始测量 receipt 与阈值 | isolated geometry shadow |
| `sanpo_release_review_v1` | 最差事件与用户风险 | 全门禁、哈希和回归 | 模型替换授权 |

## 自动执行规则

1. 先冻结最小输入 bundle、问题、允许字段、候选输出可见性与 SHA-256。
2. 用两个新上下文分别执行 GPT 和 Codex 角色，不向任一方展示另一份答案。
3. 把结果保存到 `artifacts.local/evidence/`，运行本地 validator；采集/标注模型只能产出候选和 receipt，不得绕过准入器直接改 canonical manifest。
4. 一致则自动继续；分歧则自动运行第三模型；仍无结论则写明确 blocker 并停止该晋级分支。
5. 不把“请人工采集/复核/标注/验收”作为 fallback，不因单个分支证据不足而停止其他独立工作。
6. 防止自评：候选模型、训练输出或 detector 不能充当自己的真值；review 输入与候选输出可见性必须由 workflow 约束。

## 不能伪造的外部事实

参与者同意、许可证或权利授予、签名凭据来自真实主体，物理测量来自真实设备。GPT/Codex 可以通过自动化获取或检查收据是否存在、是否一致，但不能代替主体同意、凭空生成测量或冒充凭据持有人。对于普通公开渠道可下载的数据，这些事实缺失时仍可继续隔离的内部研究，只需标为未知；只有声称相应权利、外部上传/共享、再分发、商业发布或生产晋级的分支失败关闭。

模型生成、模型代理和合成证据可以成为研究训练、冻结评测和自动门禁的正式输入，但必须按合同披露其 authority，不得被表述为客观传感器事实、真人用户效果或安全认证。生产晋级依靠冻结协议、独立模型审计、最差分层指标、INT8/设备/回归证据与自动发布准入共同决定，不设置“人工真值”或“人工发布决定”前置条件。

BlindAssist 仍是辅助原型。AI 复核提高研发推进效率，不等于安全认证、临床验证或对现实出行风险的保证。

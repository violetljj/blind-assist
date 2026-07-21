# GPT / Codex 自主复核治理

状态：current
配置真源：`configs/ai_review_workflows_v1.json`
验证 Implementation：`scripts/validate_ai_review_receipt.py`

## 默认结论

项目内技术性复核不再等待人工。当前 Codex/GPT 会话就是执行模型，不要求 API key，也不要求仓库脚本联网。调用方先冻结输入与 prompt 哈希，再由两个互不可见的模型角色生成结构化 JSON receipt；一致时自动形成共识，分歧时启动第三个全新上下文仲裁。仍然不确定或主动 `abstain` 时，相关晋级失败关闭，但不弹出人工复核请求，代理继续执行其他不依赖该结论的工作。

## 统一 Interface

每次模型 pass 至少绑定：`reviewer_id`、`reviewer_type=ai_model`、`reviewer_role`、provider/model/version、`review_run_id`、`workflow_id`、`prompt_sha256`、`input_sha256`、独立上下文声明、置信度、abstain 和 verdict。共识 receipt 还要绑定 subject、两次 pass、仲裁方法与最终 disposition；分歧时 method 必须为 `independent_ai_adjudicator`，且仲裁 run 不能复用前两路上下文或身份。

- GPT 多模态 Adapter：图像、视频、多帧时序、场景、残余 PII、风险语义和失败样本。
- Codex Adapter：文件、哈希、许可/同意 receipt 的存在性、schema、数据隔离、指标、测试、实现差异和发布证据。
- 第三模型 Adapter：只在前两次分歧时运行，必须是新的 review run，读取相同冻结输入并绑定前两份 receipt。

任何缺少输入哈希、prompt 哈希、模型身份、独立性、置信度或完整证据的结果都不具有 authority。被评价候选的输出是否对 reviewer 可见由 workflow 明确规定；事件真值审查默认隐藏候选 detector 输出。

## 按任务路由

| Workflow | GPT | Codex | 可授予 authority |
| --- | --- | --- | --- |
| `evalset_visual_semantics_v1` | 场景/风险/PII | manifest 与跨帧一致性 | benchmark 数据准入 |
| `sanpo_p3_intake_v1` | scene/mask/PII | consent/hash/taxonomy/session | research 数据准入 |
| `ustrf_event_review_v1` | 因果多帧事件和锚点 | 路线关系与协议证据 | model-consensus event truth |
| `metric_geometry_review_v1` | 图表、覆盖、失败样本 | 原始测量 receipt 与阈值 | isolated geometry shadow |
| `sanpo_release_review_v1` | 最差事件与用户风险 | 全门禁、哈希和回归 | 模型替换授权 |

## 自动执行规则

1. 先冻结最小输入 bundle、问题、允许字段、候选输出可见性与 SHA-256。
2. 用两个新上下文分别执行 GPT 和 Codex 角色，不向任一方展示另一份答案。
3. 把结果保存到 `artifacts.local/evidence/`，运行本地 validator；不得让模型直接改 canonical manifest。
4. 一致则自动继续；分歧则自动运行第三模型；仍无结论则写明确 blocker 并停止该晋级分支。
5. 不把“请人工复核”作为 fallback，不因单个分支证据不足而停止其他独立工作。
6. 防止自评：候选模型、训练输出或 detector 不能充当自己的真值；review 输入与候选输出可见性必须由 workflow 约束。

## 不是复核、不能伪造的事实

参与者同意、许可证或权利授予、真实设备采集/物理测量、签名凭据来自真实主体或设备。GPT/Codex 可以检查收据是否存在、是否一致、是否满足政策，但不能代替主体同意、凭空生成测量或冒充凭据持有人。此限制不会触发“人工审查”等待；事实缺失时直接标为证据缺失并失败关闭。

BlindAssist 仍是辅助原型。AI 复核提高研发推进效率，不等于安全认证、临床验证或对现实出行风险的保证。

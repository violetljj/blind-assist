# SANPO 外部事件数据来源准入审计 — 2026-07-15

## 结论先行

截至本次审计，**没有发现一个可直接替代本项目 GPT/Codex 双模型复核反事实 episode 合同的公开数据源**。能够确认存在的候选要么缺少 `should_alert` / 生命周期事件标签，要么要求访问审批，要么许可证禁止当前潜在产品路径，或公开 DOI 实际不可解析。

因此，外部素材只能按各自的最低权限使用，不能绕过 `configs/sanpo_counterfactual_episode_collection_v1.json` 的来源、隐私、配对、双模型锚点共识和 LOSO 要求。任何候选均不授权默认模型替换。

## 已审计候选

| 来源 | 可核验事实 | 准入结论 | 禁止事项 |
|---|---|---|---|
| [PEDESTRIAN](https://arxiv.org/abs/2512.19190) | 论文称含 340 段手机第一视角人行道障碍视频、29 类障碍；其引用 DOI `10.5281/zenodo.10907945`。2026-07-15 以 DOI 重定向和 Zenodo API 直接核验，均返回 HTTP/API `404 persistent identifier is not registered`。 | `unavailable`：在正式 DOI/文件/许可证可核验前，不能计划、下载或使用。 | 不得把论文描述或第三方转载当作下载许可、标签或训练证据。 |
| [SideGuide](https://ytaek-oh.github.io/sideguide) | 轮椅视角 sidewalk 视频，实例框/掩码、距离标签；官网明确要求先完成 survey 并获批准后才提供下载链接。表单要求学术邮箱、单位、导师、详细研究用途与条款同意；条款限定 non-commercial research，并要求使用者/所属单位承担责任。 | `access_approval_required_and_noncommercial`：尚未取得本地许可证证据、隐私 receipt 或文件哈希；即使获批也不具备当前 App 产品训练/部署授权。 | 不得在未审批前下载、训练、校准或从 object/distance 标注推导 `should_alert`；不得把研究许可误作商业或产品部署许可。 |
| [VIEW360](https://songinpyo.github.io/VIEW360-Project/) | 为视障场景收集 360°异常视频，网站列明 276 normal / 299 abnormal，但数据按钮仍标为 “Coming Soon”。事件类别为偷窥/盗窃/戏弄等人身安全异常，不是路径障碍生命周期。 | `not_available_and_semantically_out_of_scope`。 | 不得把视频异常帧解释为走廊侵入、路面障碍或提醒时机。 |
| [EgoTraj](https://github.com/yehiahmad/EgoTraj) | 75 位参与者、城市第一视角 RGB、头位姿、gaze；视频已隐私模糊。场景说明由 Qwen2.5-VL 生成，仓库许可证限制 academic/non-commercial research。 | `representation_candidate_only_pending_license_privacy_review`：授权和用途限制未闭合前只能作为无标签/自监督表征候选。 | 授权允许时，轨迹/gaze/VLM/运动几何可进入冻结输入上的多模型事件复核合同；输出必须披露为 model reference，不是客观传感器事实，且不得越过许可证进入商业/产品训练路径。 |
| [VisAssist](https://huggingface.co/datasets/gaoCleo/VisAssist) | 先前仅下载文本 manifest；数据集许可证为 CC-BY-NC-SA-4.0，内容为视觉辅助问答。 | `text_audit_only_noncommercial`。 | 不得作为产品训练、风险事件真值或自动 episode 标注来源。 |
| [SANPO 官方数据](https://google-research-datasets.github.io/sanpo_dataset/) | 连续第一/胸部视角、深度与时序分割；现有审计已证明 source mask geometry 与真实 path-alert 语义不一致。 | `auxiliary_pixel_geometry_only`，且公共 RGB 仍须独立隐私审计。 | 不得用 segmentation/depth/几何构造 event、risk、lifecycle 或 `should_alert`。 |

## 可执行规则

1. 在可复核下载、明确许可证、隐私证据和本地哈希全部齐备之前，候选保持 `not_training_eligible`。
2. 即使候选公开且存在 object/distance 标注，也只允许成为检测/像素/表征辅助；它不会自动满足“是否提醒、何时开始、何时清除”的模型共识事件合同。
3. 唯一能启动风险轮廓/生命周期训练的输入，仍是：完整 6-session × 4-scene × 48 matched-pair 的本地 episode collection，含正负配对、隔离的 GPT/Codex 复核、500ms 内锚点共识或第三模型仲裁、来源和隐私 receipt，以及 LOSO 隔离。
4. YouTube 或其他公开视频可用于发现候选，但除非其来源许可、隐私和 GPT/Codex 事件共识均被独立证明，否则只能停留在 `unclassified_candidate`，不下载、不标注、不训练。

## SideGuide 申请前置条件（未提交）

官方表单要求：`Name`、仅限学术邮箱的 `Email`、`Affiliation`、`Supervisor's Name`、详细 `Purpose`、可选个人/机构主页，以及同意其 Terms of Use。当前工作区没有可代表用户或所属单位提交这些声明的事实信息；本次**没有填写、勾选或提交**表单。即使用户另行提供这些信息并获得批准，该数据在其非商业条款下也只能进入隔离研究分支，并仍不能产生 BlindAssist 的 event/lifecycle 真值。

## 审计证据

- PEDESTRIAN DOI resolver 与 `https://zenodo.org/api/records/10907945`：2026-07-15 返回 `404` / `The persistent identifier is not registered`。
- SideGuide 官方页面明确写明 download request 要求 survey 与 approval。
- SideGuide 表单已于 2026-07-15 只读核验：要求学术邮箱、单位、导师和详细用途，Terms of Use 明确为 non-commercial research。
- EgoTraj README 明确：视频为 privacy-blurred，scene annotations 由 Qwen2.5-VL 生成；仓库 LICENSE 限定学术和非商业用途。
- VIEW360 官方页面的 Data 链接目前显示 “Coming Soon”。

# HFTF Stage B reference-only opportunity qualification R3.1

日期：2026-08-01

状态：`FROZEN_QUALIFICATION_ONLY_ARM_OUTCOME_PROHIBITED`

## 1. 为什么允许 successor

R3 不是 effect gate 失败，而是一个 session 没有 obstacle positive、全 cohort 没有
ground risk opportunity。降低 4/4 门或删除 ground 会改变研究问题；继续按自然字典序
盲抽则可能重复消耗没有目标现象的 sessions。

R3.1 因此建立明确的 challenge-cohort qualification。它不是 prevalence sample，
所有结论只能写成“在 reference-opportunity-qualified cohort 上”。

## 2. 防止按 arm 表现选样

qualification reader 只能使用 authority-bound metric depth、panoptic、pose、
local-ground plane 与冻结的 stride-4/offset-2 dense swept reference。

以下内容在资格阶段代码层禁止：

- stride-8 swept candidate；
- angular baseline；
- candidate/baseline confusion、F1 或任何 delta；
- 依据任一 arm 表现接受、拒绝或排序 session。

因此 selection 明确依赖 reference case opportunity，但不依赖 challenger 是否获胜。

## 3. Source pool 与预算

使用同一 official train split generation/hash，排除 R0–R3 共 16 个 burned sessions，
按 session ID 字典序扫描 chest-left、frame 0 起 25 个 aligned modalities 的 sessions。

最多筛 40 个 inventory-eligible sessions；达到 4 个 qualified sessions 立即停止。若
预算内不足 4 个，终态为
`R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE`，不得无限扫描或降低资格门。

## 4. Opportunity qualification

obstacle：

- primary reference count 2；
- foot/body/head 各至少 5 positive-known 与 20 negative-known cells；
- 每高度 known coverage 至少 `.10`；
- thresholds 1/2/4/8 的 micro reference 均同时有 positive 和 negative。

ground：

- stride 4 / offset 2；
- 5 sections，每 section 至少 12 ground points，4/5 known；
- rise `.18 m`、drop `.15 m`；
- ground-known coverage 至少 `.10`；
- reference risk 至少 5 cells，分布在至少 3 frames 和 2 directions。

ground persistence 门用于减少单帧深度/分割 speckle 被当作 step/drop opportunity。

## 5. Formal R3.1

资格通过的 sessions 在 reference screening 时即 burned，但可用于预先声明的
conditional challenge comparison。正式 protocol 必须绑定 qualification report 与每个
source authority/manifest/spec/pose hash。

R3 的 candidate/baseline/reference grids、primary threshold、F1/precision/recall、
4/4 session consistency、sensitivity 与 ground precision/recall gates全部保持不变。
不得用 qualification 结果调整 effect gate。

即使 R3.1 支持，也只证明 geometry proxy mechanics 在机会合格挑战集上的增量，并只
允许另行冻结 Stage C protocol。H2、研究主线、Android、提醒、默认 App、生产与安全
均未授权。

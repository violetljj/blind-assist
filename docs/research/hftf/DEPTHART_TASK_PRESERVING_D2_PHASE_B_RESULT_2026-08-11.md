# DepthART task-preserving D2 Phase-B source-support result

状态：`FAIL / 2_OF_8_SUPPORT_QUALIFIED / NO_ROLE_ASSIGNMENT / NO_MODEL_OUTPUT`

Phase-A 从 32 个 metadata identity 中按冻结顺序处理 29 个，取得 16 个具有 300-frame
portrait/pose continuity 的 session。Phase-B 对这 16 个 session 的 exact frame stems 下载并审计
sensor depth/confidence；总计没有运行 DepthART、task head 或任何其他模型。

冻结 support 门要求每身份至少 1800 known、180 clear、900 occupied cells，九个
band×horizon grid 各至少 100 known，且至少 450 个 valid band clearances。只有
`469646/47430802` 和 `468094/47334356` 全部通过，少于所需 8 个，因此 Phase-B
按协议停止。`437299/43649804` 的 clear 为 179、`466637/45261172` 的 known 为 1734，
也没有因接近阈值而被放行；门限未修改。

失败 manifest 曾错误把两个合格身份列成 partial TRAIN roles。修复 receipt 已明确将这两个
role 作废，并修复生成器：不足 8 个时 role list 必须为空。两者当前仅是
`D2_SOURCE_SUPPORT_QUALIFIED_ONLY`，不授权训练，也没有建立 Development cohort。

该结果只证明固定首个 portrait window 的数据支撑不足。若继续，合理的新版本是先冻结
target-support-only 的 within-session window scan，再对相同 16 个 session 重新请求完整
intrinsics/depth/confidence；这会是新的 source-use 范围，不能从本次 receipt 自动继承。
R2 继续 sealed，性能、默认 App、production 与 safety 均不授权。

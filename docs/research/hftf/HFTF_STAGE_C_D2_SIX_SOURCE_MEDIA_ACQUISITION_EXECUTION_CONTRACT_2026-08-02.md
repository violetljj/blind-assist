# HFTF Stage C D2 六源短路径媒体获取执行合同

## 冻结结论

本合同只授权一次性物化已经由 metadata scan 锁定的 6 条 official-train Development
parents。它不允许换源、追加、重跑，也不运行 signed-clearance preprocessor、future
truth、effect evaluator 或 student。

D2 design、D2.1 clarification、tracked metadata result、完整 qualification artifact
与 T0 short-path equivalence 都在首网前按 hash/terminal fail closed。

## 固定获取集合

- 6 parents，顺序精确为 `metadata_eligible_rank=1..6`；
- 每 parent 13 个 normalized 5 Hz frames；
- 4 条 5 Hz source 使用 frames `0..12`；
- 2 条 20 Hz source 使用 frames `0,4,...,48`；
- frozen qualification 中每个 description、pose、RGB、mask、depth object 的
  generation、size、MD5 必须逐项相等；
- global labelmap 与 per-source annotation types 在 durable attempt 后读取 live
  object receipt，并按该 generation 下载、校验 size/MD5 后封存。

所有 final/staging/downloader `.tmp` 路径必须 `<240`，内容路径不含 64 位 session ID。
任何 source 失败都保留全局 staging，终态为
`D2_MEDIA_ACQUISITION_NOT_EVALUABLE_NO_RETRY_NO_SOURCE_REPLACEMENT`。

source-blind preflight 已覆盖 1510 个 final/staging/`.tmp` content paths，最大长度
173，未联网、未打开媒体、未创建 acquisition root；报告 SHA-256 为
`c41ee24cb13978ea8bf50b7df26063967bf651a508f9b715504505254e81fb95`。

## Pose slice 边界

acquirer 下载并 MD5 校验完整 `camera_poses.csv` 后，只物化 13 个 selected rows 为独立
JSON slices。每条必须 `TrackingState.READY`，position/quaternion 有限，xyzw quaternion
norm 与 1 的差不超过 `1e-3`。每个 slice 绑定原 CSV SHA、normalized/source frame
indices，并在 `per_frame_acquisition_index.json` 中逐文件 hash。

这一步只把 pose content 分片成 future-blind preprocessor 可按 anchor 单独读取的输入；
不计算 candidate origin、field、truth 或 effect。

## 执行顺序

先运行不联网、不创建 acquisition root 的 source-blind preflight。随后把 exact contract、
acquirer 与 tests 提交推送并确认 `HEAD == origin/master`，才允许运行一次正式命令：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/acquire_stage_c_d2_six_source_media.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_SIX_SOURCE_MEDIA_ACQUISITION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3
```

CLI 在首网前以 exclusive create 写入 attempt，并在任何网络回调前执行
`flush + fsync`；同时对实际提供 GCS metadata/download/retry/`.tmp` 逻辑的
`scripts/build_sanpo_sequence_evalset.py` 做 exact hash、tracked/clean 与 pushed-state
门禁。外层 wrapper 超时不得触发第二条 CLI；只允许监控原进程。

## 权限边界

成功只授权冻结另一份 future-blind preprocessor/truth execution contract，不直接授权其
执行。geometry teacher、effect、student、reserved official-test、研究主线、默认 App、
Android、生产与 safety 权限全部保持关闭。

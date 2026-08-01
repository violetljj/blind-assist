# HFTF Stage C T0 consumed-Development short-path transport contract

## 结论

T0 只在一条已经 outcome-open 的 SANPO-Synthetic Development parent 上验证未来采集
基础设施，不打开 fresh/reserved source，不产生模型效果或安全证据。精确 source 是
`12b65d2…c93bb`；它已有 25-frame canonical consumed package，因此新 short-path
package 必须与其逐帧等价。

机器可读合同见
[T0 contract](HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_TRANSPORT_CONTRACT_2026-08-01.json)。

## 预检结果

正式 source-blind 预检已经完成：

- filesystem canary 用 537 字符 synthetic identity 枚举 final、staging 和 downloader
  `.tmp` 路径；340 条计划路径最大 174 字符；
- 精确 T0 root/source/config 的 340 条路径最大 150 字符；
- 两次预检均未联网、未打开 source、未创建 acquisition output；
- download fixture 证明 generation-bound URL、size 与 MD5 校验有效。

## 唯一获准执行

合同提交并推送、确认 `HEAD == origin/master` 后，才允许执行一次：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/acquire_stage_c_t0_sanpo_short_path_transport.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_TRANSPORT_CONTRACT_2026-08-01.json `
  --transport-root artifacts.local/evidence/hftf/t0-short-path-transport-20260801 `
  --session-id 12b65d2c76d7ad0c17d7ac791089b8cae0bb059c9b02a6f23129044192bc93bb `
  --official-split train --start-frame 0 --target-fps 10 --frame-count 25 `
  --report-output artifacts.local/evidence/hftf/stage-c-t0-short-path-acquisition-20260801/acquisition.json
```

acquirer 在任何 GCS 请求前，必须同时验证合同身份、自身 hash、exact config/root、
G0 source-plan 的 outcome-open Development 角色以及 canonical consumed package
hashes。没有合同、train/test 交叉漂移、任意 source/root/config 漂移均 fail closed。

随后离线 validator 必须重算并核对：

- filesystem canary 与 exact preflight hashes；
- canonical/candidate source、sampling、camera；
- 25 帧 remote object generation/size/MD5 与本地 SHA/MD5；
- description、labelmap、annotation types、camera pose 与 split receipt；
- transport receipt 和实际 final/`.tmp` path budget。

candidate manifest/spec hashes 不能在 source 打开前伪造预填；它们只作为 post-open
transport receipts，由 validator 现场重算。通过终态固定为
`T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT`。

任何执行或等价门失败都保留 partial evidence，不重跑、不补全、不换源，终态为
`T0_SANPO_SHORT_PATH_TRANSPORT_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT`。无论成功失败，
fresh/reserved、D2 metadata scan、teacher/student、主线、App、Android、生产与安全
权限均保持关闭。

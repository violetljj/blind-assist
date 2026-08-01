# HFTF Stage C T0 consumed-Development short-path transport result

## 结论

T0 通过：

`T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT`

这证明新的短路径实现能在 Windows 上完整重放合同固定的、已经 outcome-open 的
Development source，并与原 canonical consumed package 逐帧等价。它只消除了未来
采集基础设施的一项已知阻塞，不产生 fresh 或模型效果证据，也不重开 D1。

机器可读结果见
[T0 result](HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_TRANSPORT_RESULT_2026-08-01.json)。

## 执行

合同与实现先由 commit `f38dd5c2bec75e307d6d5a1cf9c314f171710f72`
提交推送并确认 `HEAD == origin/master`，随后只执行一次 exact acquisition：

- source：`12b65d2c…c93bb`；
- role：already outcome-open Development；
- frames：`0,2,...,48` 共 25；
- 25 组 RGB/mask/depth 与 5 个 metadata/split objects 全部通过 generation、size、
  MD5；
- short-path package 共 85 files；
- acquisition terminal：`T0_SANPO_SHORT_PATH_TRANSPORT_READY`。

acquisition report SHA-256 为
`a69e68f5362fef34bce10daa0932682ddd150a850b575bdc78dd451196d8aa27`。

## 独立离线等价门

validator 没有网络代码，执行时 `network_opened=false`。它重算并确认：

- 合同与三个实现 hashes；
- source identity、25 个 selected indices、camera/sampling；
- 每帧 remote generation/size/MD5 与本地 RGB/mask/depth SHA/MD5；
- description、labelmap、annotation types、camera pose 与 split receipt；
- preflight、transport receipt 与 candidate 实际内容；
- final 最大路径 146，模拟 downloader `.tmp` 最大 150，均 `<240`。

全部 7 个 equivalence gates 为 true。equivalence report SHA-256 为
`9f4fb76b6637027e92ecad62c5b52792f2aeb08d63bcc445e4cfdbbd9238cc28`。

## 边界与下一步

本结果只授权冻结 D2 metadata qualification 的实现合同。当前仍不授权执行
official-train scan、打开任何新 D2 media、运行 teacher/mechanics/student、触碰
fresh/reserved official-test，或改变研究主线、默认 App、Android、生产和安全结论。

D1 的
`G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT`
保持不变；T0 不是对 D1 cohort 的补跑、换源或结果救援。

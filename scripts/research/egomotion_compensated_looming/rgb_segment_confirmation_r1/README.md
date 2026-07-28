# RCLE RGB Segment Confirmation R1

状态：`HISTORICAL / CLOSED_NOT_EVALUABLE / NO_RERUN_AUTHORITY`

本目录保留 R1 的 preaccess identity lock 实现、验证器与复现入口。当前权威终态是
[RCLE RGB Segment Confirmation R1 result](../../../../docs/research/rcle/RCLE_RGB_SEGMENT_CONFIRMATION_R1_RESULT_2026-07-28.md)：

`RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE / VALID_FAIL_CLOSED_TERMINAL`

该终态为 `0` eligible RGB frames、`0` pixel decode、`0` RGB algorithm calls。
下面的历史入口只用于审计与实现复现，不授权重试 claim、访问 payload、运行 RGB
算法或改写终态。

## 研究问题

只检查 R4 已选的 OpenLORIS `corridor1-1:w004` 与 DLR
`extreme_geometry/hexagon_01:w001`。本模块不发现新数据，不改变冻结 RGB
连续 expansion 或 `CAUSAL_THREE_PAIR_CONFIRMATION_R1`，也不把跨来源的
positive/below 差异解释为角色区分或泛化。

## 两级权限

1. preaccess lock 只允许取得所选窗口与一帧前后 guard 的 opaque RGB bytes，
   记录 member/message SHA、时间戳和相机同步；禁止解码、可视化和算法运行。
2. 只有完整 frame/pair identity lock 通过独立复核并签发后，后续 formal runner
   才能获得一次性 RGB 执行权限。

OpenLORIS solid 7z 或 DLR compressed ZIP member 若不能在冻结运输预算内停止，
对应片段为 `NOT_EVALUABLE`；不得改为完整数据源下载、换窗或换源。

## 历史稳定入口

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/egomotion_compensated_looming/rgb_segment_confirmation_r1/build_preaccess_lock.py `
  --repo-root E:\linnan\linnan `
  --output artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/preaccess_lock.json
```

## 输出与边界

本地 evidence、cache 与运输临时文件只能位于 `artifacts.local/`。当前入口只读既有
metadata/geometry evidence，不发网络请求、不打开 PNG/message payload。

# RCLE Phase B Bonn Formal Entry B0 预注册

状态：`DESIGN_FROZEN / REVIEW_PENDING / EXECUTION_AUTHORIZED_IF_REVIEW_PASS / NOT_STARTED`

日期：2026-07-26

## 目标

B0 是 Phase B 的正式入口，只做六条固定 Bonn sequence 的 acquisition、
archive/member inventory 和 timestamp-only window denominator。它不解码 RGB /
depth，不读取 pose 数值，不运行 RCLE 或任何 Phase B metric。

R3 minimal-bootstrap preclaim canonical metadata authority receipt：

```text
05a283b84f62bee000447bb567eadd63b424afaa9d81f5f0d83d36a9ed02489b
```

## 固定 cohort

| Rank | Sequence | Official display size |
| ---: | --- | ---: |
| 1 | `rgbd_bonn_crowd2` | 498.4 MB |
| 2 | `rgbd_bonn_balloon_tracking` | 325.9 MB |
| 3 | `rgbd_bonn_balloon_tracking2` | 236.4 MB |
| 4 | `rgbd_bonn_moving_obstructing_box2` | 422.1 MB |
| 5 | `rgbd_bonn_balloon2` | 267.0 MB |
| 6 | `rgbd_bonn_moving_nonobstructing_box2` | 513.1 MB |

总 official display size 为 `2262.9 MB`。只允许官方
`https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/{sequence_id}.zip`
URL；不得换 sequence、换 mirror 或加入第七条。

## Canonical 输入输出

- archive：
  `artifacts.local/datasets/rcle_phase_b_bonn_b0/archives/{sequence_id}.zip`
- extracted inventory：不解压到磁盘，只读取 ZIP central directory 和允许的文本；
- evidence：
  `artifacts.local/evidence/rcle_phase_b_bonn_b0/formal_entry_b0/`
- 必须先 exclusive-create `run_claim.json`；
- 允许同一 run 内每个 URL 最多三次 bounded transport retry；
- claim 一旦创建，异常、中断或失败均禁止第二次 B0 run。

## 允许读取

1. HTTP status、headers 和 archive bytes；
2. ZIP central directory、member names、sizes、CRC；
3. 全部 member bytes 只允许为完整 CRC test 流式读取/解压；不得图像解码、
   落盘 extracted bytes、缓存、采样或人工/模型 inspection，必须单独计数；
4. `rgb.txt`、`depth.txt`、`groundtruth.txt` 的非注释行；
5. 上述三个文本每行的第一列 timestamp；
6. ZIP 完整 CRC test。

`groundtruth.txt` 允许流式读取完整文本行 bytes，但只解析和保留第一 token；
后续 pose tokens 不得数值解析、保存或统计。RGB/depth member bytes 除 CRC-only
stream 外不得读取；static map、图像观感、旧 trace/score 均禁止。

## 必需 member contract

每条 sequence 必须唯一包含并可解析：

- `rgb.txt`
- `depth.txt`
- `groundtruth.txt`
- 至少一个 `rgb/` member；
- 至少一个 `depth/` member。

缺失、重复、路径穿越、CRC 失败、HTML/error body、URL/sequence identity 不一致
或文本 timestamp 非有限/非严格递增，整条 sequence 保留为
`NOT_EVALUABLE_ARCHIVE_AUTHORITY`，不得替换。

## Window denominator

每条 sequence：

1. `t0 = max(first_rgb_timestamp, first_depth_timestamp, first_pose_timestamp)`；
2. `t1 = min(last_rgb_timestamp, last_depth_timestamp, last_pose_timestamp)`；
3. 从 `t0` 开始生成连续、不重叠半开窗口
   `[t0 + 10k, t0 + 10(k+1))`；
4. 尾部不足 `10s` 丢弃；
5. 不按画面、pose、support、metric 或名称挑窗口；
6. 零窗口 sequence 保留在 6-sequence 分母，不得替换。

第一次 B0 receipt 中物化的全部窗口是未来 Phase B 唯一完整 window denominator。

## Gate

PASS 要求：

- 6/6 official URL identity 和 local archive hash 已固定；
- 6/6 archive/member inventory receipt 可复算；
- 6/6 timestamp-only firewall 有效；
- 至少 2/6 sequence 各有至少一个完整 10s window；
- 所有失败/零窗口单位完整保留。

否则：

```text
HOLD_PHASE_B_B0_NOT_EVALUABLE_NO_REPLACEMENT_NO_RERUN
```

PASS：

```text
PHASE_B_B0_INVENTORY_PASS_B1_METRIC_PROTOCOL_MAY_BE_FROZEN
```

## 当前边界

本协议通过设计审查后即具备执行授权，但本次任务停在
`EXECUTION_AUTHORIZED / NOT_STARTED`，不实际下载或读取 payload。
B0 PASS 也不自动授权 RGB/depth decode、pose 数值、RCLE metrics、Kill Gate B、
Replay、Android、人体、安全或生产。

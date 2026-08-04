# FRESH-TF R1-A 来源回执

日期：2026-08-04
终态：`FRESH_TF_R1A_SOURCE_TRANSPORT_AND_METADATA_ADMISSION_SUPPORTED`

三份预锁定 archive 已从 TUM RGB-D 的旧官方端点完整取得并封存 SHA-256。检查仅打开
archive 文件表与 `rgb.txt`、`depth.txt`、`groundtruth.txt` 时间戳元数据；尚未打开
RGB、depth 或任何四臂 outcome。

时间戳 admission 后发现协议遗漏独立 warp-residual 上限和 cell 状态优先级；两者已在
媒体打开前补全并将最终 protocol SHA-256 重绑为
`2379D50E497ED417C6EF8BF6D9CFDD793AF64709B22AD494061E861687D345F9`。

| sequence | archive bytes | admitted RGB | RGB duration | depth Δt P95 | pose Δt P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `freiburg1_rpy` | 410,268,381 | 721 | 24.069 s | 17.057 ms | 4.846 ms |
| `freiburg1_desk` | 344,011,403 | 596 | 20.404 s | 17.573 ms | 3.688 ms |
| `freiburg3_sitting_static` | 443,506,009 | 688 | 23.728 s | 1.158 ms | 4.564 ms |

三条序列均具备 `rgb.txt`、`depth.txt`、`groundtruth.txt`，并通过每 session 至少 300 个
接纳 RGB frame 与 15 秒时长的来源门。个别首尾时间戳超过 30 ms，只排除对应 frame，
不放宽冻结的 nearest-neighbor admission 门。

这只说明 transport 与时间戳 metadata 可用。每种机制角色仍只有一个 session，低于协议
要求的两个独立 session，因此正式 R1-A 效果评价尚不可接纳。下一步只能实现并单测 C1
投影、z-buffer 与硬状态赋值，然后运行 mechanics/opportunity canary；不能据此晋级
FRESH-TF、R1-B 或 NPU 调度。

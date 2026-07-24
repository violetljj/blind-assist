# USTRF JRDB RGB/time/frame-transform access canary R0 结果（2026-07-25）

状态：`ACCESS_BLOCKED_LOGIN_REQUIRED / VALID`

权限：`ACCESS_AUDIT_ONLY / G1_CLOSED / SIGNAL_CLOSED / ROUTE_TRUTH_CLOSED / ANDROID_CLOSED / HUMAN_CLOSED / PRODUCTION_CLOSED`

## 结论

JRDB 官方 toolkit 已提供 static calibration 与 stitched image 路径合同，公开 sample structure 也声明 dataset 内有 `images/`、`calibration/` 和 `timestamps/`。但 sample archive 只有 16 个空目录、0 个 payload 文件；toolkit 本身没有真实 RGB 或独立 capture timestamp。visualiser 中名为 `timestamps` 的值来自 label JSON key，不能充当采集时钟权威。

日期化公开下载页同时明确：test labels 可公开下载，JRDB dataset 下载必须登录；当前 Chrome 与内置浏览器均无已登录 JRDB session。本阶段因此按预注册优先级终止为：

`ACCESS_BLOCKED_LOGIN_REQUIRED`

这不是 frame-transform 不存在，也不是 dataset 不可用；它表示在当前授权与会话下，无法合法物化“同一 sequence/frame 的真实 RGB + 独立 capture time + calibration”最小 canary。

## 固定证据

- JRDB 官方页：<https://jrdb.erc.monash.edu/>
- 官方 toolkit：<https://github.com/JRDB-dataset/jrdb_toolkit>
- toolkit commit：`4fbf7d6eba3255746000eb8c15f707af69561c5d`
- `JRDB_sample_structure.zip`：4,064 bytes；SHA-256 `e1225c2d9e25a76ee644143f16115bd1ee29d369f2cfa2fbcc9800d876739205`
- 日期化公开页 HTML：46,233 bytes；SHA-256 `750f284c61bf6986778055df0cbc5e894b4c737c7c6c0ffa87ab50a0ece5f265`
- sample structure：16 directories、0 files、0 unsafe path
- toolkit calibration：`defaults.yaml` 与 `cameras.yaml` 均 hash-bound
- toolkit stitched contract：`images/image_stitched/{location}/{file_index}.jpg`

## 独立验证

- config SHA-256：`3aee5b1ba18691fef2e6cc0c472af9ba33037d42e4526ff0eaa08169a914e932`
- producer PID：`57756`
- receipt SHA-256：`74b1ac930728cb4a5da70bb66c656958da117ed962183cf5c5e583578cb3dddd`
- validator PID：`52844`
- validation SHA-256：`87846f0bb91d085a341ec46012e1735bbc1d96550fe51274a309a6519139da54`
- validator：schema、stage、PID isolation、deterministic recomputation、archive safety、终态和高权限关闭全部通过

首轮运行用未归一化字符串匹配 README，因 URL 换行而错误落到较低优先级 `FRAME_IDENTITY_OR_TIME_AUTHORITY_INSUFFICIENT`；该失败 receipt 不具权限。只修复空白归一化并更新实现绑定摘要后，producer 与独立 validator 重跑得到上述有效终态。

## 当前停止点

不得猜测受限 archive URL、把标签帧号当 capture time、下载全量数据或启动 G1。若用户在保留的 JRDB Chrome 页面自行登录，下一轮只允许：

1. 先读取登录后官方 archive 清单、大小、许可和 checksum；
2. 冻结不超过一个 sequence 的 RGB/timestamp canary 与资源预算；
3. 只在同 sequence/frame identity 和 calibration 可复算后重新判断 data-pack admissibility；
4. route-role truth 与任何 signal 仍须后续独立 goal。

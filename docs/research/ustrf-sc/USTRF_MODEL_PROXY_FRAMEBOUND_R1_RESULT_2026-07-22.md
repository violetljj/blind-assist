# USTRF 模型代理 pilot 与 ARCore frame-bound R1 结果（2026-07-22）

状态：完成；按 geometry 停止门冻结 / benchmark-only / production-isolated

## 结论

本轮已把采集与审核中的人工角色全部替换为大模型/自动 Agent，并完整执行到交接计划的预注册停止点：

1. 5 个场景的 10 个正负 matched-pair episode 已由图像模型生成，两个互不可见的模型 run 独立复核，10/10 接受；机械审计重新解码并核对 1000 帧。
2. Pilot 只开放 `proxy_full_matrix_expansion_eligible=true`；`human_truth=false`，U0、训练、Android runtime 和生产权限全部为 false。
3. SM-S9280 的独占 ARCore 单 `Frame` canary 自动采集 150 行，但 raw depth/confidence、tracking、稳定 Anchor 和 valid pair 全部为 0；host verdict 为 `FREEZE_FRAME_BOUND_METRIC_GEOMETRY`。
4. 因 geometry 先决门失败，按计划不扩正式 120 episode、不运行 U0，也不以人工移动手机或人工复核回救。

## 模型代理 pilot

证据目录：`artifacts.local/evidence/ustrf-sc/model-proxy-route-event-pilot-v1-20260722-r1/`。

- 固定矩阵：1 session × 5 scene × positive/matched-negative = 10 episode / 5 pair。
- 每集：1536×1024 四面板 contact sheet，转换为 1280×720、10 秒、10 fps H.264；每集 100 个解码帧 SHA-256。
- 生成 provenance：接受的 10 张图与 2 张被拒修复候选均保留记录；早期图像工具调用没有自动嵌入逐字 prompt，故 `generation_records.jsonl` 明确标为 `posthoc_reconstructed_summary`，不伪造原 prompt。
- 审核：`gpt_multimodal_reviewer` 与 `codex_evidence_reviewer` 两次隔离 run 均未看到另一份 review 或 candidate 输出。原始 `clear/outside` 末帧措辞保留，固定规则只把等价含义归一化为合同 relation；没有语义分歧，因此未启动第三模型。
- 反绕过：17 个测试覆盖任意配置、绝对路径、重复 episode/media、坏 frame、复用 reviewer/run、替换 input、篡改 raw review、生成记录重绑定和 pilot 自行开启 U0，全部通过。

最终 manifest SHA-256：`26bedc320a7581b24e7045cafeafb79f8c7cb085d98a900541050785792b2b3c`；audit SHA-256：`39d796bb18273b871eed265dcb6692302009125f4aaa78ea2e4e8e2464964737`。

## ARCore single-Frame canary

设备证据目录：`artifacts.local/evidence/ustrf-sc/frame-bound-metric-geometry-sm-s9280-20260722-r2/`。

Canary 使用独占 `Session`，每个 `Session.update()` 的同一 `Frame` 绑定 `Frame.timestamp`、`getAndroidCameraTimestamp()`、camera image、raw depth/confidence、tracking、pose/intrinsics/transforms 与一个预期跨帧持久 Anchor。自动 instrumentation 不要求用户移动设备，不发出导航或安全动作。

| 主机重算项 | 结果 | 门 |
| --- | ---: | ---: |
| raw rows | 150 | 记录完整 |
| unique Android camera timestamp | 139 | ≥100，通过 |
| duplicate timestamp | 0 | =0，通过 |
| Frame↔camera image pair | 139 | ≥100，通过 |
| raw depth/confidence pair | 0 | ≥100，失败 |
| source-aligned fraction | 0.0 | ≥0.95，失败 |
| tracking / denominator | 0 / 139 | ≥0.95，失败 |
| valid pair | 0 | ≥100，失败 |
| persistent `INTER_FRAME_STABLE` Anchor | 0 | 必需，失败 |

`raw_frames.jsonl` SHA-256 为 `392f596462c7e9eababeffd67bb8f9a67336f849ea812beb8f97a5134363193b`；host audit SHA-256 为 `69a90af614d6ecd199cf22927026cd40e837e6b1992e7934f6ce26c9252bf7e7`。设备 instrumentation 为 `OK (1 test)` 只表示采集程序完成；validator 合法地产出 `ok=true / gate_open=false` 并以 exit 2 表示门关闭。

Host validator 另有 9 个纯临时目录合同测试，覆盖合法 100 帧开门、输入哈希篡改、重复 Camera2 时间戳、缺 raw depth、逐帧伪 Anchor、非 `INTER_FRAME_STABLE` 与不安全 authority；连同 pilot 共 26 tests 全部通过。

## 决策与后续边界

- 本轮结论不是“再移动一次手机可能就行”，而是当前无人化静置手机路线不满足冻结门；同一窗口不继续重试。
- 120 episode/60 pair 扩展与六臂 U0 保持关闭。Pilot 的 10 集不得冒充正式分母或独立真实场景。
- 下一条可接受的新证据链只能是无需人工采集/审核的新硬件或自动化深度来源，并重新预注册其 device/mount/calibration/metric geometry 合同；不得放宽 `100 / 0.95 / INTER_FRAME_STABLE`。
- App 默认模型、反馈链、训练和生产权限未改变。

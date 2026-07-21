# USTRF R1.2c 非 R1.3 seen positive 预注册（2026-07-22）

## 结论

已找到并冻结第六个合格 seen positive：`bangkok_tactile_cone_intrusion`。它来自 2026-07-19 已打开的 Bangkok Modern Center 公共步行视频，不占用 R1.3 的任何来源槽位。两名互不见结果、且不读取 detector/tracker/association 输出的模型复核均判定：同一红白交通锥进入前向步行路线，至少需要向左绕行；静帧不足以声称已经减速或必须停止。

合同验证得到两个 alertable `robust-inside` anchor：333s 的边界距离为 `28.45px`，336s 为 `61.34px`，均大于最大 `.03 × 852 = 25.56px` 不确定性。328s 只有 `1.60px` 边界余量，固定降为 `uncertain_boundary / not_gate_eligible`；339s 的可见边界代理为 robust outside，只支持清除，不伪造画外接触点。

因此，seen positive 资格计数已由 `5 + 1` 补到 `6`。但本预注册本身不授权 London FP16-768：下一步必须把 Bangkok 替换 Japan 物化为新的 R1.2c 连续清单，重跑完整六正例 oracle，确认全门一致后才能执行唯一 768 候选。

## 来源与许可

- 来源：Wikimedia Commons，POPtravel，Bangkok Modern Center 连续步行视频；
- 许可：CC BY 3.0，2026-07-22 再核验；Commons 页面记录 YouTubeReviewBot 于 2021-03-30 确认许可；
- 本地 240p 视频 SHA-256：`2d27115804fde2f1bee3dff8e12325c21889b1c454fbb94d23c25d32d92fb572`；
- 该来源在本轮预注册前已用于 r7.66、r7.86 与 r7.97a seen 诊断，因此不能再声称新 held-out，但正适合本次“不消费 R1.3”的补位目标。

## 冻结事件合同

| 字段 | 冻结值 |
| --- | --- |
| event ID | `bangkok_tactile_cone_intrusion` |
| 替换对象 | `japan_path_intrusion`；旧 Japan 排除裁决保持不可改写 |
| 连续窗口 | `328000–340000ms` |
| alertable | `333000–336000ms` |
| clear from | `339000ms` |
| 唯一目标 | tactile route 上同一红白交通锥；不得用黑色隔离柱、花盆、行人或其他锥桶替代 |
| truth | `positive_route_obstacle_requires_lateral_avoidance` |
| 几何 | 两份独立视觉路线复核的较窄凸多边形共识；`.01/.02/.03` 三档全部 inside 才算 anchor |

## 证据与权限

- 预注册合同：`configs/ustrf_crosscam_seen_positive_r12c_prereg_v1.json`；
- 验证收据：`artifacts.local/evidence/ustrf-crosscam-codex/r12c-seen-positive-prereg-v1/validation.json`，SHA-256 `94154e091bec1e80cb1accc15fe20de0472c90db96619c1948e75d98ae70d083`；
- focused tests：`3 tests OK`；含 OpenCV 的完整 crosscam 合同组：`25 tests OK`；
- R1.3 discovery/download/decode/result access/slot consumption 均为 false；
- 仍保持 benchmark-only、无训练、无 App/default backend、无生产替换权限；768、完整连续重放、soak 与 R1.3 unlock 权限继续为 false。

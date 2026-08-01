# HFTF H1 causal-advected-origin geometry teacher result R2

日期：2026-08-01

终态：`H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP`

## 1. 结论

R2 通过了全部 source/mechanics validity 门，但 multi-height 没有在 4/4 independent
sessions 上稳定非冗余：

- 4/4 authority、source independence、exact frozen set、preparation hash、usable
  anchors 与 single/multi consistency 均通过；
- current/near/far worst-session coverage 为
  `0.204191/0.184698/0.119136`，全部高于 `.15/.10/.10`；
- `03694304/03b6dc99/03d70593` 的 height disagreement 通过 `.02`，但
  `03c87279` 只有 `2/684=0.002924`；
- 因此按顺序门关闭在 `H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP`；
- future change 仅作 diagnostic，不能据其 3/4 passing 宣称 future mechanism
  formally supported。

R2 确认这个新 evidence version 的 causal rolling field 已可评价；不能把跨 R1/R2 的
差异只归因于 advection，因为 source cohort 同时改变。mandatory multi-height 轴则未
达到预冻结的跨 session 稳定性要求，不再用新 height threshold/bands 救援。

## 2. 身份与一次性执行

- protocol commit：`e7cc55d3eb230067de5771a99fe442db6f58250a`
- runner commit：`d32549776603b06b78c284f3c72519d3462e9676`
- protocol SHA-256：
  `3115ed4a9c6e03078c7e18ade2eee61b1873981e21da08b7f74b83d974fc6ffe`
- runner SHA-256：
  `5fb0248d99584198f879c2cc953232d77394a3fdaf3927d48e9d81b09e91cd57`
- result：
  `artifacts.local/evidence/hftf/h1-causal-advected-origin-geometry-teacher-r2-20260801/teacher_canary.json`
- result SHA-256：
  `600f37dea7940af5a4e2d09eb798547f3a8694b2dc4d04ce611e68f186023949`

正式运行前 HFTF suite 41 项通过，独立只读 implementation review 无 blocking
finding。output path 使用 exclusive create，未覆盖旧结果。四个 R2 sessions 现已
burned。

## 3. Frozen-denominator 结果

| Source | U | current | near | far | height | future diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `03694304…e34a` | 15 | 0.298765 | 0.229630 | 0.175926 | 39/540 = 0.072222 | 128/1,620 = 0.079012 |
| `03b6dc99…e68c` | 15 | 0.346296 | 0.416049 | 0.538889 | 11/540 = 0.020370 | 142/1,620 = 0.087654 |
| `03c87279…e3b0` | 19 | 0.204191 | 0.184698 | 0.174951 | 2/684 = 0.002924 | 28/2,052 = 0.013645 |
| `03d70593…b39e1` | 15 | 0.329012 | 0.233333 | 0.119136 | 17/540 = 0.031481 | 112/1,620 = 0.069136 |

每 horizon denominator 分别为 `1,620/1,620/2,052/1,620`；height denominators
为 `540/540/684/540`。UNKNOWN/invalid 均保留。single/multi consistency error
四者均为 `0`。

## 4. Causal-advection diagnostic

| Source | median tangent speed m/s | near origin error m | far origin error m |
| --- | ---: | ---: | ---: |
| `03694304` | 9.225 | 0.656 | 2.224 |
| `03b6dc99` | 10.024 | 1.274 | 1.244 |
| `03c87279` | 1.623 | 0.131 | 0.329 |
| `03d70593` | 9.877 | 0.345 | 2.642 |

这些值只定位 constant-velocity advection 的误差，不进入 gate。尤其不能因为三个快速
source 的 future change 较高，就把 overall change 解释为环境主体运动预测；rolling
origin 穿过静态空间本身也会产生差异。

## 5. 研究决策

R2 关闭的是“multi-height 必须作为 HFTF 核心输出”的 evidence version，不关闭
action-agnostic causal rolling-future field：

- `foot/body/head` 从 mandatory core 降为 auxiliary diagnostic；
- 下一 H1 若继续，只允许在 fresh sessions 上预冻结 single-height
  `[0.05,2.05] m` rolling-future canary；
- R3 不得修改 R2 的 origin、theta、distance、horizon、known、UNKNOWN、coverage 或
  future-change 数值门；
- R3 成功也只授权另行冻结 single-height H2 student，不恢复 multi-height claim。

当前 H2、主线、Android、提醒、默认 App、生产与安全均未授权。

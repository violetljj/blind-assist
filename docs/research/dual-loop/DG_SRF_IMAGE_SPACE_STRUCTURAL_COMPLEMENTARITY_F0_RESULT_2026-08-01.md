# DG-SRF Image-Space Structural Complementarity F0 Result

状态：`COMPLETE / VALID / STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP /
DEVELOPMENT_STANDARD / FINAL_CONFIRMATION_NOT_ACTIVATED`

日期：2026-08-01（Asia/Hong_Kong）

协议：`DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0`

冻结合同见
[F0 protocol](DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_PROTOCOL_2026-08-01.md)。

## 结论先行

固定的 Depth Anything V2 Small 单帧结构场在这 520 帧 consumed Development 上不支持
DG-SRF F1：最终终态为

```text
STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP
```

独立 validator 从 raw depth、canonical truth 和 packed A/B 重新计算 q、D1-D5、group
AP、10-fold LOSO threshold、九门、coverage 和 terminal，`29,031` 项检查一致，
`validation_status=VALID`。

这不是“相对深度对所有类别无关障碍无效”的普遍结论。它只否定本协议精确定义的
单帧、逐帧归一化、固定 `N/E/R+/R-`、等权 composite、image-space proxy 及当前
SANPO-Real 520-frame comparator。不得在相同 520 帧上通过改权重、梯度尺度、trend、
morphology、lambda、Video Depth 或时序 latch 救援。

## 主要证据

### 阈值无关结构信号

10 个 source-session 的 macro AUPRC：

| Arm | Macro AUPRC | 相对 B | 优于 B 的组数 |
| --- | ---: | ---: | ---: |
| B frozen binary DDRNet residual | 0.362109 | — | — |
| D1 proximity | 0.278070 | -0.084038 | 1/10 |
| D2 multiscale gradient | 0.359603 | -0.002506 | 5/10 |
| D3 surface-trend residual | 0.311101 | -0.051007 | 1/10 |
| D4 fixed equal-weight composite | 0.309456 | -0.052652 | 3/10 |
| D5 D4 + proxy prior | 0.281121 | -0.080987 | diagnostic only |

D1-D4 全部未满足“macro AP 高于 B 且至少 8/10 group 高于 B”的 stable-signal 定义。
D2 最接近 B，但仍略低于 B，且只在 5/10 组同方向；不能据此称为稳定信号。

D4 也没有组合增益：

```text
D4 macro exceeds every D1-D3: false
D4 strictly exceeds best D1-D3 by >1e-6: 1/10 groups
required: 8/10 groups
```

### LOSO operating-point utility

10 个 outer fold 的九组训练上下文均为
`NO_ALL_GATE_OPERATING_POINT`。冻结 maximin 规则仍选择阈值并只应用到 held-out group；
阈值为 `.30 / .35 / .40`，其中 8/10 组为 `.40`。

| Gate | 实测 | 门 | 结果 |
| --- | ---: | ---: | --- |
| FP pixel reduction vs B | 0.556665 | >= 0.30 | PASS |
| overall residual recall retention | 0.254913 | >= 0.90 | FAIL |
| minimum-group recall retention | 0.000019 | >= 0.80 | FAIL |
| boundary_step_curb retention | 0.950596 | >= 0.80 | PASS |
| obstacle retention | 0.139797 | >= 0.80 | FAIL |
| delta recall C-A | 0.078227 | >= 0.05 | PASS |
| delta FP-area fraction C-A | 0.047137 | <= 0.05 | PASS |
| residual truth component recall | 0.252938 | >= 0.50 | FAIL |
| false activation components/frame | 6.823077 | <= 3.0 | FAIL |

只通过 4/9 门。主要失败不是单纯阈值偏保守：最弱组 recall retention 接近零，
obstacle retention 只有 `.139797`，同时 false components/frame 仍为 `6.823077`。
即使 boundary retention 保住，也再次出现 boundary/obstacle 明显分裂；因此不能用
FP reduction `.556665` 或 C-A recall `.078227` 单独主张可用互补性。

## 输出健康、canary 与假信号检查

- 520/520 depth output 健康；overall 和 minimum-group q coverage 都为 `1.0`；
- 官方输出方向固定为 `RAW_LARGER_IS_NEARER`；synthetic RGB canary 4/4 同向，
  median normalized near-far margin `.707553`；
- transform-only resize/affine normalization canary 为 `PASS`；
- exact strict-loaded parameter count 为 `24,785,089`；
- producer 没有读取 canonical truth、A/B mask 或 outcome；
- evaluator 只在 raw depth 全部冻结后 late-join truth；
- D1-D4 的失败不是 q 通过回避困难帧得到：没有任何帧被排除；
- D5 proxy 的 macro AP 为 `.281121`，低于 D4；lower-center image-space prior 没有
  构成隐藏救援。

按与 detector identity 混杂的 role 分层，D4 macro AP 在三个 role 中均低于 B：

| Role | B | D4 |
| --- | ---: | ---: |
| consumed-old-blind | 0.431286 | 0.397526 |
| Development | 0.430730 | 0.330477 |
| R1-consumed-fresh | 0.258898 | 0.244401 |

这不能消除两套 YOLO detector 与 role 的完全混杂，但说明总负结果并非只由一个 role 的
反向结果造成。

## Evidence identity

冻结执行 Git：
`32650abe1c0bb974626c61adcc31a8a47fa4a793`

```text
config SHA256
581255599df706ef535de4782d63436d6a80067cf154946b28ba7cddba4e7c22

truth-minimized inference manifest SHA256
a3be11b00155c0e7bf0bdd38c675c08040a47f94a764b3cf928e8310ce3b69e9

raw depth maps SHA256
25cab1b64afe67dd15393507545020abcd0fc80f1b40ff977af7b3b3f6b2e731

depth index SHA256
845ae954c25020bcfc5abe528f54bf03d943eaf1cf1bbb9ef7893912dc0820c5

producer receipt SHA256
7cdd567935dde48f4163f14ed64712dfffd3367a5216d0aaad5d23419be6acef

evaluation result SHA256
3e61a44384983917aa8105dd4d4db0f24bc1059f1f53511ed2ee7d4640af17e8

independent validation SHA256
f8211e025da802840d6e502b13a5c19ec81f92bd763d1fd1d68d20111016d9d6
```

GPU producer inference 为 `177.349 s / 2.932 fps`，只属于本机 Development evidence，
不是 Android、Snapdragon、A568、QNN/HTP 或产品实时性证据。evaluator 为
`64.890 s`。

本地产物位于：

```text
artifacts.local/evidence/dg-srf-image-space-structural-complementarity-f0/
  prepared-v1/
  pilot-v1/
  producer-v1/
  evaluation-v1/
  validation-v1.json
```

## 证据强度与限制

强项：

- model/source/checkpoint/preprocess、数据 membership、六项机器合同和 terminal 映射在
  outcome 前冻结并推送；
- truth-minimized producer 与 truth-late evaluator 分离；
- q=0 不排除帧，且实际 coverage 为 100%；
- 每个 held-out session 不参与该 fold threshold 选择；
- 独立 validator 不 import producer、operators 或 evaluator，并从 raw depth 重算全部
  claim-bearing quantity。

限制：

- 520 帧 outcome 全部已消费，只是 Development；
- 10 组全部来自 SANPO-Real，缺 participant、route、parent-capture independence；
- B 是 frozen binary DDRNet residual，不是 continuous DDRNet score；
- 两套 YOLO detector identity 与 source role 完全混杂；
- 没有可靠统一时间轴，不评价 real-time flicker、TTC 或 event effect；
- component hit 用任意正像素相交，较宽松；
- 单目相对深度的玻璃、镜面、低光、运动模糊和细物体失真没有独立真值归因。

因此结论上限仍是：

```text
CONSUMED_DEVELOPMENT_IMAGE_SPACE_STRUCTURAL_MASK_ALIGNMENT
```

不能写为未知障碍真实召回、可通行性、风险下降、提醒效果、设备可部署或安全证据。

## 停止与后续权限

当前精确定义的 DG-SRF F0 已关闭。以下动作不获授权：

- 在同一 520 帧上调 D4 权重、gradient sigma、surface trend、dead zone 或 morphology；
- 搜索 D5 lambda 或把 proxy 改称路线/走廊；
- 引入 Video Depth、temporal propagation/latch 掩盖单帧负结果；
- 启动 F1-F5、Android、QNN/A568、risk/feedback、TTS、振动或提醒；
- 修改 production App 或默认 App。

raw depth、health ledger、group AP、failure sessions 和 validator 可作为论文负结果、
回归 fixture 或 visual-only diagnostic 资产保留。若未来提出实质不同的新问题，必须由
新数据和新协议决定；本结果不自动授权重开。

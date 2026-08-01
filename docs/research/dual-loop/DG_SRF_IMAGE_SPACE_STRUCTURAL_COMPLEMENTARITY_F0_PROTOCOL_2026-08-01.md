# DG-SRF Image-Space Structural Complementarity F0 Protocol

状态：`PROTOCOL_AND_IMPLEMENTATION_FROZEN / RESULT_NOT_RUN /
DEVELOPMENT_STANDARD / FINAL_CONFIRMATION_NOT_ACTIVATED`

日期：2026-08-01（Asia/Hong_Kong）

协议：`DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0`

## 结论先行

本轮冻结并只执行一个 Depth Anything V2 Small 单帧结构信号审计。研究问题是：预训练
相对逆深度能否在实际 YOLO 覆盖区之外，以低于冻结 raw DDRNet residual 的假激活
代价，对 canonical `boundary_step_curb / obstacle` pixels 提供跨 10 个
source-session group 稳定的图像空间互补。

这不是“未知障碍召回”、米制深度、地面高度、真实行进走廊、事件效果或安全能力评测。
正结果最多授权另立 F1 设计；F1 执行、Video Depth、端侧部署和产品融合均未授权。

## 证据角色与独立性上限

输入固定为既有 520 帧：200 帧永久降级的 R1 consumed-fresh、200 帧 Development、
120 帧 consumed-old-blind，共 10 个 SANPO-Real source-session。RGB、canonical mask
和 packed A/B mask 逐帧身份完整，但所有 outcome 均已被查看；没有
participant、route、parent-capture identity，也没有统一可靠 capture timestamp。

因此分析单位叫 source-session group，但不声称 10 个独立参与者、路线或自然事件。
所有结论限定为：

```text
CONSUMED_DEVELOPMENT
IMAGE_SPACE_STRUCTURAL_MASK_ALIGNMENT_ONLY
NOT_UNKNOWN_OBSTACLE_RECALL
NOT_CONFIRMATION_OR_PRODUCT_SAFETY_EVIDENCE
```

本协议相对已关闭 segmentation gating 引入的新因果变量是深度结构表征，不修复或重开
历史 segmentation、RCLE、中央阻塞或 R1/R2-P0 terminal。

## 模型、方向与尺度合同

唯一模型为 Depth Anything V2 Small：

- official source commit：
  `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`；
- official checkpoint revision：
  `03876f8651c73a60fe4c2c48294e09fcb6838fcf`；
- checkpoint SHA256：
  `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`；
- checkpoint bytes：`99,218,434`；
- strict-loaded exact parameter count：`24,785,089`；
- encoder `vits`，official input size `518`，Small license `Apache-2.0`。

预处理固定为 official keep-aspect lower-bound resize、14 倍数、输入 `INTER_CUBIC`、
BGR→RGB、`/255`、ImageNet normalization，恢复原图采用 bilinear
`align_corners=true`；之后只用 `INTER_LINEAR` 缩放到 `256×256` 分析空间。

官方输出语义冻结为 affine-invariant inverse depth：

```text
FROZEN_DIRECTION = RAW_LARGER_IS_NEARER
DEPTH_DIRECTION_SYNTHETIC_CANARY_REQUIRED
CANARY_MAY_VALIDATE_BUT_MAY_NOT_SELECT_OR_FLIP_DIRECTION
CROSS_FRAME_RAW_DEPTH_MAGNITUDE_COMPARISON_FORBIDDEN
METRIC_DISTANCE_INTERPRETATION_FORBIDDEN
```

四张程序生成 RGB canary 与一张已知 inverse-depth transform canary 必须在读取
canonical truth 前通过；失败直接 `NOT_EVALUABLE`，不得试相反 sign。每帧用 raw
5%/95% 分位作稳健归一化；跨帧 raw scale/shift 不进入任何特征。

## q 输出健康门

`q` 只检测数值输出是否可计算：

- finite fraction `>= .9999`；
- relative robust span `>= .001`；
- normalized standard deviation `>= .02`；
- 单端极值 plateau fraction `<= .25`。

`q=0` 的帧不删除，D1-D5 置零并继续进入 AP、utility 和 coverage 分母。玻璃、镜面、
低光、模糊或肉眼觉得错误但仍为有限动态输出的深度，不允许主观 abstain。

执行有效还要求：

```text
overall evaluable-frame coverage >= .95
minimum source-session evaluable-frame coverage >= .90
```

## 四个冻结子信号

相对近度 `P` 是逐帧稳健归一化输出。四个子信号分别再次限制到 `[0,1]`：

1. `N=P`；
2. `E`：对 `sigma=0,1.5,3.0` 三个固定尺度求 Sobel-3 梯度幅值，平均后以逐帧
   50%/95% 分位归一化；
3. `R_plus`；
4. `R_minus`。

surface trend 固定只在 `y>=floor(.45H)` 的全宽区域计算每行 `median(P)`，以
ordinary least squares 拟合二阶纵向多项式 `T(y)`，死区 `delta=.03`：

```text
R_plus  = max(0, P - T - .03)
R_minus = max(0, T - P - .03)
```

两个 residual 独立归一化、不得抵消；非有限或 rank defect 直接实现失败。拟合不能读取
truth 或 YOLO mask，且只能叫 `surface trend`，不能叫 ground、floor、height 或 plane。

先构造结构场，再做 residualization：

```text
D1 = q * N * (1 - A)
D2 = q * E * (1 - A)
D3 = q * mean(R_plus, R_minus) * (1 - A)
D4 = q * mean(N, E, R_plus, R_minus) * (1 - A)
D5 = D4 * [.25 + .75 * C_proxy]
```

其中 `A` 是该帧冻结的实际 YOLO union。D4 权重严格 `1:1:1:1`；D5 的 lower-center
梯形纯图像先验从 `.35H` 开始、上/下 half-width 为 `.10W/.35W`、`lambda=.25`。
D5 只是消融，不能支持路线或走廊主张。

## 对照身份

- A：当前 YOLO system reference，不是 residual detector；
- B：冻结的 binary raw DDRNet class `1|2` mask 减去 A 后的 residual reference；
- D1-D3：三个固定深度单方法；
- D4：固定等权结构 composite residual；
- D5：D4 + image-space proxy prior 消融。

B 已核验等于 packed 两类 candidate union，且 `A∩B=0`；它不是 continuous DDRNet
probability，因此 B 的 AP 只是 binary-score comparator。

全体并非单一 YOLO 模型：fresh+dev 400 帧和 old-blind 120 帧分别使用两个冻结 model
SHA；labels、input `320`、confidence `.35`、NMS `.45` 相同。detector identity 与
source role 完全混杂，必须报告 role-stratified AP，但该诊断不能消除混杂。

## AP、LOSO 与唯一 operating point

AP 定义为：

```text
NON_INTERPOLATED_TIE_GROUP_PRECISION_RECALL_STEP_INTEGRAL
```

对每个 source-session，把全部非 YOLO pixels 展平；同分 score 作为一个 tie group 后
累计 precision/recall。正或负类为空则该执行 `NOT_EVALUABLE`。strict advantage
epsilon 为 `1e-6`。

D4 的组合优势必须同时满足：

```text
MacroAP(D4) > MacroAP(Dj) + 1e-6, for every j in D1..D3
count_g[AP_g(D4) - max_j AP_g(Dj) > 1e-6] >= 8/10
```

stable signal 对 D1-D4 分别定义为该 arm macro AP 严格超过 B，且至少 8/10 group AP
严格超过 B；不得用一个弱单信号替代最佳单信号比较。

D4 mask utility 的阈值固定为 `.05,.10,...,.95`。每个 outer held-out session 内：

1. 只在其余九组计算九门；
2. 对每个阈值计算各门 normalized margin；
3. 选择最大化 `min_k margin_k` 的阈值；
4. 并列依次选择更高 minimum-group retention、更高 FP reduction、更低阈值；
5. 没有九门全过点仍选择 least-shortfall 点，并写
   `NO_ALL_GATE_OPERATING_POINT`；
6. 只把选择阈值应用于 held-out group，held-out outcome 不得回写选择。

lower-bound margin 为 `(value-threshold)/abs(threshold)`；upper-bound 反向。任一分母
未定义则 `NOT_EVALUABLE`。

## 九项 utility 门

```text
FP pixel reduction vs B                      >= .30
overall residual recall retention vs B       >= .90
minimum-group recall retention vs B          >= .80
boundary_step_curb recall retention vs B      >= .80
obstacle recall retention vs B                >= .80
delta recall C-A                              >= .05
delta FP-area fraction C-A                    <= .05
residual truth component recall               >= .50
false activation components/frame             <= 3.0
```

类别 recall 是同一个 class-agnostic D4 mask 分别与两个 truth class 求交，不把 DG-SRF
写成分类器。组件采用 8-connectivity，truth component 任意一个像素被命中即 hit；这项
较宽松，必须与 FP area 和 false components/frame 联合解释。

## 独立 validator 与四终态

producer 只读 truth-minimized RGB manifest。evaluator 后置读取 truth 和 A/B。
validator 不 import producer、operators 或 evaluator，从 config、raw depth、
canonical masks、source ledgers 独立复算 q、D1-D5、AP、10 个 folds、九门、coverage
和 terminal，并逐项核对 evaluator 输出。

`validation_status=VALID|INVALID` 与科学 terminal 分离。有效复算的
`NOT_EVALUABLE` 仍可为 `VALID`；任何 identity、hash、算子、fold、指标或 terminal
漂移都必须 `INVALID`，不能伪装为科学负结果。

终态精确映射：

- `STRUCTURAL_SIGNAL_SUPPORTED_FOR_F1_DESIGN`：九门全过、D4 两项组合优势通过、
  coverage 通过、validator valid 且无合同违规；
- `SIGNAL_PRESENT_BUT_COMPOSITE_NOT_READY`：任一 D1-D4 满足 stable signal，但 D4
  未形成全部组合/utility 条件；
- `STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP`：D1-D4 均无 stable signal；
- `NOT_EVALUABLE`：模型/输入/配对、方向、coverage、LOSO、validator 或机器合同失败。

## 停止条件与权限

协议和实现 push 后才允许一次正式 520-frame producer→evaluator→validator。结果后
不得用同一 520 帧改权重、梯度尺度、surface trend、morphology、lambda，或引入
Video Depth、temporal latch。F0 失败即停止当前精确定义的方法；成功只授权提交新 F1
设计，不自动授权 F1 execution。

本轮不访问新 fresh、不训练分类器、不运行 Android/QNN/A568、不修改 production
App、risk/feedback、提醒、TTS、振动或默认 App。

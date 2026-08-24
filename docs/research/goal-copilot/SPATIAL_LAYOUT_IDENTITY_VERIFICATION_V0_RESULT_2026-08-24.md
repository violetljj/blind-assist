# Spatial-Layout Identity Verification V0 Result

状态：`DEVELOPMENT / CONSUMED_SINGLE_RUN / SPATIAL_LAYOUT_IDENTITY_MIXED_WITH_COLLATERAL_DEVELOPMENT / PASSIVE_SINGLE_REFERENCE_RGB_EXACT_INSTANCE_MAINLINE_STOP / OPEN_SET_CALIBRATION_NOT_RUN / RELIABLE_VERIFIER_NOT_ESTABLISHED / NO_P1 / DEFAULT_APP_UNCHANGED`

## 判决

预注册的 analytic reciprocal spatial-layout scorer 不晋级。在 900 个全新 Washington RGB-D Object Dataset
same-instance vs same-class-distractor pairs 上，固定 DINOv2-S mean-nearest baseline 为 `702/900=78.0%` target
outrank，layout arm 只有 `558/900=62.0%`。Paired transition 为：

```text
rescue       74
collateral  218
net        -144
```

layout 只保留 `484/702=68.9%` baseline-correct controls，低于冻结的 `>=80%` 门。23 个 baseline 在三个 candidate
quantile 上全部稳定选错的 target instances 构成 69 个 stable-distractor pairs；layout 只救回 `29/69=42.0%`，
低于冻结的 `>=50%` 门。终态为：

```text
SPATIAL_LAYOUT_IDENTITY_MIXED_WITH_COLLATERAL_DEVELOPMENT
PASSIVE_SINGLE_REFERENCE_RGB_EXACT_INSTANCE_MAINLINE_STOP
OPEN_SET_CALIBRATION_NOT_RUN
RELIABLE_VERIFIER_NOT_ESTABLISHED
NO_P1
DEFAULT_APP_UNCHANGED
```

该结果只拒绝本次固定 DINO reciprocal-match graph、top-64、4×4 coverage、local-neighborhood preservation、
Procrustes residual、conflict consistency 与等权 geometric-mean arm；不证明所有 geometry/layout 方法或单张 RGB
在信息论上不可能。但结合此前 generic appearance、PDM diffusion appearance、naive multi-reference、small learned
near-identity head 均在各自冻结数据上产生不足或 collateral，本主线按预声明 stop rule 关闭：不再以新的被动单参考
RGB backbone/head/layout 变体继续 mechanism zoo。

## Pre-pixel amendment 与 roster

初始 protocol 按 README “3 video sequences” 将 candidate 推定为 video 3；官方 ZIP central directory 显示 evaluation
set 的实际序列编号为 `1/2/4`，因此初始 freeze 在 metadata-only roster gate 失败。没有 roster、pixel decode、model
execution 或 outcome。R1 amendment 只把 candidate 改为实际第三条 video 4，其他数据规则、scorer 与 gates 不变，并以
提交 `0bb1e231` 在像素访问前推送。

正式 roster 使用全部合法实体：

```text
categories          51
physical instances 300
samples            1200
paired decisions    900
reference          video 1 / q=.50
candidate          video 4 / q=.25,.50,.75
```

每个 hard negative 是同 category 的下一个数值 instance；target 与 hard 共享 candidate video 与 quantile。archive 为
`673,456,874` bytes，SHA-256 `ea6c13eb7e9302dc3a438524f06179787fa39eb27d6cb3b4af0d9bb3631ad023`。
roster 在 ZIP central directory 上冻结完成后才解码 1200 张 RGB。

## 模型、scorer 与执行身份

两臂共享 `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`、224×224 ImageNet
normalization 与最后层 16×16×384 L2-normalized patch tokens。模型文件 SHA 全部通过既有冻结检查；执行使用
PyTorch `2.11.0+cu128`、NVIDIA GeForce RTX 5060 Laptop GPU，共 `75` batches / `1200` crops。

layout arm 没有训练参数，不读取另一个 candidate、category、instance、target slot 或 outcome。`score(R,C)=score(C,R)`
在 `1800/1800` unary comparisons 上通过绝对误差 `<=1e-9`；candidate permutation invariance 为 `900/900`。
因此负结果不是 direction asymmetry 或 A/B slot bias 造成。

## 指标与 gates

| metric / gate | 结果 |
| --- | --- |
| baseline target outrank | `702/900=78.0%` |
| layout target outrank | `558/900=62.0%` |
| `rescue > collateral` | **FAIL**：`74 <= 218` |
| control retention `>=80%` | **FAIL**：`484/702=68.9%` |
| direction invariance `=100%` | PASS：`1800/1800` |
| candidate permutation invariance `=100%` | PASS：`900/900` |
| stable instances `>=5` | PASS：`23` |
| stable-distractor target outrank `>=50%` | **FAIL**：`29/69=42.0%` |

三个 candidate quantile 的 layout target outrank 为 `182/300=60.7%`、`192/300=64.0%`、`184/300=61.3%`；
baseline 分别为 `234/300=78.0%`、`239/300=79.7%`、`229/300=76.3%`。负向差异跨三个固定 view 均存在，
不是单一 quantile 驱动。最差 layout category 为 marker、camera、mushroom，均 `44.4%`；完整 category 与 pair-level
raw diagnostics 保留在正式 report。

## 收据

正式 artifacts：
`artifacts.local/evidence/public-identifiable-referent-spatial-layout-identity-verification-v0/run-20260824T081300Z/`。

| 收据 | body/file SHA-256 |
| --- | --- |
| R1 protocol freeze body | `c4db6dea206d36fc36882de720a889373e9689e23c340d70abb0c1249cbeff13` |
| R1 protocol freeze file | `257bd172f149791cd95f703321d6699e5a52814c3a7b8b435ec2e74a015f871f` |
| frozen roster body | `1a11591ec845b36083b1055d97ae8d8ed0c4406e01117b22b7f1a1a6df01e94f` |
| frozen roster file | `625e263bdbd989be87c4228387f0f4ef63f97a233934dd6e0faaedb5507567ea` |
| raw scores file | `7dee1e0f21e59c1bec164cc6a4d39e3e603f69c45bf16183643e4b67b5d2d93a` |
| final report body | `4ec73aca511eba4ab9f58350f9e4f4dad3280dda448b974c3ac9f2a7708368e8` |
| final report file | `1440e5296c2aff60f28fbb942cead0e3c8165fd1c3642ddd1c2b376fe0a88117` |

`final-report.json` 已存在，入口拒绝覆盖或对同一 roster 建立第二次 adjudication。

## 科学边界与下一路由

本实验故意没有 target-absent 或绝对阈值，所以不能从该结果推出 `NONE`、open-set calibration 或 risk--coverage
结论。由于 identity-ranking signal 自身未通过，`OPEN_SET_IDENTITY_CALIBRATION_V0` 当前不启动；先研究 calibration
只会给一个失败的 identity representation 增加第二故障层。

下一主研究边界不是另一个 passive appearance/layout backbone，而是另立协议改变信息合同：主动获取 distinctive
evidence、多视角/交互式验证、用户确认或传感器/标记提供的独立身份证据。官方 NearID checkpoint 最多保留为以后独立的
冻结 transfer canary，不是本路线 successor，也不能 rescue 本终态。P1、belief、tracker、App 与安全权威保持关闭。

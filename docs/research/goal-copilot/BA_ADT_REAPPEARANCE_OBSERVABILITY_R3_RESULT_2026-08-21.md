# BA-ADT five-window reappearance observability audit R3

状态：`VALID / DEVELOPMENT_ONLY / FIVE_WINDOW_AUDIT_COMPLETE / ORACLE_4_OF_5 / TWO_UNOBSERVABLE_THREE_TOO_SMALL / NO_VISIBLE_SCALE_SUFFICIENT_MODEL_MISS / NO_IDENTITY_AMBIGUITY_EVIDENCE / NO_DINOV2_SAM_SKY`

## 问题与边界

本轮固定使用 instance-redetection R1 failure accounting 中的五个 `NO_CANDIDATE` 窗口，不换 detector、
不调 TargetMemory/阈值、不增加 DINOv2/SAM/Sky。它回答两个诊断问题：

1. 把 ADT GT bbox 仅作为 proposal 注入后，现有 TargetMemory → verifier → 2-of-3 → state machine 能否重捕获；
2. 五个窗口分别属于不可见/重遮挡、太小、可见且尺度足够但模型不会，还是 RGB instance identity 风险。

GT-derived oracle output 显式设置 `groundtruth_argument_supported=true` 和
`formal_evaluator_must_reject=true`。现有 RGB evaluator 对它实际 fail closed，未生成正式 evaluation
output。五个窗口各自从序列起点独立重放，只在自己的固定窗口注入 GT，避免前窗干预改变后窗状态。

尺度-可见性 proxy 固定为 visibility `>=0.50`、preview bbox 最短边 `>=24 px`、至少 3 帧。24 px 在
1408→640 observer 输入上约为 10.9 px。该 proxy 是透明诊断定义，不是 learned detectability truth，
原始 visibility、bbox、blur、相机速度和 appearance 数值均保留。

## Oracle Proposal Test

| Window | GT-derived oracle | Reacquisition delay | Downstream interpretation |
|---:|---:|---:|---|
| W0 | PASS | 10 frames | verifier/2-of-3 eventually passes |
| W1 | FAIL | n/a | only one GT-visible frame; 2-of-3 is impossible |
| W2 | PASS | 42 frames | verifier/2-of-3 eventually passes |
| W3 | PASS | 20 frames | verifier/2-of-3 eventually passes |
| W4 | PASS | 17 frames | verifier/2-of-3 eventually passes |

结果是 `4/5`，不是 `5/5`。唯一失败不是尺寸清晰足够时的隐藏状态机 bug：该目标只出现 1 帧，bbox
仅 `3×3 px`、visibility `0.10`，现有 2-of-3 在定义上不可能完成。其余四窗证明下游链可以从正确
proposal 恢复，但需要 10–42 帧；因此 identity/confirmation 不是主瓶颈，也不能描述为零延迟或完全
无影响。极小、低可见 crop 会延迟 appearance gate 和连续确认。

## Five-window audit

表中 visibility 与 preview 最短边均为 `min / median / max`；640-input 最短边是 observer 实际缩放后的
近似像素数。

| Window | Class | GT invisible | GT-visible missed | Visibility | Preview min-dim px | 640-input min-dim px | R1 / YOLOE correct proposal |
|---:|---|---:|---:|---:|---:|---:|---:|
| W0 | `UNOBSERVABLE_OR_HEAVILY_OCCLUDED` | 93 | 41 | .11/.17/.24 | 5/8/10 | 2.3/3.6/4.5 | 0 / 0 |
| W1 | `UNOBSERVABLE_OR_HEAVILY_OCCLUDED` | 430 | 1 | .10/.10/.10 | 3/3/3 | 1.4/1.4/1.4 | 0 / 0 |
| W2 | `TOO_SMALL_WHEN_VISIBLE` | 75 | 60 | .13/.48/.70 | 4/6/9 | 1.8/2.7/4.1 | 0 / 0 |
| W3 | `TOO_SMALL_WHEN_VISIBLE` | 167 | 52 | .23/.91/1.00 | 2/7/10 | .9/3.2/4.5 | 0 / 0 |
| W4 | `TOO_SMALL_WHEN_VISIBLE` | 15 | 23 | .13/1.00/1.00 | 3/17/21 | 1.4/7.7/9.5 | 0 / 0 |

人工查看 contact sheet 后，绿色 GT 框与数值一致：W0/W1 是极低 visibility 的数像素目标；W2/W3
即使 visibility 上升，原 preview 最短边仍不超过 10 px；W4 最清楚，但在 640 detector 输入上最短边
仍不超过约 9.5 px。crop Laplacian 与 appearance 在这些尺寸上易被边缘和插值主导，不足以把 blur 或
identity representation 宣布为主因。五窗内可见的同 prototype distractor 均为 0；这不证明一般场景
没有 identity ambiguity，只表示当前五窗不能支持 D 类结论。

因此四类结果为：A `2`、B `3`、C `0`、D `0`（D 为“本 cohort 无证据”，不是普遍排除）。没有形成
启动 SAM/strong teacher upper-bound canary 的前提。

## Duration decomposition

五个固定 opportunity 合计：

```text
GT-invisible duration                         780 frames
GT-visible but below size/visibility proxy    177 frames
detectable-but-missed duration                  0 frames
```

这是五个 opportunity 的诊断合计，不能与全局 `longest dropout=159` 直接相减。它说明旧 `GT-visible
>=0.10` denominator 把 1–10 个 model-input pixels 的目标也计为可重捕获机会。以后应同时报告 actual
GT-invisible、below-detectability-proxy 与 detectable-but-missed，而不是只优化整段 dropout。

## Evidence identity 与决策

```text
audit_final.json          sha256 dd3ceacec8b4b48d7e0667995e359c4ab6c2ef034f73677ed335059e59097b66
contact_sheet_final.png   sha256 580e8f35a094a28845fd05550a7d8c16517a92480eb1cb483b2c2887e8cbad46
oracle_proposals.json     sha256 ade562cb27666dd19e1eccf643befb89dd53bdfa0e6e183a972f5c5f55867115
```

五个独立 oracle observation SHA-256 依次为：

```text
W0 4225fe504d818a6d2eb3a504b760009a1d28a8e9ae0b9541e11ff3ac831ef2bb
W1 5e8b082c9943af707af9f414f91d0180a041d101a018f37e0584364f8f2d0c97
W2 c3c67e68691727908efb8c15d0d395c6358036e131c9696c9e78bb9a0205ee4e
W3 dcd4f402c082fc381487991658b311c1f4afdf0ceb33e1675718239f674f2852
W4 ba3f293da2a4393f6ed7e8fbbf1ff894e36f07d7acae4f09f25fb4c409eae81c
```

机器输出位于 ignored `artifacts.local/evidence/ba_adt_reappearance_observability_r3/`。

路线决定：不继续 `DINOv2 → SAM → detector → Sky`。当前五窗首先是 source-scale/observability 限制，
不是已建立的 candidate-representation headroom。唯一 successor 是
`ADT1_SMALL_TARGET_SEARCH_SCALE_R4`：只在可重复 Development 上检验 full-preview-resolution、tiling 或
有依据的 ROI search 能否增加原 RGB 中仍存在的有效 proposal pixels，并以新 detectable-but-missed
指标评价；它不承诺恢复 W0/W1 这类实际不可观察窗口，也不授权 teacher→edge、Sky、Android/default-App
或安全结论。

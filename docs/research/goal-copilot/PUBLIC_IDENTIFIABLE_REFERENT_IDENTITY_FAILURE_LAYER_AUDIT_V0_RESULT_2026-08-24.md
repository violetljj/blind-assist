# Public Identifiable Referent Identity Failure-Layer Audit V0

状态：`REVERSIBLE_EXPLORATION / READ_ONLY_EXISTING_EVIDENCE / LEARNED_NEAR_IDENTITY_REPRESENTATION_JUSTIFIED / NEARID_NOT_RUN / NO_P1 / DEFAULT_APP_UNCHANGED`

## 问题与权限

本审计回答：现有 referent identity 错例更像是单张 RGB 没有可用区别，还是通用表示没有保留已存在的区别。
它只读取已冻结的 DINOv2-S local、matched two-reference 与 PDM hard/control artifacts。没有下载或调用模型，
没有重新编码图片、生成新 crop/feature/score、改变 threshold/crop/layer，且没有运行 NearID、P1 或 App。

旧终态保持不变：`TWO_REFERENCE_ZERO_RESCUE_THREE_COLLATERAL`、`PDM_UNARY_REJECTED`、
`RELIABLE_VERIFIER_NOT_ESTABLISHED`、absence `NOT_EVALUABLE`。本审计是 consumed Development evidence 的错误分析，
不是新的 accuracy、Confirmation、NONE、proposal、建筑入口或产品证据。

## 冻结输入收据

| 输入 | 文件 SHA-256 | body SHA-256 |
| --- | --- | --- |
| DINOv2-S local final report | `24252fcc5e771a1e651c0b7408d8b6de86549658c1d232651c2897213faac388` | `4298e6fa7f513f5ac136c4677fe5a3195c21380c1a67b19af1769b3907f61397` |
| matched two-reference final report | `3b188e7ff2e45f38d5131fa6b88cc8e02243d97df60ef68b18e8d972adbd366f` | `d4b053aad1a22cf9c21d74b3184ce58670fd40de1393a4a013d59dd1089022c9` |
| T-LESS frozen private roster | `c7b2a77c2968932ec5d79d0d7a9d81960b5e21639abd81ced880115cc5d6fc92` | `00d77f217bf96a3f6b0e66857d8f7d33d6a7cb5d1f072c106db8956c19cf8535` |
| T-LESS materialized public manifest | `309aee033b7efacfcd9edb1591377179af7a635feed70cad5e88423f75b8f747` | `d70b0f4529bd061962ffb221dca1e409dc8162e51d09cbb63974b77f6f746135` |
| T-LESS private evidence manifest | `4c69dbbdae21383c032f63b46704f0aa9536014b05640af36d699e11a1287ee2` | `2119f7697e6dc6e1d8c9322d68d5cfb6875eee02c48b38e85b81a8e2cbc261ab` |
| T-LESS DINO baseline final report | `347a4d981cd83bac700a71662aba3ad7e47680071909e02330e6a733186bdd8d` | `a500b12db5607cef6d67aeceef7d89fd20462a73268ecd5e403e86bd189ce057` |
| PDM challenger final report | `2fa11c651f3a0c917c246883a54c411f9b6d10c0c22f24394b6d3216ab90f7c3` | `92a1b25d041edb057e2664240b367062e7288d18408950b4430b8616e76852f7` |
| Frozen PDM crop manifest | `7f0c8c8743bec5319a066fd92fb63fb91087129ea88545d6eaec6649b25d42a8` | `949f4dabb9bd222428420b0a5e6da639093e52b9099118e19e34f28816a9a5c7` |

这些 manifests 封存每个 reference/frame 的 image hash、冻结 bbox、candidate slot 与 private native object identity。
审计直接查看已有 400×400 reference 和 720×540 competition frame；表中的面积按实际 720×540 raster 复算，
没有生成新的派生图像。

## 判定规则

- `NEAR_IDENTITY_REPRESENTATION_COLLAPSE`：原始像素中有可描述的物体级区别，候选尺寸与可见度足够，但 DINO/PDM
  仍反转或只留下脆弱 margin。
- `LOCAL_LAYOUT_INFORMATION_LOST`：区别主要来自部件数量、排列、连接或缺失关系；现有 DINO 双向 mean-nearest
  与 PDM masked top-1 mean score 都不保留候选内的显式空间对应。
- `PIXEL_IDENTITY_UNDERDETERMINED` 只有在现有 reference 与 candidate 中看不到稳定区别时才允许；本审计没有足够证据
  为任何 pair 签署该强归因。
- `BACKGROUND_CONTEXT_SHORTCUT` 需要冻结的背景反事实或分量分数；当前同帧候选与 raw score 不能单独建立该因果归因。
- `VIEW_OR_QUALITY_CONFOUND` 需要可定位的模糊、边界、尺度、遮挡或不可比较视角证据；不能仅凭低 margin 推断。
- 正确 control 没有观测到 residual failure，或跨面歧义无法由现有 artifacts 分解时，保持 `UNKNOWN`。

## 逐 pair 证据表

`vis` 为 target/distractor visible fraction，`area` 为各 bbox 占 720×540 的百分比；margin 始终是 target minus distractor。

| Pair | 角色与迁移 | vis / area | DINO → PDM margin | 故障层 | 冻结像素证据与边界 |
| --- | --- | --- | --- | --- | --- |
| `tless-001-pair-01` | hard；wrong→wrong | `.960/.948`; `5.74%/3.80%` | `-.19699 → -.01254` | `NEAR_IDENTITY_REPRESENTATION_COLLAPSE` | object 23 的长条多插口/侧针结构与 object 21 的紧凑插头体可见；两种表示仍选 21。 |
| `tless-001-pair-03` | low-margin hard；correct→correct | `.942/.958`; `3.85%/3.93%` | `+.01311 → +.00584` | `NEAR_IDENTITY_REPRESENTATION_COLLAPSE` | object 23 与 20 的轮廓、插口数量和针脚位置均可见，但两臂只保留近零 margin；这是脆弱保留，不是 rescue。 |
| `tless-002-pair-03` | low-margin hard；correct→collateral | `.936/.967`; `5.37%/4.27%` | `+.00613 → -.00223` | `LOCAL_LAYOUT_INFORMATION_LOST` | object 9 与 7 的顶面槽位、端子和底座布局不同；PDM 反转，两个 scorer 均无显式 layout consistency。 |
| `tless-003-pair-01` | hard；wrong→rescue | `.950/.961`; `12.20%/4.57%` | `-.00846 → +.01664` | `LOCAL_LAYOUT_INFORMATION_LOST` | object 7 与 6 的圆孔数量、连接和壳体布局可见；PDM 救回只证明该 pair 有可恢复信号，不建立可靠 verifier。 |
| `tless-003-pair-02` | low-margin hard；correct→collateral | `.914/.924`; `9.50%/8.56%` | `+.00076 → -.00494` | `LOCAL_LAYOUT_INFORMATION_LOST` | object 7 与 9 都由相似白色局部构件组成，稳定区别主要在多孔模块的排列与连接；PDM 击穿原近零正确。 |
| `tless-003-pair-03` | matched control；correct→correct | `.825/.951`; `6.89%/7.66%` | `+.01990 → +.01161` | `UNKNOWN` | control 被两臂保留；现有结果没有 residual failure，不能从较低 target visibility 反推 quality failure。 |
| `tless-004-pair-02` | matched control；correct→correct | `.859/.794`; `3.59%/7.82%` | `+.01343 → +.01757` | `UNKNOWN` | 两臂均保留 target；没有错误可归因，也不能把 visibility 差异事后写成 causal explanation。 |
| `tless-005-pair-01` | matched control；correct→collateral | `.966/.940`; `7.97%/4.77%` | `+.01684 → -.01165` | `NEAR_IDENTITY_REPRESENTATION_COLLAPSE` | object 28 的矩形三圆纹盒体与 object 5 的单圆插座在轮廓和部件上明显不同；PDM 仍反转。 |
| `tless-006-pair-03` | matched control；correct→correct | `.978/.853`; `4.35%/4.06%` | `+.01344 → +.00747` | `UNKNOWN` | 两个圆形构件均被正确排序但 margin 较小；仅凭小 margin 不能区分 identity、view 或 context。 |
| `tless-007-pair-02` | matched control；correct→collateral | `.899/.996`; `11.81%/4.14%` | `+.02113 → -.00549` | `NEAR_IDENTITY_REPRESENTATION_COLLAPSE` | object 8 的大矩形模块与 object 26 的小方形开关在尺度、外形和部件关系上可分；PDM 仍反转。 |
| `tless-007-pair-03` | hard；wrong→wrong | `.920/.969`; `5.57%/5.25%` | `-.03766 → -.01272` | `UNKNOWN` | reference 主要显示 object 8 底面，competition 主要显示前/侧面；现有 artifacts 不能区分跨面信息不足与表示失败。 |
| `tless-009-pair-02` | matched control；correct→correct | `.996/.978`; `3.46%/4.45%` | `+.01351 → +.00634` | `UNKNOWN` | 方形开关与三圆纹盒体被两臂保留；control 不产生 residual failure 归因。 |

分层计数为：`NEAR_IDENTITY_REPRESENTATION_COLLAPSE=4`、`LOCAL_LAYOUT_INFORMATION_LOST=3`、
`UNKNOWN=5`、其余三类均为 `0`。`UNKNOWN` 没有被强行写成 negative 或 single-reference insufficiency。

## 跨 probe 一致性

1. DINO local 在已消费 C2 上为 `13/17`，仍保留 stable distractor 与 3 个 collateral，说明 order-free local evidence
   有信号但没有可靠 identity authority。
2. matched two-reference 的 single=`14/14`、fixed max=`11/14`、`0 rescue / 3 collateral`；R2 更常强化 distractor，
   因此“再给一个通用 exemplar”不是可见信息不足的可靠修复。
3. T-LESS PDM 为 `1 rescue / 4 collateral`。其中至少 3 个直接错误/击穿 pair 在原始像素中具有充分可见的外形或部件差异，
   另有 3 个 pair 的主要区别落在当前 scorer 没有显式建模的布局关系。

这些事实排除了把全部残余错误统一解释成“小目标、遮挡或像素里没有区别”。它们没有排除个别 pair 的跨面信息不足，
也没有建立 background shortcut 的因果证明。

## 唯一终态

```text
LEARNED_NEAR_IDENTITY_REPRESENTATION_JUSTIFIED
```

理由是现有证据已经包含多个“像素差异可见、尺寸/可见度足够、通用 DINO/PDM unary 仍压扁或反转”的独立 pair；
因此 learned near-identity objective 是下一次最有信息量的新表示假设。`SPATIAL_LAYOUT_VERIFICATION_FIRST` 不作为首选，
因为 residual 不只发生在细粒度排列 pair；layout 应保留为 NearID 之后的独立 mechanism arm，而不能预先融合。

该终态不表示 NearID 已运行或会成功，不恢复已消费 cohort 的模型/阈值选择资格，也不授权训练、下载、NONE、P1、
Active Search 或默认 App。若用户另行授权，唯一优先的新模型实验才是使用新 train/calibration/sealed-test 分层的
`NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0`；本审计本身到此停止。

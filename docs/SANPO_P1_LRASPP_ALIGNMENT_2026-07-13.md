# SANPO P1 LR-ASPP 结构对齐审计

## 结论

P1 已完成四组、每组五个 head-only 短跑的受控验证。最终保留 **sigmoid gate + 移除 pooled-BN/ReLU6**，拒绝 OS4 detail 与 dilated OS16 semantic 作为当前候选的默认 endpoint。

- sigmoid/no pooled-BN 把最佳 seed 从 mIoU/boundary `0.4344/0.4506` 提升至 `0.4642/0.5235`，跨后端等价也为 green。
- 但固定 sampler 的三个 model seed selection range 从 `0.2685` 增至 `0.2951`；该修正提高了上限，没有解决初始化稳定性。
- OS4 detail 使两个 seed 的 boundary IoU 坍塌到 `0.0271/0.0130`。
- OS4 + dilated OS16 的最佳 selection 仅 `0.0968`；OS8 + dilated OS16 的最佳 selection 仅 `0.1549`，证明失败不只来自 OS4 skip。
- 所有候选仍为 `do_not_replace_default_model`。下一主线进入 P2 确定性 quota sampler，而不是继续调 LR、分辨率、decoder 或 boundary 权重。

## 官方结构依据

[MobileNetV3 论文 Fig.10 与 §6.4](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) 的 LR-ASPP 使用 OS16 semantic、图示 OS4 detail，并在 pooled scale 分支采用池化、1×1 卷积与 sigmoid。主 semantic 分支仍为 1×1 Conv、BN、ReLU。

[torchvision LR-ASPP 官方实现](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/segmentation/lraspp.html) 使用 OS16 high + OS8 low，并同样采用 `AdaptiveAvgPool → Conv → Sigmoid`，pooled gate 中没有 BN/ReLU。因此 sigmoid/no pooled-BN 是两份权威实现的共同点；OS4 只属于论文图示口径，不应误称为 torchvision 结构。

## 受控矩阵结果

表中 mean/min/range 只统计固定 sampler `20260711` 时的三个 model seed，用于判断初始化及关联 Torch 随机状态的稳定性。

| 变体 | Detail/Semantic | Best mIoU | Best boundary | Model-seed score mean | min | range | 决策 |
|---|---|---:|---:|---:|---:|---:|---|
| P0 原门控 | OS8/OS32 | 0.4344 | 0.4506 | 0.2715 | 0.1739 | 0.2685 | 结构不一致 |
| P1-A sigmoid/no pooled-BN | OS8/OS32 | 0.4642 | 0.5235 | 0.3071 | 0.1970 | 0.2951 | 保留结构修正，不晋级 |
| P1-B OS4 detail | OS4/OS32 | 0.3388 | 0.2386 | 0.1183 | 0.0250 | 0.2549 | 拒绝 |
| P1-C 论文端点 | OS4/OS16 dilated | 0.2636 | 0.0593 | 0.0721 | 0.0311 | 0.0657 | 拒绝 |
| P1-D torchvision endpoint 口径 | OS8/OS16 dilated | 0.2155 | 0.1209 | 0.0604 | 0.0049 | 0.1500 | 拒绝 |

P1-A 固定 model seed `20260711` 时，三个 sampler seed selection 为 `0.4921/0.4874/0.4849`，range 仅 `0.0072`；P0 关于“sampler 随机序列不是高方差主因”的结论继续成立。

## 实现

- pooled gate 改为 `GlobalAveragePooling → Conv1×1 → Sigmoid`，删除 pooled BatchNorm 和 ReLU6。
- 新增明确的 `detail_output_stride={4,8}` 与 `semantic_output_stride={16,32}` 合同。
- OS16 使用 atrous depthwise block 保留 MobileNetV3 后半段深层特征，不用裁掉后段网络的 OS16 中间层 shortcut 冒充论文结构。
- OS4 endpoint 固定为首个完整 bottleneck 的 `expanded_conv_project_bn`，避免旧空间尺寸字典误选下一 block expansion tensor。
- 训练、Torch/TensorFlow 等价、TensorFlow export 和候选质量门均绑定 architecture revision、detail stride、semantic stride 与模型文件 SHA256；旧授权 fail closed。
- 修复 export report 中硬编码 `256×256` 的输入/输出合同描述，使其跟随实际 input size。

默认保留配置为 `architecture_revision=lraspp_sigmoid_no_pooled_bn_v1`、OS8 detail、OS32 semantic。OS4/OS16 仍保留为显式审计开关，不作为默认路线。

## 证据与边界

本地产物位于（不提交）：

- `test-artifacts.local/segmentation-candidate/p1-sigmoid-no-pooled-bn-20260713/`
- `test-artifacts.local/segmentation-candidate/p1-os4-detail-os32-semantic-20260713/`
- `test-artifacts.local/segmentation-candidate/p1-os4-detail-os16-dilated-semantic-20260713/`
- `test-artifacts.local/segmentation-candidate/p1-os8-detail-os16-dilated-semantic-20260713/`

四份 training report sidecar 均与实际 SHA256 一致。保留权重的 backend-equivalence v3：`max_abs=1.7523765563964844e-05`、argmax agreement `1.0`、报告 SHA256 `06c9d331b3d4770c9b7698d1fdd5c660ebcc29fae8a8a862c19f33ae1f2985f4`。

本轮只读取 train/dev，未读取 blind 标签，未生成 INT8，未运行设备门，未修改或替换 App 模型。head-only 结果不能证明完整 fine-tune 后的生产收益。

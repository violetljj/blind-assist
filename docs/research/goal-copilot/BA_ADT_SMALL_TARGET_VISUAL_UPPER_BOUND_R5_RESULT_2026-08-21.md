# BA-ADT small-target visual upper bound R5 result

状态：`R5_CLOSED_INCONCLUSIVE / DINOV_1_OF_3_WEAK_POSITIVE / SAM31_IMAGE_CROSS_IMAGE_PROMPT_NOT_EVALUABLE / CONSUMED_DEVELOPMENT_ONLY / NO_MORE_TEACHERS / DEFAULT_APP_UNCHANGED`

## 结论

DINOv-SwinL Teacher A 在冻结的 W2/W3/W4、97 个 eligible RGB frames 上只命中 W4 的一帧：
`correct proposal windows = 1/3`，`correct candidate recall = 1/97`。唯一正确 proposal 位于 preview
frame 3307，GT 最短边为 20 px；W2/W3 的 77 帧和 W4 其余 19 帧均未命中。正确候选 confidence
为 `0.1433`，低于同帧最强错误候选的 `0.3550`；全部 97 帧共产生 1,974 个候选级
wrong-instance proposals。因此该结果只是“较大目标上的弱阳性”，没有证明 appearance-only
tiny-target redetection 已被打穿，也不授权 teacher-to-edge、Sky、蒸馏或部署。

冻结门允许在 `1/3` 后最多运行一个机制不同的 SAM 3.1 image-only Teacher B。正式固定帧推理前的
官方接口审计发现，这个 arm 无法按冻结合同实例化：官方 SAM 3.1 checkpoint
`sam3.1_multiplex.pt` 只由 multiplex video predictor builder 加载；公开 image processor 的 visual
box 必须在已经 `set_image` 的同一张图上调用，并直接作为该图的 geometric prompt。仓库没有把历史
reference image/bbox 编码后传入另一张 target image 的受支持 image-only API。

以下替代均被拒绝，且没有读取 Teacher B 固定帧 outcome：

- 在 target image 上提供 bbox 会泄漏目标位置，等价于 oracle；
- 将 exemplar 拼贴进 target 会改变冻结 RGB 输入和目标尺度；
- 把历史 exemplar 与 W2/W3/W4 组成伪视频并传播会改变 image-only 机制、时间语义与 97-frame 合同；
- 使用 SAM 3 image checkpoint、文本 `carrot`、私有 embedding hack 或第三个 teacher 都会改变冻结 arm。

因此 Teacher B 不是 `0/3`，而是 `NOT_EVALUABLE_INTERFACE`。R5 最终结论为
`R5_CLOSED_INCONCLUSIVE`：DINOv 只建立 1 帧较大目标的弱能力上界；SAM 3.1 没有产生科学结果。
按预先约定不追加 T-Rex2、DINOv2、GroundingDINO 或其他模型，也不建立 R6/R7 rescue。工程主线从
当前已消费胡萝卜窗口的 appearance-only detector 堆叠退出，转向 camera motion/VIO/SLAM/world
memory，或另立与医院入口等大尺度 destination grounding 对齐的新任务；这不是“所有 RGB 模型都不行”
的普遍结论。

## Teacher A frozen result

| 指标 | DINOv-SwinL |
|---|---:|
| correct proposal windows | 1/3 |
| correct candidate recall | 1/97 (0.0103) |
| W2 / W3 / W4 correct frames | 0/27 / 0/50 / 1/20 |
| first correct proposal latency | W4 first eligible 后 15 帧 |
| minimum target dimension at success | 20 px |
| correct confidence | 0.143310546875 |
| strongest wrong confidence on success frame | 0.35498046875 |
| candidate-level wrong-instance proposals | 1,974 |

Teacher A 使用官方 `UX-Decoder/DINOv` source revision
`53bf20d5cfdbb86fa35141a1cff432d4923599f2` 和 Swin-L checkpoint。远端只接收 RGB video、冻结
RGB-only observations 与 runner；ADT GT 仅由本地 evaluator 读取。正式 97-frame GPU stage 为
56.62 s，峰值 CUDA reserved 15,426,650,112 bytes。

## Teacher B interface audit

审计对象为官方 `facebookresearch/sam3` source revision
`8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`：

- `download_ckpt_from_hf(version="sam3.1")` 选择 `sam3.1_multiplex.pt`；
- `build_sam3_predictor(version="sam3.1")` 返回 `build_sam3_multiplex_video_predictor`；
- `Sam3Processor.add_geometric_prompt` 要求当前 state 已含 `backbone_out`，并把 bbox 加到该同图
  `geometric_prompt` 后立即 `forward_grounding`；
- 支持的独立 image builder 使用 `version="sam3"` checkpoint，而不是 SAM 3.1 checkpoint。

这是模型/API 能力边界，不是显存、CUDA、下载或远端基础设施失败。4090D 在审计时空闲，故没有必要
下载约 GB 级 checkpoint 或执行一个违反合同的伪实验。

## Evidence identity

- DINOv teacher output：`artifacts.local/evidence/ba_adt_visual_upper_bound_r5/teacher_output_dinov_swinl_final.json`，
  SHA-256 `2AD5B3DA588D65EEFD56F06D1CC07FB9625E51A452DB07D22A4488DAC6B7C979`；
- local evaluator：`artifacts.local/evidence/ba_adt_visual_upper_bound_r5/evaluation_dinov_swinl_final.json`，
  SHA-256 `1336BB277E7D754EED7BAE7C1D1C68AABFC46FB61F0EE32EFBADF5A858D42A47`；
- SAM 3.1 interface audit source hashes：`sam3/model_builder.py`
  `D71D6D3E485EC3EAE48BBC2BA676F401B5853D65C4195A91D077B04DA38121C2`，
  `sam3/model/sam3_image_processor.py`
  `D8738A0EFB6138B01C0DC5DECEFFD29DE9E675860A9A1ED3822B08766373333B`，
  `RELEASE_SAM3p1.md`
  `15816FDD90F809AA6853E9A584AD8C40ED022BA9C6D63BA2AEF326DA0281A9FA`。

全部结论限于同一个已消费 `clean_seq136 / Carrot_A` Development episode。没有 held-out、真实用户、
交互式导航、安全、端侧效率或默认 App 权限。

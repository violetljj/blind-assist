# BA-ADT small-target visual upper bound R5 protocol

状态：`FROZEN_BEFORE_TEACHER_OUTCOME / CONSUMED_DEVELOPMENT / TERMINAL_REDETECTION_GATE / NO_R6_R7_RESCUE`

## Protocol intent

`ADT1_SMALL_TARGET_VISUAL_UPPER_BOUND_R5` determines whether the fixed consumed tiny-target reappearance
windows contain exploitable RGB evidence for a materially stronger visual-query teacher. It does not optimize the
current pipeline. A strong positive result authorizes teacher-to-edge protocol design; a negative result closes
appearance-only tiny-target redetection as an immediate BlindAssist priority.

## Frozen teacher A

Teacher A 是 Meta SAM 3，通过 Ultralytics `8.4.52` 的 `SAM3VideoSemanticPredictor` 运行。权重来自
ModelScope `facebook/sam3` commit `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`；`sam3.pt` 的 LFS object
SHA-256 为 `9999E2341CEEF5E136DAA386EECB55CB414446A00AC2B55EB2DFD2F7C3CF8C9E`，长度
3,450,062,241 bytes。冻结 FP16、model input size 1008、candidate confidence floor 0.05，不扫参数。

Visual exemplar 只能由 RGB-only R1 output 决定：在首个 `ACQUIRED → LOST` segment 中选择 detector
confidence 最大且不低于 0.70 的 frame，规则冻结后得到 frame 208、confidence 0.761395、bbox
`[406.5394, 1017.8552, 444.8676, 1097.8066]`。同时提供冻结短文本 `carrot`。从 frame 0 开始读取原
full-resolution RGB；frame 208 才添加这一次 positive exemplar，之后不追加 prompt、人工点击或修正。

Teacher 输入禁止 GT target location/bbox/visibility、未来位置、固定窗口 timing、scenario answer。GT 仅由
隔离 evaluator 在完整 teacher replay 结束后读取。正式 run 必须处理完整视频；短 `--stop-after-frame`
只允许在 frame 208 附近做 outcome-blind mechanics smoke，不能进入 R5 evaluator。

## Frozen denominator and metrics

R4 的 W2/W3/W4、3-window/97-frame denominator、visibility >= 0.50、source bbox 最短边 >= 4 px、
每窗至少 3 eligible frames 全部冻结。正确 proposal 仍为 IoU >= 0.10。Primary output 只有：

- correct proposal windows `x/3`；
- candidate recall `x/97`；
- wrong-instance proposal count；
- first-correct-proposal latency；
- first success 的最小 target size；
- correct proposal confidence 减 strongest wrong candidate confidence 的 margin。

Flow、YOLO、TargetMemory、identity verifier、2-of-3 confirmation、Goal Copilot 与正式 RGB evaluator 都不
修改。R5 只判 proposal capability；R3 的 GT proposal oracle 已单独证明下游在可用 proposal 下为 4/5。

## Predeclared terminal gate

- `0/3`：关闭 appearance-only tiny-target redetection immediate priority；不蒸馏、不启 Sky、不再换 detector，
  唯一下一方向为 Last-10m destination grounding + camera motion/VIO/SLAM/world memory。
- `1/3`：只允许一个机制明显不同的 Teacher B；先解释成功窗与失败窗的 size/visibility/blur/contrast/view
  change 差异，不授权 edge optimization。
- `2/3` 或 `3/3`：只证明 RGB information 可被 stronger teacher 利用，允许另立 teacher-to-edge protocol；
  不自动授权 Sky、蒸馏、手机部署或默认 App。

无论结果为何，这三个窗口不建立 R6/R7 rescue。胡萝卜 tiny arbitrary-instance benchmark 的结论不得外推
为医院入口、电梯门、服务台、商店入口或公交站牌等更大尺度 destination grounding 的失败。

## Frozen input identity

- preview RGB SHA-256：`77F952E3AF6531A4A4D9DB5D714292545B5D4A33F5C820D22EEEEC6541B4CC32`；
- RGB-only R1 observations SHA-256：`3EECC346683D1FBE97F1CCC72A44E855261D5A63A67DA6E2FAEAFFC433AF9362`；
- R1 failure accounting SHA-256：`7D7B8672FD0E53C272DD0829A6B0F346C5A31070F8F2832443828B4D4C888590`；
- R4 evaluation SHA-256：`83F87B9CB8FA528AAAA8CCAD84380D2285755FBFAEE41383E8BDC169154ECD76`；
- evaluator-only GT SHA-256：`18297581A15FEEF097B57109BA67D52414E203F53AABA82D781E0B658CFE6A63`。

Claim ceiling 始终为 consumed Development teacher capability；不产生 held-out、edge efficiency、真实用户、
交互式导航、安全或默认 App 权限。

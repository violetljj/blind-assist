# BA-ADT small-target visual upper bound R5 Attempt 02 protocol

状态：`FROZEN_BEFORE_TEACHER_OUTCOME / ATTEMPT_01_PRE_OUTCOME_SUPERSEDED / CONSUMED_DEVELOPMENT / TERMINAL_REDETECTION_GATE`

## Amendment reason

Attempt 01 的 SAM 3 在任何 teacher inference 或固定窗口读取前被机械可行性检查否决：3.45 GB
checkpoint 与官方 video memory 路径不适合本机 8 GB GPU。下载被停止，未产生 teacher output，未消费
R5 outcome。Attempt 02 不是观察结果后的换模型；它是首个正式 teacher run 之前的实现替换。

## Frozen Teacher A

Teacher A 为 `google/owlv2-large-patch14-ensemble`，Hugging Face revision
`95e26936e865f87db1742128404b3c035d47d89d`。`model.safetensors` 长度 1,750,520,144 bytes，
SHA-256 为 `D1C2261503C55AAF400667A843A54A5167E3C696334674C4093D6D10F7F40075`。模型为
437,610,760 parameters、native 1008、patch 14，以 FP16、batch 4 tiles 运行。正式 compute 可在用户提供的
RTX 4090D remote worker 上执行；协议、输入哈希、GT evaluator 和结果验收保持本地权威。

Visual prompt 继续使用 Attempt 01 已在结果前冻结的单一 trusted exemplar：RGB-only R1 首个
`ACQUIRED -> LOST` segment 内 detector confidence 最大且不低于 0.70 的 frame 208，confidence
0.761395，bbox `[406.5394, 1017.8552, 444.8676, 1097.8066]`。使用一个 exemplar 属于原始允许的
1-5 范围，不做 exemplar 数量 sweep。

只在 R1 RGB-only observation 标记 `target_visible=false` 且已越过 exemplar frame 时调用 teacher；仍顺序
读取并输出完整 3,824 帧。每个 search frame 固定复用 R4 S2 的 2x2、20% overlap geometry，各 tile
resize 到 checkpoint native 1008。OWLv2 官方 image-guided postprocess threshold 固定 0.0、tile NMS
0.30、每 tile top 20；映射回全图后 NMS 0.50、每帧 top 50。没有阈值、尺度、tile 或 candidate budget
sweep。

Teacher 输入只有完整 RGB、R1 RGB-only search schedule 与历史 trusted exemplar。禁止 GT target
location/bbox/visibility、未来位置、固定窗口 timing 或 scenario answer。远端不接收 GT。GT 仅由完整
teacher replay 返回本地并通过隔离 evaluator 后读取。

## Frozen outcome gate

Attempt 01 的 3-window/97-frame denominator、IoU >= 0.10、指标和三分支 gate 全部原样继承：

- `0/3`：关闭 appearance-only tiny-target redetection immediate priority，转向 destination grounding 与
  VIO/SLAM/world memory；
- `1/3`：只允许一个机制明显不同的 Teacher B；
- `2/3` 或 `3/3`：允许另立 teacher-to-edge protocol，但不自动授权 Sky、蒸馏、手机部署或默认 App。

这三个窗口不建立 R6/R7 rescue。结论仅覆盖 consumed `clean_seq136 / Carrot_A` tiny-target proposal
capability，不外推到医院入口、电梯门、服务台、商店入口或公交站牌等 destination grounding。

## Frozen input identity

- preview RGB SHA-256：`77F952E3AF6531A4A4D9DB5D714292545B5D4A33F5C820D22EEEEC6541B4CC32`；
- RGB-only R1 observations SHA-256：`2B1D9D6D9B3B7548FE98DADD597D5403C56266417C747484DAF66480973C249F`；
- R1 failure accounting SHA-256：`7D7B8672FD0E53C272DD0829A6B0F346C5A31070F8F2832443828B4D4C888590`；
- R4 evaluation SHA-256：`83F87B9CB8FA528AAAA8CCAD84380D2285755FBFAEE41383E8BDC169154ECD76`；
- evaluator-only GT SHA-256：`18297581A15FEEF097B57109BA67D52414E203F53AABA82D781E0B658CFE6A63`。

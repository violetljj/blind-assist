# BlindAssist Assistive Geometry B1 A0 TRAIN execution lock

终态：`B1_A0_DEPTH_ONLY_THREE_SEED_TRAIN_EXECUTION_LOCK_PASS`

本阶段关闭了 A0 depth-only baseline 的真实 TRAIN loader、双方向 sampler、optimizer/schedule、
混合精度、梯度累积与 checkpoint/resume 执行问题。它没有启动 20-epoch 正式训练，也没有打开
Development/Confirmation outcome。

## 冻结执行合同

- 只使用 16 个 TRAIN parent、每 epoch 4,800 帧；
- seed 固定为 `17/29/43`，每 seed 20 epoch；
- 先在每个 300-frame parent 内确定性 shuffle，再做 parent-balanced round-robin；
- 随后按 portrait/landscape 分桶，严禁 mixed-shape microbatch；
- micro-batch `4`、同方向累积 `4`，effective batch `16`；
- 小于 16 的方向 remainder 延迟到下一 epoch 同方向队列，20 epoch 后 carry 必须清空；
- 每 seed 共 `6,000` optimizer steps；AdamW，base LR `2e-5`，300-step linear warmup，
  cosine 到 `0.05×`，gradient clip `1.0`；
- A0 只激活 masked log-depth 与 valid-neighbor log-gradient，assistive heads 不训练，teacher 禁用；
- checkpoint 必须含 model、optimizer、scheduler、scaler、orientation carry、Python/NumPy/Torch/CUDA
  RNG 与 protocol hash。

## 真实 TRAIN smoke

seed 17 在当前 CUDA 环境选择 BF16、关闭 TF32，分别执行了一个 portrait 与一个 landscape
16-sample optimizer step：

| 方向 | mean loss | clip 前 gradient norm | 非零 encoder/depth 参数 |
|---|---:|---:|---:|
| portrait | 0.95634 | 6.82721 | 616 |
| landscape | 1.48731 | 6.62860 | 616 |

epoch 0 计划 299 步，正确保留 portrait `4`、landscape `12` 个 carry；峰值 CUDA memory
`2,053,701,632 bytes`。训练 SelectiveScan 继续直达显式 Autograd Function，缺失 Autograd-key
警告为 0。

写出的 95,568,600-byte checkpoint 已精确恢复 model、optimizer、scheduler、scaler、sampler
与 RNG，SHA-256 为
`E458FA743234BF4B364C6071C414A4836049843E04DA0F0FD07462DC522A1FF9`。

## 保留的负证据

Attempt 1 在两个 optimizer step 和 checkpoint 写入之后，因
`torch.load(map_location=cuda)` 把 CPU RNG ByteTensor 一并搬到 CUDA，最终报：

`TypeError: RNG state must be a torch.ByteTensor`

它保持 `HOLD_CHECKPOINT_RNG_STATE_DEVICE_MISMATCH`。Attempt 02 只把 checkpoint 首次加载位置改为
CPU，再通过 `load_state_dict` 恢复到现有 CUDA 对象；没有改变模型、optimizer、数据、精度或 loss。

## 权限边界

现在只授权唯一 successor：

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_THREE_SEED_FORMAL_TRAIN_EXECUTION`

这不是模型质量 PASS。A1–A4、Development/Confirmation outcome、双教师、部署/default App、
产品与 safety 仍未授权。

# HFTF Stage C D25：THOR-MAGNI time-to-entry result

日期：2026-08-03

证据角色：Development / continuous-time representation canary

研究主线：不变

默认 App：不变

## 结论

D25 的 5 folds × current/history 共 10 个训练 runs 已完整产生。冻结 gate 仅
monotonicity 1/9 通过，终态为：

`D25_THOR_MAGNI_TIME_TO_ENTRY_INCREMENT_NOT_SUPPORTED`

history 相对等容量 current：

| horizon-macro metric | mean delta | 正折 |
|---|---:|---:|
| source-session-macro AUROC | -0.04575 | 2/5 |
| source-session-macro AP | -0.06348 | 1/5 |
| pooled AUROC | -0.03031 | 2/5 |
| pooled AP | -0.02951 | 1/5 |

各 horizon 的 source-session-macro 五折 mean：

| 首次进入 horizon | AUROC delta | AP delta | Brier delta |
|---|---:|---:|---:|
| ≤0.5 s | -0.00943 | -0.04279 | -0.00710 |
| ≤1.0 s | -0.04141 | -0.03858 | -0.00728 |
| ≤1.5 s | -0.10231 | -0.10610 | +0.03996 |
| ≤2.0 s | -0.02987 | -0.06644 | +0.02391 |

四个 horizon 的 AUROC/AP mean 没有一个为正；因此不能把 0.5/1.0 s 的 Brier
改善单独包装成 time-to-entry representation increment。

## 科学解释

D25 回答的是一个真正不同于 D23 的问题：模型不再只判断 2 秒内是否进入，而是预测五类
首次进入时间，并由 softmax 构造结构上单调的四个累计概率。标签机会充足：

- 530 个 current-negative proximity-eligible anchors；
- 五类 counts 为 `61/32/35/29/373`；
- 四个累计 horizon positives 为 `61/93/128/157`；
- 每个 horizon 在五个 held-out folds 都有正负；
- cumulative monotonicity violation 为 `0`。

因此当前负结果不是标签不可评价，也不是把系统 claim ceiling 误当算法失败。它表明：

> 在当前 D22 MobileNet + dense-flow dynamics encoder 上，把二元 onset 改成 ordinal
> time-to-entry 不会建立跨 source 的 timing representation increment。

D23 的 binary proximity source-macro robustness 仍保留；D24 的 event conversion
负结果也保持。D25 只关闭当前 encoder family 上的 time-to-entry successor，不否定
未来风险场研究问题。

## OOM 工程故障与修复

首次执行在 fold1 history arm 的任何 held-out metric 产生前触发 CUDA OOM。原因是
执行器为统一写 checkpoint，把已完成的 current 模型继续保留在 GPU 上，再启动
history，造成跨 arm allocator 累积。

修复 commit `9b65e37` 只改变资源生命周期：

- 每个 arm 评价后立即移到 CPU；
- 立即写 checkpoint；
- 删除模型、执行 garbage collection 与 CUDA cache release；
- 模型、数据、target、loss、epoch、seed、fold 和 gate 全部不变。

随后从 fold0 开始完整重跑，没有复用首次执行的 fold0 metric。修复后显存约
1.6–1.7/8.1 GiB，完整越过原 OOM 点并产生 5/5 folds。该 OOM 是可修复工程无效，
没有烧毁 cohort，也没有进入科学终态。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d25-thor-magni-time-to-entry-v0/
    report.json
    report.json.sha256
    checkpoints/
```

- report SHA-256：
  `ea7289a7684a02fe08651de828082678cc617933255663f7bac003070cf78b4b`；
- 5 current + 5 history checkpoints 均在 report 内逐项记录 SHA-256；
- report 加 checkpoints 合计约 39.7 MiB；
- 首次工程无效与有效重跑 stdout/stderr 分开保留在
  `artifacts.local/evidence/hftf/`。

## 下一科学变量

当前 THOR proximity target 对“任何人体进入 1.25 m”建模，不区分用户选择的候选方向；
它可能奖励 looming，却没有直接表达哪条行动路径会冲突。下一候选不再改变时间 head 或
dense-flow operator，而是改变 supervision geometry：

> 从 source-native 世界轨迹生成左/中/右候选恒速路径，与其他人体的真实未来轨迹计算
> counterfactual time-to-collision field，再比较 history 与 current。

这会回到 HFTF 的核心——一次输出多个候选方向的未来身体冲突——而不是继续救一个
route-agnostic proximity score。先做 outcome-open label opportunity census；只有
方向×horizon 在每折都有足够正负才冻结学生 canary。

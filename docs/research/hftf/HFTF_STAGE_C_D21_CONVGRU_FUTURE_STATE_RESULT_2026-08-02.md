# HFTF Stage C D21：ConvGRU future-state result

日期：2026-08-02

证据角色：Development / synthetic recurrent-dynamics canary

研究主线：不变

默认 App：不变

## 结论

冻结的 seed-17 gate 未通过：

`D21_CONVGRU_FUTURE_STATE_CANARY_NOT_SUPPORTED`

D21 通过 7 项检查中的 5 项。history-minus-current：

| metric | mean delta | 正折 |
|---|---:|---:|
| environment-macro cell AUROC | +0.00390 | 2/3 |
| environment-macro cell AP | +0.00356 | 2/3 |
| pooled target-macro cell AUROC | +0.00905 | 3/3 |
| pooled target-macro cell AP | +0.00992 | 3/3 |
| sample-macro AUROC | +0.01728 | 2/3 |
| sample-macro AP | +0.01345 | 2/3 |

四个 targets 的三折 mean AUROC 与 AP 仍全部同时为正；near-head AUROC/AP 为
`+.01239/+.01348`，far-head 为 `+.01781/+.01016`，四项均 3/3 folds 正。
这些是有效的 Development 分层结果，不因完整 gate 失败而被改写为工程无效或
`NOT_EVALUABLE`。

未通过的两项都是冻结的 environment-macro effect floor：

- AUROC `+.00390 < +.010`；
- AP `+.00356 < +.005`。

两项的 positive-fold 要求都已通过；target breadth 与 sample noninferiority 也
全部通过。因此失败不是路径、parser、落盘、序列化或运行中断造成的工程无效，而是
完整执行后的科学负结果：当前 ConvGRU operator 没有把 pooled/target-local 正信号
转成预定强度的跨环境分离。

## 相对 D20 的机制信息

D21 保留 D20 的相同 20-channel aligned dense-flow dynamics，只把一次性 3D collapse
换为按 `-.8→-.6→-.4→-.2 s` 递推的 16-channel ConvGRU。两臂使用相同
1,017,316 参数、30 epochs、三折与 seed17；current comparator 的 zero dynamics
在任意 recurrent weights 下保持精确零状态。

相对 D20：

- pooled cell AUROC 从 `+.00604` 增至 `+.00905`；
- pooled cell AP 保持在约 `+.010`；
- sample AUROC/AP 从 `+.01612/+.01068` 增至 `+.01728/+.01345`；
- environment-macro AP 却从 `+.03421` 降至 `+.00356`。

因此递归状态可以继续提炼总体与 head-local onset signal，但没有解决环境异质性，
反而丢失了 D20 最强的 environment-macro AP 优势。D20 继续作为当前最强
Development mechanism signal。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d21-tartanground-convgru-future-state-v0/
```

- samples：495；
- environments：15；
- folds：3；
- paired training runs：6；
- flow SHA-256：
  `10be7dfe3f50b32a89d98fb48fdfb8f72900af078433ab578276cb3814ad13df`；
- report SHA-256：
  `f511a723a2ba3210cbc3441fec45f48dab9078763763e2c320109fa20444bf98`。

## 终止与下一步

按冻结协议不扩 seeds `23/41`，也不继续搜索 hidden width、epoch、loss 或 gate。
当前 lightweight early temporal-state family 到此停止：

`D21_LIGHTWEIGHT_TEMPORAL_STATE_FAMILY_STOP`

这个停止只关闭 D20 dynamics 上继续更换轻量 temporal operator 的局部模型搜索。
它不撤销 D20 的 broad-onset Development 正结果，也不关闭 HFTF 支线。下一科学变量
应离开同一 operator family，直接检验 D20 的 dense-flow dynamics 正信号能否在独立
source 或真实事件层复现；该检验必须单独冻结为 representation/transfer 诊断，不能
被包装成主线、App 或安全主张。

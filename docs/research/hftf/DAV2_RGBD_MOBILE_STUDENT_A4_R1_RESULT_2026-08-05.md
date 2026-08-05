# DA V2 RGB-D 轻量 student A4 R1 结果

日期：2026-08-05

## 结论

`A4_RGBD_MOBILE_STUDENT_ENGINEERING_NONINFERIORITY_NOT_SUPPORTED`

A4 R1 仅通过 P1 R1 的 `4/14` 工程门，禁止 Android profile。真实 RGB-D 监督没有解决
轻量 student 的保守占用塌缩；它把 false-clear 压到 `0.096%`，代价是 false-block
`60.63%`、harmful decision change `55.72%`。

## 训练与锁定

- 可监督 train / validation：`2374 / 590`；
- validation total：`0.3255 -> 0.3240 -> 0.3162 -> 0.3126 -> 0.3098`；
- 选中 epoch：5；参数量：`1,271,281`；
- checkpoint SHA-256：`11AB69B444EE00DAE68E186A3E0A44B7420989E82A4412635D5531EE12557142`；
- P1 cache SHA-256：`EB0C884A657B3E5F19177DADD4A48BBE001A371AF783C2C4BBB87142C2CC4638`。

一次外层命令超时留下的 checkpoint SHA `202D3D...5DCE4` 没有完整 training result，已标为
不可用；最终结果来自完全相同协议的独立目录精确重跑。

## P1 R1

| 指标 | A4 R1 |
|---|---:|
| raw AbsRel | 32.48% |
| scale-aligned AbsRel | 19.82% |
| ground recovery | 98.32% |
| clearance MAE | 1.220 m |
| collision agreement | 39.27% |
| false-clear | 0.096% |
| false-block | 60.63% |
| temporal clearance-delta MAE | 0.224 m |
| harmful / beneficial change | 55.72% / 24.09% |

R1 的 false-block 与 harmful-change 门准确阻止了“低 false-clear 等于更安全”的错误结论。
host CUDA median `7.88 ms`、P95 `8.59 ms` 仅为诊断，不是真机 App latency，且质量失败后
不得用速度救回候选。

## 后续约束

不再对 A4 改 loss、seed、epoch、阈值或 checkpoint。下一项若继续，应切换机制到 canonical
DA V2 的选择性混合精度/token 降本，或先取得新的设备侧量距/最终相机监督；不能继续在这个
已消费 cohort 上搜索轻量 head。

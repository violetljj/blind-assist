# GRAIL M0 ProcTHOR Native-Interaction V2 Protocol

日期：2026-08-25（Asia/Hong_Kong）

状态：`FROZEN_BEFORE_V2_RUNTIME_OUTCOME / ONE_SHOT / STOP_BEFORE_M1_ON_ANY_GATE_FAIL`

V2 继承 V1 的 source、runtime、target definition、controller 参数、输出 interface、counterfactual、门槛与 claim ceiling。唯一语义修正为：当一个目标在固定 1.75 m 查询半径内没有 native reachable position 时，不调用 `GetInteractablePoses`，直接输出显式 `NONE`。这恢复了原协议已经声明的 set-valued pose-or-NONE interface，不改变信息源或成功门槛。

冻结 source 为 ProcTHOR-10K test SHA-256 `9a9fa6f134e76fe87f3fd92c00883651cf9fadf4e9ad4072d6d73be229f001dc`，AI2-THOR `5.0.0 / f0825767cd50d69f666c7f282e54abfe58f1e917`，Docker image ID `sha256:36bc6640b8ecebd35b748712a44411455e09f7d3b984c9bb6d9c82dd2f4b9211`。选择 salt 为 `BLINDASSIST_GRAIL_PROCTHOR_NATIVE_M0_V2`，排除 index 0 以及 V1 roster：`906,498,394,518,795,325,161,676,421,298,500,343`。

V2 roster 在任何 V2 Unity/teacher outcome 前冻结为：

```text
753, 145, 285, 366, 945, 188, 856, 482, 87, 605, 591, 631
```

完整 rank、house hash、参数与门槛见 [`procthor_native_m0_manifest_v2.json`](../../../scripts/research/grail/procthor_native_m0_manifest_v2.json)，manifest SHA-256 必须为 `256455eda03725ab1e5ace1700b01558b8a1c5f7ef9ab357db6190ad4eade5e5`。

一次性门槛保持：houses >=8、targets >=128、types >=6、nonempty coverage >=80%、oracle pose/path=100%、local stability >=90%、action canary >=8 且 100%、structured NONE >=8 且 false commit=0、每个 nonempty target 至少两类反事实且 100% rejected。任一失败即 `STOP_BEFORE_M1`，不得在 V2 roster 调参、改过滤或重跑。

Claim ceiling 仍仅为 synthetic ProcTHOR 3D 与 AI2-THOR native reachable/interactable-pose/action mechanics；不建立 RGB、自然场景、真实相机、用户、产品或安全证据。

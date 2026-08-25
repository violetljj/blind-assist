# GRAIL M0 ProcTHOR Native-Interaction Protocol

日期：2026-08-25（Asia/Hong_Kong）

状态：`FROZEN_BEFORE_HELD_OUT_RUNTIME_OUTCOME / PROCTHOR_10K_TEST / AI2THOR_NATIVE_REACHABLE_AND_INTERACTABLE_POSES / 12_HOUSES / ONE_SHOT / STOP_BEFORE_M1_ON_ANY_GATE_FAIL / DEFAULT_APP_UNCHANGED`

## 为什么更换信息源

ARKitScenes natural mesh + OBB 的 derived teacher 在 fresh 分母只有 `20/79` 非空 set，已经关闭；本协议不调该 cohort 的 floor、clearance、face gap 或采样距离。新 teacher 改用 ProcTHOR-10K 3D houses 与 AI2-THOR simulator-native `GetReachablePositions`、`GetInteractablePoses` 和 Open/Toggle action response。它增加的是可达拓扑、可交互 pose set 与动作语义，不是另一个 mesh 阈值。

## Source 与 split

- 官方 source：`allenai/procthor-10k@439193522244720b86d8c81cde2e51e3a4d150cf`，Apache-2.0；
- Development：官方 `val` split；held-out：官方 `test` split，building/house source-disjoint；
- `test.jsonl.gz` SHA-256：`9a9fa6f134e76fe87f3fd92c00883651cf9fadf4e9ad4072d6d73be229f001dc`；
- AI2-THOR：`5.0.0 / f0825767cd50d69f666c7f282e54abfe58f1e917`；
- runtime：Linux64 + Xvfb + Mesa software GL + FIFO；冻结 image ID `sha256:36bc6640b8ecebd35b748712a44411455e09f7d3b984c9bb6d9c82dd2f4b9211`。

目标只纳入 stationary、non-pickupable 且 `openable` 或 `toggleable` 的 simulator objects。输出是 standing、horizon=0 的 set-valued `(x,z,yaw)` 或 `NONE`。`GetInteractablePoses` 查询只使用 native reachable positions；控制器固定 0.25 m grid、30° yaw、1.5 m visibility distance。

## Development 证据

`val index 0` 只用于调通：32 targets、6 types，native pose coverage=`32/32`，oracle pose/path=`32/32`，局部稳定=`30/32`，每 scene 一个非 Doorway Open/Toggle canary=`1/1`，结构化反事实=`93/93`。Doorway `CloseObject` 在 software GL 超过 backend timeout，因此真实动作只作为每 scene 一个非 Doorway canary；GRAIL closed-loop 定义为 native reachable path 进入 native interactable pose，不把门动画 transport 混入 arrival。

## Frozen roster

在任何 `test` Unity/teacher outcome 前，以 immutable dataset hash、house index、canonical house hash 和固定 salt 做 SHA-256 排序；排除历史曾被 runtime 工作引用但未产生 topology outcome 的 index 0。12 个 index 为：

```text
906, 498, 394, 518, 795, 325, 161, 676, 421, 298, 500, 343
```

完整 rank 与 house hash 见 [`manifest`](../../../scripts/research/grail/procthor_native_m0_manifest_v1.json)。manifest SHA-256 必须为 `b08007652099cbb6efd544ae0c46da4099c0f26920b14336dced6550fd2b0a60`。

## 一次性门槛

| 门 | 阈值 |
|---|---:|
| held-out houses | >=8（冻结 12） |
| stationary actionable targets | >=128 |
| target types | >=6 |
| nonempty native pose coverage | >=80% |
| oracle pose success | 100% nonempty denominator |
| native reachable-path completion | 100% nonempty denominator |
| local pose-set stability | >=90% nonempty denominator |
| non-Doorway action canaries | >=8 且 100% action+revert |
| structured `NONE` | >=8 且 false commit=0 |
| structured counterfactuals | 每个 nonempty target >=2 families 且 100% rejected |

任一门失败即 `STOP_BEFORE_M1`，不在本 test roster 修改 target filter、yaw、radius、visibility distance、稳定阈值、counterfactual 或 gate 后重跑；只有全部通过才授权 M1。

## 复现

```powershell
E:\codex-tools\bin\docker.cmd build -t blindassist-grail-ai2thor-native:5.0.0 scripts/research/grail/procthor_runtime
E:\codex-tools\bin\docker.cmd run --rm --name grail-procthor-native-m0 `
  -e LIBGL_ALWAYS_SOFTWARE=1 `
  -v "${PWD}\scripts\research\grail:/work:ro" `
  -v "${PWD}\artifacts.local\evidence\grail-m0\procthor-native-teacher\data:/data:ro" `
  -v "${PWD}\artifacts.local\evidence\grail-m0\procthor-native-teacher:/evidence" `
  -v "${PWD}\artifacts.local\evidence\grail-m0\procthor-native-teacher\ai2thor-cache:/root/.ai2thor" `
  blindassist-grail-ai2thor-native:5.0.0 python /work/run_grail_procthor_native_m0.py `
  --dataset /data/test.jsonl.gz `
  --manifest /work/procthor_native_m0_manifest_v1.json `
  --output /evidence/output/formal-report-v1.json
```

Claim ceiling：synthetic ProcTHOR 3D + AI2-THOR native reachable/interactable-pose/action mechanics；不建立 RGB student、自然场景迁移、真实相机、用户、产品或安全证据。默认 App 不变。

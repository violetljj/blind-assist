# GRAIL M0 ProcTHOR Native-Interaction V2 Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`EVALUABLE / ONE_SHOT / ALL_GATES_PASS / NATIVE_TEACHER_UPPER_BOUND_ESTABLISHED / M1_AUTHORIZED / DEFAULT_APP_UNCHANGED`

正式 V2 从已推送 commit `9dd2c6686724d7aee6c5ab6d3113c53403a83830`、clean worktree、冻结 manifest SHA-256 `256455eda03725ab1e5ace1700b01558b8a1c5f7ef9ab357db6190ad4eade5e5` 与冻结 Docker image ID `sha256:36bc6640b8ecebd35b748712a44411455e09f7d3b984c9bb6d9c82dd2f4b9211` 唯一执行。正式 report SHA-256 为 `bd4198a7a8a4588b483cc053bd9c19e754a1040f5a78a146b8a319cda838dbb2`。

## 结果

| 指标 | V2 held-out 结果 | 门槛 |
|---|---:|---:|
| scenes | 12 | >=8 |
| stationary actionable targets | 205 | >=128 |
| target types | 7 | >=6 |
| native nonempty pose coverage | 199/205 = 97.1% | >=80% |
| oracle pose success | 199/199 | 100% |
| native reachable-path completion | 199/199 | 100% |
| local pose-set stability | 191/199 = 96.0% | >=90% |
| non-Doorway action + revert canary | 12/12 | >=8 and 100% |
| structured NONE denominator / false commit | 18 / 0 | >=8 / 0 |
| structured counterfactual rejection | 572/572 | 100% |
| nonempty targets with >=2 counterfactual families | 199/199 | 100% |

终态为：

```text
GRAIL_M0_PROCTHOR_NATIVE_INTERACTION_TEACHER_UPPER_BOUND_ESTABLISHED
```

## 裁决

M0 证明在 ProcTHOR synthetic 3D 与 AI2-THOR simulator-native reachable/interactable-pose/action 语义中，set-valued interaction-pose-or-NONE 任务有清晰 teacher/oracle 上界。因此 M1 frozen-encoder 的 `B0/B1/B2/GRAIL` 未见场景、未见实例比较获得授权。

这不证明 RGB grounding、自然场景迁移、真实相机、用户、产品或安全效果；`GetInteractablePoses` 是 simulator-native visibility-distance/FOV 交互位置语义，不等同于开放世界的人类功能可供性 truth。M1 必须分别报告 interaction-pose success、wrong-target pose、absence false commit 和 candidate permutation；只有 M1 清晰超过最强简单基线且不破坏后三个门，才允许 M2 temporal belief、遮挡重捕获与 Android 三环境测试。

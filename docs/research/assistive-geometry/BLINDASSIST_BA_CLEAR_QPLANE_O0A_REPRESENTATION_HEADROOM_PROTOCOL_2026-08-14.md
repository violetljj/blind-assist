# BA-Clear/Q-Plane O0-A 表示上界协议

状态：`FROZEN_REPRESENTATION_AUDIT_ONLY / TRAINING_NOT_AUTHORIZED / FRESH_OUTCOME_UNREAD`

本轮只回答：面向冻结 body/path query 的三参数逆深度射线—平面残差，能否在完全不增加
`UNKNOWN` 的情况下，严格优于 global scale、global affine 与 shared global ray-plane。

## 冻结表示

对 `q = band × horizon`：

```text
r(p)    = K_eff^-1 [u,v,1]^T
rho0(p) = 1 / d0(p)
rhoq(p) = rho0(p)
          + mq(p) * clip(theta_q^T [1,r_lateral(p),r_up(p)], -0.20, +0.20)
```

`theta_q` 只有 3 个自由度。`mq` 由冻结 query 几何和 DepthART 基础几何确定；不得输出
per-pixel correction，不得保存统一 corrected dense depth。每个 query 只在内存中构造临时几何，
调用冻结 clearance reducer 后立即丢弃。

## 两阶段防火墙

1. Phase A 只读取 DepthART 深度、source-native depth、`K_eff` 与 TUM pose/gravity；只用
   source-native support-plane pixels 拟合几何残差。不得调用 clearance reducer，也不得派生
   false-block、false-clear 或三态 outcome。
2. Phase A 完整生成 `candidate-plan.json` 并写入 SHA-256 后，Phase B 才重新加载该计划，
   派生 source task truth 并评价各臂。

TUM pose 采用最近时间戳，冻结最大关联差为 `0.12 s`；120 帧中仅首帧因 pose 轨迹晚于 RGB
起点而使用 `0.101 s` 关联，其余帧均不超过 `0.03 s`。逐帧 delta 必须进入候选计划回执。

拟合像素必须满足 source support-plane 高度绝对值 `<=0.045 m`；task evaluation obstacle cell
高度为 `[0.08,2.00] m`。由于相机下视场在部分 `1.0 m` query 内没有可见地面，拟合在
`[0.2,3.0] m` source support 上按 `exp(-|forward-horizon|/0.75)` 加权，低于 `0.02` 的数值尾部
不进入拟合；临时修正 mask 与 task horizon 不扩张。逐帧逐 query 的拟合/评价像素交集必须为零。

## 六臂与负控

- A0：Frozen DepthART；
- A1：global scale-only geometry oracle；
- A2：global inverse-depth scale+shift affine oracle；
- A3：shared global ray-plane residual oracle；
- A4：query-local ray-plane residual oracle；
- A5：source-native depth task ceiling；
- 负控：cyclic shuffled-query、`10°` wrong gravity、冻结 wrong `K`，以及与 A3 同义的
  shared-theta/globalized Q-Plane。

wrong-gravity 与 wrong-`K` 固定使用对应正常 query 的 source-support 拟合像素 ID，只扰动
ray basis、query mask 和临时修正坐标，避免把“支持像素消失”混成坐标负控效果。

所有 oracle 参数只从 source-native inverse-depth geometry residual 拟合，绝不直接优化
clearance 或状态标签。

## 冻结晋级门

O0-A 只有同时满足机器协议中的全部门才 PASS，包括：A4 严格优于 A1/A2/A3；至少 `3/4`
parent 的 false-block 优于 A3；任一 parent 的 false-clear 增幅不超过 `0.5 pp`；coverage 与 A0
逐 parent 完全相同；至少闭合 A3→A5 clearance gap 的 `20%`；优于 shuffled-query 至少
`0.005 m`；收益覆盖至少 `2/3` bands 与 `2/3` horizons。

PASS 也只允许另立 O0-B runtime-identifiability 协议；不自动授权训练。FAIL 则关闭 Q-Plane，
不创建 learned head。机器可读协议为同名 JSON。

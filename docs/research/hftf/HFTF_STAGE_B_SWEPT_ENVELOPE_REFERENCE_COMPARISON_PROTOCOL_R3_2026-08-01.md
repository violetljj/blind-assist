# HFTF Stage B swept-envelope reference comparison protocol R3

日期：2026-08-01

状态：`FROZEN_RESULT_NOT_RUN`

## 1. Source authority

四个 source preparation 中固定的 sessions 已全部下载为 25-frame hash-bound replay。
4/4 frozen-canonical authority 均为：

- `HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED`；
- canonical transform rank 1；
- source-derived vertical `+Z`；
- local-ground plane 25/25；
- standard-body proxy admitted。

正式 machine-readable protocol 绑定每个 authority report、manifest、dataset spec 与
camera poses SHA-256。任何 byte 或 session set 不一致都在 field outcome 前 fail
closed。

## 2. Obstacle comparison

candidate、baseline、reference 与 R3 gates 完全继承 outcome 前冻结的 source
preparation：

- candidate：stride 8 / offset 4 swept envelope；
- baseline：同一 points 的 angular point-support；
- reference：不相交 stride 4 / offset 2 dense swept geometry proxy；
- shared mask：swept-prism known；
- primary reference count：2；
- thresholds 1/2/4/8 全部保留。

正式 obstacle gates：

- 每 session 每 height known coverage `>=.10`；
- cohort micro-F1 delta `>=+.10`；
- precision delta `>=+.10`；
- recall delta `>=-.02`；
- 4/4 session micro-F1 delta `>=+.05`；
- primary 下 foot/body/head F1 均超过 baseline；
- 四 thresholds 的 cohort F1 与 paired correctness 方向均支持 candidate。

## 3. Ground comparison

ground 仍使用 5 sections、`.18 m` rise、`.15 m` drop 与 missing=UNKNOWN。

- candidate ground：stride 8 / offset 4，每 section 至少 3 points；
- disjoint reference ground：stride 4 / offset 2，按 4× sampling density 把每
  section minimum 固定为 12 points；
- candidate/reference 每 session ground-known coverage 均 `>=.10`；
- shared-known coverage `>=.08`。

若 cohort reference ground risk opportunity 为 0，只能得到 obstacle partial terminal。
若 opportunity 存在，candidate ground precision 与 recall 各须 `>=.80`；否则停止为
`R3_GROUND_PROXY_NOT_SUPPORTED_STOP`。

## 4. Ordered decision

1. source/reference/known readiness；
2. obstacle reference-relative gain；
3. ground opportunity；
4. ground agreement；
5. full Stage B proxy terminal。

任何前序门失败，不解释后序项。即使 full Stage B 通过，也只授权另行冻结 future
Stage C teacher；不自动训练 student 或替换研究主线。

## 5. 权限

正式 R3 outcome 只有在 runner 实现、测试、提交并推送后才可一次执行。当前不授权
future Stage C、student/H2、研究主线、Android、提醒、默认 App、生产或安全 claim。

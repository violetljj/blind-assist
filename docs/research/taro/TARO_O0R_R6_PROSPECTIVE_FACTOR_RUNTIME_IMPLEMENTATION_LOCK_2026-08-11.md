# TARO O0R R6 prospective factor-runtime implementation lock

状态：`IMPLEMENTATION_FROZEN / SYNTHETIC_8_OF_8_PASS / REAL_EXECUTION_FALSE`

机器合同：[JSON](TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK_2026-08-11.json)

## 已实现

`prospective_factor_runtime.py` 把 R6-confirmed factor policy 实现为不接受 FARO/truth/task metric/outcome
参数的 source-only API：

- raw DepthART 与 AppleDepth/confidence 内推 source scale；
- raw candidate source geometry 拟合 baseline support；
- Apple plane 有效时 exact-own SUPPORT，并在 anchored candidate 上提取 BOUNDARY；
- QUERY_CLEARANCE 始终由 raw candidate + baseline support 计算；
- 每个 factor 都绑定 owner depth SHA；BOUNDARY/QUERY_CLEARANCE 额外绑定 source-surface pixel-ID SHA；
- query frame 按 repair 固定为 direct → baseline → unavailable，最后一种仍输出九个 UNKNOWN slots；
- output 不附 uncertainty，不运行 reducer，所有 slot 的 final state authority 都是 false。

## Synthetic evidence

8/8 focused mutation tests PASS。1440×1920 analytic floor fixture 内部恢复 source scale `1.27110935399`，
baseline/direct support height 分别为 `0.96000000529 m` 与 `1.200056157011 m`；9/9 SUPPORT、9/9 BOUNDARY、
6/9 QUERY_CLEARANCE mechanics 可评估，其余 3 个保持 UNKNOWN。全 support-unavailable fixture 保留 9/9 UNKNOWN。

tests 还证明：public builder 没有 result-side 参数；candidate array mutation、错误 factor-depth lineage、少于九个
slots 和 R6 untouched parent role 都 fail closed；重复构建与 canonical roundtrip 一致。

## 尚未证明

本锁没有读取任何真实 frame，也没有枚举 R6 untouched outcome 做 implementation/formation/tuning。未来 runner 仍须
验证 upstream source receipt 与 candidate hash；source-defined surface accuracy、source-surface-specific uncertainty、
最终三态、外部泛化、部署、产品和安全均未建立。

## 唯一后继

`TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_FORMATION_REPLAY_LOCK`

后继只允许先冻结 existing 24 formation parents 的 source-first replay、upstream lineage、Phase-A completion、
FARO-after-seal scoring、资源预算与 non-promotable gates；尚不授权实际 replay。

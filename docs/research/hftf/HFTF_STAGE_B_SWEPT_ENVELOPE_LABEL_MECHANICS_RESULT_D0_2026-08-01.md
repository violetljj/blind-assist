# HFTF Stage B swept-envelope label mechanics result D0

日期：2026-08-01

终态：`STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_ADMITTED_FOR_FRESH_R3`

证据角色：`CONSUMED_MECHANICS_AUDIT_ONLY`

## 1. 结论

冻结的 synthetic human-envelope 标签 mechanics 已满足进入全新 sources 上 formal
H1 R3 的最低条件：

- 7/7 structural canaries 通过；
- 4/4 burned R2 source authority、manifest、spec 与 pose binding 复核通过；
- `UNKNOWN -> SAFE` violation 为 0；
- 四个 session 的 foot/body/head 均有 known cells，且均出现 height-specific output；
- 相对旧 angular point-support，新增 209 个 swept-collision cells。

这只证明标签 mechanics 可执行、非退化且遵守人体横向包络与三态防火墙。它不证明
风险真值、Stage B 对独立 reference 的准确率增益、短时 future 增量、student 可学性、
用户效果或主线替换。

## 2. 绑定结果

本地不可覆盖报告：

`artifacts.local/evidence/hftf/stage-b-swept-envelope-label-mechanics-d0-20260801/mechanics.json`

SHA-256：

`52114e9fbf500f703188de14f41f0f88e6a0cc3a081421d1011bc9192554e57f`

执行实现 commit：`be79f83`

实现文件 SHA-256：

`a41395ae0eafaa5d4a35b65236f25cbf269293c7c1e82a891b99b9e8e4a94735`

冻结 mechanics protocol SHA-256：

`a69d25d77f1e2b72f407980f005c758b965517fd032562a009f91746ea1e0e6a`

首次调用在内存报告完成后因 NumPy boolean 无法 JSON 序列化而停止，没有形成有效
终态。实现随后改为原生 boolean，并在独占创建文件前完成全部 JSON 序列化；删除该
不完整文件后才执行上述成功结果。

## 3. Per-session 结果

每个 session 的冻结 denominator 为 25 frames × 6 directions × 6 distance bins =
900 cells/height。

| session | foot/body/head known | height disagreement | unique swept collisions | ground risk / ground UNKNOWN |
| --- | --- | ---: | ---: | ---: |
| `03694304` | 151 / 279 / 334 | 70/900 = 7.778% | 65 | 0 / 719 |
| `03b6dc99` | 200 / 345 / 371 | 11/900 = 1.222% | 45 | 0 / 680 |
| `03c87279` | 131 / 252 / 189 | 5/900 = 0.556% | 44 | 0 / 741 |
| `03d70593` | 117 / 267 / 408 | 25/900 = 2.778% | 55 | 0 / 765 |

总计 height disagreement 111 cells，unique swept collisions 209 cells。报告中的
dynamic provenance 数量是点对 swept-prism 的重复 assignment count，不是独立动态
对象或运动事件数量，不能支持 dynamic-motion claim。

## 4. 重要限制

真实 burned sources 上没有 ground step/drop risk cell，且 3,600 个 foot cells 中有
2,905 个 ground-UNKNOWN。synthetic fixture 已验证 step/drop 会进入 foot layer，
但真实场景机会与足部 ground reference 的有效性尚未获得支持。

因此 formal R3 必须：

1. 使用全新、outcome-blind 冻结的 source sessions；
2. 把 swept-envelope candidate 与旧 angular point-support 放在相同 known cells 上，
   对独立、预冻结的高密度 geometry reference 比较；
3. 单列 obstacle collision、ground coverage/risk 与 UNKNOWN，不能用更多 positive
   cells 本身冒充准确率增益；
4. Stage B 没有达到冻结的 reference-relative 增益门时停止，不进入 future Stage C；
5. 即使 R3 通过，也只授权另行冻结 future teacher protocol，不自动授权 H2。

## 5. 权限

当前只新增 `fresh-source formal H1 R3 protocol/source preparation` 权限。H2、
student training、研究主线、Android、提醒、默认 App、生产与安全 claim 均未授权。

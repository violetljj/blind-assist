# USTRF canonical observation source-authority data pack R0 结果（2026-07-25）

状态：`AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING / VALID`

权限：`SOURCE_AUTHORITY_CANARY_ONLY / G1_CLOSED / SIGNAL_CLOSED / ROUTE_TRUTH_CLOSED / ANDROID_CLOSED / HUMAN_CLOSED / PRODUCTION_CLOSED`

## 结论

JRDB 官方 test-label archive 证明了一个比现有 41-sequence pack 更强的 source-authority 候选：stitched 2D GT 的 `truncated` 与 `occlusion` 是逐对象、非恒定的 source-native annotation；官方 sensor 文档同时给出 robot base、camera/LiDAR transform、raw/stitched image dimensions 和 cylindrical projection。

但当前 canary 没有 RGB frame identity、capture timestamp 或 route-role truth，因此只能终止为：

`AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING`

它足以推翻“公开来源不存在 truncation authority”的更强悲观判断，但不能解除父 G0 对当前数据的 `SOURCE_AUTHORITY_ABSENT`，不能回填旧 11 event，也不能启动 G1 或 signal。

## 来源与边界

- JRDB 官方页面：<https://jrdb.erc.monash.edu/>
- label/schema：<https://jrdb.erc.monash.edu/benchmark/>
- dataset scope/license：<https://jrdb.erc.monash.edu/dataset/>
- sensor setup：<https://download.cs.stanford.edu/downloads/jrdb/Sensor_setup_JRDB.pdf>

官方资料将 JRDB描述为 human-comparable social mobile robot 的室内/室外 stationary + moving perspective 数据；2D tracking schema 明列 truncation/occlusion、逐帧 track id 与 pixel bbox，项目页面标注 CC BY-NC-SA 3.0。本地用途因此固定为 non-commercial research-only，不产生再分发或生产权限。

nuScenes 只做 metadata 负对照：其 image size/time、calibrated sensor 和 ego pose 是权威 metadata，但 `visibility_token` 是六相机聚合可见比例，不是 per-camera truncation，且 vehicle perspective 不符合本 canary 的 human-comparable source role，故未下载、未准入。schema：<https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md>。

## 下载与 archive safety

JRDB 官方公开 test-label 直链：

`https://jrdb.erc.monash.edu/static/downloads/test_labels.zip`

| 项目 | 结果 |
| --- | ---: |
| 下载字节 | 498,600,976 |
| SHA-256 | `a6247ef79a5d37033accf0c96128943fb4116ebad9fc10663a8464440646b10d` |
| ZIP entries | 723 |
| unsafe/rooted/traversal path | 0 |
| uncompressed bytes | 5,207,575,872 |

审计未解压 archive。压缩包下载低于 2 GiB 门，但完整解压体积超过该门；本轮只流式读取 `labels_2d_stitched/*.json`，没有下载 RGB/point cloud。

官方 sensor PDF 为 557,765 bytes，SHA-256 `39b888686160c4d0beedc11e15d68c1de840dfd834adc1c91392aa87b0495d7d`。

## Stitched label canary

producer 对 archive 内全部 27 条 stitched test sequence 逐文件流式解析，validator 在不同 PID 再解析一次：

| 指标 | 结果 |
| --- | ---: |
| sequence | 27 |
| frame | 27,661 |
| object | 956,803 |
| unique sequence-local track | 1,781 |
| truncation false | 925,799 |
| truncation true | 30,889 |
| truncation missing | 115 |
| truncation source coverage | 99.9879808% |
| fully visible | 279,644 |
| mostly visible | 138,908 |
| severely occluded | 298,523 |
| fully occluded | 239,613 |
| occlusion missing | 115 |

115 个 missing 没有被默认成 false；完整 data pack 必须保留 missing/abstain。`truncated` 同时出现 true 与 false，证明它在实际 GT 中不是全局占位常量。occlusion 与 truncation 保持两个字段，不相互替代。

## 验证与终态

- config SHA-256：`9889fe7c3c3a68bab839a7999c81eeee6b51fdd6cb7fa65c742ee7f5906e40b3`
- producer PID：`55872`
- receipt SHA-256：`d87466d6b7b026973a9719111502bb5150a9e02aabee131c556fc6bc444075c0`
- validator PID：`54468`
- validation SHA-256：`a6efdab6fb0eb832e9a723c2fe1271fab595d39d69aa9237e633ebdb6bd1edab`
- validator checks：archive safety、schema、非恒定 truncation、deterministic recomputation、PID isolation、终态与全部高权限关闭均为 true
- 两个 research implementation 的 `py_compile` 通过，并由根级 stable research adapter 调用；结构门不新增 implementation-path 告警

一次早期运行因遇到 `truncated=null` fail closed；协议没有将 null 当 false，而是版本化为显式 missing 计数后重跑完整 archive。另一次早期 bug 是对 `ZipInfo` 直接排序，修复为 filename-key 排序后才形成上述权威 receipt；失败尝试不具终态权限。

## 下一合法边界

要从 canary 进入 `SOURCE_DATA_PACK_ADMISSIBLE_FOR_NEW_DISCOVERY`，至少还需：

1. 对同一 sequence/frame identity 物化最小 RGB canary 与 capture timestamp；
2. 绑定 raw camera → stitched panorama → label frame 的官方或可验证 transform；
3. 在任何 signal 前另立 candidate-blind route-role/event availability 与独立 truth 计划；
4. 对 115 个 missing truncation 明确 abstain，不缩分母回救；
5. 重新冻结下载/解压预算与 JRDB 登录/条款边界。

JRDB 全量下载页面要求登录；本轮没有账户授权，也没有绕过登录。当前停止在 labels/calibration canary 是合法的可观测性进展，不是算法失败。

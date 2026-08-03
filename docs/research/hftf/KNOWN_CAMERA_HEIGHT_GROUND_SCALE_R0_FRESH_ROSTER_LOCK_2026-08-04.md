# Known camera height ground scale R0 fresh roster lock

日期：2026-08-04

状态：`FRESH_ROSTER_4_LOCKED / MEDIA_UNOPENED / ORACLE_HEIGHT_MECHANISM_ONLY`

在读取新媒体、DA 输出、sensor depth 或任何效果 outcome 前，metadata-only planner 从
ARKitScenes official Validation visits 中机械选择了 4 个 parent。它排除了 Spatial
Calibration Head R1 的全部 24 个 visit 和跨 official fold 的 `381879`；选择后仍有
178 个 eligible visits，但没有 reserve 或 outcome 后换源权限。

| visit_id | video_id | selection rank prefix |
| --- | --- | --- |
| `468286` | `47331319` | `000b2bd68043` |
| `466192` | `45260898` | `00ce7e03d1a2` |
| `422826` | `42897538` | `05298a04f5b9` |
| `470831` | `47331963` | `061a67906b3e` |

完整 ignored roster：

```text
artifacts.local/evidence/hftf/known-camera-height-ground-scale-r0-roster-20260804/roster.json
SHA-256 76CB4AC76F60D815A9E09360A936FF215058B77F5557D9AB2A62B9CABEF3905D
```

绑定：

- protocol SHA-256：`040FF50AA7E62E37EDF7AD8DEA198426703A13DD43157CDC632D4E294C14B97B`；
- ARKitScenes metadata SHA-256：
  `F93CD6A1EC0AEA5E103313F3BB4660744B011A1EAA8AA44A992E2C7C2966B145`；
- predecessor roster SHA-256：
  `7CE2D9931723EF7517531F7389FF1DFA0E4BF9BD4C8291A9E72A5BBFF7102EEC`。

本 cohort 的 `H` 将从 source truth 派生，因此只检验“如果每帧相机高度已知，地面几何
能否恢复 DA 尺度”的机制上限。ARKitScenes 是手持相机，不是固定眼镜安装；任何通过
结果都必须终止为 `ORACLE_HEIGHT_MECHANISM_SUPPORTED / WEARABLE_CONFIRMATION_NOT_EVALUABLE`，
不能称为最终佩戴路线已通过。

当前 `media_bytes_read=false`、`metric_truth_opened=false`、`outcomes_opened=false`。
下一步只允许对这 4 个固定 video 做 HEAD asset qualification；任何缺失触发
`COHORT_NOT_EVALUABLE_NO_REPLACEMENT`。

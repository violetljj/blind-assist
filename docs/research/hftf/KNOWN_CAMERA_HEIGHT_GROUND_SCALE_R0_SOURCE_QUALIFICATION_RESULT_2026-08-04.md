# Known camera height ground scale R0 source qualification result

日期：2026-08-04

终态：`HOLD_SOURCE_AUTHORITY_NO_REPLACEMENT`

## 结论

新算子与门已经冻结，4 个 fresh ARKitScenes visits 也完成了 parent-disjoint 锁定和
600 帧 hash-bound 下载；但 source-authority 资格门失败，因此本 cohort 在运行 DA 或
任何效果比较前停止。

ARKitScenes raw 没有独立 ground identity。预先降级后的 reader 只允许用 official
trajectory `+Z` gravity、confidence-2 sensor depth 和下方 ROI 形成水平面高度 proxy，
每个 parent 需至少 `90/150` 帧有效。结果为：

| visit / video | valid proxy | fraction | median H proxy | IQR | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `468286 / 47331319` | 63/150 | 0.420 | 1.327 m | 0.190 m | FAIL |
| `466192 / 45260898` | 105/150 | 0.700 | 1.101 m | 0.119 m | PASS |
| `422826 / 42897538` | 102/150 | 0.680 | 1.286 m | 0.164 m | PASS |
| `470831 / 47331963` | 19/150 | 0.127 | 0.969 m | 0.149 m | FAIL |

主要拒答是 `[0.8,2.2] m` 内的 gravity-plane support 不足；另有少量 pose timestamp、
mode、candidate 与 residual 拒答。按冻结规则，不改变 world vertical、pose tolerance、
height range、ROI、mode/support 门，也不换 video 或补 reserve。

## 证据边界

- `candidate_or_da_outputs_read=false`；
- `effect_metrics_computed=false`；
- Spatial Head sealed metric truth 仍未打开；
- 该结果不支持也不否定已知高度尺度恢复算法，只说明这组 ARKitScenes 手持相机 raw
  source 不能为本实验提供足够、独立的 ground-height authority；
- 4 段中 proxy height IQR 为 `0.119–0.190 m`，也再次说明手持采集不能冒充固定眼镜
  安装高度。

完整结果：

```text
artifacts.local/evidence/hftf/known-camera-height-ground-scale-r0-source-qualification-20260804/qualification.json
SHA-256 D3D457703BE16467FBEF9F7D20A19D5782C22A40FF9EE4D279E6F32040C7E88E
```

媒体 manifest SHA-256：
`8652ECD7B76A3428097EAE9628513F77A7F3B4E4A6A9210683BFF988750115DE`。

## 下一条合法路线

不能把本 cohort 改成更宽松的 R0.1。下一信息增量必须来自新的、先验上具有显式固定
camera/robot height 与 camera extrinsic 的 source population，或最终眼镜相机的独立
卷尺高度/内参 receipt。优先审计既有 TartanGround differential-drive catalog 中尚未
消费、且 metadata 可绑定 `robot_height + lcam_front extrinsic` 的 environments；它只
能形成 synthetic mechanism evidence，仍不能替代真实佩戴确认。

# Public functional-truth substrate audit

状态：`REVERSIBLE_EXPLORATION / OFFICIAL_LOCAL_RENDER_NOT_EVALUABLE / WEBGL_RENDER_TRANSPORT_PASS / ABOTN_ARRIVAL_TRUTH_ONLY / FUNCTIONAL_PIXEL_REGION_NOT_ESTABLISHED / NO_ALGORITHM_SUCCESSOR / DEFAULT_APP_UNCHANGED`

## 问题与结论

在已封存 public-real 8×89 不补抽、不换 teacher/provider/prompt/threshold 的边界下，核对无需人工采集即可取得的
独立 public truth source。当前可自动取得的公开来源能分别覆盖 metric arrival、first-visible pose、门区域或商铺名称
区域的一部分，但没有一个来源同时提供可执行 RGB、命名 POI、source-native functional entrance region 与 metric
arrival truth。因此不运行 baseline，也不建立算法 successor。

## ABotN-POIBench source canary

固定 [ABotN-POIBench dataset](https://huggingface.co/datasets/acvlab/ABotN-POIBench) revision
`fbb62cc3382d8ff84f7fe3b6a3e7d48e4c21e974` 与 [ABot-Navigation evaluator](https://github.com/amap-cvlab/ABot-Navigation/tree/ABotN-Bench)
revision `2a0aefb56f1e2d315bba924239e9e8ad9dca9d92`。只下载 163 份 task JSON；不下载 9.66 GB 3DGS
payload，不运行 renderer/teacher/provider。

| 项目 | 结果 |
|---|---:|
| scene / task | `11 / 163` |
| 非空 named goal | `163/163` |
| metric endpoint | `163/163` |
| trajectory | `163/163` |
| label endpoint 与 trajectory endpoint 在 0.25 m 内一致 | `163/163` |
| 显式 entrance frame / bbox / mask / polygon | `0/163` |
| distinct goal labels | `159` |

该来源可承担内部可逆的 named-POI metric-arrival canary，但不能承担单帧 functional entrance-region truth；禁止把
轨迹终点投成伪入口框。官方 evaluator 还把 `target_position` 与 `distance_to_goal` 交给 agent；未来若有可执行 render
host，BlindAssist provider envelope 必须移除这两个 evaluator-private 字段，只允许 evaluator 使用 metric goal。

许可只达到保守边界：仓库 README 声明 Apache-2.0，但固定 dataset card 没有 license 字段，dataset root 只有
`.gitattributes`，固定代码仓库 root 也没有 `LICENSE` 文件。因此允许在项目内做带 provenance 的可逆研究下载，不推导
再分发、商用或产品许可。

## Render runtime canary

[官方 renderer 文档](https://github.com/amap-cvlab/ABot-Navigation/blob/ABotN-Bench/render_server/README.md)要求
Linux、CUDA 编译扩展和每 GPU 至少 24 GB VRAM。本机为 Windows、RTX 5060 Laptop 8151 MiB、CUDA PyTorch 可用但
无 CUDA compiler；旧 remote endpoint 的只读 preflight 为 connection refused。故在 scene payload 下载前关闭为：

```text
NOT_EVALUABLE_LOCAL_RENDER_RUNTIME_VRAM_BELOW_OFFICIAL_MINIMUM
scene_payloads_downloaded=0
render_calls=0
teacher_calls=0
provider_calls=0
```

## Official release-tree pixel inventory

对同一固定 dataset revision 的完整 463-entry tree 做了补充审计，而不是只按扩展名猜测是否存在 RGB。390 个文件中
共有 182 张 PNG：163 张 `annotations/<scene>/png/traj_*_poi_*` 成功轨迹可视化、8 张 `png_failed/failed_*`
失败轨迹可视化、11 张 occupancy map；JPG、视频和其他 media 均为 0。封存任务
`20260227163550/traj_0` 的同名“大众浴池”PNG 也已下载并绑定，内容类型是 2850×1710 俯视轨迹可视化，绝非
camera observation 或店面参考图。

因此官方 release 中可直接复用的 pre-rendered observation RGB 为 `0`，不能用 annotation PNG 或 occupancy map
替代官方 renderer，也不能把它们暴露给 provider。本次 render/teacher/provider/baseline/episode-rerun 均为 0；本机
唯一已配置 AutoDL worker 的只读 SSH preflight 仍不可达，没有启动或创建远端资源。

曾短暂启动 Docker Desktop 只验证 NVIDIA passthrough，验证后已停止；没有遗留 container 或 Docker process。

## Unofficial WebGL transport canary

官方 CUDA renderer 仍保持上述 `NOT_EVALUABLE`；没有把 alternate renderer 冒充官方实现。为避免把 24 GB host
availability 误当作 real-pixel substrate 本身，另行固定
[`GaussianSplats3D`](https://github.com/mkkellogg/GaussianSplats3D) npm `0.4.7`，只下载最小 scene
`20260227163550` 的标准 INRIA 3DGS PLY（463,431,940 bytes，1,868,671 vertices）和 outcome 前按规则选定的
`traj_0.json`。Renderer 只接收 scene 与 initial camera；goal、endpoint、`distance_to_goal`、teacher output 和 private
truth 均未进入 browser envelope。

固定 WebGL 配置使用 cross-origin isolation、shared-memory CPU sort、SH degree 0 和 alpha gate 5。GPU sort 的早期
诊断曾出现 silent `instanceCount=0`，已作为 transport failure 拒绝；最终冻结配置得到：

| gate | 结果 |
|---|---:|
| retained splats submitted | `1,865,491 / 1,865,491` |
| canvas | `1280 × 720` |
| PNG bytes | `701,156` |
| luma mean / stddev | `118.79 / 55.23` |
| black / white fraction | `0.000047 / 0.007347` |
| sampled distinct RGB | `21,830` |
| renderer / teacher / provider / baseline calls | `1 / 0 / 0 / 0` |

终态为 `WEBGL_RENDER_TRANSPORT_CANARY_PASS`。它只证明公开 ABotN real 3DGS 能在本机生成非退化 RGB，且
provider firewall 可在 baseline 前成立；不证明 WebGL 与官方 CUDA renderer 像素等价，不提供 functional entrance
region，也不产生 selection accuracy、arrival success 或算法 successor。

## 其他候选边界

- [ABotN Short-Horizon OVON](https://huggingface.co/datasets/acvlab/ABotN-Short-Horizon-OVON)公开包只有 2,443 个
  first-visible start poses；标准 OVON episodes、HM3D meshes 与 semantic instance truth 需另行取得，本机不存在这些
  资产，不能自动物化。
- [DoorFront](https://doorfront.org/)的数据模型包含 pano、相机 POV、地址、门框和 human labels，语义最匹配；但
  公开实现的所有 image/label read routes 都经过 token middleware，网页要求账户，且未发布 dataset export。不得绕过
  认证。
- Shop-sign / POI captioning 数据可提供店名与 sign quadrilateral，但 sign 不是 functional entrance，不得升级为入口
  truth。

## Evidence 与下一边界

- source audit：`artifacts.local/evidence/abotn-poibench-truth-source-v0/audit.json`
  SHA-256 `3B00CB3EFBA4FAA0B1B74E9094536344DA62C6B19569B427EFC5DF4F576618D0`；
- runtime audit：`artifacts.local/evidence/abotn-poibench-truth-source-v0/render-runtime-audit.json`
  SHA-256 `412CA0D746E7C6ADC32F8708AB4FB9FAF1A28639693A2BAA0165107E0FA59A5A`；
- WebGL receipt：`artifacts.local/evidence/abotn-webgl-render-canary-v0/receipt.json`
  SHA-256 `E8FE0003AE9C6E3CD89BAA3891DCDEAF5D92587F6D918ED6E50008BF9A6D3E05`；
- frozen initial RGB：`artifacts.local/evidence/abotn-webgl-render-canary-v0/initial_view.png`
  SHA-256 `A56565D02B9AF540C15460012B07A825419241497E3468026C730DC1478634E8`；
- source annotation manifest SHA-256：`b90201c38a4660f765f9c68233e79f824dcb03ea8d7feb804b6e78cbf79a2779`。
- official pixel availability receipt：
  `artifacts.local/evidence/abotn-official-pixel-availability-v0/receipt.json`，SHA-256
  `29185CC630C549644A20DAAB4E95E4E5C2116271435066B9E7B65D920C373A6C`；封存任务轨迹图 SHA-256
  `F587AE2F952090985CF291B6B97ED7C2DA9ECCC817B8989383A0B2B84A4C8699`。

下一可执行边界已收窄为：冻结同一 task 的 arrival-only provider-firewall canary，并在任何 baseline 前确认 provider
只能看到 goal + rendered pixels + public camera state，不能看到 endpoint 或 distance-to-goal；这仍不建立 entrance-region
accuracy。若目标仍是 frame-level selection/failure
attribution，则必须取得 source-native functional entrance-region dataset/export。两者都不授权修改 V0、重开 8×89、
P1、模型 sweep 或默认 App。

Claim ceiling：`WEBGL_RENDER_TRANSPORT_AND_ARRIVAL_SUBSTRATE_MECHANICS_ONLY_NOT_OFFICIAL_RENDERER_EQUIVALENCE`。

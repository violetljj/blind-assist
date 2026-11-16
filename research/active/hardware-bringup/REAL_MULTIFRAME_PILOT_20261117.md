# 真实深度回放中的多帧记忆小试

2026-11-17，EXPLORE。判断短时空间记忆能否在已有真实输入上补漏且控制持续误报。只比较当帧、0.8 秒历史、同样历史加当前深度射线清除。共用 0.30–2.00 m 前向范围、±0.30 m 宽度、原头身高度带和 2/3 确认。历史点按相对位姿投到当前走廊，每张源帧独立达标才增加证据；清除仅作用于旧点，当前有效射线须比旧点远 8 cm。缺帧、超过 0.32 秒的间断或跟踪组件变化重置历史。没有训练、新录制或 App 变动。

## 样本

| 来源 | 能用来检查的现象 | 本轮用途 |
| --- | --- | --- |
| 已有手机实录 `phone-new-capture-20261012`，`073554041Z-a26add25` 与 `073635160Z-0260c80d` | 同一细杆正前、身侧通过；侧过背景另有人 | RGB/CNH/IMU 实录候选，缺少同步公制相机平移。本轮用于筛选 |
| 已有手机实录 `phone-imu-capture-20261012`，`020610867Z-4ea92c3c` 与 `023924340Z-b272b630` | 人靠近、退开、再次入镜；同大厅空场。前者有 440 张 RGB 和 115 个有效 ToF 帧 | 近静止相机；早先标签没有逐段区分离场时刻。本轮用于筛选 |
| 已消费 LaMAR CAB 三条 HoloLens 录制 | 正前障碍、头向有障碍而实际走向侧过、其余几何可判空 | 原生 ToF + 局部跟踪，独立 NavVis 和后续位移只用于计分。复用原标签全部 86 段，不按结果挑片。[原对照](LAMAR_NAVVIS_CORRIDOR_DEV_20261116.md) |
| TUM RGB-D `freiburg3/walking_xyz` | 手持 RGB-D 的三次真人穿过近场后离开 | 注册的公制深度 + 外部动捕位姿，约 5 Hz 取 144 对帧。人工先看 RGB 及公制深度覆盖，再锁定进入/离开索引；无独立人物距离真值。预测只看深度和位姿。[数据与格式](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) · [格式](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) |

## 结果

| 片段结果 | 单帧 | 加 0.8 秒记忆 | 记忆加射线清除 |
| --- | ---: | ---: | ---: |
| LaMAR 正前漏报（32 段） | 15 | 14 | 14 |
| LaMAR 侧向/转头误报（20 段） | 15 | 15 | 15 |
| LaMAR 其余走廊空段误报（34 段） | 5 | 7 | 7 |
| TUM 人在盒内的告警采样（13） | 11 | 12 | 12 |
| TUM 人离开盒后的告警采样（16） | 4 | 10 | 10 |

LaMAR 唯一补回的正例只有一个可判抽样时刻（`hl_2022-06-20-14-33-15-200` 的 108.002494 s），其他已检出段没有首个可判报警提前。TUM 三次都检出，第一次比单帧提前约 0.20 秒。三个离场窗口（缓存帧 29–32、38–42、74–80）从首个离场帧到首个无报警帧的时间分别约 0.20 / 0.20 / 0.40 秒，记忆后约 0.60 / 0.40 / 1.00 秒，射线清除相同。这里测的是报警状态，不等同于手机语音长度。两套结果均为开发回放，也没有达到能替换当前基线的收益。下一步可在这批真人离场片上验证实际人物掩码或轨迹门控，当前帧应保留完整提醒依据。

## 复现与文件

- [LaMAR 回放](host/real_multiframe_pilot_20261117.py) · [TUM 适配与计分](host/real_multiframe_tum_20261117.py)。LaMAR 输入为 `work/lamar-independent-heading-dev-20261116-codex`，复用其 `runtime/packages` 中的 Embree 绑定。Python 环境为 `E:/codex-tools/tools/venvs/blindassist-torch-gpu`，仅 CPU。TUM 用 `prepare`、`score` 两个入口，参考索引从同目录 `tum_reference.json` 读取。
- `artifacts.local/work/real-multiframe-pilot-20261117-codex/` 保留短计划、片段参考、逐时刻预测、比较和原始 TUM 归档。TUM 从 [官方发布](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz) 下载 527,550,055 字节，SHA256 `1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d`。RGB 配对最大时差 6.4 ms。原始归档可重建下采样缓存。
- 单帧 LaMAR 输出在原比较的 456 个时刻逐带一致。TUM 三处切断后续帧的前缀结果一致，复算了全部 29 个事件采样。静态标签使用真实深度以外的扫描参考，但动态数据只有一条手持录制，三次穿过共享人物和环境。

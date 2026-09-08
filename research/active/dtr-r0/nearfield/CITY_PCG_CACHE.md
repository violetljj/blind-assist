# City PCG 缓存复用

保留每台机器自己的项目缓存，避免重复资产处理和着色器编译。当前启动器将
`UE-LocalDataCachePath` 固定到所选项目的 `DerivedDataCache`；同一项目的后续
build/capture 使用相同目录。不要把 DDC/Zen 或已经验证可复用的 `Saved`、
`Intermediate` 数据当作一次性临时目录清理；尤其保留
`Intermediate/CachedAssetRegistry`。发生具体损坏时才定位到受影响缓存处理，
不做习惯性的整目录清空。源地图、材质、spec 和验收回执也必须保留。

主机 `street200-build-6/editor.log` 已记录：

- `UE-LocalDataCachePath` 指向
  `F:/ba-data/blindassist-artifacts-20260805/unreal/CitySample/DerivedDataCache`；
- Zen 数据目录是该目录下的 `Zen`，`ZenLocal` 状态为 `OK`；
- 从 `Intermediate/CachedAssetRegistry` 读取 334.1 MiB 既有索引缓存。

该次 build 内部操作为 13.156 秒，进程总时长为 47.672 秒，释放回执为
`released: true`。这些是本次复用记录，不是冷启动对照或长期吞吐基准；
不存在 `DDC.ddp` 的日志也不代表 Zen 缓存没有启用。

**保留缓存文件不等于保留常驻进程。** 每次任务仍通过已有进程生命周期工具
退出并释放其拥有的 UE/子进程；不要为了“暖缓存”额外保留编辑器或启动大规模
预热。共享或其他任务的服务不能按进程名一并停止。

副机继续使用独立 `CitySampleSliceV2` 项目及它自己的
`project/DerivedDataCache`，不要临时变换项目根目录，也不编辑或同步完整
CitySample 工作工程。复用已校验的资源增量即可；缓存、机器绝对路径和原始
大体量输出不需要反复跨机复制。必要的新组合只做约定的小检查。

当前采集器已支持同一个 spec 的多个 cases 在**一次 UE 进程**中顺序采集，
包括本次 24 帧套件。优先一批多 cases，不为每一帧重开 UE；静态 settling
和 native 导出检查仍按各 case 执行。

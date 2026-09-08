# 城市实验场：采集与使用

在仓库根目录执行：

```powershell
pwsh -NoProfile -File tools/collect_city_field.ps1
```

默认使用 `native-field-v2.json`。每次新建带时间的输出目录，依次运行街区加载、
路线采集、标签检查、异常帧筛选和数据导出。运行期间不要另开同一机器的 Unreal
采集任务。完成后终端打印数据路径；原始数据和失败原因均保留，不覆盖上次结果。

仅检查配置、不启动 Unreal：

```powershell
pwsh -NoProfile -File tools/collect_city_field.ps1 -CompileOnly
```

换机器时创建忽略追踪的 `artifacts.local/city-field-machine.json`，填写 `python`、
`project`、`engine`、`plugin` 四个绝对路径。Python 需要已有采集依赖和 CUDA PyTorch。
当前机器已配置。`-Plan`、`-Output`、`-MachineConfig` 可显式指定其他配置。

## 直接可用的数据

输出目录下的 `ready/` 是经过筛选的数据：

- `frames.jsonl`：一行一帧，含路线、区域、划分、场景类型、障碍条件和配对身份。
- `samples/<id>/rgb.png`：RGB 图像。
- `samples/<id>/depth.npy`：原生 float32 轴向深度，单位米。
- `samples/<id>/labels.npz`：near、support 和当前可靠目标的标签；`-1` 仍是未知。
- `samples/<id>/metadata.json`：相机内参、实际位姿、地面高度、目标身份及相对几何。
- `pairs.json`：保留下来的障碍/对照帧；只有 `usable=true` 的组具有双方。
- `excluded.json`：被排除的帧及原因。
- `summary.json`：保留数量、排除数量、各路线与场景覆盖。

`report/` 提供路线示意图和逐路线图像总览。看这些图即可检查实际环境和障碍摆放。
当前采集为按路线排列的静态稳定观察，不是连续行走视频。clear 表示移除了配置的
障碍，城市原有物体仍然存在。

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("填入本次输出路径/ready")
for line in (root / "frames.jsonl").read_text().splitlines():
    frame = json.loads(line)
    rgb = Image.open(root / frame["rgb"])
    depth = np.load(root / frame["depth"])
    with np.load(root / frame["labels"]) as labels:
        near, support = labels["near"], labels["support"]
```

## 自动筛选规则

缺失/损坏的 RGB 或深度、空白图像、无有效深度、加载未就绪、未配对、位姿异常、
地面检查异常、near 未知、当前障碍标签不可靠的帧不进入 `ready/`。
天空等局部未知像素不导致整帧丢弃，使用者继续按 `-1` 忽略这些像素。
跨划分身份重复或源文件变更等整体矛盾会停止导出，避免产出混乱的数据集。

也可对已有完整采集单独执行导出：

```powershell
python tools/export_city_field_ready.py <采集目录> --output <新的artifacts.local目录>
```

路线、障碍和数据是否好用是实验场验收条件；某个模型是否检出障碍不是采集门槛。

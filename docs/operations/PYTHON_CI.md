# Python 工具验证与冻结文件的跨平台字节检查

更新：2026-09-24。

[Python tools 工作流](../../.github/workflows/python-tools.yml)在相关路径变更时，
用 Ubuntu、Windows 和 Python 3.11 执行两组测试：

```sh
python -m unittest discover -s tools -p "test_*.py"
python -m unittest discover -s tools/data -p "test_*.py"
```

两条命令都需要；`unittest` 不会自动递归进入没有 `__init__.py` 的 `tools/data`。
依赖为 CPU PyTorch 2.6.0、NumPy 2.1.3、Pillow 11.1.0、OpenEXR 3.3.3。
先建立 `artifacts.local/tmp`，测试只使用临时夹具，不下载模型或基准数据，
不启动 UE/CARLA，不访问 GPU（`TASK_NOT_GPU_SUITABLE`）。
这覆盖工具单元测试，不等于全部研究实验测试、设备验证或冻结证据重新评估。

## 本次实际验证

- 从 Git `HEAD` 导出干净副本，排除当前未跟踪研究文件和工作区修改；
  Windows Python 3.11.9 下，原有 `tools` 144 项、`tools/data` 22 项全部通过。
- 新增字节校验器测试另通过 1 项，涵盖原始字节成功、CRLF 改写失败、
  内容改写失败、文件缺失失败，并确认校验不会修复或改写文件。
  将其加入干净副本后，完整 `tools` 145 项再次全部通过（与数据工具合计 167 项）。
- 独立 Git 检出设置 `core.autocrlf=true`，CARLA 目录 397 个已跟踪文件
  与提交中的原始字节完全相同。Ubuntu runner 尚未在本机执行；工作流负责持续覆盖。
- 日志与只读诊断保存在忽略目录 `artifacts.local/maintenance/python-ci-20260924/`。

## 字节检查能证明什么

```sh
python tools/check_research_checkout_bytes.py
```

在干净检出上比较 `research/active/dtr-r0/carla/` 全部已跟踪普通文件与 `HEAD`
blob 的 SHA-256；不同操作系统都必须与同一提交一致。工作区存在有意修改时失败是预期行为。
它不修改换行符、不更新协议里的预期哈希，也不证明历史协议的每一个冻结锁仍匹配当前源码。
范围限于 CARLA；其他研究目录的历史 Windows 启动脚本不在此项冻结字节验证内。

## 两处历史冻结锁差异

对干净副本中协议 JSON 的 `frozen_component_sha256` 字典做只读诊断，
共 48 个唯一 `(文件, 预期哈希)`：46 个匹配，2 个内容不匹配，
没有仅因 LF/CRLF 导致的差异。此统计不包含协议里其他格式的哈希字段。

两项均来自 `research/active/dtr-r0/carla/dtr_carla_ivca_c1_protocol.json`：

| 文件 | 冻结 SHA-256 |
| --- | --- |
| `dtr_carla_x24_plan_adherent_predictor.py` | `C37F2667EC119A855F69DE01132ABDD32D37C387B8CD7AFF5879046498631997` |
| `dtr_carla_x25_rigid_footprint_predictor.py` | `EE9A37ED6978C9E187D70E607288A129710B2C3C823F7D3E37CE6B1ECC088895` |

2026-09-05 的提交 `1560e3e8a794eb2ffb22e89d9c1d5b9127577f2d` 给 X24
运动拟合增加可选窗口参数，并让 X25 传入该参数。读取该提交父版本的两个 Git blob，
其原始 SHA-256 分别与上述冻结值精确一致。因此，这两处是可定位的历史源码更新，
不是跨平台换行问题。此次保留当前源码和历史锁；如果重放该冻结协议，必须恢复对应历史
版本及其完整依赖，不能直接使用当前源码并把旧锁改成新值。

验证临时 Python 环境和检出工作文件已释放；共享克隆的隐藏 `.git` 元数据仍在
`artifacts.local/tmp/python-ci-20260924/git-checkout/.git`，因不使用强制删除而保留。
它属于本轮验证诊断，没有运行进程；后续需在允许处理隐藏文件的清理任务中移除。

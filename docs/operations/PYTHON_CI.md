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

## 冻结协议原始字节修复（2026-09-24）

上面的检出一致性不等于冻结锁一致性。补充核对发现 C17 至 C44 的 28 份
协议，本地 CRLF 字节匹配既有锁，而 Git 中的 LF 字节不匹配。例如 C44
本地为 163,339 字节，Git 原版本为 157,644 字节。此前只扫描
`frozen_component_sha256` 的诊断没有覆盖这些协议引用，不能据此断言不存在换行问题。

这些精确路径在 `.gitattributes` 中设置 `-text`，提交原始 CRLF 字节；
`.editorconfig` 同步保留其换行和空白。没有修改任何冻结哈希或协议内容。

```sh
python tools/check_frozen_carla_bytes.py
```

Ubuntu/Windows CI 在干净检出上执行此项检查。清单
`tools/frozen_carla_byte_locks.json` 只保存文件与原协议哈希字段的引用，
不另造预期哈希；检查直接读取原锁、按原始字节计算 SHA-256，不做换行归一化。
配套回归测试覆盖换行改写、内容损坏、缺文件、失效锁引用和空清单。
这覆盖选定的 28 份协议，不等于整个历史源码闭包已经恢复。

修复后，本机从待提交索引分别以 `core.autocrlf=true` 和 `false` 生成干净检出：
两种配置下 28 个原锁全部匹配，397 个 CARLA 文件全部与索引原字节一致，
新增的 2 项回归测试均通过。逐文件比较也确认 28 份协议相对旧提交仅恢复 CRLF，
JSON 内容及其中哈希不变。日志保留在
`artifacts.local/maintenance/frozen-rules-20260924/`；这不是 Ubuntu runner 的实测结果。

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

同一 X24 历史值也出现在 Final Reckoning 的 `implementation_locks`。
当前 `test_validate_dtr_final_reckoning_roster.py` 的完整锁验证仍在 X24 失败；
这是已知源码版本差异，不能通过替换旧哈希、回退当前算法或跳过断言伪装修复。
新增 CI 冻结协议字节检查独立报告其范围，并不声称这个完整历史测试通过。

验证临时 Python 环境和检出工作文件已释放；共享克隆的隐藏 `.git` 元数据仍在
`artifacts.local/tmp/python-ci-20260924/git-checkout/.git`，因不使用强制删除而保留。
它属于本轮验证诊断，没有运行进程；后续需在允许处理隐藏文件的清理任务中移除。

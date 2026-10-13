# 新头戴录制：冻结 v1/v2/v3 执行入口

2026-10-12 工程准备；2026-10-13 录制前加入[冻结 v3](FROZEN_FUSION_EVAL_V3_PROTOCOL.md)，同一批 raw/labels 三版都跑、并列报告。目标是消除输入遗漏与结果遗漏；不开展新评价、不调整科学逻辑。以[冻结协议](FROZEN_FUSION_EVAL_V1_PROTOCOL.md)为唯一模型、标签与判据依据。

## 录制与标签

1. 用户完成新头戴录制并导出 ZIP。结束电脑预览再连接 H3 手机采集（H3 只允许一个 TCP 消费者）。保持当前安装；每段显式开始、停止。至少覆盖空通道 `quiet`、正前走近障碍 `ahead`，另录低头地面 `calibration`、头高横杆 `head`、侧杆 `side`。这些是类别覆盖提示，不增加协议的样本量要求。
2. 原始 ZIP、解包、完整性记录和结果都存入 `artifacts.local/work/` 下的新目录。`raw` 应直接包含本次所有 `capture-*` 目录，避免混入旧批次。标签准备前可先运行 `python research/active/hardware-bringup/host/check_frozen_fusion_inputs.py --raw <本次raw路径> --input-only`；`INPUTS_ONLY_COMPLETE` 只表示导入输入检查完成，不能代替下方带标签与结果的检查。
3. 仅查看 RGB 缩略图，按协议给每个目录写标签；其他情形写 `unknown`，不可删除不利片段。标签格式如下，保存为 UTF-8 **无 BOM**（原冻结脚本要求）：

```json
{"capture-实际目录名": {"class": "quiet", "scene": "现场场景描述"}}
```

4. **标签写好后提交 Git，再运行评价**，这是冻结协议原有要求。原始数据与结果留在忽略目录；可将本次标签的小型 JSON 保存为本路线下新的受版本控制文件，在命令中使用其路径。记录标签提交号。预检不能证明已提交或标签正确。

## 命令

从仓库根目录执行 PowerShell；将三个输入变量替换为本次真实路径。输出目录必须使用新名称，避免原脚本覆盖已有结果。评价使用 `config/local.toml` 的 `research_python` 启动器（含 CUDA 环境设置），不要使用 `hardware-bringup/venv` 或默认 `python` 直接推理。2026-10-12 已验证该启动器可运行 v1 `--help`，OpenCV 4.10.0、Torch 2.11.0+cu130、CUDA 可用；没有运行新评价。

```powershell
$py = python -c "import pathlib,tomllib; print(tomllib.loads(pathlib.Path('config/local.toml').read_text(encoding='utf-8-sig'))['research_python'])"
$hostTools = '.\research\active\hardware-bringup\host'
$raw = '.\artifacts.local\work\本次录制\raw'
$labels = '.\research\active\hardware-bringup\本次已提交标签.json'
$out = '.\artifacts.local\work\本次冻结评价'

& $py "$hostTools\frozen_fusion_eval_v1.py" --help
if ($LASTEXITCODE -ne 0) { throw '先修复 Python 导入环境；--help 不运行评价' }
& $py "$hostTools\check_frozen_fusion_inputs.py" --raw $raw --labels $labels
if ($LASTEXITCODE -ne 0) { throw '输入不完整，按输出修复导入或标签后再评价' }
if (Test-Path -LiteralPath $out) { throw '输出路径已存在；保留原结果，不覆盖' }
& $py "$hostTools\frozen_fusion_eval_v1.py" --raw $raw --labels $labels --out $out
if ($LASTEXITCODE -ne 0) { throw 'v1 执行失败；保留日志和部分输出，先查失败原因' }
& $py "$hostTools\frozen_fusion_eval_v2.py" $out
if ($LASTEXITCODE -ne 0) { throw 'v2 执行失败；保留 v1 结果，先查失败原因' }
& $py "$hostTools\frozen_fusion_eval_v3.py" --raw $raw --labels $labels --out $out
if ($LASTEXITCODE -ne 0) { throw 'v3 执行失败；保留 v1/v2 结果，先查失败原因' }
& $py "$hostTools\check_frozen_fusion_inputs.py" --raw $raw --labels $labels --reports $out
if ($LASTEXITCODE -ne 0) { throw '结果覆盖不完整，不作总体通过结论' }
```

预检只读、使用标准库，不加载模型。除了目录与标签一一对应、类别和非空文件，它还严格解析 JSONL，检查 ToF 索引对原始字节的完整覆盖、有效 CNH 的 8×8×16 结构和数值、IMU 重力引用，以及所有 RGB 图片的存在性和导出索引已有的 bytes/SHA256。带 BOM 的标签会在评价前报错，须在标签提交前保存为 UTF-8 无 BOM。坏索引或 payload 会给出目录、文件及索引行号，优先从原始 ZIP 重新导入，不能删行绕过检查。

`input_diagnostics` 按段给出有效 CNH、RGB、有效 IMU 和未匹配数量；没有 IMU 引用、旧 IMU schema 缺重力字段、没有 100 ms 内 RGB 接收匹配时给出警告，不补齐输入或改变冻结判据。缺重力与坏引用分开，合法旧格式不会被当作损坏。计数分母是全部有效 CNH，尚未经过模型就绪门槛。已索引的无效 boot/config/截断尾记录保留，不当作有效 CNH。接收时间接近不代表曝光同步。预检不解码图片、不替代原始 ZIP/传输校验、完整时钟诊断或标签提交核查。

原 v1 会忽略缺失的标签目录，两版摘要会排除没有就绪帧的片段，所以结果仍需最后一次 `--reports` 检查。`COMPLETE` 仅表示检查覆盖完整，科学判据仍以两版逐段输出为准，失败也可以是完整结果。

v1/v2 均完整报告所有片段、就绪帧分母及各自判据；主张必须同时满足所有 `ahead` 与所有 `quiet`。缺类或零就绪帧写“无法完整评价”，不得使用摘要中的剩余片段宣称通过。`side`/`head` 单列，不改变总体主张。新数据评价后不改标签、模型、阈值或判据。

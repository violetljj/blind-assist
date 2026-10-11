# 有标注 CNH 引导脚本：独立模拟串口核验

2026-10-11，EXPLORE。**11 个测试 PASS**；这次只回放已有串口字节、注入主机时钟与中断，不连接 COM，不采集、不刷写、不运行真实标注或模型，也不访问保护480/test。

测试代码：[`host/test_real_labeled_capture.py`](host/test_real_labeled_capture.py)。独立检查采集及标注接口后修复的行为，未由测试修改采集/标注实现。

## 来源与留证

- 原录制：`artifacts.local/hardware-bringup/captures/cnh-h3-1mhz-20260927/raw.bin`，2,800,855 bytes，186 个既有 H3 帧。
- 原 SHA256：`ee3a93e91dde441baaf4665af3b23cb6a7e86693cf437933013a67ace7994e06`；测试前后相同。
- 最终留证：`artifacts.local/work/real-labeled-capture-prep-20261011/independent-replay-mx07u5re/`；逐次 raw/frames/events/issues/metadata、session/inventory 全部保留。
- 命令日志：同 prep 目录下 `independent-replay-test.log`，机器收据 `independent-replay-receipt.json`。首次测试因测量提示接口正在调整而 mock 未满足输入，已中止；随后改为显式测量回答并最终完整通过，旧测试目录保留。
- `synthetic-stale-before-boot.bin` 仅把原录制末尾5个帧前置于原 boot，用于模拟 RESET 前缓存；它是**合成串口调度 fixture**，不是新录制。缺身份、错误尺寸/配置、失帧率、低帧率和重复序号判据使用合成 summary mutation，不伪称实机异常。标注接口仅用一帧合成 envelope，不给真实录制生成标签。

## 实际覆盖

| Case | 核验结果 |
|---|---|
| 真实字节自检与引导 | H3 boot/config、8×8×16、既有序号/时钟通过；前后照片、距离、高度、俯角、ambient 元数据和距离/ambient 8×8显示可用 |
| 异常自检 | 缺/错身份，错config/尺寸，帧不足，3.81Hz，丢帧>2%，固件错误、重置、重复及 invalid_frame 均拒绝 |
| RESET旧帧隔离 | 原raw保留旧缓存与序号重置；门控只统计最后boot后的新启动窗口，可通过 |
| 30s窗口完整性 | 独立重新打开原录制，虚拟30.184s保存154帧；每字节SHA/长度、逐原始sensor对象、逐帧UTC+monotonic时间戳与重新解码一致 |
| 全12段会话 | 6条件×暗/亮全部提示与保存；本case每段虚拟1s，12段均有帧，流程complete；10段needs_review因同一既有场景与摆放条件不匹配，警告不被隐藏 |
| 录制中断/续录/重录 | 部分raw与解析帧保留，interrupted保留；resume新attempt，redo旧成功attempt标superseded且raw字节不变 |
| 倒计时中断 | 使用默认5s倒计时；中断后会话及attempt明确interrupted_before_recording，不遗留recording状态，串口关闭 |
| 显式实测输入 | 空Enter不能把建议数值当测量；未知开阔距离可None |
| JPEG不可用 | 注入Wi-Fi不可用，首末相机receipt留unavailable，CNH全流程完成，不发网络请求 |
| 相机地址边界 | 公网IP、DNS名、HTTPS和非stream路径拒绝，避免外部上传/错误目标 |
| 标注骨架 | 合成HEAD/BODY杆POS、外侧clear、擦边pass、误差跨边界UNKNOWN、未审核背景UNKNOWN；nested sensor seq/ms和UTC/monotonic字段不丢失 |

相机可用时真实MJPEG抓取、实机串口拔插/RESET和30秒物理时钟精度尚未验证；这些留待用户实际采集时观察。本次虚拟时钟与原固件时间戳只证明准备脚本流程，不能证明硬件可靠性或真实标签正确性。

## 命令与预算

```powershell
python -m unittest discover -s research/active/hardware-bringup/host -p test_real_labeled_capture.py -v
```

最终 `Ran 11 tests in 2.748s / OK`，最终命令 wall 3.322s。独立核验全部读写、前次中止/修正与重复检查累计保守 **≤80 command-wall s / 分配180s**；计入主任务CPU≤1200 command-wall s。只用CPU，无设备采集和GPU进程。`git diff --check` 针对测试文件通过。

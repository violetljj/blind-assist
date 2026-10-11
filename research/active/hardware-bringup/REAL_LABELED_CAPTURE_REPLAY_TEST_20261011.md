# 宿舍最简 CNH 引导脚本：独立回放核验

2026-10-11，EXPLORE，协议 `simple-dorm-v1`。**8个测试全部PASS**；只用既有录制和合成串口/时钟fixture，不连接COM、不采集、不刷写、不联网、不访问保护480/test，不生成真实录制标签。测试见 [`host/test_real_labeled_capture.py`](host/test_real_labeled_capture.py)。此前复杂24段版的旧回放payload保留，但不作为当前11段版验收结果。

## 来源与留证

- H3启动/帧原字节：`artifacts.local/hardware-bringup/captures/cnh-h3-1mhz-20260927/raw.bin`，2,800,855B，SHA256 `ee3a93e91dde441baaf4665af3b23cb6a7e86693cf437933013a67ace7994e06`。
- 稳定平面探距原字节：`artifacts.local/hardware-bringup/captures/simple-20260927T164234Z-cc4d0e/01-background-attempt1/tof/raw.bin`，2,324,611B，SHA256 `47c2699e2d9f6db03342ddb09cd6ac2d4c4917bfe9add1fb5b670735d35a4740`。两源测试前后hash相同。
- 当前payload：`artifacts.local/work/real-labeled-capture-simple-prep-20261011/independent-simple-ud2w_jnp/`。原始保存、frames/events/issues、metadata、capture.json、inventory、book-probe与失败启动门控都保留。
- 同prep目录 `independent-simple-test.log`、`independent-simple-receipt.json`。先前测试误假设H3启动录制具有稳定中央距离，失败证据保留；诊断实际中央读数不稳定后，稳定2s测距改用已知平面录制，未知/不稳行为继续单独核验。

## 实际覆盖

| Case | 结果 |
|---|---|
| 白天9段＋夜间2段全流程 | 同root独立启动续夜间，11段均保存帧，3次startup、最终complete；本case探距虚拟0.4s、录制虚拟1s，默认倒计时3s。1段needs_review保留，不假称真实摆放通过 |
| 启动自检拒绝 | 真实H3 boot/config与64×16通过；缺/错身份、错误config/尺寸、帧不足、3.81Hz、丢帧>2%、重复、固件错误及invalid_frame拒绝；异常summary mutation明确为合成fixture |
| 2s探距与30s保存 | 稳定平面2s窗口聚合中央27/28/35/36的status5、nb_target>0距离；独立重新打开H3原录制，虚拟30.184s保存154帧，raw长度/hash、逐sensor对象、UTC/monotonic时间戳与重解码一致 |
| 跳过、重录、中断续录 | s标skipped，r与--redo编号新attempt；旧成功段superseded但raw不变；部分raw与帧、interrupted保留，resume完成白天剩余段 |
| 跨日夜间目录 | --night默认选最新未完成夜间的simple manifest，即使日期变化；旧复杂协议排除，已完成夜间目录不被误选 |
| 第9段独立重启 | 先close、提示拔插、再打开并保存新startup gate；合成删boot的第二连接被拒绝，失败raw/gate保留，不沿用第一次身份 |
| 整窗口距离与QC | 跨帧跳动标unknown，status无效不用；丢帧/低帧率提示；准备距离稳定但capture中央未知显示无法核对。杆5–8触发书替代探距；book-probe单独保存，提示拿走书，book/准备帧序号与正式capture不交叉 |
| 后处理弱标签 | 合成metadata验证11条件映射、condition_label/expected_response；near-no-echo1/10的条件弱clear、空过道4/9/11未知距离UNKNOWN、杆缺几何/None尺寸UNKNOWN、needs_review UNKNOWN，均标CONDITION_WEAK_APPROXIMATE与independent_ground_truth=false |

高度只问一次，Enter记录120cm和 `default_estimate`；照片可在摆好同一prompt输入，无测量表。源码检查无camera/网络客户端入口，测试不发网络请求。真实串口RESET/拔插恢复、真实探距是否对应目标和物理30秒时钟尚未验证；虚拟回放只证明准备脚本流程。固件距离为弱参考，条件弱标签不能称独立实机真值、总体误报率或检出率。

## 命令和预算

```powershell
python -m unittest discover -s research/active/hardware-bringup/host -p test_real_labeled_capture.py -v
```

最终 `Ran 8 tests in 4.116s / OK`，最终测试＋源码检查命令wall **4.560s**；本轮独立核验含失败、修正和留证累计保守 **≤45 command-wall s / 分配180s**，计入主任务CPU≤900 command-wall s。只用CPU，无设备/网络/GPU进程。

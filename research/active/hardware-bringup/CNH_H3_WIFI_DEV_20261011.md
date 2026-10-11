# H3 CNH Wi-Fi 接入记录

2026-10-11，EXPLORE 工程接入。用户在原“不刷写”准备任务之后明确授权新固件，并要求两板换接充电宝后继续预览。本次已刷入完整 CNH 无线固件，RGB 与 ToF 已在本机采集台同时更新；十段正式标注采集尚未开始。

## 实现与设备

- 新增 `firmware/tof_cnh_h3_wifi/tof_cnh_h3_wifi.ino`，原H3固件保留。64 zone × 16 bin、start_bin=0、sub_sample=8、名义5Hz、20ms integration、autonomous、I²C 1MHz；完整直方图及固件距离/信号/ambient 经TCP3335 NDJSON输出，USB可选，无USB主机时避免输出阻塞。
- ToF XIAO ESP32-S3：COM5，USB序列98:A3:16:F7:88:1C，8MB flash。实际地址 `192.168.5.10`，固件 `xiao-cnh-h3-wifi-v1`。RGB Atom地址 `192.168.5.9:81/stream`。两板加入现有2.4GHz局域网；密码仅经USB配置到设备NVS，不写代码、日志或Git。
- 复用现有8×8×16解码，采集脚本增加显式私网IPv4 TCP输入，USB流程保留；身份/layout/帧率/丢帧门槛沿用。连接事件与实际缓存boot分开，第9段必须boot_id变化，重连TCP不能代替物理重启。
- 采集台新增Wi-Fi选择与内存预览，不写采集目录；正式开始前释放单客户端连接。预览断线/1.5s无有效帧先清空旧读数，再按1→2→3s重连；正式录制异常仍留失败证据，不静默重启录制。关服务按钮释放连接、线程和8791端口。

## 刷写与恢复

按既有工具链编译（m5stack ESP32 core3.3.8、dio、hwcdc、CDCOnBoot）。首次Arduino自动原型/C链接冲突失败已保留并修复。最终app大小1,048,832 bytes，SHA256 `13c6366927ce0a7347da1dc2bb0fa6ca4c7e2c1490b691294d81785f6a614742`。esptool ROM模式仅写app0 `0x10000`，hash核验通过；bootloader/partition table未重写，Wi-Fi配置另写设备NVS。

原H3 app备份：`artifacts.local/hardware-bringup/backups/xiao-h3-before-wifi-20261011/h3-app-before.bin`，430,160 bytes，前430,144 bytes与原编译app SHA256 `3e68971e8e498b522a1961a92dae4676485c56abf5c0320f39eca7a14fd461c9`一致，后16 bytes为padding。两次全flash备份尝试分别失败/因低速主动停止，日志与部分数据保留，不能当作完整备份。需要回退时可将已核验原app写回同一app0地址。

## 实际核验

12秒无线检查收到61帧，全部8×8×16；设备时间计算5.1038Hz，missing/duplicate/reset/device error/解析issue均0，归一化直方图有限且无全零格。实际config的CNH bytes=5468、read_bytes=6916；这是新构建实际读出长度，不宣称二进制读取长度与旧构建相同。boot_id=`f47b73c57b4d83ad`。

页面实查：`camera.available=true`，RGB持续更新；ToF预览更新29帧时记录状态，`phase=idle`、`worker_active=false`，正式录制未启动。独立短mock断线→清空→重连→停止释放检查PASS，command-wall1.405s。只做针对改动的核验，未跑模型或全仓测试。

证据：`artifacts.local/work/cnh-h3-wifi-20261011/`（刷写、配置、失败日志、`wifi-check/startup-01/gate.json`、链路原字节/解析帧、`web-wifi-receipt.json`、`web-rgb-tof.jpg`）；编译manifest与日志在 `artifacts.local/hardware-bringup/build/tof_cnh_h3_wifi/`。61帧属于链路检查，不能并入十段标注数据。

## 使用与限制

```powershell
& .\artifacts.local\hardware-bringup\venv\Scripts\python.exe .\research\active\hardware-bringup\host\real_labeled_capture_web.py --tof-host 192.168.5.10
```

打开 `http://127.0.0.1:8791/`；当前服务与内存预览留给用户操作，点“关闭本地服务”释放。换接充电宝需两板仍在同一局域网覆盖内；实际充电宝切换尚未验证。DHCP地址变化时需更新页面ToF地址并重新预览。

RGB/ToF相对安装固定，整套刚性转动不会改变相对关系；原始zone与RGB画面上下左右尚未标定。下一步用书在画面中心附近左/右/上/下逐一确认旋转与翻转，确认之前保持原始区序。整体俯角相对地面的变化仍影响HEAD/BODY几何，方向确认不等于完整外参标定。

本次仅证明当次短时链路与预览工作；不证明测距准确性、曝光同步、报警效果或安全性。冻结M3/原5格/S及阈值未改，未访问保护480/test，没有新标注录制。CPU沿用准备任务≤900 command-wall秒上限，无GPU，未提高上限。

## 后续实时方向观察

同日用户移动瓶子时，观察到RGB右侧目标对应原始网格上部，RGB左上目标对应原始网格右侧。采集台新增可选“按当前画面对齐 · 初步观察”，显示映射为 `(display_row, display_col)=(7-raw_col, 7-raw_row)`；默认仍原始区序，本次页面已手动选用观察方向。只改变CSS位置，点击仍使用原zone的CNH，保存字节/帧/模型输入完全不变。64格实际DOM位置唯一、四角对应检查通过；截图 `web-orientation-observed.jpg`。这是当前安装的粗方向候选，未完成四方向定点或完整外参标定，不能作为逐query几何真值。

用户随后要求画面旋转与镜像，新增每次90°左右旋转和左右镜像开关；Canvas在解码后变换并交换宽高，不裁剪。候选映射先对齐，再做同样旋转及显示横轴镜像；原始区序模式维持原排列。共享CameraView只有存在本页显式quarterTurns属性时启用变换，其他页面默认路径保留。现场页面已设顺时针90°、镜像开，实际Canvas480×640、持续呈现1607帧时核验，64网格位置仍唯一，收据 `web-rotation-mirror.jpg`。这些检查只验证显示变换与更新，没有完成物理方向定点确认；两路变换相同不证明外参正确，也不证明曝光同步。正式标注采集仍未开始。

随后用户固定目标并明确发送“右”“上”，已完成两个方向的实物粗对照。右侧黑色书本对应显示网格右半部的距离失效区域，不能据此报告有效近距离。上方书本对应网格上半部的近回波：设备seq5003/5006/5008，可信近距分别185/216/194、188/214/187、252/195/207/235/229mm，下半部均无50–500mm的status5近回波。现场设置仍quarterTurns=1、mirrorX=true，候选显示合成为 `(7-raw_row,7-raw_col)`；截图 `orientation-right-probe.jpg`、`orientation-up-probe.jpg`。当前显示左右/上下的粗方向对应得到支持；未做完整四方向重复、像素投影/外参或尺寸距离精度标定，不能推广为逐query标注真值。目标黑色导致不少UNKNOWN，右侧对照是影响区域而非有效距离验证。

用户随后明确要求记住当前固定安装、改变后再确认。页面默认恢复observed方向、顺时针90°、镜像开，并用本机浏览器localStorage `cnh.capture.view.v1`保存后续显示选择；刷新已实查恢复。该设置不改变采集帧和模型输入。用户又明确授权开始真实十段采集，本机引导已启动并停在安装高度估计输入；这次授权替代此前准备任务“不采集”的限制，正式段需按摆放提示确认后开始。

## 首段实际录制与声音引导（15:53起）

用户改为头戴约155cm、有自然抖动，并授权代理根据实时RGB判断摆放后开始。`operator-context.json`和正式段metadata记录head_worn/static_scene=false、俯角未测。场景尽量不动，但观测会随头动，帧方差包含运动混杂，不能直接解释为严格静态仪器噪声。

采集台新增本机声音引导：开始一声、结束两声、中文步骤播报；浏览器声音按钮已开启，物理听到与否未独立确认。摆放等待曾导致TCP消费者阻塞、固件断开，三个失败准备段完整保留。网页等待时持续排空不保存的字节，WiFiStream遇EOF自动重连；TCP重新连接保留transport_connection/cached实际boot_id，不能算物理重启。

第一段阳台朝外30s已存`real-labeled-capture-20261011/01-near_no_echo/attempt-04`：143帧、4.767Hz、序号缺10帧，自动标needs_review；ambient中位482.38（CNH归一化单位，不是lux）。RGB见外楼、天空、栏杆在下方；固件中央多数UNKNOWN，偶有1–3mm读数，不能据此把开放区域标为真实毫米级表面，也不能仅凭UNKNOWN证明全zone空。随后第二段同室白墙自动测距约0.69225m（弱参考），开始30s录制。原字节与帧保留本地，未运行冻结模型或后处理标注。此次只做Python编译与实际引导/保存检验，不扩大测试。

第二段近墙30s完成153帧、5.10273Hz、零缺序号、quality_warnings=[]；录制中央固件距离中位0.623m，和预读约0.692m分别保留。第一段原字节SHA256与metadata一致、解析行数143一致，收据`first-segment-receipt.json`。当前引导已到第三段远墙，后续进展追加记录。

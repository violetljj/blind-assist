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

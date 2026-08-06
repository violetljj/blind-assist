# AtomS3R 固件 ToF 关闭归因 R25 结果（2026-08-06）

## 结论

已编译并刷写 ToF 关闭诊断固件，确认设备可正常启动；但手机端 10 秒对照命令被安全审批拦截，未取得可用于延迟结论的逐帧数据。因此本实验不晋升、不宣称 ToF 是否参与设备端瓶颈。

## 已确认

- ToF 关闭固件编译通过：Flash 约 32%，动态内存静态占用约 18%
- 刷写到确认的 AtomS3R COM5 成功
- `/api/status`：`sampling_enabled=false`、ToF `NOT_READY`
- Wi-Fi、相机、SVGA/XGA 控制服务仍正常

## 处理

已刷回 ToF 开启的稳定 r11 固件，并恢复：

```text
resolution = SVGA 800×600
jpeg_quality = 10
frame_buffer_count = 2
grab_mode = LATEST
tof sampling = true
tof status = VALID
```

由于缺少 ToF-off 的手机端阶段耗时证据，不能把该路线列为已排除或已优化；后续若要再做，需先获得明确的测试执行许可。

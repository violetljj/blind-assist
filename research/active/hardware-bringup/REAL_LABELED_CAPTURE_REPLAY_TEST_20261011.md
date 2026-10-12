# 单会话10段 CNH：4/4b 配对改动核验

2026-10-11，EXPLORE，协议 `simple-dorm-v2`。**本轮只运行1个 focused 回放测试，PASS**。完整启动自检、2秒探距、30秒保存、书替代探距隔离与弱标签的旧8项PASS保留在 Git `a03b1715` 及 `artifacts.local/work/real-labeled-capture-simple-prep-20261011/`，此次不重复运行。

本轮用 [`host/test_real_labeled_capture.py`](host/test_real_labeled_capture.py) 只检查改动：单会话顺序 `1,2,3,4,4b,5,6,7,8,9`；4/4b相邻，共用 `ambient_pair_id=empty-corridor-fixed-setup`，metadata记录high/low及实际CNH ambient聚合；提示4b保持第4段原位。注入录制中断后保留部分raw、resume补齐10段；`--redo 4b`只追加该段，旧暗档标superseded，raw字节保持不变。

原录制仍为 `artifacts.local/hardware-bringup/captures/cnh-h3-1mhz-20260927/raw.bin`，前后SHA256一致：`ee3a93e91dde441baaf4665af3b23cb6a7e86693cf437933013a67ace7994e06`。只注入模拟主机时钟和中断，探距虚拟0.3s、正式录制虚拟0.6s；不将回放的high/low字段当真实两档ambient效果。无COM连接、网络、采集、刷写、模型或真实标注执行。

本轮payload：`artifacts.local/work/real-labeled-capture-paired-prep-20261011/focused-paired-qcsvvsj7/`，含完整attempt、startup、inventory与 `focused-receipt.json`；同prep目录 `focused-test.log`。旧所有回放payload保留。

```powershell
python -m unittest discover -s research/active/hardware-bringup/host -p test_real_labeled_capture.py -v
```

实际 `Ran 1 test in 1.014s / OK`，测试命令wall **1.548s**；本轮核验读写合计保守≤10 command-wall s。真实同原位明暗配对与硬件采集效果仍待用户录制，本任务只交付准备结果。

## 本机采集台接入

复用原simple_capture页面样式、8×8距离网格与CNH，新增web桥接同一GuidedCapture，不另写采集器。独立focused mock PASS：idle不打开设备；点击start才打开mock；prompt队列回答对应；stop释放mock串口并退出worker。未连接真实设备。HTTP GET页面200，API状态idle、worker_active=false，10段及4/4b high/low次序正确，仅监听127.0.0.1:8791；本次没有创建真实capture.json。只读启动收据保存在paired-prep目录web-idle-receipt.json。

本地页面服务留给用户开始录制，实际listener PID46708（launcher36580），页面“关闭本地服务”可释放；其console/error/PID保留在paired-prep目录。

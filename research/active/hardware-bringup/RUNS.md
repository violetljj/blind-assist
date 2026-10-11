# Hardware support run log

| Date | Code version | Configuration changes | Metrics (units/denominators) | Conclusion / result link |
| --- | --- | --- | --- |
| 2026-10-11 | 本机十段采集台实时相机接入 | 用户要求实时画面；复用CameraFeed/CameraView、同源MJPEG，保持ToF显式开始 | 实际页面显示画面并更新380帧；API camera.available=true，ToF idle/worker=false；仅语法/依赖和实际画面focused检查，无模型执行 | 相机预览首末图/元数据独立保留于artifacts.local/work/real-labeled-camera-preview，不并入标注CNH。web-camera-receipt.json保留当前状态/PID；服务留给用户使用，页面关闭服务释放 |
| 2026-10-11 | `simple-dorm-v2`；10段同日准备 | 用户改为4后插入同位置暗档4b，取消夜间两段；4/4b实际ambient配对，5/9恢复明档；不采集、不刷写、不运行标注 | 10×30s=5分钟录制，约20分钟含摆放；1个focused回放PASS（命令wall1.548s），仅检查顺序/配对/重录与续录；本机可视化采集台桥接同一脚本；沿用CPU≤900command-wall秒预算，无GPU | [操作单](REAL_LABELED_CAPTURE_SHEET_20261011.md)、[回放收据](REAL_LABELED_CAPTURE_REPLAY_TEST_20261011.md)。旧版录制目录不会静默混用；距离/几何保持弱参考，拉帘改变回波的混杂在标注计划注明 |

| 2026-10-11 | `host/real_labeled_capture.py` / `real_labeled_annotation.py` / `test_real_labeled_capture.py`，宿舍简版 | EXPLORE；目标用真实静态弱标签校准模拟器并查看冻结M3/5格/S响应；准备11条件×30s、3s倒计时，中央固件测距/必要时书本代理，安装高一次估计 | 独立8case PASS；11段日夜流程/3startup，虚拟30s保存154帧，书probe隔离、跨日夜间续录与失败留证通过；CPU保守≤300/900command-wall秒，无GPU | 旧11段一页操作与回放收据保留于Git `a03b1715`；旧24段材料保留于Git `ada649c3`；当前仅准备/模拟串口，不采集/刷写/真实标注/模型修改，不读保护480/test |
| 2026-10-11 | `host/real_labeled_capture.py` / `real_labeled_annotation.py` / `test_real_labeled_capture.py` | EXPLORE首次标注录制准备；6条件×2照明×2独立开始，30s/段、5s倒计时；COM5只读H3自检/原字节与实测照片关联 | 独立模拟串口11case PASS；12段流程每段虚拟1s，10段needs_review为场景不匹配；另30s虚拟collect154帧；CPU保守≤300/1200command-wall秒，无GPU | 旧24段引导/实测几何准备已备，材料与11case收据保留于Git `ada649c3` 的同名文件及旧prep payload。本任务无采集/刷写/真实标注/模型修改，未读保护480/test。 |
| 2026-10-11 | `host/cnh_sim2real_*.py` (source hashes in payload) | EXPLORE，既有13真实CNH片段；far-only模板/差分floor，冻结三模型两路径及既定两格诊断 | 12922decoded/12726满窗；旧M3 H/B7875/1951、5格10795/6661、S10799/6726；新各12726/12726；bias340/1024、floorR².0024；CPU保守≤400/1800command-wall、GPU≤150/300s | 适配原型完成，物理合同/迁移未修复；独立raw/forward PASS；仅真实回放，重训/标注采集仅草案。[报告](SIM2REAL_INPUT_CONTRACT_DEV_20261011.md) |

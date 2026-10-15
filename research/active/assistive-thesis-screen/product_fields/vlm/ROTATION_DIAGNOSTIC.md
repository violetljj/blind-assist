# 180 度朝向：事后 Development 诊断

本分支在主线程目视检查既有评价图中的 6 张之后提出，其中 14955、12716、
19585 的实体包装在 EXIF 校正后仍倒置。因此它明确是 **posthoc Development**
诊断，不是未见确认，不替换原两臂比较，也不依据本次成绩调整提示或继续搜角度。

对原 image_receipts 的全部 96 图统一先做 `ImageOps.exif_transpose`，再做
`PIL.Image.Transpose.ROTATE_180` 像素置换；不选择错误图片子集、不重编码JPEG。
使用原 image-only 的模型 revision、提示、像素上限、精度及生成参数；没有OCR输入，
不读商品真值或质量标签。前3图启动canary计入同一个run，后续跳过这3图续写93图。

新增GPU进程墙钟上限 **300秒**；既有模型进程377.953秒与探针保守60秒另计，
全过程仍在原1800秒预算内。无需下载。输出固定为
`artifacts.local/work/product-fields-dev-20261015/vlm/rotated_180/`。
每图保留原始图像hash、EXIF状态和附加旋转记录、原始生成及token logprob。

证据特征后续仍用原EXIF像素上的OCR作字串支持，不能把原OCR几何位置当作旋转图坐标；
对照预测为原 image-only。此次只询问朝向是否造成可恢复的读字错误，
不能据此声称旋转策略已泛化或真实补拍已有收益。

执行完成：96/96图成功，0推理/解析错误；新增模型进程墙钟190.734/300秒，
包含首3图canary及重载。推理中位1.531秒/图、P95 1.907秒/图，最高Torch
allocated 4,726,926,336 bytes。非空product/brand/variety为96/96/95，
只是生成覆盖，不是正确率。全量token文本重建及字段logprob重算通过。

全部三臂模型进程累计568.687秒；加环境探针保守60秒仍为628.687/1800秒。
新增输出SHA256为
`fa90d18add9f699833915059bf36da9c39f67c3a90742225ffe5a868d97aace0`。
`rotation_protocol_check.json`核验参数仅增加旋转/事后证据身份三项，原两份
预测SHA不变。新总表为`summary_with_rotation.json`，原`summary.json`保留。
已通知独立证据代理接续。本分支没有读取真值，正确性收益由上层单独评价。

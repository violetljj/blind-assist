package com.linnan.blindassist.vision

import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.tensorflow.lite.DataType
import kotlin.math.max
import kotlin.math.min

internal data class YoloDecodeResult(
    val detections: List<Detection>,
    val warning: String? = null
)

internal object YoloOutputDecoder {
    fun parse(
        raw: FloatArray,
        shape: IntArray,
        dataType: DataType,
        letterbox: LetterboxInfo,
        labels: List<String>,
        confidenceThreshold: Float,
        iouThreshold: Float
    ): YoloDecodeResult {
        if (dataType != DataType.FLOAT32) {
            return YoloDecodeResult(emptyList(), "模型输出类型不支持：$dataType")
        }
        if (shape.size != 3) {
            return YoloDecodeResult(emptyList(), "模型输出形状不支持：${shape.contentToString()}")
        }

        val dim1 = shape[1]
        val dim2 = shape[2]
        val channelsFirst = dim1 <= dim2 && dim1 >= 5
        val channels = if (channelsFirst) dim1 else dim2
        val predictions = if (channelsFirst) dim2 else dim1
        val classCount = min(labels.size, channels - BOX_CHANNELS)
        if (classCount <= 0) {
            return YoloDecodeResult(emptyList(), "模型输出类别数异常：${shape.contentToString()}")
        }

        val frameSize = FrameSize(letterbox.sourceWidth, letterbox.sourceHeight)
        val detections = mutableListOf<Detection>()
        if (channelsFirst) {
            for (prediction in 0 until predictions) {
                var bestClass = -1
                var bestScore = 0f
                var classOffset = BOX_CHANNELS * predictions + prediction
                for (classId in 0 until classCount) {
                    val score = raw[classOffset]
                    if (score > bestScore) {
                        bestScore = score
                        bestClass = classId
                    }
                    classOffset += predictions
                }
                if (bestClass < 0 || bestScore < confidenceThreshold) continue
                addDetection(
                    detections,
                    bestClass,
                    bestScore,
                    raw[prediction],
                    raw[predictions + prediction],
                    raw[2 * predictions + prediction],
                    raw[3 * predictions + prediction],
                    letterbox,
                    frameSize,
                    labels
                )
            }
        } else {
            for (prediction in 0 until predictions) {
                val predictionOffset = prediction * channels
                var bestClass = -1
                var bestScore = 0f
                val classEnd = predictionOffset + BOX_CHANNELS + classCount
                var classOffset = predictionOffset + BOX_CHANNELS
                while (classOffset < classEnd) {
                    val score = raw[classOffset]
                    if (score > bestScore) {
                        bestScore = score
                        bestClass = classOffset - predictionOffset - BOX_CHANNELS
                    }
                    classOffset += 1
                }
                if (bestClass < 0 || bestScore < confidenceThreshold) continue
                addDetection(
                    detections,
                    bestClass,
                    bestScore,
                    raw[predictionOffset],
                    raw[predictionOffset + 1],
                    raw[predictionOffset + 2],
                    raw[predictionOffset + 3],
                    letterbox,
                    frameSize,
                    labels
                )
            }
        }
        return YoloDecodeResult(nms(detections, iouThreshold))
    }

    private fun addDetection(
        detections: MutableList<Detection>,
        bestClass: Int,
        bestScore: Float,
        rawCx: Float,
        rawCy: Float,
        rawWidth: Float,
        rawHeight: Float,
        letterbox: LetterboxInfo,
        frameSize: FrameSize,
        labels: List<String>
    ) {
        val cx = normalizeCoordinate(rawCx, letterbox.inputSize)
        val cy = normalizeCoordinate(rawCy, letterbox.inputSize)
        val width = normalizeCoordinate(rawWidth, letterbox.inputSize)
        val height = normalizeCoordinate(rawHeight, letterbox.inputSize)
        val sourceBox = mapToSource(
            BoundingBox(
                left = cx - width / 2f,
                top = cy - height / 2f,
                right = cx + width / 2f,
                bottom = cy + height / 2f
            ),
            letterbox
        ).clamped(frameSize)
        if (sourceBox.width <= 1f || sourceBox.height <= 1f) return
        detections += Detection(
            classId = bestClass,
            label = labels.getOrElse(bestClass) { "class_$bestClass" },
            confidence = bestScore,
            boundingBox = sourceBox,
            frameSize = frameSize
        )
    }

    private fun normalizeCoordinate(value: Float, inputSize: Int): Float {
        return if (value <= 1.5f) value * inputSize else value
    }

    private fun mapToSource(box: BoundingBox, info: LetterboxInfo): BoundingBox {
        return BoundingBox(
            left = (box.left - info.dx) / info.scale,
            top = (box.top - info.dy) / info.scale,
            right = (box.right - info.dx) / info.scale,
            bottom = (box.bottom - info.dy) / info.scale
        )
    }

    private fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.size <= 1) return detections
        val ordered = detections.sortedByDescending { it.confidence }
        val suppressed = BooleanArray(ordered.size)
        val result = ArrayList<Detection>(ordered.size)
        for (currentIndex in ordered.indices) {
            if (suppressed[currentIndex]) continue
            val current = ordered[currentIndex]
            result += current
            for (nextIndex in currentIndex + 1 until ordered.size) {
                if (!suppressed[nextIndex]) {
                    val next = ordered[nextIndex]
                    if (current.classId == next.classId &&
                        iou(current.boundingBox, next.boundingBox) > iouThreshold
                    ) {
                        suppressed[nextIndex] = true
                    }
                }
            }
        }
        return result
    }

    private fun iou(a: BoundingBox, b: BoundingBox): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private const val BOX_CHANNELS = 4
}

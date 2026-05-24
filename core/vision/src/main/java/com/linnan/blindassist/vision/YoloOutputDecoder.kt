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

        fun value(prediction: Int, channel: Int): Float {
            return if (channelsFirst) {
                raw[channel * predictions + prediction]
            } else {
                raw[prediction * channels + channel]
            }
        }

        val frameSize = FrameSize(letterbox.sourceWidth, letterbox.sourceHeight)
        val detections = mutableListOf<Detection>()
        for (prediction in 0 until predictions) {
            var bestClass = -1
            var bestScore = 0f
            for (classId in 0 until classCount) {
                val score = value(prediction, BOX_CHANNELS + classId)
                if (score > bestScore) {
                    bestScore = score
                    bestClass = classId
                }
            }
            if (bestClass < 0 || bestScore < confidenceThreshold) continue

            val cx = normalizeCoordinate(value(prediction, 0), letterbox.inputSize)
            val cy = normalizeCoordinate(value(prediction, 1), letterbox.inputSize)
            val width = normalizeCoordinate(value(prediction, 2), letterbox.inputSize)
            val height = normalizeCoordinate(value(prediction, 3), letterbox.inputSize)

            val modelBox = BoundingBox(
                left = cx - width / 2f,
                top = cy - height / 2f,
                right = cx + width / 2f,
                bottom = cy + height / 2f
            )

            val sourceBox = mapToSource(modelBox, letterbox).clamped(frameSize)
            if (sourceBox.width <= 1f || sourceBox.height <= 1f) continue

            detections += Detection(
                classId = bestClass,
                label = labels.getOrElse(bestClass) { "class_$bestClass" },
                confidence = bestScore,
                boundingBox = sourceBox,
                frameSize = frameSize
            )
        }
        return YoloDecodeResult(nms(detections, iouThreshold))
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
        val result = mutableListOf<Detection>()
        val remaining = detections.sortedByDescending { it.confidence }.toMutableList()

        while (remaining.isNotEmpty()) {
            val current = remaining.removeAt(0)
            result += current
            val iterator = remaining.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (current.classId == next.classId && iou(current.boundingBox, next.boundingBox) > iouThreshold) {
                    iterator.remove()
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

package com.linnan.blindassist.vision

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.util.Log
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.session.DetectorMetrics
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class TfliteYoloDetector(
    context: Context,
    private val modelAssetName: String = MODEL_ASSET,
    labelsAssetName: String = LABELS_ASSET,
    private val inputSize: Int = INPUT_SIZE,
    private val confidenceThreshold: Float = CONFIDENCE_THRESHOLD,
    private val iouThreshold: Float = IOU_THRESHOLD
) : ObjectDetector {
    val labels: List<String> = context.assets.open(labelsAssetName)
        .bufferedReader()
        .useLines { lines -> lines.map { it.trim() }.filter { it.isNotEmpty() }.toList() }

    override val isReady: Boolean get() = interpreter != null
    override val statusMessage: String
        get() = loadError?.message ?: runtimeWarning ?: if (isReady) "模型已加载" else "模型未加载"
    var lastPreprocessMs: Long = 0L
        private set
    var lastInferenceMs: Long = 0L
        private set
    var lastPostprocessMs: Long = 0L
        private set
    var lastTotalDetectMs: Long = 0L
        private set

    private var gpuDelegate: GpuDelegate? = null
    private var interpreter: Interpreter? = null
    private val preprocessor = ImagePreprocessor(inputSize)
    private var outputBuffer: ByteBuffer? = null
    private var outputFloats: FloatArray = FloatArray(0)
    private var loadError: Throwable? = null
    @Volatile
    private var runtimeWarning: String? = null

    init {
        try {
            val loadedInterpreter = createInterpreter(context)
            validateInputTensor(loadedInterpreter)
            validateOutputTensor(loadedInterpreter)
            interpreter = loadedInterpreter
        } catch (error: Throwable) {
            Log.w(TAG, "Failed to load TFLite model: $modelAssetName", error)
            loadError = IllegalStateException(
                "缺少模型文件：assets/$modelAssetName。请先导出 YOLO11n TFLite FP16 模型。",
                error
            )
        }
    }

    private fun createInterpreter(context: Context): Interpreter {
        val model = loadMappedAsset(context, modelAssetName)
        val gpuOptions = Interpreter.Options().apply {
            setNumThreads(4)
            maybeAttachGpuDelegate(this)
        }
        return try {
            Interpreter(model, gpuOptions)
        } catch (gpuError: Throwable) {
            Log.w(TAG, "GPU interpreter failed, falling back to CPU.", gpuError)
            gpuDelegate?.close()
            gpuDelegate = null
            runtimeWarning = "GPU 不可用，已回退 CPU"
            Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
        }
    }

    override fun detect(bitmap: Bitmap): DetectorFrameResult {
        val frameSize = FrameSize(bitmap.width, bitmap.height)
        val localInterpreter = interpreter ?: return DetectorFrameResult(
            detections = emptyList(),
            frameSize = frameSize,
            metrics = currentMetrics()
        )
        val totalStart = System.nanoTime()
        val preprocessStart = totalStart
        val input = preprocessor.prepare(bitmap)
        lastPreprocessMs = elapsedMs(preprocessStart)

        val outputTensor = localInterpreter.getOutputTensor(0)
        val localOutputBuffer = reusableOutputBuffer(outputTensor.numBytes())

        val start = System.nanoTime()
        localInterpreter.run(input.buffer, localOutputBuffer)
        lastInferenceMs = elapsedMs(start)
        localOutputBuffer.rewind()

        val floats = reusableOutputFloats(outputTensor.numElements())
        localOutputBuffer.asFloatBuffer().get(floats)

        val postprocessStart = System.nanoTime()
        val detections = parseOutput(
            raw = floats,
            shape = outputTensor.shape(),
            dataType = outputTensor.dataType(),
            letterbox = input.letterbox
        )
        lastPostprocessMs = elapsedMs(postprocessStart)
        lastTotalDetectMs = elapsedMs(totalStart)
        return DetectorFrameResult(
            detections = detections,
            frameSize = frameSize,
            metrics = currentMetrics()
        )
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    private fun currentMetrics(): DetectorMetrics {
        return DetectorMetrics(
            totalMs = lastTotalDetectMs,
            preprocessMs = lastPreprocessMs,
            inferenceMs = lastInferenceMs,
            postprocessMs = lastPostprocessMs,
            fps = 0f,
            modelStatus = statusMessage
        )
    }

    private fun maybeAttachGpuDelegate(options: Interpreter.Options) {
        try {
            val compatibilityList = CompatibilityList()
            if (compatibilityList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate(compatibilityList.bestOptionsForThisDevice)
                options.addDelegate(gpuDelegate)
            }
        } catch (error: Throwable) {
            Log.w(TAG, "GPU delegate unavailable, falling back to CPU.", error)
        }
    }

    private fun reusableOutputBuffer(numBytes: Int): ByteBuffer {
        val current = outputBuffer
        if (current == null || current.capacity() != numBytes) {
            outputBuffer = ByteBuffer
                .allocateDirect(numBytes)
                .order(ByteOrder.nativeOrder())
        }
        return outputBuffer!!.also { it.rewind() }
    }

    private fun reusableOutputFloats(numElements: Int): FloatArray {
        if (outputFloats.size != numElements) {
            outputFloats = FloatArray(numElements)
        }
        return outputFloats
    }

    private fun elapsedMs(startNanos: Long): Long {
        return (System.nanoTime() - startNanos) / 1_000_000L
    }

    private fun validateInputTensor(interpreter: Interpreter) {
        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape()
        require(inputTensor.dataType() == DataType.FLOAT32) {
            "模型输入必须是 FLOAT32，实际为 ${inputTensor.dataType()}"
        }
        require(shape.contentEquals(intArrayOf(1, inputSize, inputSize, 3))) {
            "模型输入必须是 [1,$inputSize,$inputSize,3]，实际为 ${shape.contentToString()}"
        }
    }

    private fun validateOutputTensor(interpreter: Interpreter) {
        val outputTensor = interpreter.getOutputTensor(0)
        val shape = outputTensor.shape()
        require(outputTensor.dataType() == DataType.FLOAT32) {
            "模型输出必须是 FLOAT32，实际为 ${outputTensor.dataType()}"
        }
        require(shape.size == 3) {
            "模型输出需要是 raw YOLO 三维张量，实际为 ${shape.contentToString()}。请使用 nms=False 导出。"
        }
        val dim1 = shape[1]
        val dim2 = shape[2]
        require((dim1 >= MIN_YOLO_CHANNELS && dim2 > dim1) || (dim2 >= MIN_YOLO_CHANNELS && dim1 > dim2)) {
            "模型输出不像 raw YOLO 张量，实际为 ${shape.contentToString()}。请使用 nms=False 导出。"
        }
    }

    private fun parseOutput(
        raw: FloatArray,
        shape: IntArray,
        dataType: DataType,
        letterbox: LetterboxInfo
    ): List<Detection> {
        if (dataType != DataType.FLOAT32) {
            runtimeWarning = "模型输出类型不支持：$dataType"
            return emptyList()
        }
        if (shape.size != 3) {
            runtimeWarning = "模型输出形状不支持：${shape.contentToString()}"
            return emptyList()
        }

        val dim1 = shape[1]
        val dim2 = shape[2]
        val channelsFirst = dim1 <= dim2 && dim1 >= 5
        val channels = if (channelsFirst) dim1 else dim2
        val predictions = if (channelsFirst) dim2 else dim1
        val classCount = min(labels.size, channels - BOX_CHANNELS)
        if (classCount <= 0) {
            runtimeWarning = "模型输出类别数异常：${shape.contentToString()}"
            return emptyList()
        }
        runtimeWarning = null

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
        return nms(detections)
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

    private fun nms(detections: List<Detection>): List<Detection> {
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

    private fun loadMappedAsset(context: Context, assetName: String): MappedByteBuffer {
        val descriptor: AssetFileDescriptor = context.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            val channel = stream.channel
            return channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    companion object {
        const val MODEL_ASSET = "yolo11n_fp16_320.tflite"
        const val LABELS_ASSET = "coco_labels.txt"
        const val INPUT_SIZE = 320
        const val CONFIDENCE_THRESHOLD = 0.35f
        const val IOU_THRESHOLD = 0.45f
        private const val MIN_YOLO_CHANNELS = 5
        private const val BOX_CHANNELS = 4
        private const val TAG = "TfliteYoloDetector"
    }
}

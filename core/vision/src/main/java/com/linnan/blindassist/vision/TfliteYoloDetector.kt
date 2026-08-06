package com.linnan.blindassist.vision

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.os.SystemClock
import android.os.Trace
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
import java.nio.FloatBuffer
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
    private val iouThreshold: Float = IOU_THRESHOLD,
    val executionBackend: DetectorExecutionBackend = DetectorExecutionBackend.CPU_XNNPACK,
    private val externalInterpreterOptionsFactory: (() -> Interpreter.Options)? = null,
    private val externalBackendCloser: (() -> Unit)? = null
) : ObjectDetector {
    val labels: List<String>

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
    private var outputFloatBuffer: FloatBuffer? = null
    private var outputFloats: FloatArray = FloatArray(0)
    private var outputBytes: Int = 0
    private var outputElements: Int = 0
    private var outputShape: IntArray = intArrayOf()
    private var outputDataType: DataType = DataType.FLOAT32
    private var loadError: Throwable? = null
    @Volatile
    private var runtimeWarning: String? = null
    private val lifecycleLock = Any()

    init {
        var loadedLabels = emptyList<String>()
        var loadedInterpreter: Interpreter? = null
        try {
            loadedLabels = loadLabels(context, labelsAssetName)
            val candidateInterpreter = createInterpreter(context)
            loadedInterpreter = candidateInterpreter
            validateInputTensor(candidateInterpreter)
            validateOutputTensor(candidateInterpreter)
            val outputTensor = candidateInterpreter.getOutputTensor(0)
            outputBytes = outputTensor.numBytes()
            outputElements = outputTensor.numElements()
            outputShape = outputTensor.shape()
            outputDataType = outputTensor.dataType()
            interpreter = candidateInterpreter
            loadedInterpreter = null
            Log.i(TAG, "Detector ready backend=${executionBackend.wireName}")
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            closeInterpreter(loadedInterpreter)
            closeGpuDelegate()
            closeExternalBackend()
            Log.w(TAG, "Failed to load TFLite model: $modelAssetName", error)
            loadError = IllegalStateException(
                "模型初始化失败：请确认 assets/$modelAssetName 和 assets/$labelsAssetName 存在，且输入/输出张量符合 YOLO11n TFLite FP16 约定。",
                error
            )
        }
        labels = loadedLabels
    }

    private fun createInterpreter(context: Context): Interpreter {
        val model = loadMappedAsset(context, modelAssetName)
        externalInterpreterOptionsFactory?.let { optionsFactory ->
            DetectorBackendPolicy.requireExternalBackendInjectionAuthorized(
                context.packageName,
                executionBackend
            )
            runtimeWarning = "外部隔离后端：${executionBackend.wireName}"
            return Interpreter(model, optionsFactory())
        }
        DetectorBackendPolicy.requireProductionAuthorized(executionBackend)
        runtimeWarning = "LiteRT CPU 兼容模式"
        return Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
    }

    override fun detect(bitmap: Bitmap): DetectorFrameResult {
        val frameSize = FrameSize(bitmap.width, bitmap.height)
        return detectFrame(
            frameSize = frameSize,
            sourceFrame = null,
            prepareInput = { preprocessor.prepare(bitmap) }
        )
    }

    override fun detect(frame: VisionFrame): DetectorFrameResult {
        val frameSize = FrameSize(frame.displayWidth(), frame.displayHeight())
        return detectFrame(
            frameSize = frameSize,
            sourceFrame = frame.frameStamp,
            prepareInput = {
                when (frame) {
                    is NativeImageVisionFrame -> {
                        val bitmap = frame.nativeImage as? Bitmap
                            ?: error("Unsupported native image: ${frame.nativeImage.javaClass.name}")
                        preprocessor.prepare(bitmap)
                    }
                    is RgbaVisionFrame -> preprocessor.prepare(frame)
                    else -> error("Only bitmap or RGBA camera frames are supported by the realtime detector")
                }
            }
        )
    }

    private fun detectFrame(
        frameSize: FrameSize,
        sourceFrame: FrameStamp?,
        prepareInput: () -> ModelInput
    ): DetectorFrameResult {
        synchronized(lifecycleLock) {
            val localInterpreter = interpreter ?: return DetectorFrameResult(
                detections = emptyList(),
                frameSize = frameSize,
                metrics = currentMetrics(),
                sourceFrame = sourceFrame
            )
            val totalStart = SystemClock.elapsedRealtimeNanos()
            val preprocessStart = totalStart
            val input = traced(TRACE_PREPROCESS) { prepareInput() }
            val preprocessComplete = SystemClock.elapsedRealtimeNanos()
            lastPreprocessMs = elapsedMs(preprocessStart, preprocessComplete)

            val localOutputBuffer = reusableOutputBuffer(outputBytes)

            val qnnEnqueue = SystemClock.elapsedRealtimeNanos()
            traced(TRACE_QNN_EXECUTE) {
                localInterpreter.run(input.buffer, localOutputBuffer)
            }
            val qnnComplete = SystemClock.elapsedRealtimeNanos()
            lastInferenceMs = elapsedMs(qnnEnqueue, qnnComplete)

            val floats = traced(TRACE_OUTPUT_READ) {
                localOutputBuffer.rewind()
                reusableOutputFloats(outputElements).also {
                    reusableOutputFloatBuffer(localOutputBuffer).get(it)
                }
            }
            val outputReadComplete = SystemClock.elapsedRealtimeNanos()

            val detections = traced(TRACE_POSTPROCESS) {
                parseOutput(
                    raw = floats,
                    shape = outputShape,
                    dataType = outputDataType,
                    letterbox = input.letterbox
                )
            }
            val postprocessComplete = SystemClock.elapsedRealtimeNanos()
            lastPostprocessMs = elapsedMs(outputReadComplete, postprocessComplete)
            lastTotalDetectMs = elapsedMs(totalStart, postprocessComplete)
            return DetectorFrameResult(
                detections = detections,
                frameSize = frameSize,
                metrics = currentMetrics(),
                sourceFrame = sourceFrame,
                stageTiming = DetectorStageTiming(
                    preprocessStartNs = preprocessStart,
                    preprocessCompleteNs = preprocessComplete,
                    preprocessLetterboxDrawStartNs = input.timing?.letterboxDrawStartNs,
                    preprocessLetterboxDrawCompleteNs = input.timing?.letterboxDrawCompleteNs,
                    preprocessBitmapPixelsCompleteNs = input.timing?.pixelsReadCompleteNs,
                    preprocessInputWriteCompleteNs = input.timing?.inputWriteCompleteNs,
                    qnnEnqueueNs = qnnEnqueue,
                    qnnCompleteNs = qnnComplete,
                    outputReadCompleteNs = outputReadComplete,
                    postprocessCompleteNs = postprocessComplete
                )
            )
        }
    }

    override fun close() {
        synchronized(lifecycleLock) {
            closeInterpreter(interpreter)
            interpreter = null
            closeGpuDelegate()
            closeExternalBackend()
        }
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
            FatalThrowables.rethrowIfFatal(error)
            Log.w(TAG, "GPU delegate unavailable, falling back to CPU.", error)
        }
    }

    private fun loadLabels(context: Context, labelsAssetName: String): List<String> {
        return context.assets.open(labelsAssetName)
            .bufferedReader()
            .useLines { lines -> lines.map { it.trim() }.filter { it.isNotEmpty() }.toList() }
    }

    private fun closeInterpreter(interpreterToClose: Interpreter?) {
        try {
            interpreterToClose?.close()
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            Log.w(TAG, "Failed to close TFLite interpreter cleanly.", error)
        }
    }

    private fun closeGpuDelegate() {
        try {
            gpuDelegate?.close()
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            Log.w(TAG, "Failed to close TFLite GPU delegate cleanly.", error)
        } finally {
            gpuDelegate = null
        }
    }

    private fun reusableOutputBuffer(numBytes: Int): ByteBuffer {
        val current = outputBuffer
        if (current == null || current.capacity() != numBytes) {
            outputBuffer = ByteBuffer
                .allocateDirect(numBytes)
                .order(ByteOrder.nativeOrder())
            outputFloatBuffer = outputBuffer!!.asFloatBuffer()
        }
        return outputBuffer!!.also { it.rewind() }
    }

    private fun reusableOutputFloatBuffer(buffer: ByteBuffer): FloatBuffer {
        val current = outputFloatBuffer
        if (current == null || current.capacity() * Float.SIZE_BYTES != buffer.capacity()) {
            outputFloatBuffer = buffer.asFloatBuffer()
        }
        return outputFloatBuffer!!.also { it.rewind() }
    }

    private fun reusableOutputFloats(numElements: Int): FloatArray {
        if (outputFloats.size != numElements) {
            outputFloats = FloatArray(numElements)
        }
        return outputFloats
    }

    private fun elapsedMs(startNanos: Long, endNanos: Long): Long {
        return (endNanos - startNanos) / 1_000_000L
    }

    private inline fun <T> traced(name: String, block: () -> T): T {
        val tracing = try {
            Trace.beginSection(name)
            true
        } catch (_: RuntimeException) {
            false
        }
        return try {
            block()
        } finally {
            if (tracing) {
                try {
                    Trace.endSection()
                } catch (_: RuntimeException) {
                    // Android Trace is unavailable in local JVM tests.
                }
            }
        }
    }

    private fun VisionFrame.displayWidth(): Int {
        return if (rotationDegrees.normalizedRotation() % 180 == 0) width else height
    }

    private fun VisionFrame.displayHeight(): Int {
        return if (rotationDegrees.normalizedRotation() % 180 == 0) height else width
    }

    private fun Int.normalizedRotation(): Int {
        val normalized = this % 360
        return if (normalized < 0) normalized + 360 else normalized
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
        val result = YoloOutputDecoder.parse(
            raw = raw,
            shape = shape,
            dataType = dataType,
            letterbox = letterbox,
            labels = labels,
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold
        )
        runtimeWarning = result.warning
        return result.detections
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
        private const val TRACE_PREPROCESS = "BlindAssist.YoloPreprocess"
        private const val TRACE_QNN_EXECUTE = "BlindAssist.QnnExecute"
        private const val TRACE_OUTPUT_READ = "BlindAssist.YoloOutputRead"
        private const val TRACE_POSTPROCESS = "BlindAssist.YoloPostprocess"
    }

    private fun closeExternalBackend() {
        try {
            externalBackendCloser?.invoke()
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            Log.w(TAG, "Failed to close external TFLite backend cleanly.", error)
        }
    }
}

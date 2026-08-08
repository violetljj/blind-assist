package com.linnan.blindassist.benchmark

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import com.linnan.blindassist.vision.ImagePreprocessor
import com.linnan.blindassist.vision.LetterboxInfo
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/** Test-APK-only detector that keeps production CPU defaults untouched while profiling GPU/size OFATs. */
internal class UstrfBenchmarkYoloDetector(
    context: Context,
    modelAssetName: String,
    labelsAssetName: String,
    private val inputSize: Int,
    private val confidenceThreshold: Float,
    private val iouThreshold: Float,
    requestedBackend: Backend,
) : Closeable {
    enum class Backend { CPU, GPU_DELEGATE, CPU_FALLBACK }

    val labels = context.assets.open(labelsAssetName).bufferedReader().useLines { lines ->
        lines.map(String::trim).filter(String::isNotEmpty).toList()
    }
    val isReady: Boolean get() = true
    val statusMessage: String get() = actualBackend.name.lowercase()
    val interpreter: Interpreter
    val actualBackend: Backend
    var lastPreprocessMs = 0L; private set
    var lastInferenceMs = 0L; private set
    var lastPostprocessMs = 0L; private set
    var lastTotalDetectMs = 0L; private set
    private val preprocessor = ImagePreprocessor(inputSize)
    private var gpuDelegate: GpuDelegate? = null
    private var outputBuffer: ByteBuffer? = null
    private var outputFloats = FloatArray(0)

    init {
        val model = loadMappedAsset(context, modelAssetName)
        if (requestedBackend == Backend.CPU) {
            actualBackend = Backend.CPU
            interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
        } else {
            var attached = false
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                val compatibility = CompatibilityList()
                if (compatibility.isDelegateSupportedOnThisDevice) {
                    gpuDelegate = GpuDelegate(compatibility.bestOptionsForThisDevice)
                    addDelegate(gpuDelegate)
                    attached = true
                }
            }
            interpreter = try {
                Interpreter(model, options)
            } catch (_: Throwable) {
                gpuDelegate?.close(); gpuDelegate = null; attached = false
                Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
            }
            actualBackend = if (attached) Backend.GPU_DELEGATE else Backend.CPU_FALLBACK
        }
        validateTensors()
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val totalStart = System.nanoTime()
        val preprocessStart = totalStart
        val input = preprocessor.prepare(bitmap)
        lastPreprocessMs = elapsed(preprocessStart)
        val tensor = interpreter.getOutputTensor(0)
        val buffer = reusableBuffer(tensor.numBytes())
        val inferenceStart = System.nanoTime()
        interpreter.run(input.buffer, buffer)
        lastInferenceMs = elapsed(inferenceStart)
        buffer.rewind()
        val floats = reusableFloats(tensor.numElements())
        buffer.asFloatBuffer().get(floats)
        val postprocessStart = System.nanoTime()
        val detections = parse(floats, tensor.shape(), tensor.dataType(), input.letterbox)
        lastPostprocessMs = elapsed(postprocessStart)
        lastTotalDetectMs = elapsed(totalStart)
        return detections
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }

    private fun validateTensors() {
        val input = interpreter.getInputTensor(0)
        check(input.dataType() == DataType.FLOAT32)
        check(input.shape().contentEquals(intArrayOf(1, inputSize, inputSize, 3)))
        val output = interpreter.getOutputTensor(0)
        check(output.dataType() == DataType.FLOAT32 && output.shape().size == 3)
    }

    private fun parse(raw: FloatArray, shape: IntArray, dataType: DataType,
                      letterbox: LetterboxInfo): List<Detection> {
        check(dataType == DataType.FLOAT32 && shape.size == 3)
        val dim1 = shape[1]; val dim2 = shape[2]
        val channelsFirst = dim1 <= dim2 && dim1 >= 5
        val channels = if (channelsFirst) dim1 else dim2
        val predictions = if (channelsFirst) dim2 else dim1
        val classCount = min(labels.size, channels - BOX_CHANNELS)
        check(classCount > 0)
        fun value(prediction: Int, channel: Int) = if (channelsFirst) {
            raw[channel * predictions + prediction]
        } else raw[prediction * channels + channel]
        val frameSize = FrameSize(letterbox.sourceWidth, letterbox.sourceHeight)
        val detections = mutableListOf<Detection>()
        for (prediction in 0 until predictions) {
            var bestClass = -1; var bestScore = 0f
            for (classId in 0 until classCount) {
                val score = value(prediction, BOX_CHANNELS + classId)
                if (score > bestScore) { bestScore = score; bestClass = classId }
            }
            if (bestClass < 0 || bestScore < confidenceThreshold) continue
            val cx = normalize(value(prediction, 0)); val cy = normalize(value(prediction, 1))
            val width = normalize(value(prediction, 2)); val height = normalize(value(prediction, 3))
            val source = mapToSource(BoundingBox(cx - width / 2f, cy - height / 2f,
                cx + width / 2f, cy + height / 2f), letterbox).clamped(frameSize)
            if (source.width <= 1f || source.height <= 1f) continue
            detections += Detection(bestClass, labels[bestClass], bestScore, source, frameSize)
        }
        return nms(detections)
    }

    private fun normalize(value: Float) = if (value <= 1.5f) value * inputSize else value
    private fun mapToSource(box: BoundingBox, info: LetterboxInfo) = BoundingBox(
        (box.left - info.dx) / info.scale, (box.top - info.dy) / info.scale,
        (box.right - info.dx) / info.scale, (box.bottom - info.dy) / info.scale,
    )
    private fun nms(detections: List<Detection>): List<Detection> {
        val result = mutableListOf<Detection>()
        val remaining = detections.sortedByDescending { it.confidence }.toMutableList()
        while (remaining.isNotEmpty()) {
            val current = remaining.removeAt(0); result += current
            val iterator = remaining.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (current.classId == next.classId && iou(current.boundingBox, next.boundingBox) > iouThreshold) iterator.remove()
            }
        }
        return result
    }
    private fun iou(a: BoundingBox, b: BoundingBox): Float {
        val intersection = max(0f, min(a.right, b.right) - max(a.left, b.left)) *
            max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }
    private fun reusableBuffer(bytes: Int): ByteBuffer {
        if (outputBuffer?.capacity() != bytes) outputBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
        return checkNotNull(outputBuffer).also(ByteBuffer::rewind)
    }
    private fun reusableFloats(elements: Int): FloatArray {
        if (outputFloats.size != elements) outputFloats = FloatArray(elements)
        return outputFloats
    }
    private fun elapsed(start: Long) = (System.nanoTime() - start) / 1_000_000L
    private fun loadMappedAsset(context: Context, assetName: String): MappedByteBuffer {
        val descriptor: AssetFileDescriptor = context.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
        }
    }
    private companion object { const val BOX_CHANNELS = 4 }
}

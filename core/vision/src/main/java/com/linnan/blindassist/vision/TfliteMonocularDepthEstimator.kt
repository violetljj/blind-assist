package com.linnan.blindassist.vision

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class TfliteMonocularDepthEstimator(
    context: Context,
    private val modelAssetName: String = MODEL_ASSET,
    private val closerIsLarger: Boolean = true
) : DepthEstimator {
    override val isReady: Boolean get() = interpreter != null
    override val statusMessage: String
        get() = loadError?.message ?: if (isReady) "depth model ready" else "depth model not loaded"

    var lastPreprocessMs: Long = 0L
        private set
    var lastInferenceMs: Long = 0L
        private set
    var lastPostprocessMs: Long = 0L
        private set
    var lastTotalMs: Long = 0L
        private set

    private var interpreter: Interpreter? = null
    private var outputBuffer: ByteBuffer? = null
    private var outputFloats: FloatArray = FloatArray(0)
    private var loadError: Throwable? = null
    private val lifecycleLock = Any()

    init {
        var loadedInterpreter: Interpreter? = null
        try {
            loadedInterpreter = Interpreter(
                loadMappedAsset(context, modelAssetName),
                Interpreter.Options().setNumThreads(4)
            )
            validateInputTensor(loadedInterpreter)
            validateOutputTensor(loadedInterpreter)
            interpreter = loadedInterpreter
            loadedInterpreter = null
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            closeInterpreter(loadedInterpreter)
            Log.w(TAG, "Failed to load depth model: $modelAssetName", error)
            loadError = IllegalStateException(
                "Depth model initialization failed for assets/$modelAssetName",
                error
            )
        }
    }

    override fun estimate(bitmap: Bitmap): DepthFrameResult {
        synchronized(lifecycleLock) {
            val localInterpreter = interpreter ?: return emptyResult()
            val totalStart = System.nanoTime()
            val inputTensor = localInterpreter.getInputTensor(0)
            val outputTensor = localInterpreter.getOutputTensor(0)
            val inputShape = inputTensor.shape()
            val inputHeight = inputShape[1]
            val inputWidth = inputShape[2]

            val preprocessStart = System.nanoTime()
            val input = prepareInput(bitmap, inputWidth, inputHeight)
            lastPreprocessMs = elapsedMs(preprocessStart)

            val localOutputBuffer = reusableOutputBuffer(outputTensor.numBytes())
            val inferenceStart = System.nanoTime()
            localInterpreter.run(input, localOutputBuffer)
            lastInferenceMs = elapsedMs(inferenceStart)
            localOutputBuffer.rewind()

            val postprocessStart = System.nanoTime()
            val floats = reusableOutputFloats(outputTensor.numElements())
            localOutputBuffer.asFloatBuffer().get(floats)
            val depthMap = normalizeDepth(floats, outputTensor.shape())
            lastPostprocessMs = elapsedMs(postprocessStart)
            lastTotalMs = elapsedMs(totalStart)
            return DepthFrameResult(depthMap, currentMetrics())
        }
    }

    override fun close() {
        synchronized(lifecycleLock) {
            closeInterpreter(interpreter)
            interpreter = null
        }
    }

    private fun emptyResult(): DepthFrameResult {
        return DepthFrameResult(
            RelativeDepthMap(width = 1, height = 1, closeness = floatArrayOf(0f)),
            currentMetrics()
        )
    }

    private fun currentMetrics(): DepthEstimatorMetrics {
        return DepthEstimatorMetrics(
            totalMs = lastTotalMs,
            preprocessMs = lastPreprocessMs,
            inferenceMs = lastInferenceMs,
            postprocessMs = lastPostprocessMs,
            modelStatus = statusMessage
        )
    }

    private fun prepareInput(bitmap: Bitmap, inputWidth: Int, inputHeight: Int): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val buffer = ByteBuffer
            .allocateDirect(4 * inputWidth * inputHeight * CHANNELS)
            .order(ByteOrder.nativeOrder())
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixel = scaled.getPixel(x, y)
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                buffer.putFloat((pixel and 0xFF) / 255f)
            }
        }
        if (scaled !== bitmap) {
            scaled.recycle()
        }
        buffer.rewind()
        return buffer
    }

    private fun normalizeDepth(raw: FloatArray, outputShape: IntArray): RelativeDepthMap {
        val (height, width) = outputMapSize(outputShape)
        val count = min(raw.size, width * height)
        var minValue = Float.POSITIVE_INFINITY
        var maxValue = Float.NEGATIVE_INFINITY
        for (index in 0 until count) {
            val value = raw[index]
            if (value.isFinite()) {
                minValue = min(minValue, value)
                maxValue = max(maxValue, value)
            }
        }
        val range = maxValue - minValue
        val normalized = FloatArray(width * height)
        if (!range.isFinite() || range <= 1e-6f) {
            return RelativeDepthMap(width, height, normalized)
        }
        for (index in 0 until count) {
            val value = ((raw[index] - minValue) / range).coerceIn(0f, 1f)
            normalized[index] = if (closerIsLarger) value else 1f - value
        }
        return RelativeDepthMap(width, height, normalized)
    }

    private fun outputMapSize(shape: IntArray): Pair<Int, Int> {
        return when (shape.size) {
            2 -> shape[0] to shape[1]
            3 -> shape[1] to shape[2]
            4 -> if (shape[3] == 1) shape[1] to shape[2] else shape[2] to shape[3]
            else -> error("Unsupported depth output shape: ${shape.contentToString()}")
        }
    }

    private fun validateInputTensor(interpreter: Interpreter) {
        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape()
        require(inputTensor.dataType() == DataType.FLOAT32) {
            "Depth model input must be FLOAT32, actual=${inputTensor.dataType()}"
        }
        require(shape.size == 4 && shape[0] == 1 && shape[3] == CHANNELS) {
            "Depth model input must be NHWC [1,H,W,3], actual=${shape.contentToString()}"
        }
    }

    private fun validateOutputTensor(interpreter: Interpreter) {
        val outputTensor = interpreter.getOutputTensor(0)
        val shape = outputTensor.shape()
        require(outputTensor.dataType() == DataType.FLOAT32) {
            "Depth model output must be FLOAT32, actual=${outputTensor.dataType()}"
        }
        require(shape.size in 2..4) {
            "Depth model output must be a dense depth map, actual=${shape.contentToString()}"
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

    private fun closeInterpreter(interpreterToClose: Interpreter?) {
        try {
            interpreterToClose?.close()
        } catch (error: Throwable) {
            FatalThrowables.rethrowIfFatal(error)
            Log.w(TAG, "Failed to close depth interpreter cleanly.", error)
        }
    }

    private fun loadMappedAsset(context: Context, assetName: String): MappedByteBuffer {
        val descriptor: AssetFileDescriptor = context.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    private fun elapsedMs(startNanos: Long): Long {
        return (System.nanoTime() - startNanos) / 1_000_000L
    }

    companion object {
        const val MODEL_ASSET = "depth/depth_anything_v2_small_fp32.tflite"
        private const val CHANNELS = 3
        private const val TAG = "TfliteDepthEstimator"
    }
}

package com.linnan.blindassist.hftf

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.floor

internal object Dav2PreprocessContract {
    const val INPUT_WIDTH = 640
    const val INPUT_HEIGHT = 480
    const val OUTPUT_WIDTH = 686
    const val OUTPUT_HEIGHT = 518
    const val CHANNELS = 3
    const val INPUT_BYTES = INPUT_WIDTH * INPUT_HEIGHT * CHANNELS
    const val PLANE = OUTPUT_WIDTH * OUTPUT_HEIGHT
    const val OUTPUT_ELEMENTS = PLANE * CHANNELS
    val means = floatArrayOf(0.485f, 0.456f, 0.406f)
    val inverseStd = floatArrayOf(1f / 0.229f, 1f / 0.224f, 1f / 0.225f)
}

internal class Dav2ReferencePreprocessor {
    fun resizeRgb(rgb: ByteArray, outputHwc: FloatArray) {
        require(rgb.size == Dav2PreprocessContract.INPUT_BYTES)
        require(outputHwc.size == Dav2PreprocessContract.OUTPUT_ELEMENTS)
        val scaleX = Dav2PreprocessContract.INPUT_WIDTH.toDouble() / Dav2PreprocessContract.OUTPUT_WIDTH
        val scaleY = Dav2PreprocessContract.INPUT_HEIGHT.toDouble() / Dav2PreprocessContract.OUTPUT_HEIGHT
        for (row in 0 until Dav2PreprocessContract.OUTPUT_HEIGHT) {
            val sourceY = (row + 0.5) * scaleY - 0.5
            val baseY = floor(sourceY).toInt()
            for (column in 0 until Dav2PreprocessContract.OUTPUT_WIDTH) {
                val sourceX = (column + 0.5) * scaleX - 0.5
                val baseX = floor(sourceX).toInt()
                for (channel in 0 until Dav2PreprocessContract.CHANNELS) {
                    var weighted = 0.0
                    for (dy in -1..2) {
                        val y = (baseY + dy).coerceIn(0, Dav2PreprocessContract.INPUT_HEIGHT - 1)
                        val wy = cubicWeight(sourceY - baseY - dy)
                        for (dx in -1..2) {
                            val x = (baseX + dx).coerceIn(0, Dav2PreprocessContract.INPUT_WIDTH - 1)
                            val pixel = rgb[(y * Dav2PreprocessContract.INPUT_WIDTH + x) * 3 + channel]
                                .toInt() and 0xff
                            weighted += pixel * wy * cubicWeight(sourceX - baseX - dx)
                        }
                    }
                    outputHwc[(row * Dav2PreprocessContract.OUTPUT_WIDTH + column) * 3 + channel] =
                        (weighted / 255.0).toFloat()
                }
            }
        }
    }

    fun rgbToNchw(inputHwc: FloatArray, outputNchw: FloatArray) {
        for (pixel in 0 until Dav2PreprocessContract.PLANE) {
            outputNchw[pixel] = inputHwc[pixel * 3]
            outputNchw[Dav2PreprocessContract.PLANE + pixel] = inputHwc[pixel * 3 + 1]
            outputNchw[2 * Dav2PreprocessContract.PLANE + pixel] = inputHwc[pixel * 3 + 2]
        }
    }

    fun normalizeInPlace(nchw: FloatArray) {
        for (channel in 0 until Dav2PreprocessContract.CHANNELS) {
            val start = channel * Dav2PreprocessContract.PLANE
            val end = start + Dav2PreprocessContract.PLANE
            val mean = Dav2PreprocessContract.means[channel]
            val inverseStd = Dav2PreprocessContract.inverseStd[channel]
            for (index in start until end) nchw[index] = (nchw[index] - mean) * inverseStd
        }
    }

    fun normalize(inputNchw: FloatArray, outputNchw: FloatArray) {
        for (channel in 0 until Dav2PreprocessContract.CHANNELS) {
            val start = channel * Dav2PreprocessContract.PLANE
            val end = start + Dav2PreprocessContract.PLANE
            val mean = Dav2PreprocessContract.means[channel]
            val inverseStd = Dav2PreprocessContract.inverseStd[channel]
            for (index in start until end) outputNchw[index] = (inputNchw[index] - mean) * inverseStd
        }
    }

    fun preprocess(rgb: ByteArray, hwc: FloatArray, nchw: FloatArray): FloatArray {
        resizeRgb(rgb, hwc)
        rgbToNchw(hwc, nchw)
        normalizeInPlace(nchw)
        return nchw
    }

    private fun cubicWeight(value: Double): Double {
        val x = abs(value)
        val a = -0.75
        return when {
            x <= 1.0 -> (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
            x < 2.0 -> a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
            else -> 0.0
        }
    }
}

internal class Dav2KotlinTablePreprocessor {
    val output: ByteBuffer = ByteBuffer.allocateDirect(Dav2PreprocessContract.OUTPUT_ELEMENTS * 4)
        .order(ByteOrder.nativeOrder())
    private val floats: FloatBuffer = output.asFloatBuffer()
    private val xIndices = IntArray(Dav2PreprocessContract.OUTPUT_WIDTH * 4)
    private val xWeights = FloatArray(Dav2PreprocessContract.OUTPUT_WIDTH * 4)
    private val yIndices = IntArray(Dav2PreprocessContract.OUTPUT_HEIGHT * 4)
    private val yWeights = FloatArray(Dav2PreprocessContract.OUTPUT_HEIGHT * 4)

    init {
        buildTable(
            Dav2PreprocessContract.INPUT_WIDTH,
            Dav2PreprocessContract.OUTPUT_WIDTH,
            xIndices,
            xWeights,
        )
        buildTable(
            Dav2PreprocessContract.INPUT_HEIGHT,
            Dav2PreprocessContract.OUTPUT_HEIGHT,
            yIndices,
            yWeights,
        )
    }

    fun preprocess(rgb: ByteArray): ByteBuffer {
        require(rgb.size == Dav2PreprocessContract.INPUT_BYTES)
        val width = Dav2PreprocessContract.OUTPUT_WIDTH
        val sourceWidth = Dav2PreprocessContract.INPUT_WIDTH
        for (row in 0 until Dav2PreprocessContract.OUTPUT_HEIGHT) {
            val yOffset = row * 4
            for (column in 0 until width) {
                val xOffset = column * 4
                var red = 0f
                var green = 0f
                var blue = 0f
                for (yi in 0 until 4) {
                    val sourceRow = yIndices[yOffset + yi] * sourceWidth
                    val wy = yWeights[yOffset + yi]
                    for (xi in 0 until 4) {
                        val pixel = (sourceRow + xIndices[xOffset + xi]) * 3
                        val weight = wy * xWeights[xOffset + xi] / 255f
                        red += (rgb[pixel].toInt() and 0xff) * weight
                        green += (rgb[pixel + 1].toInt() and 0xff) * weight
                        blue += (rgb[pixel + 2].toInt() and 0xff) * weight
                    }
                }
                val index = row * width + column
                floats.put(index, (red - Dav2PreprocessContract.means[0]) * Dav2PreprocessContract.inverseStd[0])
                floats.put(
                    Dav2PreprocessContract.PLANE + index,
                    (green - Dav2PreprocessContract.means[1]) * Dav2PreprocessContract.inverseStd[1],
                )
                floats.put(
                    2 * Dav2PreprocessContract.PLANE + index,
                    (blue - Dav2PreprocessContract.means[2]) * Dav2PreprocessContract.inverseStd[2],
                )
            }
        }
        output.rewind()
        return output
    }

    private fun buildTable(
        inputSize: Int,
        outputSize: Int,
        indices: IntArray,
        weights: FloatArray,
    ) {
        val scale = inputSize.toFloat() / outputSize
        for (destination in 0 until outputSize) {
            val source = (destination + 0.5f) * scale - 0.5f
            val base = floor(source).toInt()
            for (tap in 0 until 4) {
                val delta = tap - 1
                indices[destination * 4 + tap] = (base + delta).coerceIn(0, inputSize - 1)
                weights[destination * 4 + tap] = cubicWeight(source - base - delta)
            }
        }
    }

    private fun cubicWeight(value: Float): Float {
        val x = abs(value)
        val a = -0.75f
        return when {
            x <= 1f -> (a + 2f) * x * x * x - (a + 3f) * x * x + 1f
            x < 2f -> a * x * x * x - 5f * a * x * x + 8f * a * x - 4f * a
            else -> 0f
        }
    }
}

internal class Dav2NativePreprocessor : AutoCloseable {
    val fp32Output: ByteBuffer = ByteBuffer.allocateDirect(Dav2PreprocessContract.OUTPUT_ELEMENTS * 4)
        .order(ByteOrder.nativeOrder())
    val fp16Output: ByteBuffer = ByteBuffer.allocateDirect(Dav2PreprocessContract.OUTPUT_ELEMENTS * 2)
        .order(ByteOrder.nativeOrder())
    val resizedHwcFp32Output: ByteBuffer = ByteBuffer.allocateDirect(Dav2PreprocessContract.OUTPUT_ELEMENTS * 4)
        .order(ByteOrder.nativeOrder())
    private var handle = nativeCreate()

    fun preprocessFp32(rgb: ByteArray): ByteBuffer {
        check(handle != 0L)
        fp32Output.clear()
        nativeRun(handle, rgb, fp32Output, false)
        fp32Output.position(0)
        fp32Output.limit(Dav2PreprocessContract.OUTPUT_ELEMENTS * 4)
        return fp32Output
    }

    fun preprocessFp16(rgb: ByteArray): ByteBuffer {
        return preprocessFp16CanonicalStrict(rgb)
    }

    /** Reproduces the official Python/OpenCV operation order before the final FP32 cast. */
    fun preprocessFp32Canonical(rgb: ByteArray): ByteBuffer {
        check(handle != 0L)
        fp32Output.clear()
        nativeRunOfficial(handle, rgb, fp32Output)
        fp32Output.position(0)
        fp32Output.limit(Dav2PreprocessContract.OUTPUT_ELEMENTS * 4)
        return fp32Output
    }

    fun preprocessFp16CanonicalStrict(rgb: ByteArray): ByteBuffer {
        preprocessFp32Canonical(rgb)
        fp16Output.clear()
        nativeConvertFp32ToFp16(fp32Output, fp16Output, Dav2PreprocessContract.OUTPUT_ELEMENTS)
        fp16Output.position(0)
        fp16Output.limit(Dav2PreprocessContract.OUTPUT_ELEMENTS * 2)
        return fp16Output
    }

    /**
     * Runs the admitted FP32 OpenCV/NEON path first, then performs an integer,
     * bit-exact IEEE-754 round-to-nearest-ties-to-even binary32 -> binary16
     * conversion. This deliberately does not rely on the active FP rounding
     * mode, the compiler's fast-math transformations, or ARM FCVT rounding.
     */
    fun preprocessFp16Strict(rgb: ByteArray): ByteBuffer {
        check(handle != 0L)
        fp32Output.clear()
        nativeRun(handle, rgb, fp32Output, false)
        fp32Output.position(0)
        fp32Output.limit(Dav2PreprocessContract.OUTPUT_ELEMENTS * 4)
        fp16Output.clear()
        nativeConvertFp32ToFp16(fp32Output, fp16Output, Dav2PreprocessContract.OUTPUT_ELEMENTS)
        fp16Output.position(0)
        fp16Output.limit(Dav2PreprocessContract.OUTPUT_ELEMENTS * 2)
        return fp16Output
    }

    /** Retained only as a diagnostic control for the previously failed fused arm. */
    fun preprocessFp16Fused(rgb: ByteArray): ByteBuffer {
        check(handle != 0L)
        fp16Output.clear()
        nativeRun(handle, rgb, fp16Output, true)
        fp16Output.position(0)
        fp16Output.limit(Dav2PreprocessContract.OUTPUT_ELEMENTS * 2)
        return fp16Output
    }

    /** Copies the last OpenCV cubic-resize result before normalization/CHW packing. */
    fun copyLastResizedHwcFp32(): ByteBuffer {
        check(handle != 0L)
        resizedHwcFp32Output.clear()
        nativeCopyLastResizedHwcFp32(handle, resizedHwcFp32Output)
        resizedHwcFp32Output.position(0)
        resizedHwcFp32Output.limit(Dav2PreprocessContract.OUTPUT_ELEMENTS * 4)
        return resizedHwcFp32Output
    }

    fun convertFp32ToFp16Strict(input: ByteBuffer, elements: Int): ByteBuffer {
        check(handle != 0L)
        require(input.isDirect) { "FP32 input must be a direct buffer" }
        require(elements >= 0 && input.limit() >= elements * 4)
        require(fp16Output.capacity() >= elements * 2)
        nativeConvertFp32ToFp16(input, fp16Output, elements)
        fp16Output.position(0)
        fp16Output.limit(elements * 2)
        return fp16Output
    }

    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    private external fun nativeCreate(): Long
    private external fun nativeRun(handle: Long, input: ByteArray, output: ByteBuffer, fp16: Boolean)
    private external fun nativeRunOfficial(handle: Long, input: ByteArray, output: ByteBuffer)
    private external fun nativeConvertFp32ToFp16(input: ByteBuffer, output: ByteBuffer, elements: Int)
    private external fun nativeCopyLastResizedHwcFp32(handle: Long, output: ByteBuffer)
    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("dav2_preprocess_native")
        }
    }
}

internal fun floatToHalfBits(value: Float): Short = android.util.Half.toHalf(value)

internal fun halfBitsToFloat(value: Short): Float = android.util.Half.toFloat(value)

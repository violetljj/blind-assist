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
    private var handle = nativeCreate()

    fun preprocessFp32(rgb: ByteArray): ByteBuffer {
        check(handle != 0L)
        nativeRun(handle, rgb, fp32Output, false)
        fp32Output.rewind()
        return fp32Output
    }

    fun preprocessFp16(rgb: ByteArray): ByteBuffer {
        check(handle != 0L)
        nativeRun(handle, rgb, fp16Output, true)
        fp16Output.rewind()
        return fp16Output
    }

    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    private external fun nativeCreate(): Long
    private external fun nativeRun(handle: Long, input: ByteArray, output: ByteBuffer, fp16: Boolean)
    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("dav2_preprocess_native")
        }
    }
}

internal fun floatToHalfBits(value: Float): Short {
    val bits = value.toRawBits()
    val sign = (bits ushr 16) and 0x8000
    var exponent = ((bits ushr 23) and 0xff) - 127 + 15
    var mantissa = bits and 0x7fffff
    if (exponent <= 0) {
        if (exponent < -10) return sign.toShort()
        mantissa = (mantissa or 0x800000) shr (1 - exponent)
        return (sign or ((mantissa + 0x1000) shr 13)).toShort()
    }
    if (exponent >= 31) return (sign or 0x7c00 or if (mantissa != 0) 0x0200 else 0).toShort()
    mantissa += 0x1000
    if ((mantissa and 0x800000) != 0) {
        mantissa = 0
        exponent++
        if (exponent >= 31) return (sign or 0x7c00).toShort()
    }
    return (sign or (exponent shl 10) or (mantissa shr 13)).toShort()
}

internal fun halfBitsToFloat(value: Short): Float {
    val bits = value.toInt() and 0xffff
    val sign = (bits and 0x8000) shl 16
    var exponent = (bits ushr 10) and 0x1f
    var mantissa = bits and 0x03ff
    val floatBits = when {
        exponent == 0 -> {
            if (mantissa == 0) {
                sign
            } else {
                while ((mantissa and 0x0400) == 0) {
                    mantissa = mantissa shl 1
                    exponent--
                }
                mantissa = mantissa and 0x03ff
                sign or ((exponent + 127 - 15 + 1) shl 23) or (mantissa shl 13)
            }
        }
        exponent == 0x1f -> sign or 0x7f800000.toInt() or (mantissa shl 13)
        else -> sign or ((exponent + 127 - 15) shl 23) or (mantissa shl 13)
    }
    return Float.fromBits(floatBits)
}

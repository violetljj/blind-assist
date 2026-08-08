package com.linnan.blindassist.benchmark

import android.content.Context
import android.graphics.Bitmap
import com.linnan.blindassist.risk.DenseSemanticMask
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

/**
 * Benchmark-APK-only adapter for the four-class INT8 traversability contract.
 * It deliberately lives outside :app so it cannot change the production YOLO
 * detector path or ship in the normal application package.
 */
class TfliteTraversabilitySegmenter(
    context: Context,
    assetName: String,
    private val expectedSize: Int = INPUT_SIZE
) : AutoCloseable {
    private val interpreter: Interpreter
    private val inputType: DataType
    private val outputType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val outputWidth: Int
    private val outputHeight: Int

    init {
        val model = loadMappedAsset(context, assetName)
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))
        interpreter.allocateTensors()
        val input = interpreter.getInputTensor(0)
        val output = interpreter.getOutputTensor(0)
        require(input.shape().contentEquals(intArrayOf(1, expectedSize, expectedSize, 3))) {
            "Expected [1,$expectedSize,$expectedSize,3] segmentation input, got ${input.shape().toList()}"
        }
        require(output.shape().size == 4 && output.shape()[0] == 1 && output.shape()[3] == CLASS_COUNT) {
            "Expected [1,H,W,$CLASS_COUNT] segmentation output, got ${output.shape().toList()}"
        }
        inputType = input.dataType()
        outputType = output.dataType()
        require(inputType == DataType.UINT8 || inputType == DataType.INT8) { "INT8 model required; input=$inputType" }
        require(outputType == DataType.UINT8 || outputType == DataType.INT8) { "INT8 model required; output=$outputType" }
        inputScale = input.quantizationParams().scale
        inputZeroPoint = input.quantizationParams().zeroPoint
        require(inputScale > 0f) { "Quantized input scale must be positive" }
        require(output.quantizationParams().scale > 0f) { "Quantized output scale must be positive" }
        outputHeight = output.shape()[1]
        outputWidth = output.shape()[2]
    }

    fun segment(source: Bitmap): DenseSemanticMask {
        val bitmap = if (source.width == expectedSize && source.height == expectedSize) source else {
            Bitmap.createScaledBitmap(source, expectedSize, expectedSize, true)
        }
        try {
            val pixels = IntArray(expectedSize * expectedSize)
            bitmap.getPixels(pixels, 0, expectedSize, 0, 0, expectedSize, expectedSize)
            val input = ByteBuffer.allocateDirect(expectedSize * expectedSize * 3).order(ByteOrder.nativeOrder())
            pixels.forEach { pixel ->
                putQuantized(input, (pixel shr 16) and 0xFF)
                putQuantized(input, (pixel shr 8) and 0xFF)
                putQuantized(input, pixel and 0xFF)
            }
            input.rewind()
            val output = ByteBuffer.allocateDirect(outputHeight * outputWidth * CLASS_COUNT).order(ByteOrder.nativeOrder())
            interpreter.run(input, output)
            output.rewind()
            val classes = IntArray(outputHeight * outputWidth)
            for (index in classes.indices) {
                var bestClass = 0
                var bestScore = Int.MIN_VALUE
                repeat(CLASS_COUNT) { classIndex ->
                    val score = if (outputType == DataType.UINT8) output.get().toInt() and 0xFF else output.get().toInt()
                    if (score > bestScore) {
                        bestScore = score
                        bestClass = classIndex
                    }
                }
                classes[index] = bestClass
            }
            return DenseSemanticMask(outputWidth, outputHeight, classes)
        } finally {
            if (bitmap !== source) bitmap.recycle()
        }
    }

    override fun close() = interpreter.close()

    private fun putQuantized(buffer: ByteBuffer, channel: Int) {
        val q = (channel / inputScale + inputZeroPoint).roundToInt()
        buffer.put(
            when (inputType) {
                DataType.UINT8 -> q.coerceIn(0, 255).toByte()
                DataType.INT8 -> q.coerceIn(-128, 127).toByte()
                else -> error("validated above")
            }
        )
    }

    companion object {
        const val INPUT_SIZE = 256
        const val CLASS_COUNT = 4

        private fun loadMappedAsset(context: Context, assetName: String): MappedByteBuffer {
            val descriptor = context.assets.openFd(assetName)
            FileInputStream(descriptor.fileDescriptor).use { stream ->
                return stream.channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }
    }
}

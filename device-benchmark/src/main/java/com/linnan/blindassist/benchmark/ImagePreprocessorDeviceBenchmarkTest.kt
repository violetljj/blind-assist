package com.linnan.blindassist.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.ImagePreprocessor
import com.linnan.blindassist.vision.RgbaVisionFrame
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.ceil

@RunWith(AndroidJUnit4::class)
class ImagePreprocessorDeviceBenchmarkTest {
    @Test
    fun prepareRgba1440x1080Rotation90_reportsNanosecondTiming() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val frame = SyntheticRgbaFrame(
            width = SOURCE_WIDTH,
            height = SOURCE_HEIGHT,
            rotationDegrees = ROTATION_DEGREES,
            rowStride = SOURCE_WIDTH * RGBA_CHANNELS
        )
        val preprocessor = ImagePreprocessor(INPUT_SIZE)

        repeat(WARMUP_ITERATIONS) {
            preprocessor.prepare(frame)
        }

        val samplesNanos = LongArray(MEASURED_ITERATIONS)
        repeat(MEASURED_ITERATIONS) { index ->
            val startedAt = System.nanoTime()
            preprocessor.prepare(frame)
            samplesNanos[index] = System.nanoTime() - startedAt
        }

        val firstHash = sha256(preprocessor.prepare(frame).buffer)
        val secondHash = sha256(preprocessor.prepare(frame).buffer)
        assertEquals("preprocessor output changed for identical input", firstHash, secondHash)
        assertTrue("preprocessor returned no timing samples", samplesNanos.isNotEmpty())

        val result = JSONObject()
            .put("schema", "blindassist_image_preprocessor_device_benchmark_v1")
            .put("device_under_test", android.os.Build.MODEL)
            .put("source_width", SOURCE_WIDTH)
            .put("source_height", SOURCE_HEIGHT)
            .put("source_row_stride", frame.rowStride)
            .put("source_pixel_stride", frame.pixelStride)
            .put("rotation_degrees", ROTATION_DEGREES)
            .put("input_size", INPUT_SIZE)
            .put("warmup_iterations", WARMUP_ITERATIONS)
            .put("measured_iterations", MEASURED_ITERATIONS)
            .put("prepare_p50_us", percentileNanos(samplesNanos, 50.0) / NANOS_PER_MICROSECOND)
            .put("prepare_p95_us", percentileNanos(samplesNanos, 95.0) / NANOS_PER_MICROSECOND)
            .put("prepare_p99_us", percentileNanos(samplesNanos, 99.0) / NANOS_PER_MICROSECOND)
            .put("prepare_max_us", samplesNanos.max() / NANOS_PER_MICROSECOND)
            .put("output_sha256", firstHash)

        val output = File(
            checkNotNull(instrumentation.targetContext.getExternalFilesDir(null)),
            RESULT_RELATIVE_PATH
        )
        output.parentFile?.mkdirs()
        output.writeText(result.toString(2), Charsets.UTF_8)
    }

    private fun percentileNanos(samples: LongArray, percentile: Double): Double {
        val ordered = samples.sortedArray()
        val index = ceil(ordered.size * percentile / 100.0).toInt().coerceIn(1, ordered.size) - 1
        return ordered[index].toDouble()
    }

    private fun sha256(buffer: ByteBuffer): String {
        val source = buffer.duplicate().also { it.rewind() }
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(source)
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private class SyntheticRgbaFrame(
        override val width: Int,
        override val height: Int,
        override val rotationDegrees: Int,
        override val rowStride: Int
    ) : RgbaVisionFrame {
        override val pixelStride: Int = RGBA_CHANNELS
        override val buffer: ByteBuffer = ByteBuffer
            .allocateDirect(rowStride * height)
            .order(ByteOrder.nativeOrder())
            .also { target ->
                for (y in 0 until height) {
                    for (x in 0 until width) {
                        val offset = y * rowStride + x * pixelStride
                        target.put(offset, ((x * 31 + y * 7) and 0xFF).toByte())
                        target.put(offset + 1, ((x * 13 + y * 29) and 0xFF).toByte())
                        target.put(offset + 2, ((x * 19 + y * 17) and 0xFF).toByte())
                        target.put(offset + 3, 0xFF.toByte())
                    }
                }
                target.rewind()
            }

        override fun close() = Unit
    }

    private companion object {
        const val SOURCE_WIDTH = 1440
        const val SOURCE_HEIGHT = 1080
        const val ROTATION_DEGREES = 90
        const val INPUT_SIZE = 320
        const val RGBA_CHANNELS = 4
        const val WARMUP_ITERATIONS = 200
        const val MEASURED_ITERATIONS = 1_000
        const val NANOS_PER_MICROSECOND = 1_000.0
        const val RESULT_RELATIVE_PATH = "benchmark-results/image-preprocessor-device-benchmark.json"
    }
}

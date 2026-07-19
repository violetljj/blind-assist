package com.linnan.blindassist.benchmark

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.model.BoundingBox
import com.linnan.blindassist.model.Detection
import com.linnan.blindassist.model.FrameSize
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

/**
 * Measures only the candidate's post-YOLO increment: feature extraction, INT8 input conversion
 * and an untrained TCN inference. This test has no camera, labels, risk interpretation or alert
 * side effect; the model is a performance fixture, not a candidate safety model.
 */
@RunWith(AndroidJUnit4::class)
class YoloImuEventHeadIntegrationTest {
    @Test
    fun logsPostYoloCandidateIncrementWithoutAlerting() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val model = mapAsset(context, MODEL_ASSET)
        Interpreter(model, Interpreter.Options().setNumThreads(1).setUseXNNPACK(true)).use { interpreter ->
            interpreter.allocateTensors()
            val window = YoloImuCausalWindow()
            repeat(CONTEXT) { index ->
                window.append(frameAtIndex(index.toLong() + 1L))
            }
            repeat(WARMUP) { index -> runIncrement(interpreter, window, (index + CONTEXT + 1).toLong()) }
            val samples = DoubleArray(MEASURED) { index ->
                val start = SystemClock.elapsedRealtimeNanos()
                runIncrement(interpreter, window, (index + WARMUP + CONTEXT + 1).toLong())
                (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
            }.sorted()
            val p50 = samples[samples.size / 2]
            val p95 = samples[(samples.size * 95 / 100).coerceAtMost(samples.lastIndex)]
            Log.i(TAG, "samples=$MEASURED p50_ms=%.4f p95_ms=%.4f".format(java.util.Locale.US, p50, p95))
            assertTrue("p95_ms=$p95", p95 < INCREMENTAL_P95_BUDGET_MS)
        }
    }

    private fun runIncrement(interpreter: Interpreter, window: YoloImuCausalWindow, timestampNanos: Long) {
        val sequence = requireNotNull(window.append(frameAtIndex(timestampNanos)))
        val motion = quantize(sequence.motionSequence, MOTION_SCALE, MOTION_ZERO)
        val spatial = quantize(sequence.spatialSequence, SPATIAL_SCALE, SPATIAL_ZERO)
        val outputs = hashMapOf<Int, Any>(
            0 to output(8), 1 to output(4), 2 to output(1), 3 to output(3), 4 to output(1)
        )
        interpreter.runForMultipleInputsOutputs(arrayOf(motion, spatial), outputs)
    }

    private fun frameAtIndex(index: Long): YoloImuFeatureFrame = YoloImuFeatureFrame(
        timestampNanos = index * FRAME_INTERVAL_NANOS,
        detections = listOf(
            detection(390f, 580f, 570f, 900f, 1),
            detection(90f, 520f, 250f, 760f, 2),
            detection(700f, 440f, 820f, 680f, 3),
            detection(470f, 230f, 560f, 410f, 4)
        ),
        imu = YoloImuMotion(0.01f, -0.01f, 0f, 0.002f, observed = true)
    )

    private fun detection(left: Float, top: Float, right: Float, bottom: Float, classId: Int) = Detection(
        classId = classId,
        label = "benchmark",
        confidence = 0.9f,
        boundingBox = BoundingBox(left, top, right, bottom),
        frameSize = FrameSize(1_000, 1_000)
    )

    private fun quantize(values: FloatArray, scale: Float, zero: Int): ByteBuffer = ByteBuffer.allocateDirect(values.size).order(ByteOrder.nativeOrder()).apply {
        values.forEach { value -> put((value / scale).roundToInt().plus(zero).coerceIn(-128, 127).toByte()) }
        rewind()
    }

    private fun output(count: Int): ByteBuffer = ByteBuffer.allocateDirect(count).order(ByteOrder.nativeOrder())

    private fun mapAsset(context: android.content.Context, asset: String): MappedByteBuffer =
        context.assets.openFd(asset).use { descriptor ->
            java.io.FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }

    private companion object {
        const val MODEL_ASSET = "corridor_causal_tcn_int8_v0.tflite"
        const val CONTEXT = 8
        const val WARMUP = 100
        const val MEASURED = 1_000
        const val INCREMENTAL_P95_BUDGET_MS = 1.0
        const val FRAME_INTERVAL_NANOS = 100_000_000L
        const val MOTION_SCALE = 0.015686275f
        const val MOTION_ZERO = -1
        const val SPATIAL_SCALE = 0.0039215595f
        const val SPATIAL_ZERO = -128
        const val TAG = "YoloImuEventHeadBench"
    }
}

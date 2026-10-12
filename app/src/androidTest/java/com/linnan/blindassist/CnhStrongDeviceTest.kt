package com.linnan.blindassist

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.linnan.blindassist.vision.CnhStrongEngine
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Synthetic arithmetic/ORT parity only: consumes no captured scene or evaluation data.
 * Push exporter outputs into targetContext.filesDir/fusion-experiment before running.
 */
@RunWith(AndroidJUnit4::class)
class CnhStrongDeviceTest {
    @Test fun frozenFiveSeedStrongMatchesPythonOnDevice() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.noBackupFilesDir, "fusion-experiment")
        fun buffer(name: String) = ByteBuffer.wrap(File(directory, name).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        fun floats(name: String): FloatArray {
            val b = buffer(name)
            return FloatArray(b.remaining() / 4) { b.float }
        }
        val hist = floats("fixture_histogram.bin")
        val ambient = floats("fixture_ambient.bin")
        val scoreBuffer = buffer("fixture_scores.bin")
        val scores = DoubleArray(scoreBuffer.remaining() / 8) { scoreBuffer.double }
        var maxError = 0.0
        val timings = mutableListOf<Double>()
        CnhStrongEngine(File(directory, "cnh_strong_m3.onnx"), File(directory, "cnh_strong_calibration.bin")).use { engine ->
            listOf(-13.25 to 7.5, 0.0 to 0.0, -35.0 to -12.0).forEachIndexed { pose, (pitch, roll) ->
                engine.reset()
                repeat(8) { frame ->
                    val start = System.nanoTime()
                    val result = engine.process(hist.copyOfRange(frame * 1024, (frame + 1) * 1024),
                        ambient.copyOfRange(frame * 64, (frame + 1) * 64), frame.toLong(), frame * 200L, pitch, roll)
                    timings += (System.nanoTime() - start) / 1e6
                    assertEquals(frame == 7, result.ready)
                    assertEquals(frame + 1, result.historyLength)
                    if (frame >= 3) {
                        val expectedHead = scores[(pose * 8 + frame) * 2]
                        val expectedBody = scores[(pose * 8 + frame) * 2 + 1]
                        maxError = maxOf(maxError, abs(expectedHead - result.headScore), abs(expectedBody - result.bodyScore))
                        assertEquals(expectedHead, result.headScore, 2e-4)
                        assertEquals(expectedBody, result.bodyScore, 2e-4)
                        assertEquals(frame == 7 && expectedHead >= 8.768392178320116, result.headStrong)
                        assertEquals(frame == 7 && expectedBody >= 8.768392178320116, result.bodyStrong)
                    }
                }
            }
        }
        Log.i("CnhStrongDeviceTest", "synthetic parity maxScoreError=$maxError; processMillis=$timings")
    }
}
